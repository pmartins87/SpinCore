from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import torch

import spincore.r7_5_representation_v3_stage as stage
from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_stage import (
    frozen_config,
    load_phase2_v3_runtime,
    new_phase2_v3_runtime,
    save_phase2_v3_runtime,
)
from spincore.r7_5_representation_v3_stage_contract import (
    DOMAINS,
    ITERATIONS,
    TORCH_THREADS,
    TRAINING_SEEDS,
    deck_seed as production_deck_seed,
)
from spincore.solver import SolverLibrary

SCHEMA = "SPINCORE_R7_5_3C_SAMPLING_SPLIT_STAGE_CELL_V1"
REPRESENTATIONS = (H2_FINAL, H3_FINAL)
COLLECTOR_XOR = 0xC0FFEE
FIXED_LEARNING_SEED = 1801739323


def _fresh_traversal_rng(seed: int) -> random.Random:
    return random.Random(int(seed) ^ COLLECTOR_XOR)


def _restore_traversal_rng(state: dict, traversal_seed: int) -> random.Random:
    if int(state.get("diagnostic_traversal_seed", -1)) != int(traversal_seed):
        raise RuntimeError("diagnostic traversal-seed identity drift")
    raw = state.get("diagnostic_traversal_rng_state")
    if raw is None:
        raise RuntimeError("missing diagnostic traversal RNG state")
    rng = random.Random()
    rng.setstate(raw)
    return rng


def _run_iteration_with_deck(*, deck_schedule_seed: int, **kwargs):
    original = stage.deck_seed
    stage.deck_seed = lambda _ignored_training_seed, global_root, iteration: production_deck_seed(
        int(deck_schedule_seed), int(global_root), int(iteration)
    )
    try:
        return stage.run_one_phase2_v3_iteration(**kwargs)
    finally:
        stage.deck_seed = original


def main() -> int:
    ap = argparse.ArgumentParser(description="R7.5.3C deck-vs-traversal sampling split staged worker")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    ap.add_argument("--domain", choices=DOMAINS, required=True)
    ap.add_argument("--deck-seed", type=int, choices=TRAINING_SEEDS, required=True)
    ap.add_argument("--traversal-seed", type=int, choices=TRAINING_SEEDS, required=True)
    ap.add_argument("--target-iteration", type=int, choices=tuple(range(1, ITERATIONS + 1)), required=True)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--checkpoint-out", type=Path, required=True)
    ap.add_argument("--report-out", type=Path, required=True)
    ap.add_argument("--execution-sha", required=True)
    args = ap.parse_args()

    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("torch thread contract drift")
    config = frozen_config()
    solver = SolverLibrary(args.solver)

    if args.resume:
        bundle, session, behavior, _spec, state = load_phase2_v3_runtime(
            args.resume,
            repo_root=args.repo_root,
            solver=solver,
            representation=args.representation,
            domain=args.domain,
            training_seed=FIXED_LEARNING_SEED,
            config=config,
            execution_sha=args.execution_sha,
        )
        if int(state.get("diagnostic_fixed_learning_seed", -1)) != FIXED_LEARNING_SEED:
            raise RuntimeError("fixed learning-seed identity drift")
        if int(state.get("diagnostic_deck_seed", -1)) != int(args.deck_seed):
            raise RuntimeError("diagnostic deck-seed identity drift")
        traversal_rng = _restore_traversal_rng(state, int(args.traversal_seed))
    else:
        if int(args.target_iteration) != 1:
            raise SystemExit("fresh sampling-split cell must start at iteration 1")
        bundle, session, behavior, _spec, state = new_phase2_v3_runtime(
            args.repo_root,
            solver=solver,
            representation=args.representation,
            domain=args.domain,
            training_seed=FIXED_LEARNING_SEED,
            config=config,
        )
        traversal_rng = _fresh_traversal_rng(int(args.traversal_seed))
        state["diagnostic_fixed_learning_seed"] = FIXED_LEARNING_SEED
        state["diagnostic_deck_seed"] = int(args.deck_seed)
        state["diagnostic_traversal_seed"] = int(args.traversal_seed)
        state["diagnostic_rng_split"] = {
            "deck": "exact solver deck schedule only",
            "traversal": "collector stochastic opponent/action and strategy-path draws only",
            "learning_memory": "fixed at one precommitted original training seed",
            "final_policy": "not trained here; one common fixed learner is applied later",
        }

    session.collector.rng = traversal_rng
    started = time.perf_counter()
    iteration_report = _run_iteration_with_deck(
        deck_schedule_seed=int(args.deck_seed),
        bundle=bundle,
        session=session,
        behavior=behavior,
        state=state,
        config=config,
        target_iteration=int(args.target_iteration),
    )
    compute_seconds = time.perf_counter() - started
    state["diagnostic_traversal_rng_state"] = session.collector.rng.getstate()

    save_started = time.perf_counter()
    save_phase2_v3_runtime(
        args.checkpoint_out,
        bundle=bundle,
        behavior=behavior,
        state=state,
        config=config,
        execution_sha=args.execution_sha,
        finalized=False,
        final_report=None,
    )
    checkpoint_seconds = time.perf_counter() - save_started

    payload = {
        "schema": SCHEMA,
        "execution_sha": args.execution_sha,
        "representation": args.representation,
        "domain": args.domain,
        "deck_seed": int(args.deck_seed),
        "traversal_seed": int(args.traversal_seed),
        "fixed_learning_seed": FIXED_LEARNING_SEED,
        "target_iteration": int(args.target_iteration),
        "config": config.to_dict(),
        "iteration_report": iteration_report,
        "compute_wall_seconds": float(compute_seconds),
        "checkpoint_write_seconds": float(checkpoint_seconds),
        "strategy_memory": {"items": len(bundle.pol_mem.items), "seen": int(bundle.pol_mem.seen)},
        "advantage_memory": {"items": len(bundle.adv_mem.items), "seen": int(bundle.adv_mem.seen)},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "diagnostic_only": True,
        "representation_winner": None,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
