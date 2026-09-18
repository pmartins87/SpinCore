#!/usr/bin/env python3
from __future__ import annotations

"""Read-only mechanics smoke for the HU-preflop board-averaging collector.

The same deterministic HU roots are collected twice from the Stage-B model:
canonical K=1 and experimental K=4. The smoke verifies the causal contract:

- sample count/order/identity are unchanged;
- canonical-board postflop targets are unchanged;
- at least one preflop target changes because it is averaged across boards;
- K=4 consumes more traversal nodes;
- no checkpoint/reservoir/optimizer state is saved.

This is not a strength test.
"""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.lean_functional_training import (  # noqa: E402
    _root_deck_seed,
    _root_policy_seed,
    load_checkpoint,
)
from spincore.lean_parallel import (  # noqa: E402
    RootJob,
    _collect_chunk,
    _sample_street,
    _worker_init,
)
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler  # noqa: E402
from spincore.solver import SolverLibrary  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--roots", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _same_identity(a, b) -> bool:
    return (
        a.observation == b.observation
        and tuple(a.legal) == tuple(b.legal)
        and float(a.weight) == float(b.weight)
        and int(a.iteration) == int(b.iteration)
    )


def _target_max_abs_delta(a, b) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a.target, b.target))


def main() -> int:
    args = parse_args()
    if int(args.roots) <= 0:
        raise SystemExit("--roots must be positive")

    solver_path = args.solver.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    solver = SolverLibrary(solver_path)
    seed, config, completed, _sampler0, runtimes, _history, finalized = load_checkpoint(
        checkpoint, solver=solver
    )
    if not finalized or int(completed) != int(config.iterations):
        raise SystemExit("expected finalized Stage-B checkpoint")
    if int(config.exact_opponent_levels) != 0:
        raise SystemExit("smoke expects Stage-B exact_opponent_levels=0")

    runtime = runtimes["TRUE_HEADS_UP"]
    iteration = int(completed) + 1
    sampler = LegacyScenarioSampler(
        seed=int(seed) ^ 0xB04D5A0F,
        config=LegacyScenarioConfig(heads_up_prob=1.0),
    )
    jobs = []
    for root_index in range(int(args.roots)):
        episode = sampler.sample_episode(force_domain="TRUE_HEADS_UP")
        jobs.append(
            RootJob(
                root_index=int(root_index),
                episode=episode,
                deck_seed=_root_deck_seed(
                    int(seed), "TRUE_HEADS_UP", int(iteration), int(root_index)
                ),
                policy_seed=_root_policy_seed(
                    int(seed), "TRUE_HEADS_UP", int(iteration), int(root_index)
                ),
            )
        )

    model_state = {
        key: value.detach().cpu().clone()
        for key, value in runtime.bundle.advantage.state_dict().items()
    }
    model_ready = bool(runtime.bundle.counters.get("advantage_ready", 0))

    # _collect_chunk is normally entered inside a spawned worker. Initialize the
    # same worker-local solver/thread contract explicitly for this read-only smoke.
    _worker_init(str(solver_path))

    canonical = _collect_chunk(
        "TRUE_HEADS_UP",
        int(runtime.bundle.seed),
        model_state,
        model_ready,
        int(iteration),
        0,
        1,
        tuple(jobs),
    )
    averaged = _collect_chunk(
        "TRUE_HEADS_UP",
        int(runtime.bundle.seed),
        model_state,
        model_ready,
        int(iteration),
        0,
        4,
        tuple(jobs),
    )

    if len(canonical) != len(averaged) or len(canonical) != len(jobs):
        raise RuntimeError("root-count drift between K1 and K4")

    total_nodes_k1 = 0
    total_nodes_k4 = 0
    total_samples = 0
    preflop_samples = 0
    preflop_changed = 0
    postflop_samples = 0
    postflop_nonidentical = 0
    max_preflop_target_delta = 0.0
    rows = []

    for one, four in zip(canonical, averaged):
        if int(one.root_index) != int(four.root_index):
            raise RuntimeError("root ordering drift")
        if len(one.samples) != len(four.samples):
            raise RuntimeError("sample-count drift between K1 and K4")

        root_pre = root_changed = root_post = root_post_bad = 0
        root_max_delta = 0.0
        for s1, s4 in zip(one.samples, four.samples):
            if not _same_identity(s1, s4):
                raise RuntimeError("sample identity/order drift between K1 and K4")
            delta = _target_max_abs_delta(s1, s4)
            street = _sample_street(s1)
            if street == 0:
                preflop_samples += 1
                root_pre += 1
                max_preflop_target_delta = max(max_preflop_target_delta, delta)
                root_max_delta = max(root_max_delta, delta)
                if delta > 1e-12:
                    preflop_changed += 1
                    root_changed += 1
            else:
                postflop_samples += 1
                root_post += 1
                if delta > 1e-12:
                    postflop_nonidentical += 1
                    root_post_bad += 1

        total_nodes_k1 += int(one.nodes)
        total_nodes_k4 += int(four.nodes)
        total_samples += len(one.samples)
        rows.append(
            {
                "root_index": int(one.root_index),
                "nodes_k1": int(one.nodes),
                "nodes_k4": int(four.nodes),
                "sample_count": len(one.samples),
                "preflop_samples": int(root_pre),
                "preflop_changed_targets": int(root_changed),
                "postflop_samples": int(root_post),
                "postflop_nonidentical_targets": int(root_post_bad),
                "max_preflop_target_abs_delta": float(root_max_delta),
            }
        )

    if preflop_samples <= 0:
        raise RuntimeError("smoke found no HU preflop Advantage samples")
    if preflop_changed <= 0 or not math.isfinite(max_preflop_target_delta):
        raise RuntimeError("K4 did not change any preflop target")
    if postflop_nonidentical != 0:
        raise RuntimeError("K4 changed canonical-board postflop targets")
    if total_nodes_k4 <= total_nodes_k1:
        raise RuntimeError("K4 did not increase traversal-node cost")

    report = {
        "schema": "SPINCORE_LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_V1",
        "checkpoint": str(checkpoint),
        "completed_iteration": int(completed),
        "prospective_iteration": int(iteration),
        "read_only": True,
        "optimizer_steps": 0,
        "training_memory_writes": 0,
        "roots_compared": len(jobs),
        "k1": 1,
        "k4": 4,
        "total_samples_per_arm": int(total_samples),
        "preflop_samples": int(preflop_samples),
        "preflop_changed_targets": int(preflop_changed),
        "preflop_changed_fraction": float(preflop_changed / preflop_samples),
        "postflop_samples": int(postflop_samples),
        "postflop_nonidentical_targets": int(postflop_nonidentical),
        "max_preflop_target_abs_delta": float(max_preflop_target_delta),
        "nodes_k1": int(total_nodes_k1),
        "nodes_k4": int(total_nodes_k4),
        "node_multiplier_k4_over_k1": float(total_nodes_k4 / total_nodes_k1),
        "contract": {
            "same_root_jobs": True,
            "same_sample_count": True,
            "same_sample_order_and_identity": True,
            "canonical_postflop_targets_unchanged": True,
            "preflop_targets_board_averaged": True,
            "canonical_rng_progression_preserved_by_implementation": True,
        },
        "rows": rows,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== HU PREFLOP BOARD-AVERAGING MECHANICS SMOKE ===")
    print(
        f"roots={len(jobs)} samples={total_samples} preflop={preflop_samples} "
        f"changed={preflop_changed} postflop={postflop_samples}"
    )
    print(
        f"nodes_k1={total_nodes_k1} nodes_k4={total_nodes_k4} "
        f"multiplier={report['node_multiplier_k4_over_k1']:.3f}"
    )
    print(
        f"max_preflop_target_abs_delta={max_preflop_target_delta:.8f} "
        f"postflop_nonidentical={postflop_nonidentical}"
    )
    print("LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
