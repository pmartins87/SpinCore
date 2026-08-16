from __future__ import annotations

import argparse
from dataclasses import replace
import json
import platform
import time
from pathlib import Path

import torch

from spincore.r7_5_representation_v3_stage import (
    finalize_phase2_v3_seed,
    frozen_config,
    load_phase2_v3_runtime,
    new_phase2_v3_runtime,
    run_one_phase2_v3_iteration,
    save_phase2_v3_runtime,
)
from spincore.r7_5_representation_v3_stage_contract import (
    DOMAINS,
    ITERATIONS,
    REPRESENTATIONS,
    ROOTS_PER_ITERATION,
    TORCH_THREADS,
    TRAINING_SEEDS,
    validate_phase2_v3_contract,
)
from spincore.solver import SolverLibrary

SCHEMA = "SPINCORE_R7_5_3C_CHANCE_COVERAGE_X4_STAGED_WORKER_V1"
COVERAGE_MULTIPLIER = 4
EFFECTIVE_ROOTS_PER_ITERATION = ROOTS_PER_ITERATION * COVERAGE_MULTIPLIER


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5.3C winner-independent x4 chance-coverage worker")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--training-seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--target-iteration", type=int, choices=tuple(range(1, ITERATIONS + 1)), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if not args.execution_sha.strip():
        raise SystemExit("--execution-sha is required")
    if int(args.target_iteration) == ITERATIONS and not args.finalize:
        raise SystemExit("final x4 iteration must include --finalize")
    if args.finalize and int(args.target_iteration) != ITERATIONS:
        raise SystemExit("--finalize is legal only on final iteration")

    contract = validate_phase2_v3_contract(
        repo_root,
        representation=str(args.representation),
        domain=str(args.domain),
        training_seed=int(args.training_seed),
    )
    base_config = frozen_config()
    effective_config = replace(base_config, roots_per_iteration=EFFECTIVE_ROOTS_PER_ITERATION)
    if base_config.roots_per_iteration != ROOTS_PER_ITERATION or effective_config.roots_per_iteration != 256:
        raise RuntimeError("chance-coverage multiplier contract drift")

    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("frozen Phase 2 torch thread contract was not applied")
    solver = SolverLibrary(args.solver)

    if args.resume:
        bundle, session, behavior, _spec, state = load_phase2_v3_runtime(
            args.resume,
            repo_root=repo_root,
            solver=solver,
            representation=str(args.representation),
            domain=str(args.domain),
            training_seed=int(args.training_seed),
            config=base_config,
            execution_sha=str(args.execution_sha),
        )
        if int(state.get("chance_coverage_multiplier", -1)) != COVERAGE_MULTIPLIER:
            raise RuntimeError("chance-coverage multiplier identity drift")
        if int(state.get("effective_roots_per_iteration", -1)) != EFFECTIVE_ROOTS_PER_ITERATION:
            raise RuntimeError("effective root-count identity drift")
    else:
        if int(args.target_iteration) != 1:
            raise SystemExit("fresh x4 worker must start at iteration 1")
        bundle, session, behavior, _spec, state = new_phase2_v3_runtime(
            repo_root,
            solver=solver,
            representation=str(args.representation),
            domain=str(args.domain),
            training_seed=int(args.training_seed),
            config=base_config,
        )
        state["chance_coverage_multiplier"] = COVERAGE_MULTIPLIER
        state["effective_roots_per_iteration"] = EFFECTIVE_ROOTS_PER_ITERATION
        state["chance_coverage_semantics"] = (
            "Original independent training seeds and production deck_seed function; "
            "only balanced roots per iteration increase from 64 to 256."
        )

    started = time.perf_counter()
    iteration_report = run_one_phase2_v3_iteration(
        bundle=bundle,
        session=session,
        behavior=behavior,
        state=state,
        config=effective_config,
        target_iteration=int(args.target_iteration),
    )
    if int(iteration_report.get("roots_added", -1)) != EFFECTIVE_ROOTS_PER_ITERATION:
        raise RuntimeError("x4 iteration did not collect exactly 256 roots")

    final_report = None
    if args.finalize:
        final_report = finalize_phase2_v3_seed(
            bundle=bundle,
            behavior=behavior,
            session=session,
            state=state,
            config=effective_config,
        )
        if int(final_report.get("roots", -1)) != EFFECTIVE_ROOTS_PER_ITERATION * ITERATIONS:
            raise RuntimeError("x4 final report root count drift")
    compute_seconds = time.perf_counter() - started

    save_started = time.perf_counter()
    # Checkpoint identity deliberately retains the original frozen Phase-2 config.
    # The x4 root override is recorded in stage state and the final report. This
    # lets the authoritative loader remain unchanged while the remediation is
    # explicit and auditable.
    save_phase2_v3_runtime(
        args.checkpoint_out,
        bundle=bundle,
        behavior=behavior,
        state=state,
        config=base_config,
        execution_sha=str(args.execution_sha),
        finalized=bool(args.finalize),
        final_report=final_report,
    )
    checkpoint_seconds = time.perf_counter() - save_started

    payload = {
        "schema": SCHEMA,
        "execution_sha": str(args.execution_sha),
        "representation": str(args.representation),
        "domain": str(args.domain),
        "training_seed": int(args.training_seed),
        "target_iteration": int(args.target_iteration),
        "base_phase2_config": base_config.to_dict(),
        "effective_training_config": effective_config.to_dict(),
        "chance_coverage_multiplier": COVERAGE_MULTIPLIER,
        "independent_training_seed_preserved": True,
        "production_deck_seed_semantics_preserved": True,
        "model_identity": contract["live_model"],
        "iteration_report": iteration_report,
        "compute_wall_seconds": float(compute_seconds),
        "checkpoint_write_seconds": float(checkpoint_seconds),
        "checkpoint_to_checkpoint_seconds": float(compute_seconds + checkpoint_seconds),
        "finalized": bool(args.finalize),
        "final_report": final_report,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
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
