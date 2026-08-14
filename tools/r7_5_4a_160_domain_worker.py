from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from spincore.r7_5_action_stage import (
    ActionStageConfig,
    finalize_stage_seed,
    load_stage_runtime,
    new_stage_runtime,
    run_one_stage_iteration,
    save_stage_runtime,
)
from spincore.r7_5_action_stage_contract import (
    ADVANTAGE_STEPS,
    AUDIT_SIZE,
    BATCH_SIZE,
    ENSEMBLE_SIZE,
    EPSILON_CAP,
    EPSILON_SCALE,
    EXACT_OPPONENT_LEVELS,
    ITERATIONS,
    LEARNING_RATE,
    POLICY_STEPS,
    RESERVOIR_CAPACITY,
    ROOTS_PER_ITERATION,
    TORCH_THREADS,
    validate_action_stage_contract,
)
from spincore.solver import SolverLibrary

SCHEMA = "SPINCORE_R7_5_4A_160_STAGED_WORKER_V1"


def frozen_config() -> ActionStageConfig:
    return ActionStageConfig(
        roots_per_iteration=ROOTS_PER_ITERATION,
        total_iterations=ITERATIONS,
        exact_opponent_levels=EXACT_OPPONENT_LEVELS,
        reservoir_capacity=RESERVOIR_CAPACITY,
        advantage_steps=ADVANTAGE_STEPS,
        policy_steps=POLICY_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        ensemble_size=ENSEMBLE_SIZE,
        audit_size=AUDIT_SIZE,
        epsilon_scale=EPSILON_SCALE,
        epsilon_cap=EPSILON_CAP,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Contract-locked staged R7.5.4A 160 worker")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--domain", choices=("TRUE_HEADS_UP", "THREE_HANDED"), required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--target-iteration", type=int, choices=(1, 2, 3, 4, 5), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    validate_action_stage_contract(
        repo_root,
        candidate_id=str(args.candidate),
        training_seed=int(args.training_seed),
    )
    config = frozen_config()
    if int(args.target_iteration) == ITERATIONS and not args.finalize:
        raise SystemExit("iteration 5 must include --finalize")
    if args.finalize and int(args.target_iteration) != ITERATIONS:
        raise SystemExit("--finalize is legal only on iteration 5")
    if not str(args.execution_sha).strip():
        raise SystemExit("--execution-sha is required")

    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("frozen torch thread contract was not applied")
    solver = SolverLibrary(args.solver)

    if args.resume:
        bundle, session, behavior, _spec, state = load_stage_runtime(
            args.resume,
            repo_root=repo_root,
            solver=solver,
            candidate_id=str(args.candidate),
            domain=str(args.domain),
            training_seed=int(args.training_seed),
            config=config,
            execution_sha=str(args.execution_sha),
        )
    else:
        if int(args.target_iteration) != 1:
            raise SystemExit("fresh worker must start at iteration 1")
        bundle, session, behavior, _spec, state = new_stage_runtime(
            repo_root,
            solver=solver,
            candidate_id=str(args.candidate),
            domain=str(args.domain),
            training_seed=int(args.training_seed),
            config=config,
        )

    started = time.perf_counter()
    iteration_report = run_one_stage_iteration(
        bundle=bundle,
        session=session,
        behavior=behavior,
        state=state,
        config=config,
        target_iteration=int(args.target_iteration),
    )
    final_report = None
    if args.finalize:
        final_report = finalize_stage_seed(
            bundle=bundle,
            behavior=behavior,
            session=session,
            state=state,
            config=config,
        )
    compute_wall_seconds = time.perf_counter() - started

    save_started = time.perf_counter()
    save_stage_runtime(
        args.checkpoint_out,
        bundle=bundle,
        behavior=behavior,
        state=state,
        config=config,
        execution_sha=str(args.execution_sha),
        finalized=bool(args.finalize),
        final_report=final_report,
    )
    checkpoint_write_seconds = time.perf_counter() - save_started

    payload = {
        "schema": SCHEMA,
        "execution_sha": str(args.execution_sha),
        "candidate_id": str(args.candidate),
        "domain": str(args.domain),
        "training_seed": int(args.training_seed),
        "target_iteration": int(args.target_iteration),
        "config": config.to_dict(),
        "iteration_report": iteration_report,
        "compute_wall_seconds": float(compute_wall_seconds),
        "checkpoint_write_seconds": float(checkpoint_write_seconds),
        "checkpoint_to_checkpoint_stage_seconds": float(compute_wall_seconds + checkpoint_write_seconds),
        "finalized": bool(args.finalize),
        "final_report": final_report,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
