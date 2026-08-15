from __future__ import annotations

import argparse
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
    TORCH_THREADS,
    TRAINING_SEEDS,
    validate_phase2_v3_contract,
)
from spincore.solver import SolverLibrary

SCHEMA = "SPINCORE_R7_5_3C_PHASE2_STAGED_WORKER_V1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Contract-locked R7.5.3C Phase 2 worker")
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
        raise SystemExit("final Phase 2 iteration must include --finalize")
    if args.finalize and int(args.target_iteration) != ITERATIONS:
        raise SystemExit("--finalize is legal only on the final Phase 2 iteration")

    contract = validate_phase2_v3_contract(
        repo_root,
        representation=str(args.representation),
        domain=str(args.domain),
        training_seed=int(args.training_seed),
    )
    config = frozen_config()
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
            config=config,
            execution_sha=str(args.execution_sha),
        )
    else:
        if int(args.target_iteration) != 1:
            raise SystemExit("fresh Phase 2 worker must start at iteration 1")
        bundle, session, behavior, _spec, state = new_phase2_v3_runtime(
            repo_root,
            solver=solver,
            representation=str(args.representation),
            domain=str(args.domain),
            training_seed=int(args.training_seed),
            config=config,
        )

    started = time.perf_counter()
    iteration_report = run_one_phase2_v3_iteration(
        bundle=bundle,
        session=session,
        behavior=behavior,
        state=state,
        config=config,
        target_iteration=int(args.target_iteration),
    )
    final_report = None
    if args.finalize:
        final_report = finalize_phase2_v3_seed(
            bundle=bundle,
            behavior=behavior,
            session=session,
            state=state,
            config=config,
        )
    compute_seconds = time.perf_counter() - started

    save_started = time.perf_counter()
    save_phase2_v3_runtime(
        args.checkpoint_out,
        bundle=bundle,
        behavior=behavior,
        state=state,
        config=config,
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
        "config": config.to_dict(),
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
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
