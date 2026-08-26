from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from spincore.r7_5_action_stage import (
    ActionStageConfig,
    finalize_stage_seed,
    load_stage_runtime,
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
    POSTFLOP_TRAINING_SEEDS,
    RESERVOIR_CAPACITY,
    ROOTS_PER_ITERATION,
    TORCH_THREADS,
    validate_action_stage_contract,
)
from spincore.r7_5_action_stage_recovery import (
    PARTIAL_PHASE,
    collect_stage_root_chunk,
    fit_collected_stage_iteration,
    load_partial_collection_runtime,
    save_partial_collection_runtime,
    save_recovered_stage_runtime,
)
from spincore.solver import SolverLibrary


SCHEMA = "SPINCORE_R7_5_4A_DENSE3H_RECOVERY_WORKER_V1"
CANDIDATE = "PF_DENSE_REFERENCE"
DOMAIN = "THREE_HANDED"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _checkpoint_metadata(path: Path) -> tuple[str, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return (
        str(dict(payload.get("progress") or {}).get("phase")),
        dict(dict(payload.get("extra") or {}).get("recovery_provenance") or {}),
    )


def _provenance(args) -> dict:
    return {
        "source_training_run_id": int(args.source_training_run_id),
        "source_iteration1_artifact_id": int(args.source_iteration1_artifact_id),
        "source_iteration1_artifact_digest": str(args.source_iteration1_artifact_digest),
        "source_iteration1_checkpoint_sha256": str(args.source_checkpoint_sha256),
        "candidate_id": CANDIDATE,
        "domain": DOMAIN,
        "training_seed": int(args.training_seed),
        "intervention": "MECHANICAL_MID_ITERATION_CHECKPOINT_ONLY",
        "root_order_changed": False,
        "deck_seed_formula_changed": False,
        "reservoir_semantics_changed": False,
        "optimizer_semantics_changed": False,
        "policy_semantics_changed": False,
    }


def _assert_provenance(expected: dict, actual: dict) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise RuntimeError(f"recovery provenance mismatch: {key}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Linux-native mechanical recovery for the three timed-out dense 3H cells"
    )
    parser.add_argument("--mode", choices=("collect", "fit"), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--target-iteration", type=int, choices=(2, 3, 4, 5), required=True)
    parser.add_argument("--root-budget", type=int, default=2)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--source-execution-sha", required=True)
    parser.add_argument("--recovery-execution-sha", required=True)
    parser.add_argument("--source-training-run-id", type=int, required=True)
    parser.add_argument("--source-iteration1-artifact-id", type=int, required=True)
    parser.add_argument("--source-iteration1-artifact-digest", required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    args = parser.parse_args()

    if int(args.training_seed) not in tuple(int(x) for x in POSTFLOP_TRAINING_SEEDS):
        raise SystemExit("recovery seed is outside the frozen R7.5.4A matrix")
    if int(args.root_budget) != 2:
        raise SystemExit("mechanical recovery chunk budget is frozen at 2")
    if not args.resume.is_file():
        raise SystemExit(f"resume checkpoint missing: {args.resume}")

    repo_root = args.repo_root.resolve()
    validate_action_stage_contract(
        repo_root, candidate_id=CANDIDATE, training_seed=int(args.training_seed)
    )
    config = frozen_config()
    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("frozen torch thread contract was not applied")
    solver = SolverLibrary(args.solver)
    provenance = _provenance(args)

    input_sha256 = _sha256(args.resume)
    phase, loaded_recovery_provenance = _checkpoint_metadata(args.resume)
    if (
        args.mode == "collect"
        and int(args.target_iteration) == 2
        and phase == "post_advantage_fit"
        and input_sha256 != str(args.source_checkpoint_sha256)
    ):
        raise RuntimeError("original iteration-1 checkpoint SHA-256 mismatch")
    if int(args.target_iteration) > 2 and phase == "post_advantage_fit":
        if loaded_recovery_provenance.get("recovery_execution_sha") != str(
            args.recovery_execution_sha
        ):
            raise RuntimeError("recovered prior-iteration execution SHA mismatch")
        for key, value in provenance.items():
            if loaded_recovery_provenance.get(key) != value:
                raise RuntimeError(f"recovered prior-iteration provenance mismatch: {key}")
    started = time.perf_counter()
    finalized = False
    final_report = None

    if args.mode == "collect":
        if phase == "post_advantage_fit":
            bundle, session, behavior, _spec, state = load_stage_runtime(
                args.resume,
                repo_root=repo_root,
                solver=solver,
                candidate_id=CANDIDATE,
                domain=DOMAIN,
                training_seed=int(args.training_seed),
                config=config,
                execution_sha=str(args.source_execution_sha),
            )
            partial = None
        elif phase == PARTIAL_PHASE:
            (
                bundle,
                session,
                behavior,
                _spec,
                state,
                partial,
                _loaded_provenance,
            ) = load_partial_collection_runtime(
                args.resume,
                repo_root=repo_root,
                solver=solver,
                candidate_id=CANDIDATE,
                domain=DOMAIN,
                training_seed=int(args.training_seed),
                config=config,
                source_execution_sha=str(args.source_execution_sha),
                recovery_execution_sha=str(args.recovery_execution_sha),
            )
            _assert_provenance(provenance, _loaded_provenance)
        else:
            raise SystemExit(f"collect mode cannot resume checkpoint phase {phase!r}")

        partial, operation_report = collect_stage_root_chunk(
            bundle=bundle,
            session=session,
            state=state,
            config=config,
            target_iteration=int(args.target_iteration),
            root_budget=int(args.root_budget),
            partial=partial,
        )
        save_partial_collection_runtime(
            args.checkpoint_out,
            bundle=bundle,
            behavior=behavior,
            state=state,
            partial=partial,
            config=config,
            source_execution_sha=str(args.source_execution_sha),
            recovery_execution_sha=str(args.recovery_execution_sha),
            recovery_provenance=provenance,
        )
        roots_collected = int(partial["roots_collected"])
    else:
        if phase != PARTIAL_PHASE:
            raise SystemExit("fit mode requires a completed partial-collection checkpoint")
        (
            bundle,
            session,
            behavior,
            _spec,
            state,
            partial,
            _loaded_provenance,
        ) = load_partial_collection_runtime(
            args.resume,
            repo_root=repo_root,
            solver=solver,
            candidate_id=CANDIDATE,
            domain=DOMAIN,
            training_seed=int(args.training_seed),
            config=config,
            source_execution_sha=str(args.source_execution_sha),
            recovery_execution_sha=str(args.recovery_execution_sha),
        )
        _assert_provenance(provenance, _loaded_provenance)
        operation_report = fit_collected_stage_iteration(
            bundle=bundle,
            session=session,
            behavior=behavior,
            state=state,
            config=config,
            target_iteration=int(args.target_iteration),
            partial=partial,
        )
        finalized = int(args.target_iteration) == int(config.total_iterations)
        if finalized:
            final_report = finalize_stage_seed(
                bundle=bundle,
                behavior=behavior,
                session=session,
                state=state,
                config=config,
            )
        save_recovered_stage_runtime(
            args.checkpoint_out,
            bundle=bundle,
            behavior=behavior,
            state=state,
            config=config,
            source_execution_sha=str(args.source_execution_sha),
            recovery_execution_sha=str(args.recovery_execution_sha),
            recovery_provenance=provenance,
            finalized=finalized,
            final_report=final_report,
        )
        roots_collected = int(config.roots_per_iteration)

    wall_seconds = time.perf_counter() - started
    output_sha256 = _sha256(args.checkpoint_out)
    payload = {
        "schema": SCHEMA,
        "mode": str(args.mode),
        "candidate_id": CANDIDATE,
        "domain": DOMAIN,
        "training_seed": int(args.training_seed),
        "target_iteration": int(args.target_iteration),
        "root_budget": int(args.root_budget),
        "roots_collected": roots_collected,
        "source_execution_sha": str(args.source_execution_sha),
        "recovery_execution_sha": str(args.recovery_execution_sha),
        "input_checkpoint_sha256": input_sha256,
        "output_checkpoint_sha256": output_sha256,
        "operation_report": operation_report,
        "finalized": finalized,
        "final_report": final_report,
        "wall_seconds": float(wall_seconds),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
        "recovery_provenance": provenance,
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
