from __future__ import annotations

import argparse
from dataclasses import replace
import json
import platform
import time
from pathlib import Path

import torch

import spincore.r7_5_representation_v3_stage as stage
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3_checkpoint import (
    RepresentationV3Progress,
    load_representation_v3_checkpoint,
    save_representation_v3_checkpoint,
)
from spincore.r7_5_representation_v3_stage import (
    finalize_phase2_v3_seed,
    frozen_config,
    load_phase2_v3_runtime,
    new_phase2_v3_runtime,
    run_one_phase2_v3_iteration,
    save_phase2_v3_runtime,
)
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    DOMAINS,
    EXACT_OPPONENT_LEVELS,
    ITERATIONS,
    MODEL_FINGERPRINTS,
    REPRESENTATIONS,
    ROOTS_PER_ITERATION,
    TORCH_THREADS,
    TRAINING_SEEDS,
    deck_seed,
    validate_phase2_v3_contract,
)
from spincore.r7_5_representation_v3_uncertainty import V3UncertaintyDampedPolicyMixture
from spincore.solver import SolverLibrary

SCHEMA = "SPINCORE_R7_5_3C_CHANCE_COVERAGE_X4_STAGED_WORKER_V2"
CHUNK_SCHEMA = "SPINCORE_R7_5_3C_CHANCE_COVERAGE_X4_PARTIAL_COLLECTION_V1"
COVERAGE_MULTIPLIER = 4
ROOTS_PER_CHUNK = ROOTS_PER_ITERATION
EFFECTIVE_ROOTS_PER_ITERATION = ROOTS_PER_ITERATION * COVERAGE_MULTIPLIER


def _aggregate_geometry(chunks: list[dict]) -> dict:
    visits = sum(int(row["branch_geometry"]["advantage_decision_visits"]) for row in chunks)
    nominal = sum(int(row["branch_geometry"]["nominal_aggressive_branches"]) for row in chunks)
    effective = sum(int(row["branch_geometry"]["effective_unique_aggressive_branches"]) for row in chunks)
    return {
        "advantage_decision_visits": visits,
        "nominal_aggressive_branches": nominal,
        "effective_unique_aggressive_branches": effective,
        "nominal_aggressive_branches_per_decision": float(nominal / visits) if visits else 0.0,
        "effective_unique_aggressive_branches_per_decision": float(effective / visits) if visits else 0.0,
    }


def _save_partial(
    path: Path,
    *,
    bundle,
    behavior,
    state: dict,
    base_config,
    representation: str,
    domain: str,
    training_seed: int,
    target_iteration: int,
    root_chunk: int,
    execution_sha: str,
) -> None:
    extra = {
        "partial_schema": CHUNK_SCHEMA,
        "stage_config": base_config.to_dict(),
        "stage_state": dict(state),
        "target_iteration": int(target_iteration),
        "root_chunk": int(root_chunk),
        "behavior_model_states": [model.state_dict() for model in behavior.models],
        "behavior_stats": behavior.stats(),
    }
    save_representation_v3_checkpoint(
        path,
        bundle,
        RepresentationV3Progress(
            iteration=int(state["completed_iteration"]),
            global_root=int(state["global_root"]),
            advantage_optimizer_step=int(bundle.counters["adv_optimizer_steps"]),
            policy_optimizer_step=int(bundle.counters["policy_optimizer_steps"]),
            phase="x4_partial_collect",
        ),
        domain=domain,
        action_candidate=ACTION_CANDIDATE,
        execution_sha=execution_sha,
        architecture_fingerprint_sha256=MODEL_FINGERPRINTS[representation],
        extra=extra,
    )


def _load_partial(
    path: Path,
    *,
    repo_root: Path,
    solver,
    representation: str,
    domain: str,
    training_seed: int,
    target_iteration: int,
    previous_chunk: int,
    execution_sha: str,
    base_config,
):
    bundle, progress, spec, extra = load_representation_v3_checkpoint(
        path,
        repo_root=repo_root,
        expected_domain=domain,
        expected_representation=representation,
        expected_seed=training_seed,
        expected_action_candidate=ACTION_CANDIDATE,
        expected_execution_sha=execution_sha,
        expected_architecture_fingerprint_sha256=MODEL_FINGERPRINTS[representation],
        device="cpu",
    )
    if progress.phase != "x4_partial_collect":
        raise RuntimeError("expected x4 partial-collection checkpoint")
    if extra.get("partial_schema") != CHUNK_SCHEMA:
        raise RuntimeError("wrong x4 partial checkpoint schema")
    if dict(extra.get("stage_config") or {}) != base_config.to_dict():
        raise RuntimeError("x4 partial base-config drift")
    if int(extra.get("target_iteration", -1)) != int(target_iteration):
        raise RuntimeError("x4 partial target-iteration drift")
    if int(extra.get("root_chunk", -1)) != int(previous_chunk):
        raise RuntimeError("x4 partial chunk identity drift")
    state = dict(extra.get("stage_state") or {})
    if int(progress.iteration) != int(state.get("completed_iteration", -1)):
        raise RuntimeError("x4 partial progress iteration drift")
    if int(progress.global_root) != int(state.get("global_root", -1)):
        raise RuntimeError("x4 partial global-root drift")

    behavior = V3UncertaintyDampedPolicyMixture(
        representation=representation,
        device="cpu",
        epsilon_scale=base_config.epsilon_scale,
        epsilon_cap=base_config.epsilon_cap,
    )
    models = []
    for index, state_dict in enumerate(list(extra.get("behavior_model_states") or [])):
        _cfg, model = stage._make_v3_model(representation, 0x520000 + index)
        model.load_state_dict(state_dict)
        models.append(model)
    behavior.models = models
    behavior.restore_stats(dict(extra.get("behavior_stats") or {}))
    session = stage._make_session(solver, bundle, spec, behavior)
    return bundle, session, behavior, spec, state


def _collect_chunk(*, session, bundle, state: dict, target_iteration: int, roots: int) -> dict:
    scenarios = action_scenario_cycle(str(state["domain"]))
    scenario_counts = list(state["scenario_counts"])
    global_root = int(state["global_root"])
    session.collector.reset_telemetry()
    roots_before = int(bundle.counters["roots"])
    nodes_before = int(bundle.counters["nodes"])
    adv_before = int(bundle.adv_mem.seen)
    pol_before = int(bundle.pol_mem.seen)
    started = time.perf_counter()
    for _ in range(int(roots)):
        scenario_index = global_root % len(scenarios)
        scenario_counts[scenario_index] += 1
        session.collect_root(
            scenarios[scenario_index],
            iteration=int(target_iteration),
            exact_opponent_levels=EXACT_OPPONENT_LEVELS,
            deck_seed=deck_seed(int(state["training_seed"]), global_root, int(target_iteration)),
        )
        global_root += 1
    seconds = time.perf_counter() - started
    state["global_root"] = global_root
    state["scenario_counts"] = scenario_counts
    return {
        "roots": int(bundle.counters["roots"]) - roots_before,
        "nodes": int(bundle.counters["nodes"]) - nodes_before,
        "advantage_seen": int(bundle.adv_mem.seen) - adv_before,
        "strategy_seen": int(bundle.pol_mem.seen) - pol_before,
        "tree_collection_seconds": float(seconds),
        "branch_geometry": session.collector.telemetry_snapshot(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5.3C winner-independent x4 chance-coverage worker")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--training-seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--target-iteration", type=int, choices=tuple(range(1, ITERATIONS + 1)), required=True)
    parser.add_argument("--root-chunk", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if not args.execution_sha.strip():
        raise SystemExit("--execution-sha is required")
    if args.finalize and not (int(args.target_iteration) == ITERATIONS and int(args.root_chunk) == 4):
        raise SystemExit("--finalize is legal only on iteration 3 chunk 4")
    if int(args.target_iteration) == ITERATIONS and int(args.root_chunk) == 4 and not args.finalize:
        raise SystemExit("final x4 chunk must include --finalize")

    contract = validate_phase2_v3_contract(
        repo_root,
        representation=str(args.representation),
        domain=str(args.domain),
        training_seed=int(args.training_seed),
    )
    base_config = frozen_config()
    effective_config = replace(base_config, roots_per_iteration=EFFECTIVE_ROOTS_PER_ITERATION)
    fit_only_config = replace(base_config, roots_per_iteration=0)
    if base_config.roots_per_iteration != 64 or effective_config.roots_per_iteration != 256:
        raise RuntimeError("chance-coverage multiplier contract drift")

    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("frozen Phase 2 torch thread contract was not applied")
    solver = SolverLibrary(args.solver)

    if int(args.root_chunk) == 1:
        if int(args.target_iteration) == 1:
            if args.resume:
                raise SystemExit("iteration 1 chunk 1 must start fresh")
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
                "each Phase-2 iteration collects four sequential 64-root chunks before one unchanged Advantage fit."
            )
        else:
            if not args.resume:
                raise SystemExit("later iteration chunk 1 requires previous finalized-iteration checkpoint")
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
        if int(state["completed_iteration"]) != int(args.target_iteration) - 1:
            raise RuntimeError("x4 iteration start identity drift")
        state["x4_pending_iteration"] = {
            "iteration": int(args.target_iteration),
            "roots_before": int(bundle.counters["roots"]),
            "nodes_before": int(bundle.counters["nodes"]),
            "advantage_seen_before": int(bundle.adv_mem.seen),
            "strategy_seen_before": int(bundle.pol_mem.seen),
            "chunks": [],
        }
    else:
        if not args.resume:
            raise SystemExit("x4 chunk 2-4 requires previous partial checkpoint")
        bundle, session, behavior, _spec, state = _load_partial(
            args.resume,
            repo_root=repo_root,
            solver=solver,
            representation=str(args.representation),
            domain=str(args.domain),
            training_seed=int(args.training_seed),
            target_iteration=int(args.target_iteration),
            previous_chunk=int(args.root_chunk) - 1,
            execution_sha=str(args.execution_sha),
            base_config=base_config,
        )

    pending = dict(state.get("x4_pending_iteration") or {})
    if int(pending.get("iteration", -1)) != int(args.target_iteration):
        raise RuntimeError("missing x4 pending-iteration state")
    chunks = list(pending.get("chunks") or [])
    if len(chunks) != int(args.root_chunk) - 1:
        raise RuntimeError("x4 chunk history length drift")

    chunk_report = _collect_chunk(
        session=session,
        bundle=bundle,
        state=state,
        target_iteration=int(args.target_iteration),
        roots=ROOTS_PER_CHUNK,
    )
    if int(chunk_report["roots"]) != ROOTS_PER_CHUNK:
        raise RuntimeError("x4 chunk did not collect exactly 64 roots")
    chunks.append(chunk_report)
    pending["chunks"] = chunks
    state["x4_pending_iteration"] = pending

    iteration_report = None
    final_report = None
    fit_seconds = 0.0
    if int(args.root_chunk) < 4:
        _save_partial(
            args.checkpoint_out,
            bundle=bundle,
            behavior=behavior,
            state=state,
            base_config=base_config,
            representation=str(args.representation),
            domain=str(args.domain),
            training_seed=int(args.training_seed),
            target_iteration=int(args.target_iteration),
            root_chunk=int(args.root_chunk),
            execution_sha=str(args.execution_sha),
        )
    else:
        fit_started = time.perf_counter()
        iteration_report = run_one_phase2_v3_iteration(
            bundle=bundle,
            session=session,
            behavior=behavior,
            state=state,
            config=fit_only_config,
            target_iteration=int(args.target_iteration),
        )
        fit_seconds = time.perf_counter() - fit_started
        roots_added = int(bundle.counters["roots"]) - int(pending["roots_before"])
        nodes_added = int(bundle.counters["nodes"]) - int(pending["nodes_before"])
        adv_added = int(bundle.adv_mem.seen) - int(pending["advantage_seen_before"])
        pol_added = int(bundle.pol_mem.seen) - int(pending["strategy_seen_before"])
        manual_tree_seconds = sum(float(row["tree_collection_seconds"]) for row in chunks)
        patched = dict(iteration_report)
        patched.update({
            "roots_added": roots_added,
            "nodes_added": nodes_added,
            "advantage_seen_added": adv_added,
            "strategy_seen_added": pol_added,
            "tree_collection_seconds": manual_tree_seconds,
            "nodes_per_root": float(nodes_added / roots_added),
            "advantage_samples_per_root": float(adv_added / roots_added),
            "strategy_samples_per_root": float(pol_added / roots_added),
            "tree_seconds_per_root": float(manual_tree_seconds / roots_added),
            "branch_geometry": _aggregate_geometry(chunks),
            "chance_coverage_chunks": chunks,
        })
        if roots_added != EFFECTIVE_ROOTS_PER_ITERATION:
            raise RuntimeError(f"x4 iteration root total drift: {roots_added}")
        # run_one recorded only its zero-root fit call; replace it with the full
        # four-chunk iteration report and add the manually accumulated tree time.
        state["iteration_reports"][-1] = patched
        state["tree_collection_seconds_total"] = float(state["tree_collection_seconds_total"]) + manual_tree_seconds
        state.pop("x4_pending_iteration", None)
        iteration_report = patched

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

    payload = {
        "schema": SCHEMA,
        "execution_sha": str(args.execution_sha),
        "representation": str(args.representation),
        "domain": str(args.domain),
        "training_seed": int(args.training_seed),
        "target_iteration": int(args.target_iteration),
        "root_chunk": int(args.root_chunk),
        "roots_per_chunk": ROOTS_PER_CHUNK,
        "chance_coverage_multiplier": COVERAGE_MULTIPLIER,
        "effective_roots_per_iteration": EFFECTIVE_ROOTS_PER_ITERATION,
        "independent_training_seed_preserved": True,
        "production_deck_seed_semantics_preserved": True,
        "model_identity": contract["live_model"],
        "chunk_report": chunk_report,
        "iteration_completed": int(args.root_chunk) == 4,
        "iteration_report": iteration_report,
        "fit_and_finalize_wall_seconds": float(fit_seconds),
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
    print(json.dumps({
        "representation": payload["representation"],
        "domain": payload["domain"],
        "training_seed": payload["training_seed"],
        "target_iteration": payload["target_iteration"],
        "root_chunk": payload["root_chunk"],
        "chunk_roots": chunk_report["roots"],
        "iteration_completed": payload["iteration_completed"],
        "finalized": payload["finalized"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
