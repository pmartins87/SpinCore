from __future__ import annotations

"""Final winner-independent R7.5.3C x16 chance-coverage cell worker.

Frozen by validation/R7_5_3C_FINAL_CONTINGENCY_X16_FREEZE_20260818.json.
Each Phase-2 iteration collects sixteen contiguous 64-root chunks (1024 roots)
before exactly one unchanged Advantage fit. The two original training seeds,
production deck_seed function, global-root order, scenario cycle, model budgets,
and hard gates are unchanged.
"""

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

SCHEMA = "SPINCORE_R7_5_3C_FINAL_CHANCE_COVERAGE_X16_STAGED_WORKER_V1"
CHUNK_SCHEMA = "SPINCORE_R7_5_3C_FINAL_CHANCE_COVERAGE_X16_PARTIAL_COLLECTION_V1"
PARTIAL_PHASE = "x16_partial_collect"
COVERAGE_MULTIPLIER = 16
ROOTS_PER_CHUNK = ROOTS_PER_ITERATION
CHUNKS_PER_ITERATION = 16
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


def _fit_only_iteration(*, bundle, session, behavior, state: dict, config, target_iteration: int) -> dict:
    """Frozen Phase-2 Advantage fit with no additional roots.

    This is the mechanically corrected zero-root fit path already admitted for
    x4, embedded here so x16 never executes the old root-normalized reporting
    division. The caller replaces zero-root placeholder fields with the true
    1024-root iteration totals immediately after the fit.
    """
    if int(config.roots_per_iteration) != 0:
        raise ValueError("x16 fit-only helper is legal only for zero-root fit calls")
    expected = int(state["completed_iteration"]) + 1
    if int(target_iteration) != expected:
        raise ValueError(f"Phase 2 must advance exactly one iteration: expected {expected}")
    if not 1 <= int(target_iteration) <= config.total_iterations:
        raise ValueError("Phase 2 target iteration out of range")

    global_root = int(state["global_root"])
    scenario_counts = list(state["scenario_counts"])
    adv_seen_before = int(bundle.adv_mem.seen)
    pol_seen_before = int(bundle.pol_mem.seen)

    fit_started = time.perf_counter()
    reset_seed = stage.primary_reset_seed(int(state["training_seed"]), int(target_iteration))
    session.reset_advantage_network(init_seed=reset_seed, lr=config.learning_rate)
    session.train_advantage(steps=config.advantage_steps, batch_size=config.batch_size)
    primary_nrmse = stage.audit_v3_advantage_model(
        bundle.advantage,
        bundle.adv_mem.items,
        representation=str(state["representation"]),
        sample_size=config.audit_size,
        seed=int(state["training_seed"]) ^ (int(target_iteration) * 0x45D9F3B),
    )
    models = [bundle.advantage]
    member_reports = [{
        "member": 0,
        "role": "PRIMARY_AUTHORITATIVE_COUPLED_RNG",
        "init_seed": int(reset_seed),
        "optimizer_steps": config.advantage_steps,
        "final_weighted_nrmse": float(primary_nrmse),
    }]
    for member in (1, 2, 3):
        init_seed, batch_seed = stage.side_member_seeds(
            int(state["training_seed"]), int(target_iteration), member
        )
        member_started = time.perf_counter()
        model, fit_report = stage.fit_independent_v3_advantage_member(
            bundle.adv_mem.items,
            representation=str(state["representation"]),
            init_seed=init_seed,
            batch_seed=batch_seed,
            steps=config.advantage_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
        )
        nrmse = stage.audit_v3_advantage_model(
            model,
            bundle.adv_mem.items,
            representation=str(state["representation"]),
            sample_size=config.audit_size,
            seed=int(state["training_seed"]) ^ (int(target_iteration) * 0x13579B) ^ (member * 0x2468AC),
        )
        member_reports.append({
            **fit_report,
            "member": member,
            "role": "SIDE_MEMBER_DOES_NOT_PERTURB_PRIMARY_RNG",
            "final_weighted_nrmse": float(nrmse),
            "fit_seconds": float(time.perf_counter() - member_started),
        })
        models.append(model)
    behavior.models = models
    ensemble_nrmse = stage.ensemble_v3_advantage_nrmse(
        models,
        bundle.adv_mem.items,
        representation=str(state["representation"]),
        sample_size=config.audit_size,
        seed=int(state["training_seed"]) ^ (int(target_iteration) * 0x5EEDBEEF),
    )
    fit_seconds = time.perf_counter() - fit_started
    regret_proxy = stage._mean_positive_regret_proxy(
        bundle.adv_mem.items,
        sample_size=config.audit_size,
        seed=int(state["training_seed"]) ^ (int(target_iteration) * 0x27D4EB2D),
    )

    report = {
        "iteration": int(target_iteration),
        "roots_added": 0,
        "nodes_added": 0,
        "nodes_per_root": 0.0,
        "tree_collection_seconds": 0.0,
        "tree_seconds_per_root": 0.0,
        "advantage_fit_seconds": float(fit_seconds),
        "advantage_seen_added": int(bundle.adv_mem.seen) - adv_seen_before,
        "strategy_seen_added": int(bundle.pol_mem.seen) - pol_seen_before,
        "advantage_samples_per_root": 0.0,
        "strategy_samples_per_root": 0.0,
        "branch_geometry": session.collector.telemetry_snapshot(),
        "regret_proxy": regret_proxy,
        "ensemble_weighted_nrmse": float(ensemble_nrmse),
        "ensemble_advantage_gate_pass": bool(ensemble_nrmse <= stage.ADVANTAGE_NRMSE_MAX),
        "members": member_reports,
        "behavior_stats_after_fit": behavior.stats(),
        "peak_rss_bytes": stage._peak_rss_bytes(),
        "x16_fit_only_reporting_guard": True,
    }
    state["completed_iteration"] = int(target_iteration)
    state["global_root"] = global_root
    state["scenario_counts"] = scenario_counts
    state["iteration_reports"] = list(state["iteration_reports"]) + [report]
    state["advantage_fit_seconds_total"] = float(state["advantage_fit_seconds_total"]) + fit_seconds
    return report


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
            phase=PARTIAL_PHASE,
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
    if progress.phase != PARTIAL_PHASE:
        raise RuntimeError("expected x16 partial-collection checkpoint")
    if extra.get("partial_schema") != CHUNK_SCHEMA:
        raise RuntimeError("wrong x16 partial checkpoint schema")
    if dict(extra.get("stage_config") or {}) != base_config.to_dict():
        raise RuntimeError("x16 partial base-config drift")
    if int(extra.get("target_iteration", -1)) != int(target_iteration):
        raise RuntimeError("x16 partial target-iteration drift")
    if int(extra.get("root_chunk", -1)) != int(previous_chunk):
        raise RuntimeError("x16 partial chunk identity drift")
    state = dict(extra.get("stage_state") or {})
    if int(progress.iteration) != int(state.get("completed_iteration", -1)):
        raise RuntimeError("x16 partial progress iteration drift")
    if int(progress.global_root) != int(state.get("global_root", -1)):
        raise RuntimeError("x16 partial global-root drift")

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
    parser = argparse.ArgumentParser(description="R7.5.3C final winner-independent x16 chance-coverage worker")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--training-seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--target-iteration", type=int, choices=tuple(range(1, ITERATIONS + 1)), required=True)
    parser.add_argument("--root-chunk", type=int, choices=tuple(range(1, CHUNKS_PER_ITERATION + 1)), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if not args.execution_sha.strip():
        raise SystemExit("--execution-sha is required")
    if args.finalize and not (int(args.target_iteration) == ITERATIONS and int(args.root_chunk) == CHUNKS_PER_ITERATION):
        raise SystemExit("--finalize is legal only on iteration 3 chunk 16")
    if int(args.target_iteration) == ITERATIONS and int(args.root_chunk) == CHUNKS_PER_ITERATION and not args.finalize:
        raise SystemExit("final x16 chunk must include --finalize")

    contract = validate_phase2_v3_contract(
        repo_root,
        representation=str(args.representation),
        domain=str(args.domain),
        training_seed=int(args.training_seed),
    )
    base_config = frozen_config()
    effective_config = replace(base_config, roots_per_iteration=EFFECTIVE_ROOTS_PER_ITERATION)
    fit_only_config = replace(base_config, roots_per_iteration=0)
    if base_config.roots_per_iteration != 64 or effective_config.roots_per_iteration != 1024:
        raise RuntimeError("x16 chance-coverage multiplier contract drift")

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
                "each Phase-2 iteration collects sixteen sequential 64-root chunks before one unchanged Advantage fit."
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
                raise RuntimeError("x16 chance-coverage multiplier identity drift")
        if int(state["completed_iteration"]) != int(args.target_iteration) - 1:
            raise RuntimeError("x16 iteration start identity drift")
        state["x16_pending_iteration"] = {
            "iteration": int(args.target_iteration),
            "roots_before": int(bundle.counters["roots"]),
            "nodes_before": int(bundle.counters["nodes"]),
            "advantage_seen_before": int(bundle.adv_mem.seen),
            "strategy_seen_before": int(bundle.pol_mem.seen),
            "chunks": [],
        }
    else:
        if not args.resume:
            raise SystemExit("x16 chunk 2-16 requires previous partial checkpoint")
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

    pending = dict(state.get("x16_pending_iteration") or {})
    if int(pending.get("iteration", -1)) != int(args.target_iteration):
        raise RuntimeError("missing x16 pending-iteration state")
    chunks = list(pending.get("chunks") or [])
    if len(chunks) != int(args.root_chunk) - 1:
        raise RuntimeError("x16 chunk history length drift")

    chunk_report = _collect_chunk(
        session=session,
        bundle=bundle,
        state=state,
        target_iteration=int(args.target_iteration),
        roots=ROOTS_PER_CHUNK,
    )
    if int(chunk_report["roots"]) != ROOTS_PER_CHUNK:
        raise RuntimeError("x16 chunk did not collect exactly 64 roots")
    chunks.append(chunk_report)
    pending["chunks"] = chunks
    state["x16_pending_iteration"] = pending

    iteration_report = None
    final_report = None
    fit_seconds = 0.0
    if int(args.root_chunk) < CHUNKS_PER_ITERATION:
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
        iteration_report = _fit_only_iteration(
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
            raise RuntimeError(f"x16 iteration root total drift: {roots_added}")
        state["iteration_reports"][-1] = patched
        state["tree_collection_seconds_total"] = float(state["tree_collection_seconds_total"]) + manual_tree_seconds
        state.pop("x16_pending_iteration", None)
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
                raise RuntimeError("x16 final report root count drift")

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
        "chunks_per_iteration": CHUNKS_PER_ITERATION,
        "chance_coverage_multiplier": COVERAGE_MULTIPLIER,
        "effective_roots_per_iteration": EFFECTIVE_ROOTS_PER_ITERATION,
        "independent_training_seed_preserved": True,
        "production_deck_seed_semantics_preserved": True,
        "model_identity": contract["live_model"],
        "chunk_report": chunk_report,
        "iteration_completed": int(args.root_chunk) == CHUNKS_PER_ITERATION,
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
