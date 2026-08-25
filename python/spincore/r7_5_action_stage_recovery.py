from __future__ import annotations

"""Mechanically resumable collection for the frozen R7.5.4A action stage.

The accepted 160-root worker checkpoints only after a complete 32-root
iteration.  Three PF_DENSE_REFERENCE/THREE_HANDED cells exceeded the hosted
runner limit during iteration 2.  This module splits only the tree-collection
portion of an iteration.  The root order, deck seeds, policy calls, reservoir
updates, optimizer sequence and final checkpoint schema remain the frozen ones.
"""

import time
from pathlib import Path

from spincore.r7_5_action_checkpoint import (
    ActionProgress,
    load_action_checkpoint,
    save_action_checkpoint,
)
from spincore.r7_5_action_fit import (
    audit_action_advantage_model,
    ensemble_action_advantage_nrmse,
    fit_independent_action_advantage_member,
)
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_action_stage import (
    ActionStageConfig,
    _ensemble_checkpoint_rows,
    _make_session,
    _peak_rss_bytes,
    behavior_stats,
    restore_behavior_stats,
)
from spincore.r7_5_action_stage_contract import (
    PHASE,
    SELECTED_REPRESENTATION,
    deck_seed,
    primary_reset_seed,
    side_member_seeds,
)
from spincore.r7_5_action_uncertainty import ActionUncertaintyDampedPolicyMixture
from spincore_nn.action_models import make_advantage_action_model


PARTIAL_COLLECTION_SCHEMA = "SPINCORE_R7_5_4A_PARTIAL_COLLECTION_V1"
RECOVERY_PROVENANCE_SCHEMA = "SPINCORE_R7_5_4A_DENSE3H_RECOVERY_PROVENANCE_V1"
PARTIAL_PHASE = "partial_tree_collection"


def _identity(state: dict) -> tuple[str, str, int]:
    return (
        str(state.get("candidate_id")),
        str(state.get("domain")),
        int(state.get("training_seed", -1)),
    )


def _raw_geometry(snapshot: dict) -> dict[str, int]:
    return {
        "advantage_decision_visits": int(snapshot["advantage_decision_visits"]),
        "nominal_aggressive_branches": int(snapshot["nominal_aggressive_branches"]),
        "effective_unique_aggressive_branches": int(
            snapshot["effective_unique_aggressive_branches"]
        ),
    }


def _geometry_snapshot(raw: dict[str, int]) -> dict[str, int | float]:
    visits = int(raw["advantage_decision_visits"])
    nominal = int(raw["nominal_aggressive_branches"])
    effective = int(raw["effective_unique_aggressive_branches"])
    return {
        "advantage_decision_visits": visits,
        "nominal_aggressive_branches": nominal,
        "effective_unique_aggressive_branches": effective,
        "nominal_aggressive_branches_per_decision": (
            float(nominal) / visits if visits else 0.0
        ),
        "effective_unique_aggressive_branches_per_decision": (
            float(effective) / visits if visits else 0.0
        ),
    }


def _new_partial(state: dict, bundle, target_iteration: int) -> dict:
    expected = int(state["completed_iteration"]) + 1
    if int(target_iteration) != expected:
        raise ValueError(
            f"partial stage must advance exactly one iteration: expected {expected}, "
            f"got {target_iteration}"
        )
    return {
        "schema": PARTIAL_COLLECTION_SCHEMA,
        "target_iteration": int(target_iteration),
        "roots_collected": 0,
        "roots_before": int(bundle.counters["roots"]),
        "nodes_before": int(bundle.counters["nodes"]),
        "advantage_seen_before": int(bundle.adv_mem.seen),
        "strategy_seen_before": int(bundle.pol_mem.seen),
        "tree_collection_seconds": 0.0,
        "branch_geometry_raw": {
            "advantage_decision_visits": 0,
            "nominal_aggressive_branches": 0,
            "effective_unique_aggressive_branches": 0,
        },
        "chunk_reports": [],
    }


def _validate_partial(
    partial: dict,
    *,
    state: dict,
    bundle,
    config: ActionStageConfig,
    target_iteration: int,
) -> None:
    if partial.get("schema") != PARTIAL_COLLECTION_SCHEMA:
        raise ValueError("wrong partial-collection schema")
    if int(partial.get("target_iteration", -1)) != int(target_iteration):
        raise ValueError("partial-collection target iteration mismatch")
    if int(state["completed_iteration"]) + 1 != int(target_iteration):
        raise ValueError("partial-collection completed-iteration mismatch")
    roots_collected = int(partial.get("roots_collected", -1))
    if not 0 <= roots_collected <= int(config.roots_per_iteration):
        raise ValueError("partial-collection root count outside frozen iteration")
    if int(bundle.counters["roots"]) - int(partial["roots_before"]) != roots_collected:
        raise ValueError("partial-collection root accounting drift")


def collect_stage_root_chunk(
    *,
    bundle,
    session,
    state: dict,
    config: ActionStageConfig,
    target_iteration: int,
    root_budget: int,
    partial: dict | None = None,
) -> tuple[dict, dict]:
    """Collect the next consecutive roots without fitting an Advantage model."""
    if int(root_budget) <= 0:
        raise ValueError("positive partial root budget required")
    if not 1 <= int(target_iteration) <= int(config.total_iterations):
        raise ValueError("target iteration outside frozen stage range")
    if partial is None:
        partial = _new_partial(state, bundle, int(target_iteration))
    else:
        partial = dict(partial)
    _validate_partial(
        partial,
        state=state,
        bundle=bundle,
        config=config,
        target_iteration=int(target_iteration),
    )

    remaining = int(config.roots_per_iteration) - int(partial["roots_collected"])
    roots_this_chunk = min(int(root_budget), remaining)
    if roots_this_chunk <= 0:
        raise ValueError("partial iteration already has all frozen roots")

    scenarios = action_scenario_cycle(str(state["domain"]))
    scenario_counts = list(state["scenario_counts"])
    global_root = int(state["global_root"])
    session.collector.reset_telemetry()
    chunk_roots_before = int(bundle.counters["roots"])
    chunk_nodes_before = int(bundle.counters["nodes"])
    chunk_adv_before = int(bundle.adv_mem.seen)
    chunk_pol_before = int(bundle.pol_mem.seen)

    started = time.perf_counter()
    for _ in range(roots_this_chunk):
        scenario_index = global_root % len(scenarios)
        scenario_counts[scenario_index] += 1
        session.collect_root(
            scenarios[scenario_index],
            iteration=int(target_iteration),
            exact_opponent_levels=int(config.exact_opponent_levels),
            deck_seed=deck_seed(
                int(state["training_seed"]), global_root, int(target_iteration)
            ),
        )
        global_root += 1
    elapsed = time.perf_counter() - started

    chunk_geometry = _raw_geometry(session.collector.telemetry_snapshot())
    aggregate_geometry = dict(partial["branch_geometry_raw"])
    for key, value in chunk_geometry.items():
        aggregate_geometry[key] = int(aggregate_geometry[key]) + int(value)

    state["global_root"] = global_root
    state["scenario_counts"] = scenario_counts
    roots_added = int(bundle.counters["roots"]) - chunk_roots_before
    if roots_added != roots_this_chunk:
        raise RuntimeError("partial root accounting drift")
    chunk_report = {
        "chunk_index": len(partial["chunk_reports"]) + 1,
        "roots_added": roots_added,
        "roots_collected_after": int(partial["roots_collected"]) + roots_added,
        "nodes_added": int(bundle.counters["nodes"]) - chunk_nodes_before,
        "advantage_seen_added": int(bundle.adv_mem.seen) - chunk_adv_before,
        "strategy_seen_added": int(bundle.pol_mem.seen) - chunk_pol_before,
        "tree_collection_seconds": float(elapsed),
        "branch_geometry": _geometry_snapshot(chunk_geometry),
    }
    partial["roots_collected"] = int(partial["roots_collected"]) + roots_added
    partial["tree_collection_seconds"] = (
        float(partial["tree_collection_seconds"]) + float(elapsed)
    )
    partial["branch_geometry_raw"] = aggregate_geometry
    partial["chunk_reports"] = list(partial["chunk_reports"]) + [chunk_report]
    _validate_partial(
        partial,
        state=state,
        bundle=bundle,
        config=config,
        target_iteration=int(target_iteration),
    )
    return partial, chunk_report


def fit_collected_stage_iteration(
    *,
    bundle,
    session,
    behavior,
    state: dict,
    config: ActionStageConfig,
    target_iteration: int,
    partial: dict,
) -> dict:
    """Run the frozen post-collection fit after all 32 roots are present."""
    _validate_partial(
        partial,
        state=state,
        bundle=bundle,
        config=config,
        target_iteration=int(target_iteration),
    )
    if int(partial["roots_collected"]) != int(config.roots_per_iteration):
        raise ValueError("cannot fit before all frozen roots are collected")

    fit_started = time.perf_counter()
    reset_seed = primary_reset_seed(int(state["training_seed"]), int(target_iteration))
    # Keep this sequence byte-for-byte aligned with run_one_stage_iteration.
    session.reset_advantage_network(
        init_seed=reset_seed, lr=float(config.learning_rate)
    )
    session.train_advantage(
        steps=int(config.advantage_steps), batch_size=int(config.batch_size)
    )
    primary_nrmse = audit_action_advantage_model(
        bundle.advantage,
        bundle.adv_mem.items,
        selected_representation=SELECTED_REPRESENTATION,
        sample_size=int(config.audit_size),
        seed=int(state["training_seed"]) ^ (int(target_iteration) * 0x45D9F3B),
    )
    models = [bundle.advantage]
    member_reports = [
        {
            "member": 0,
            "role": "PRIMARY_AUTHORITATIVE_COUPLED_RNG",
            "init_seed": int(reset_seed),
            "optimizer_steps": int(config.advantage_steps),
            "final_weighted_nrmse": float(primary_nrmse),
        }
    ]
    for member in (1, 2, 3):
        init_seed, batch_seed = side_member_seeds(
            int(state["training_seed"]), int(target_iteration), member
        )
        member_started = time.perf_counter()
        model, member_report = fit_independent_action_advantage_member(
            bundle.adv_mem.items,
            selected_representation=SELECTED_REPRESENTATION,
            init_seed=init_seed,
            batch_seed=batch_seed,
            steps=int(config.advantage_steps),
            batch_size=int(config.batch_size),
            learning_rate=float(config.learning_rate),
        )
        nrmse = audit_action_advantage_model(
            model,
            bundle.adv_mem.items,
            selected_representation=SELECTED_REPRESENTATION,
            sample_size=int(config.audit_size),
            seed=(
                int(state["training_seed"])
                ^ (int(target_iteration) * 0x13579B)
                ^ (member * 0x2468AC)
            ),
        )
        member_reports.append(
            {
                **member_report,
                "member": member,
                "role": "SIDE_MEMBER_DOES_NOT_PERTURB_PRIMARY_RNG",
                "optimizer_steps": int(config.advantage_steps),
                "final_weighted_nrmse": float(nrmse),
                "fit_seconds": float(time.perf_counter() - member_started),
            }
        )
        models.append(model)
    fit_seconds = time.perf_counter() - fit_started
    behavior.models = models
    ensemble_nrmse = ensemble_action_advantage_nrmse(
        models,
        bundle.adv_mem.items,
        selected_representation=SELECTED_REPRESENTATION,
        sample_size=int(config.audit_size),
        seed=int(state["training_seed"]) ^ (int(target_iteration) * 0x5EEDBEEF),
    )

    roots_added = int(bundle.counters["roots"]) - int(partial["roots_before"])
    if roots_added != int(config.roots_per_iteration):
        raise RuntimeError("root accounting drift in recovered action iteration")
    nodes_added = int(bundle.counters["nodes"]) - int(partial["nodes_before"])
    adv_seen_added = int(bundle.adv_mem.seen) - int(partial["advantage_seen_before"])
    pol_seen_added = int(bundle.pol_mem.seen) - int(partial["strategy_seen_before"])
    tree_seconds = float(partial["tree_collection_seconds"])
    geometry = _geometry_snapshot(dict(partial["branch_geometry_raw"]))
    report = {
        "iteration": int(target_iteration),
        "roots_added": roots_added,
        "nodes_added": nodes_added,
        "tree_collection_seconds": tree_seconds,
        "tree_seconds_per_root": float(tree_seconds / roots_added),
        "advantage_fit_seconds": float(fit_seconds),
        "advantage_seen_added": adv_seen_added,
        "strategy_seen_added": pol_seen_added,
        "advantage_samples_per_root": float(adv_seen_added / roots_added),
        "strategy_samples_per_root": float(pol_seen_added / roots_added),
        "branch_geometry": geometry,
        "ensemble_weighted_nrmse": float(ensemble_nrmse),
        "ensemble_advantage_gate_pass": bool(ensemble_nrmse <= 0.75),
        "members": member_reports,
        "peak_rss_bytes": _peak_rss_bytes(),
        "mechanical_recovery": {
            "collection_chunks": len(partial["chunk_reports"]),
            "chunk_root_counts": [
                int(row["roots_added"]) for row in partial["chunk_reports"]
            ],
        },
    }
    state["completed_iteration"] = int(target_iteration)
    state["iteration_reports"] = list(state["iteration_reports"]) + [report]
    state["tree_collection_seconds_total"] = (
        float(state["tree_collection_seconds_total"]) + tree_seconds
    )
    state["advantage_fit_seconds_total"] = (
        float(state["advantage_fit_seconds_total"]) + fit_seconds
    )
    return report


def _rebuild_behavior(
    *, bundle, spec, solver, state: dict, config: ActionStageConfig, extra: dict
):
    behavior = ActionUncertaintyDampedPolicyMixture(
        selected_representation=SELECTED_REPRESENTATION,
        device="cpu",
        epsilon_scale=float(config.epsilon_scale),
        epsilon_cap=float(config.epsilon_cap),
    )
    rows = list(extra.get("behavior_ensemble") or [])
    completed = int(state["completed_iteration"])
    if completed <= 0 or len(rows) != 4:
        raise ValueError("partial recovery requires a completed four-member ensemble")
    if int(rows[0].get("member", -1)) != 0:
        raise ValueError("partial recovery primary member mismatch")
    models = [bundle.advantage]
    for member in (1, 2, 3):
        row = rows[member]
        init_seed, batch_seed = side_member_seeds(
            int(state["training_seed"]), completed, member
        )
        if int(row.get("member", -1)) != member:
            raise ValueError("partial recovery side-member ordering mismatch")
        if int(row.get("init_seed", -1)) != init_seed:
            raise ValueError("partial recovery side-member init seed mismatch")
        if int(row.get("batch_seed", -1)) != batch_seed:
            raise ValueError("partial recovery side-member batch seed mismatch")
        _, model = make_advantage_action_model(
            SELECTED_REPRESENTATION, device="cpu", seed=int(init_seed)
        )
        model.load_state_dict(row["state_dict"])
        models.append(model)
    behavior.models = models
    restore_behavior_stats(behavior, dict(extra.get("behavior_stats") or {}))
    session = _make_session(solver, bundle, spec, behavior, config)
    return session, behavior


def save_partial_collection_runtime(
    path: str | Path,
    *,
    bundle,
    behavior,
    state: dict,
    partial: dict,
    config: ActionStageConfig,
    source_execution_sha: str,
    recovery_execution_sha: str,
    recovery_provenance: dict,
) -> None:
    if int(state["completed_iteration"]) <= 0:
        raise ValueError("partial recovery requires an accepted completed iteration")
    _validate_partial(
        partial,
        state=state,
        bundle=bundle,
        config=config,
        target_iteration=int(partial["target_iteration"]),
    )
    extra = {
        "execution_sha": str(source_execution_sha),
        "stage_config": config.to_dict(),
        "stage_state": dict(state),
        "behavior_ensemble": _ensemble_checkpoint_rows(
            behavior.models,
            int(state["training_seed"]),
            int(state["completed_iteration"]),
        ),
        "behavior_stats": behavior_stats(behavior),
        "final_report": None,
        "partial_collection": dict(partial),
        "recovery_provenance": {
            "schema": RECOVERY_PROVENANCE_SCHEMA,
            "source_execution_sha": str(source_execution_sha),
            "recovery_execution_sha": str(recovery_execution_sha),
            **dict(recovery_provenance),
        },
    }
    save_action_checkpoint(
        path,
        bundle,
        ActionProgress(
            iteration=int(partial["target_iteration"]),
            phase=PARTIAL_PHASE,
            root_index=int(partial["roots_collected"]),
            advantage_optimizer_step=int(bundle.counters["adv_optimizer_steps"]),
            policy_optimizer_step=int(bundle.counters["policy_optimizer_steps"]),
        ),
        action_phase=PHASE,
        extra=extra,
    )


def load_partial_collection_runtime(
    path: str | Path,
    *,
    repo_root: str | Path,
    solver,
    candidate_id: str,
    domain: str,
    training_seed: int,
    config: ActionStageConfig,
    source_execution_sha: str,
    recovery_execution_sha: str,
):
    bundle, progress, spec, extra = load_action_checkpoint(
        path, repo_root=repo_root, device="cpu"
    )
    state = dict(extra.get("stage_state") or {})
    partial = dict(extra.get("partial_collection") or {})
    provenance = dict(extra.get("recovery_provenance") or {})
    if extra.get("execution_sha") != str(source_execution_sha):
        raise ValueError("partial recovery source execution SHA mismatch")
    if dict(extra.get("stage_config") or {}) != config.to_dict():
        raise ValueError("partial recovery stage config mismatch")
    if _identity(state) != (str(candidate_id), str(domain), int(training_seed)):
        raise ValueError("partial recovery identity mismatch")
    if progress.phase != PARTIAL_PHASE:
        raise ValueError("checkpoint is not a partial collection")
    if provenance.get("schema") != RECOVERY_PROVENANCE_SCHEMA:
        raise ValueError("partial recovery provenance schema mismatch")
    if provenance.get("recovery_execution_sha") != str(recovery_execution_sha):
        raise ValueError("partial recovery execution SHA mismatch")
    if int(progress.iteration) != int(partial.get("target_iteration", -1)):
        raise ValueError("partial recovery progress iteration mismatch")
    if int(progress.root_index) != int(partial.get("roots_collected", -1)):
        raise ValueError("partial recovery progress root mismatch")
    _validate_partial(
        partial,
        state=state,
        bundle=bundle,
        config=config,
        target_iteration=int(progress.iteration),
    )
    session, behavior = _rebuild_behavior(
        bundle=bundle,
        spec=spec,
        solver=solver,
        state=state,
        config=config,
        extra=extra,
    )
    return bundle, session, behavior, spec, state, partial, provenance


def save_recovered_stage_runtime(
    path: str | Path,
    *,
    bundle,
    behavior,
    state: dict,
    config: ActionStageConfig,
    source_execution_sha: str,
    recovery_execution_sha: str,
    recovery_provenance: dict,
    finalized: bool = False,
    final_report: dict | None = None,
) -> None:
    iteration = int(state["completed_iteration"])
    if iteration <= 0:
        raise ValueError("cannot save recovered stage before a completed iteration")
    provenance = {
        "schema": RECOVERY_PROVENANCE_SCHEMA,
        "source_execution_sha": str(source_execution_sha),
        "recovery_execution_sha": str(recovery_execution_sha),
        **dict(recovery_provenance),
    }
    if final_report is not None:
        final_report = {
            **dict(final_report),
            "mechanical_recovery_provenance": dict(provenance),
        }
    extra = {
        "execution_sha": str(source_execution_sha),
        "stage_config": config.to_dict(),
        "stage_state": dict(state),
        "behavior_ensemble": _ensemble_checkpoint_rows(
            behavior.models, int(state["training_seed"]), iteration
        ),
        "behavior_stats": behavior_stats(behavior),
        "final_report": final_report,
        "recovery_provenance": provenance,
    }
    save_action_checkpoint(
        path,
        bundle,
        ActionProgress(
            iteration=iteration,
            phase="post_policy_fit" if finalized else "post_advantage_fit",
            root_index=int(config.roots_per_iteration),
            advantage_optimizer_step=int(bundle.counters["adv_optimizer_steps"]),
            policy_optimizer_step=int(bundle.counters["policy_optimizer_steps"]),
        ),
        action_phase=PHASE,
        extra=extra,
    )
