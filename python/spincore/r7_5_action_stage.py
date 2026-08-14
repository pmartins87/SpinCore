from __future__ import annotations

from dataclasses import asdict, dataclass
import resource
import sys
import time
from pathlib import Path

import torch

from spincore.deep_cfr import icm_delta_utility
from spincore.r7_5_action_checkpoint import (
    ActionProgress,
    load_action_checkpoint,
    save_action_checkpoint,
)
from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_action_fit import (
    audit_action_advantage_model,
    audit_action_policy_model,
    ensemble_action_advantage_nrmse,
    fit_independent_action_advantage_member,
)
from spincore.r7_5_action_scenarios import action_scenario_cycle, scenario_descriptor
from spincore.r7_5_action_stage_contract import (
    PAYOUT,
    PHASE,
    SELECTED_REPRESENTATION,
    deck_seed,
    primary_reset_seed,
    side_member_seeds,
)
from spincore.r7_5_action_training import ActionDeepCFRSession, make_action_bundle
from spincore.r7_5_action_uncertainty import ActionUncertaintyDampedPolicyMixture
from spincore_nn.action_models import make_advantage_action_model

STAGE_STATE_SCHEMA = "SPINCORE_R7_5_ACTION_STAGE_STATE_V1"
FINAL_REPORT_SCHEMA = "SPINCORE_R7_5_ACTION_DOMAIN_FINAL_REPORT_V1"


@dataclass(frozen=True)
class ActionStageConfig:
    roots_per_iteration: int
    total_iterations: int
    exact_opponent_levels: int
    reservoir_capacity: int
    advantage_steps: int
    policy_steps: int
    batch_size: int
    learning_rate: float
    ensemble_size: int
    audit_size: int
    epsilon_scale: float
    epsilon_cap: float

    def __post_init__(self) -> None:
        if self.roots_per_iteration <= 0 or self.total_iterations <= 0:
            raise ValueError("positive action stage root/iteration counts required")
        if self.exact_opponent_levels < 0:
            raise ValueError("nonnegative exact-opponent level required")
        if self.reservoir_capacity <= 0 or self.batch_size <= 0 or self.audit_size <= 0:
            raise ValueError("positive action stage memory/batch/audit sizes required")
        if self.advantage_steps < 0 or self.policy_steps < 0:
            raise ValueError("nonnegative optimizer-step counts required")
        if self.learning_rate <= 0.0:
            raise ValueError("positive learning rate required")
        if self.ensemble_size != 4:
            raise ValueError("accepted R7.5 action behavior requires exactly four ensemble members")
        if self.epsilon_scale != 1.75 or self.epsilon_cap != 0.5:
            raise ValueError("accepted uncertainty coefficients changed")

    def to_dict(self) -> dict:
        return asdict(self)


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def behavior_stats(behavior: ActionUncertaintyDampedPolicyMixture) -> dict:
    return {
        "calls": int(behavior.calls),
        "epsilon_sum": float(behavior.epsilon_sum),
        "epsilon_max": float(behavior.epsilon_max),
        "disagreement_sum": float(behavior.disagreement_sum),
        "raw_epsilon_max": float(behavior.raw_epsilon_max),
        "cap_hit_calls": int(behavior.cap_hit_calls),
        "epsilon_ge_010_calls": int(behavior.epsilon_ge_010_calls),
        "epsilon_ge_025_calls": int(behavior.epsilon_ge_025_calls),
    }


def restore_behavior_stats(behavior: ActionUncertaintyDampedPolicyMixture, values: dict) -> None:
    for key, value in values.items():
        if not hasattr(behavior, key):
            raise ValueError(f"unknown uncertainty statistic {key!r}")
        setattr(behavior, key, value)


def _make_session(solver, bundle, action_spec, behavior, config: ActionStageConfig):
    session = ActionDeepCFRSession(
        solver_library=solver,
        bundle=bundle,
        action_spec=action_spec,
        terminal_utility=icm_delta_utility(PAYOUT),
        device="cpu",
    )
    session.collector.policy = behavior
    session.collector.rng = bundle.batch_rng
    return session


def new_stage_runtime(
    repo_root: str | Path,
    *,
    solver,
    candidate_id: str,
    domain: str,
    training_seed: int,
    config: ActionStageConfig,
):
    root = Path(repo_root)
    spec = postflop_candidate_specs(root)[str(candidate_id)]
    bundle = make_action_bundle(
        int(training_seed),
        domain=str(domain),
        selected_representation=SELECTED_REPRESENTATION,
        action_spec=spec,
        device="cpu",
        reservoir_capacity=int(config.reservoir_capacity),
        lr=float(config.learning_rate),
    )
    behavior = ActionUncertaintyDampedPolicyMixture(
        selected_representation=SELECTED_REPRESENTATION,
        device="cpu",
        epsilon_scale=float(config.epsilon_scale),
        epsilon_cap=float(config.epsilon_cap),
    )
    session = _make_session(solver, bundle, spec, behavior, config)
    scenarios = action_scenario_cycle(str(domain))
    state = {
        "schema": STAGE_STATE_SCHEMA,
        "candidate_id": str(candidate_id),
        "domain": str(domain),
        "training_seed": int(training_seed),
        "completed_iteration": 0,
        "global_root": 0,
        "scenario_counts": [0] * len(scenarios),
        "iteration_reports": [],
        "tree_collection_seconds_total": 0.0,
        "advantage_fit_seconds_total": 0.0,
        "policy_fit_seconds_total": 0.0,
    }
    return bundle, session, behavior, spec, state


def _ensemble_checkpoint_rows(models, training_seed: int, iteration: int) -> list[dict]:
    if len(models) != 4:
        raise ValueError("cannot checkpoint incomplete four-member action ensemble")
    rows = [
        {
            "member": 0,
            "role": "PRIMARY_AUTHORITATIVE_COUPLED_RNG",
            "init_seed": primary_reset_seed(training_seed, iteration),
        }
    ]
    for member in (1, 2, 3):
        init_seed, batch_seed = side_member_seeds(training_seed, iteration, member)
        rows.append(
            {
                "member": member,
                "role": "SIDE_MEMBER_DOES_NOT_PERTURB_PRIMARY_RNG",
                "init_seed": int(init_seed),
                "batch_seed": int(batch_seed),
                "state_dict": models[member].state_dict(),
            }
        )
    return rows


def save_stage_runtime(
    path: str | Path,
    *,
    bundle,
    behavior,
    state: dict,
    config: ActionStageConfig,
    execution_sha: str,
    finalized: bool = False,
    final_report: dict | None = None,
) -> None:
    if not execution_sha.strip():
        raise ValueError("immutable execution SHA required for staged checkpoint")
    iteration = int(state["completed_iteration"])
    if iteration <= 0:
        raise ValueError("cannot checkpoint before first completed iteration")
    extra = {
        "execution_sha": str(execution_sha),
        "stage_config": config.to_dict(),
        "stage_state": dict(state),
        "behavior_ensemble": _ensemble_checkpoint_rows(behavior.models, int(state["training_seed"]), iteration),
        "behavior_stats": behavior_stats(behavior),
        "final_report": final_report,
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


def load_stage_runtime(
    path: str | Path,
    *,
    repo_root: str | Path,
    solver,
    candidate_id: str,
    domain: str,
    training_seed: int,
    config: ActionStageConfig,
    execution_sha: str,
):
    bundle, progress, spec, extra = load_action_checkpoint(path, repo_root=repo_root, device="cpu")
    state = dict(extra.get("stage_state") or {})
    if state.get("schema") != STAGE_STATE_SCHEMA:
        raise ValueError("wrong staged action state schema")
    if extra.get("execution_sha") != str(execution_sha):
        raise ValueError("staged action execution SHA mismatch")
    if dict(extra.get("stage_config") or {}) != config.to_dict():
        raise ValueError("staged action config mismatch")
    identity = (
        state.get("candidate_id"),
        state.get("domain"),
        int(state.get("training_seed", -1)),
    )
    if identity != (str(candidate_id), str(domain), int(training_seed)):
        raise ValueError("staged action identity mismatch")
    if int(progress.iteration) != int(state["completed_iteration"]):
        raise ValueError("staged action progress mismatch")
    if progress.phase == "post_policy_fit":
        raise ValueError("cannot resume already finalized action seed")
    if progress.phase != "post_advantage_fit":
        raise ValueError("unexpected staged action checkpoint phase")

    behavior = ActionUncertaintyDampedPolicyMixture(
        selected_representation=SELECTED_REPRESENTATION,
        device="cpu",
        epsilon_scale=float(config.epsilon_scale),
        epsilon_cap=float(config.epsilon_cap),
    )
    rows = list(extra.get("behavior_ensemble") or [])
    if len(rows) != 4:
        raise ValueError("staged checkpoint missing four-member ensemble")
    models = [bundle.advantage]
    if int(rows[0].get("member", -1)) != 0:
        raise ValueError("staged checkpoint primary member mismatch")
    for member in (1, 2, 3):
        row = rows[member]
        init_seed, batch_seed = side_member_seeds(training_seed, int(state["completed_iteration"]), member)
        if int(row.get("member", -1)) != member:
            raise ValueError("staged checkpoint side-member ordering mismatch")
        if int(row.get("init_seed", -1)) != init_seed or int(row.get("batch_seed", -1)) != batch_seed:
            raise ValueError("staged checkpoint side-member seed mismatch")
        _, model = make_advantage_action_model(
            SELECTED_REPRESENTATION,
            device="cpu",
            seed=int(init_seed),
        )
        model.load_state_dict(row["state_dict"])
        models.append(model)
    behavior.models = models
    restore_behavior_stats(behavior, dict(extra.get("behavior_stats") or {}))
    session = _make_session(solver, bundle, spec, behavior, config)
    return bundle, session, behavior, spec, state


def run_one_stage_iteration(
    *,
    bundle,
    session,
    behavior,
    state: dict,
    config: ActionStageConfig,
    target_iteration: int,
) -> dict:
    expected = int(state["completed_iteration"]) + 1
    if int(target_iteration) != expected:
        raise ValueError(f"stage must advance exactly one iteration: expected {expected}, got {target_iteration}")
    if not 1 <= int(target_iteration) <= int(config.total_iterations):
        raise ValueError("target iteration outside frozen stage range")

    scenarios = action_scenario_cycle(str(state["domain"]))
    scenario_counts = list(state["scenario_counts"])
    global_root = int(state["global_root"])
    session.collector.reset_telemetry()
    roots_before = int(bundle.counters["roots"])
    nodes_before = int(bundle.counters["nodes"])
    adv_seen_before = int(bundle.adv_mem.seen)
    pol_seen_before = int(bundle.pol_mem.seen)

    tree_started = time.perf_counter()
    for _ in range(int(config.roots_per_iteration)):
        scenario_index = global_root % len(scenarios)
        scenario_counts[scenario_index] += 1
        session.collect_root(
            scenarios[scenario_index],
            iteration=int(target_iteration),
            exact_opponent_levels=int(config.exact_opponent_levels),
            deck_seed=deck_seed(int(state["training_seed"]), global_root, int(target_iteration)),
        )
        global_root += 1
    tree_seconds = time.perf_counter() - tree_started
    geometry = session.collector.telemetry_snapshot()

    fit_started = time.perf_counter()
    reset_seed = primary_reset_seed(int(state["training_seed"]), int(target_iteration))
    session.reset_advantage_network(init_seed=reset_seed, lr=float(config.learning_rate))
    session.train_advantage(steps=int(config.advantage_steps), batch_size=int(config.batch_size))
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
        model, report = fit_independent_action_advantage_member(
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
            seed=int(state["training_seed"]) ^ (int(target_iteration) * 0x13579B) ^ (member * 0x2468AC),
        )
        member_reports.append(
            {
                **report,
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

    roots_added = int(bundle.counters["roots"]) - roots_before
    if roots_added != int(config.roots_per_iteration):
        raise RuntimeError("root accounting drift in staged action iteration")
    nodes_added = int(bundle.counters["nodes"]) - nodes_before
    adv_seen_added = int(bundle.adv_mem.seen) - adv_seen_before
    pol_seen_added = int(bundle.pol_mem.seen) - pol_seen_before
    report = {
        "iteration": int(target_iteration),
        "roots_added": roots_added,
        "nodes_added": nodes_added,
        "tree_collection_seconds": float(tree_seconds),
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
    }
    state["completed_iteration"] = int(target_iteration)
    state["global_root"] = global_root
    state["scenario_counts"] = scenario_counts
    state["iteration_reports"] = list(state["iteration_reports"]) + [report]
    state["tree_collection_seconds_total"] = float(state["tree_collection_seconds_total"]) + tree_seconds
    state["advantage_fit_seconds_total"] = float(state["advantage_fit_seconds_total"]) + fit_seconds
    return report


def finalize_stage_seed(*, bundle, behavior, session, state: dict, config: ActionStageConfig) -> dict:
    if int(state["completed_iteration"]) != int(config.total_iterations):
        raise ValueError("cannot finalize before all staged iterations")
    started = time.perf_counter()
    losses = session.train_average_policy(steps=int(config.policy_steps), batch_size=int(config.batch_size))
    policy_seconds = time.perf_counter() - started
    if len(losses) != int(config.policy_steps):
        raise RuntimeError("AveragePolicy optimizer-step count drift")
    state["policy_fit_seconds_total"] = float(state["policy_fit_seconds_total"]) + policy_seconds

    policy_tv = audit_action_policy_model(
        bundle.policy,
        bundle.pol_mem.items,
        selected_representation=SELECTED_REPRESENTATION,
        sample_size=int(config.audit_size),
        seed=int(state["training_seed"]) ^ 0x2468ACE0,
    )
    ensemble_nrmse = ensemble_action_advantage_nrmse(
        behavior.models,
        bundle.adv_mem.items,
        selected_representation=SELECTED_REPRESENTATION,
        sample_size=int(config.audit_size),
        seed=int(state["training_seed"]) ^ 0x13572468,
    )
    roots = int(bundle.counters["roots"])
    nodes = int(bundle.counters["nodes"])
    visits = sum(int(row["branch_geometry"]["advantage_decision_visits"]) for row in state["iteration_reports"])
    effective = sum(int(row["branch_geometry"]["effective_unique_aggressive_branches"]) for row in state["iteration_reports"])
    nominal = sum(int(row["branch_geometry"]["nominal_aggressive_branches"]) for row in state["iteration_reports"])
    scenarios = action_scenario_cycle(str(state["domain"]))
    total_compute = (
        float(state["tree_collection_seconds_total"])
        + float(state["advantage_fit_seconds_total"])
        + float(state["policy_fit_seconds_total"])
    )
    return {
        "schema": FINAL_REPORT_SCHEMA,
        "candidate_id": str(state["candidate_id"]),
        "domain": str(state["domain"]),
        "training_seed": int(state["training_seed"]),
        "selected_representation": SELECTED_REPRESENTATION,
        "iterations": int(config.total_iterations),
        "roots_per_iteration": int(config.roots_per_iteration),
        "roots": roots,
        "nodes": nodes,
        "nodes_per_root": float(nodes / roots),
        "tree_collection_seconds": float(state["tree_collection_seconds_total"]),
        "tree_seconds_per_root": float(state["tree_collection_seconds_total"] / roots),
        "seconds_per_root": float(state["tree_collection_seconds_total"] / roots),
        "advantage_fit_seconds": float(state["advantage_fit_seconds_total"]),
        "policy_fit_seconds": float(state["policy_fit_seconds_total"]),
        "full_training_compute_seconds": total_compute,
        "full_training_seconds_per_root": float(total_compute / roots),
        "advantage_samples": len(bundle.adv_mem.items),
        "advantage_seen": int(bundle.adv_mem.seen),
        "strategy_samples": len(bundle.pol_mem.items),
        "strategy_seen": int(bundle.pol_mem.seen),
        "advantage_samples_per_root": float(bundle.adv_mem.seen / roots),
        "strategy_samples_per_root": float(bundle.pol_mem.seen / roots),
        "advantage_decision_visits": visits,
        "nominal_aggressive_branches_per_decision": float(nominal / visits) if visits else 0.0,
        "effective_unique_aggressive_branches_per_decision": float(effective / visits) if visits else 0.0,
        "peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in state["iteration_reports"]),
        "ensemble_advantage_weighted_nrmse": float(ensemble_nrmse),
        "policy_weighted_mean_tv": float(policy_tv),
        "advantage_gate_pass": bool(ensemble_nrmse <= 0.75),
        "policy_gate_pass": bool(policy_tv <= 0.12),
        "scenario_counts": [
            {"scenario": scenario_descriptor(ep), "root_count": int(count)}
            for ep, count in zip(scenarios, state["scenario_counts"])
        ],
        "all_scenarios_exercised": all(int(count) > 0 for count in state["scenario_counts"]),
        "uncertainty_runtime_statistics": {
            **behavior_stats(behavior),
            "mean_epsilon": float(behavior.epsilon_sum / behavior.calls) if behavior.calls else 0.0,
            "mean_disagreement": float(behavior.disagreement_sum / behavior.calls) if behavior.calls else 0.0,
        },
        "iteration_reports": list(state["iteration_reports"]),
        "primary_advantage_optimizer_steps": int(bundle.counters["adv_optimizer_steps"]),
        "side_advantage_optimizer_steps": int(3 * config.advantage_steps * config.total_iterations),
        "average_policy_optimizer_steps": int(bundle.counters["policy_optimizer_steps"]),
        "strategic_selection_permitted_at_160": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
