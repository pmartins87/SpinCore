from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import resource
import sys
import time
from pathlib import Path

from spincore.deep_cfr import icm_delta_utility
from spincore.r7 import stratified_audit_indices
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3 import (
    H2_FINAL,
    H3_FINAL,
    RepresentationV3DeepCFRSession,
    make_representation_v3_bundle,
)
from spincore.r7_5_representation_v3_checkpoint import (
    RepresentationV3Progress,
    load_representation_v3_checkpoint,
    save_representation_v3_checkpoint,
)
from spincore.r7_5_representation_v3_fit import (
    audit_v3_advantage_model,
    audit_v3_policy_model,
    ensemble_v3_advantage_nrmse,
    fit_independent_v3_advantage_member,
)
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    ADVANTAGE_NRMSE_MAX,
    ADVANTAGE_STEPS,
    AUDIT_SIZE,
    BATCH_SIZE,
    CROSS_SEED_MEAN_TV_MAX,
    CROSS_SEED_OBSERVATIONS,
    CROSS_SEED_P95_TV_MAX,
    ENSEMBLE_SIZE,
    EPSILON_CAP,
    EPSILON_SCALE,
    EXACT_OPPONENT_LEVELS,
    ITERATIONS,
    LEARNING_RATE,
    MODEL_FINGERPRINTS,
    PAYOUT,
    POLICY_STEPS,
    POLICY_TV_MAX,
    RESERVOIR_CAPACITY,
    ROOTS_PER_ITERATION,
    deck_seed,
    primary_reset_seed,
    side_member_seeds,
    validate_phase2_v3_contract,
)
from spincore.r7_5_representation_v3_uncertainty import V3UncertaintyDampedPolicyMixture
from spincore_nn.models_v3_final import make_h2_final_v3, make_h3_final_v3

STAGE_STATE_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_STAGE_STATE_V1"
FINAL_REPORT_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_DOMAIN_FINAL_REPORT_V1"


@dataclass(frozen=True)
class Phase2V3StageConfig:
    roots_per_iteration: int = ROOTS_PER_ITERATION
    total_iterations: int = ITERATIONS
    exact_opponent_levels: int = EXACT_OPPONENT_LEVELS
    reservoir_capacity: int = RESERVOIR_CAPACITY
    advantage_steps: int = ADVANTAGE_STEPS
    policy_steps: int = POLICY_STEPS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    ensemble_size: int = ENSEMBLE_SIZE
    audit_size: int = AUDIT_SIZE
    epsilon_scale: float = EPSILON_SCALE
    epsilon_cap: float = EPSILON_CAP

    def to_dict(self) -> dict:
        return asdict(self)


def frozen_config() -> Phase2V3StageConfig:
    return Phase2V3StageConfig()


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _make_v3_model(representation: str, seed: int):
    if representation == H2_FINAL:
        return make_h2_final_v3(device="cpu", seed=int(seed))
    if representation == H3_FINAL:
        return make_h3_final_v3(device="cpu", seed=int(seed))
    raise ValueError("unsupported final V3 representation")


def _make_session(solver, bundle, action_spec, behavior):
    session = RepresentationV3DeepCFRSession(
        solver_library=solver,
        bundle=bundle,
        action_spec=action_spec,
        terminal_utility=icm_delta_utility(PAYOUT),
        device="cpu",
    )
    session.collector.policy = behavior
    session.collector.rng = bundle.batch_rng
    return session


def _mean_positive_regret_proxy(memory_items, *, sample_size: int, seed: int) -> dict[str, float | int]:
    indices = stratified_audit_indices(len(memory_items), int(sample_size), int(seed))
    if not indices:
        return {"samples": 0, "weighted_mean_positive_action_advantage": math.inf}
    numerator = denominator = 0.0
    positive_actions = legal_actions = 0
    for index in indices:
        sample = memory_items[index]
        values = [
            float(sample.target[action])
            for action, enabled in enumerate(sample.legal)
            if int(enabled)
        ]
        if not values:
            continue
        weight = max(0.0, float(sample.weight))
        numerator += weight * (sum(max(0.0, value) for value in values) / len(values))
        denominator += weight
        positive_actions += sum(value > 0.0 for value in values)
        legal_actions += len(values)
    return {
        "samples": len(indices),
        "weighted_mean_positive_action_advantage": (
            float(numerator / denominator) if denominator > 0.0 else math.inf
        ),
        "positive_action_fraction": (
            float(positive_actions / legal_actions) if legal_actions else math.inf
        ),
    }


def _ensemble_rows(models, *, training_seed: int, iteration: int) -> list[dict]:
    if len(models) != ENSEMBLE_SIZE:
        raise ValueError("cannot checkpoint incomplete V3 four-member ensemble")
    rows = [{
        "member": 0,
        "role": "PRIMARY_AUTHORITATIVE_COUPLED_RNG",
        "init_seed": primary_reset_seed(training_seed, iteration),
    }]
    for member in (1, 2, 3):
        init_seed, batch_seed = side_member_seeds(training_seed, iteration, member)
        rows.append({
            "member": member,
            "role": "SIDE_MEMBER_DOES_NOT_PERTURB_PRIMARY_RNG",
            "init_seed": int(init_seed),
            "batch_seed": int(batch_seed),
            "state_dict": models[member].state_dict(),
        })
    return rows


def new_phase2_v3_runtime(
    repo_root: str | Path,
    *,
    solver,
    representation: str,
    domain: str,
    training_seed: int,
    config: Phase2V3StageConfig,
):
    contract = validate_phase2_v3_contract(
        repo_root,
        representation=representation,
        domain=domain,
        training_seed=training_seed,
    )
    if config.to_dict() != frozen_config().to_dict():
        raise ValueError("Phase 2 V3 stage config is not frozen config")
    spec = contract["action_spec"]
    bundle = make_representation_v3_bundle(
        representation,
        int(training_seed),
        device="cpu",
        reservoir_capacity=config.reservoir_capacity,
        lr=config.learning_rate,
    )
    behavior = V3UncertaintyDampedPolicyMixture(
        representation=representation,
        device="cpu",
        epsilon_scale=config.epsilon_scale,
        epsilon_cap=config.epsilon_cap,
    )
    session = _make_session(solver, bundle, spec, behavior)
    scenarios = action_scenario_cycle(domain)
    state = {
        "schema": STAGE_STATE_SCHEMA,
        "representation": representation,
        "domain": domain,
        "training_seed": int(training_seed),
        "action_candidate": ACTION_CANDIDATE,
        "completed_iteration": 0,
        "global_root": 0,
        "scenario_counts": [0] * len(scenarios),
        "iteration_reports": [],
        "tree_collection_seconds_total": 0.0,
        "advantage_fit_seconds_total": 0.0,
        "policy_fit_seconds_total": 0.0,
    }
    return bundle, session, behavior, spec, state


def save_phase2_v3_runtime(
    path: str | Path,
    *,
    bundle,
    behavior,
    state: dict,
    config: Phase2V3StageConfig,
    execution_sha: str,
    finalized: bool = False,
    final_report: dict | None = None,
) -> None:
    iteration = int(state["completed_iteration"])
    if iteration <= 0:
        raise ValueError("cannot checkpoint Phase 2 before first completed iteration")
    representation = str(state["representation"])
    extra = {
        "stage_config": config.to_dict(),
        "stage_state": dict(state),
        "behavior_ensemble": _ensemble_rows(
            behavior.models,
            training_seed=int(state["training_seed"]),
            iteration=iteration,
        ),
        "behavior_stats": behavior.stats(),
        "final_report": final_report,
    }
    save_representation_v3_checkpoint(
        path,
        bundle,
        RepresentationV3Progress(
            iteration=iteration,
            global_root=int(state["global_root"]),
            advantage_optimizer_step=int(bundle.counters["adv_optimizer_steps"]),
            policy_optimizer_step=int(bundle.counters["policy_optimizer_steps"]),
            phase="post_policy_fit" if finalized else "post_advantage_fit",
        ),
        domain=str(state["domain"]),
        action_candidate=ACTION_CANDIDATE,
        execution_sha=str(execution_sha),
        architecture_fingerprint_sha256=MODEL_FINGERPRINTS[representation],
        extra=extra,
    )


def load_phase2_v3_runtime(
    path: str | Path,
    *,
    repo_root: str | Path,
    solver,
    representation: str,
    domain: str,
    training_seed: int,
    config: Phase2V3StageConfig,
    execution_sha: str,
):
    validate_phase2_v3_contract(
        repo_root,
        representation=representation,
        domain=domain,
        training_seed=training_seed,
    )
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
    if dict(extra.get("stage_config") or {}) != config.to_dict():
        raise ValueError("Phase 2 staged config mismatch")
    state = dict(extra.get("stage_state") or {})
    if state.get("schema") != STAGE_STATE_SCHEMA:
        raise ValueError("wrong Phase 2 staged state schema")
    identity = (
        state.get("representation"), state.get("domain"),
        int(state.get("training_seed", -1)), state.get("action_candidate"),
    )
    if identity != (representation, domain, int(training_seed), ACTION_CANDIDATE):
        raise ValueError("Phase 2 staged identity mismatch")
    if int(progress.iteration) != int(state["completed_iteration"]):
        raise ValueError("Phase 2 staged progress mismatch")
    if int(progress.global_root) != int(state["global_root"]):
        raise ValueError("Phase 2 staged global-root mismatch")
    if progress.phase == "post_policy_fit":
        raise ValueError("cannot resume finalized Phase 2 seed")
    if progress.phase != "post_advantage_fit":
        raise ValueError("unexpected Phase 2 checkpoint phase")

    behavior = V3UncertaintyDampedPolicyMixture(
        representation=representation,
        device="cpu",
        epsilon_scale=config.epsilon_scale,
        epsilon_cap=config.epsilon_cap,
    )
    rows = list(extra.get("behavior_ensemble") or [])
    if len(rows) != ENSEMBLE_SIZE or int(rows[0].get("member", -1)) != 0:
        raise ValueError("Phase 2 checkpoint ensemble identity mismatch")
    models = [bundle.advantage]
    iteration = int(state["completed_iteration"])
    for member in (1, 2, 3):
        row = rows[member]
        init_seed, batch_seed = side_member_seeds(training_seed, iteration, member)
        if int(row.get("member", -1)) != member:
            raise ValueError("Phase 2 side-member ordering mismatch")
        if int(row.get("init_seed", -1)) != init_seed or int(row.get("batch_seed", -1)) != batch_seed:
            raise ValueError("Phase 2 side-member seed mismatch")
        _, model = _make_v3_model(representation, init_seed)
        model.load_state_dict(row["state_dict"])
        models.append(model)
    behavior.models = models
    behavior.restore_stats(dict(extra.get("behavior_stats") or {}))
    session = _make_session(solver, bundle, spec, behavior)
    return bundle, session, behavior, spec, state


def run_one_phase2_v3_iteration(
    *,
    bundle,
    session,
    behavior,
    state: dict,
    config: Phase2V3StageConfig,
    target_iteration: int,
) -> dict:
    expected = int(state["completed_iteration"]) + 1
    if int(target_iteration) != expected:
        raise ValueError(f"Phase 2 must advance exactly one iteration: expected {expected}")
    if not 1 <= int(target_iteration) <= config.total_iterations:
        raise ValueError("Phase 2 target iteration out of range")

    scenarios = action_scenario_cycle(str(state["domain"]))
    scenario_counts = list(state["scenario_counts"])
    global_root = int(state["global_root"])
    session.collector.reset_telemetry()
    roots_before = int(bundle.counters["roots"])
    nodes_before = int(bundle.counters["nodes"])
    adv_seen_before = int(bundle.adv_mem.seen)
    pol_seen_before = int(bundle.pol_mem.seen)

    tree_started = time.perf_counter()
    for _ in range(config.roots_per_iteration):
        scenario_index = global_root % len(scenarios)
        scenario_counts[scenario_index] += 1
        session.collect_root(
            scenarios[scenario_index],
            iteration=int(target_iteration),
            exact_opponent_levels=config.exact_opponent_levels,
            deck_seed=deck_seed(int(state["training_seed"]), global_root, int(target_iteration)),
        )
        global_root += 1
    tree_seconds = time.perf_counter() - tree_started
    geometry = session.collector.telemetry_snapshot()

    fit_started = time.perf_counter()
    reset_seed = primary_reset_seed(int(state["training_seed"]), int(target_iteration))
    session.reset_advantage_network(init_seed=reset_seed, lr=config.learning_rate)
    session.train_advantage(steps=config.advantage_steps, batch_size=config.batch_size)
    primary_nrmse = audit_v3_advantage_model(
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
        init_seed, batch_seed = side_member_seeds(
            int(state["training_seed"]), int(target_iteration), member
        )
        member_started = time.perf_counter()
        model, fit_report = fit_independent_v3_advantage_member(
            bundle.adv_mem.items,
            representation=str(state["representation"]),
            init_seed=init_seed,
            batch_seed=batch_seed,
            steps=config.advantage_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
        )
        nrmse = audit_v3_advantage_model(
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
    ensemble_nrmse = ensemble_v3_advantage_nrmse(
        models,
        bundle.adv_mem.items,
        representation=str(state["representation"]),
        sample_size=config.audit_size,
        seed=int(state["training_seed"]) ^ (int(target_iteration) * 0x5EEDBEEF),
    )
    fit_seconds = time.perf_counter() - fit_started
    regret_proxy = _mean_positive_regret_proxy(
        bundle.adv_mem.items,
        sample_size=config.audit_size,
        seed=int(state["training_seed"]) ^ (int(target_iteration) * 0x27D4EB2D),
    )

    roots_added = int(bundle.counters["roots"]) - roots_before
    if roots_added != config.roots_per_iteration:
        raise RuntimeError("Phase 2 root accounting drift")
    nodes_added = int(bundle.counters["nodes"]) - nodes_before
    adv_seen_added = int(bundle.adv_mem.seen) - adv_seen_before
    pol_seen_added = int(bundle.pol_mem.seen) - pol_seen_before
    report = {
        "iteration": int(target_iteration),
        "roots_added": roots_added,
        "nodes_added": nodes_added,
        "nodes_per_root": float(nodes_added / roots_added),
        "tree_collection_seconds": float(tree_seconds),
        "tree_seconds_per_root": float(tree_seconds / roots_added),
        "advantage_fit_seconds": float(fit_seconds),
        "advantage_seen_added": adv_seen_added,
        "strategy_seen_added": pol_seen_added,
        "advantage_samples_per_root": float(adv_seen_added / roots_added),
        "strategy_samples_per_root": float(pol_seen_added / roots_added),
        "branch_geometry": geometry,
        "regret_proxy": regret_proxy,
        "ensemble_weighted_nrmse": float(ensemble_nrmse),
        "ensemble_advantage_gate_pass": bool(ensemble_nrmse <= ADVANTAGE_NRMSE_MAX),
        "members": member_reports,
        "behavior_stats_after_fit": behavior.stats(),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    state["completed_iteration"] = int(target_iteration)
    state["global_root"] = global_root
    state["scenario_counts"] = scenario_counts
    state["iteration_reports"] = list(state["iteration_reports"]) + [report]
    state["tree_collection_seconds_total"] = float(state["tree_collection_seconds_total"]) + tree_seconds
    state["advantage_fit_seconds_total"] = float(state["advantage_fit_seconds_total"]) + fit_seconds
    return report


def finalize_phase2_v3_seed(
    *,
    bundle,
    behavior,
    session,
    state: dict,
    config: Phase2V3StageConfig,
) -> dict:
    if int(state["completed_iteration"]) != config.total_iterations:
        raise ValueError("cannot finalize Phase 2 before all iterations")
    started = time.perf_counter()
    losses = session.train_average_policy(
        steps=config.policy_steps,
        batch_size=config.batch_size,
    )
    policy_seconds = time.perf_counter() - started
    if len(losses) != config.policy_steps:
        raise RuntimeError("Phase 2 AveragePolicy step-count drift")
    state["policy_fit_seconds_total"] = float(state["policy_fit_seconds_total"]) + policy_seconds

    policy_tv = audit_v3_policy_model(
        bundle.policy,
        bundle.pol_mem.items,
        representation=str(state["representation"]),
        sample_size=config.audit_size,
        seed=int(state["training_seed"]) ^ 0x71A5BEEF,
    )
    return {
        "schema": FINAL_REPORT_SCHEMA,
        "representation": str(state["representation"]),
        "domain": str(state["domain"]),
        "training_seed": int(state["training_seed"]),
        "action_candidate": ACTION_CANDIDATE,
        "iterations": config.total_iterations,
        "roots": int(bundle.counters["roots"]),
        "nodes": int(bundle.counters["nodes"]),
        "advantage_samples_seen": int(bundle.adv_mem.seen),
        "strategy_samples_seen": int(bundle.pol_mem.seen),
        "advantage_optimizer_steps_primary": int(bundle.counters["adv_optimizer_steps"]),
        "average_policy_optimizer_steps": int(bundle.counters["policy_optimizer_steps"]),
        "final_policy_weighted_mean_tv": float(policy_tv),
        "final_policy_gate_pass": bool(policy_tv <= POLICY_TV_MAX),
        "iteration_reports": list(state["iteration_reports"]),
        "behavior_stats": behavior.stats(),
        "tree_collection_seconds_total": float(state["tree_collection_seconds_total"]),
        "advantage_fit_seconds_total": float(state["advantage_fit_seconds_total"]),
        "policy_fit_seconds_total": float(state["policy_fit_seconds_total"]),
        "peak_rss_bytes": _peak_rss_bytes(),
        "cross_seed_gate_thresholds": {
            "observations_per_seed": CROSS_SEED_OBSERVATIONS,
            "mean_tv_max": CROSS_SEED_MEAN_TV_MAX,
            "p95_tv_max": CROSS_SEED_P95_TV_MAX,
        },
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
