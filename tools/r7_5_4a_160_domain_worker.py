from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path

import torch

import r7_4_stability_pilot_worker as r74
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
from spincore.r7_5_action_training import ActionDeepCFRSession, make_action_bundle
from spincore.r7_5_action_uncertainty import ActionUncertaintyDampedPolicyMixture
from spincore.solver import SolverLibrary
from spincore_nn.action_models import make_advantage_action_model


SCHEMA = "SPINCORE_R7_5_4A_160_DOMAIN_STAGE_V1"
FINAL_SCHEMA = "SPINCORE_R7_5_4A_160_DOMAIN_FINAL_V1"
RUNNER_FREEZE_SCHEMA = "SPINCORE_R7_5_4_RUNNER_IMPLEMENTATION_FREEZE_V1"
TRAINING_FREEZE_SCHEMA = "SPINCORE_R7_5_4_TRAINING_IMPLEMENTATION_FREEZE_V1"
PREFLIGHT_SCHEMA = "SPINCORE_R7_5_4_STRATEGIC_PREFLIGHT_V5"
REP_RESULT_SCHEMA = "SPINCORE_R7_5_3_REPRESENTATION_ABLATION_RESULT_V1"
SELECTED_REPRESENTATION = "C0_V1_FROZEN_CONTROL"
PHASE = "R7_5_4A_POSTFLOP"
ROOT_LEVEL = 160
ROOTS_PER_ITERATION = 32
ITERATIONS = 5
EXACT_OPPONENT_LEVELS = 2
RESERVOIR_CAPACITY = 100000
ADVANTAGE_STEPS = 4096
POLICY_STEPS = 16384
BATCH_SIZE = 256
LEARNING_RATE = 0.001
ENSEMBLE_SIZE = 4
AUDIT_SIZE = 2048
EPSILON_SCALE = 1.75
EPSILON_CAP = 0.5
TORCH_THREADS = 2
PAYOUT = (0.5, 0.3, 0.2)
POSTFLOP_TRAINING_SEEDS = (1737995611, 645939859, 1311335590)
MEMBER_INIT_XOR = 0x0E115EED
MEMBER_BATCH_XOR = 0xBA7C8A11


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def primary_reset_seed(training_seed: int, iteration: int) -> int:
    return (int(training_seed) ^ (int(iteration) * 0x9E3779B1)) & 0x7FFFFFFF


def side_member_seeds(training_seed: int, iteration: int, member: int) -> tuple[int, int]:
    if int(member) not in (1, 2, 3):
        raise ValueError("R7.5.4A side member must be 1, 2 or 3")
    init_seed = (
        int(training_seed)
        ^ MEMBER_INIT_XOR
        ^ (int(iteration) * 0x9E3779B1)
        ^ (int(member) * 0x045D9F3B)
    ) & 0x7FFFFFFF
    batch_seed = (
        int(training_seed)
        ^ MEMBER_BATCH_XOR
        ^ (int(iteration) * 0x85EBCA77)
        ^ (int(member) * 0xC2B2AE3D)
    ) & ((1 << 64) - 1)
    return int(init_seed), int(batch_seed)


def _behavior_stats(behavior: ActionUncertaintyDampedPolicyMixture) -> dict:
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


def _restore_behavior_stats(behavior: ActionUncertaintyDampedPolicyMixture, payload: dict) -> None:
    for key, value in payload.items():
        if not hasattr(behavior, key):
            raise ValueError(f"unknown uncertainty behavior statistic {key!r}")
        setattr(behavior, key, value)


def _validate_repo_contract(repo_root: Path, *, candidate: str, seed: int, execution_sha: str) -> dict:
    validation = repo_root / "validation"
    runner = _read(validation / "R7_5_4_RUNNER_IMPLEMENTATION_FREEZE_20260814.json")
    training = _read(validation / "R7_5_4_TRAINING_IMPLEMENTATION_FREEZE.json")
    preflight = _read(validation / "R7_5_4A_160_STRATEGIC_PREFLIGHT.json")
    rep = _read(validation / "R7_5_3_REPRESENTATION_ABLATION_RESULT.json")
    v3 = _read(validation / "R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V3.json")
    cost = _read(validation / "R7_5_4_COST_TELEMETRY_SEMANTIC_FREEZE_20260814.json")

    if runner.get("schema") != RUNNER_FREEZE_SCHEMA:
        raise ValueError("R7.5.4A runner freeze schema mismatch")
    if training.get("schema") != TRAINING_FREEZE_SCHEMA:
        raise ValueError("R7.5.4A training freeze schema mismatch")
    if preflight.get("schema") != PREFLIGHT_SCHEMA or not bool(preflight.get("ready_to_start")):
        raise ValueError("R7.5.4A 160 durable strategic preflight is not PASS")
    if preflight.get("selected_representation") != SELECTED_REPRESENTATION:
        raise ValueError("preflight representation differs from C0 winner")
    if rep.get("schema") != REP_RESULT_SCHEMA or not bool(rep.get("r7_5_3_representation_ablation_pass")):
        raise ValueError("R7.5.3 representation gate is not PASS")
    if rep.get("selected_candidate") != SELECTED_REPRESENTATION:
        raise ValueError("R7.5.4A worker only supports durable C0 winner")
    if list(v3["seed_derivation"]["postflop_training_seeds"]) != list(POSTFLOP_TRAINING_SEEDS):
        raise ValueError("R7.5.4A V3 training seed drift")
    if int(seed) not in POSTFLOP_TRAINING_SEEDS:
        raise ValueError("R7.5.4A worker received non-frozen postflop training seed")

    rt = runner["training"]
    exact = {
        "reservoir_capacity": RESERVOIR_CAPACITY,
        "final_fit_audit_size": AUDIT_SIZE,
        "iterations": ITERATIONS,
        "exact_opponent_levels": EXACT_OPPONENT_LEVELS,
        "advantage_optimizer_steps_per_member_per_iteration": ADVANTAGE_STEPS,
        "average_policy_optimizer_steps": POLICY_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "ensemble_size": ENSEMBLE_SIZE,
        "epsilon_scale": EPSILON_SCALE,
        "epsilon_cap": EPSILON_CAP,
    }
    for key, expected in exact.items():
        if rt.get(key) != expected:
            raise ValueError(f"R7.5.4A runner freeze drift for {key}: {rt.get(key)!r} != {expected!r}")
    level = runner["root_levels"][str(ROOT_LEVEL)]
    if int(level["roots_per_iteration"]) != ROOTS_PER_ITERATION:
        raise ValueError("R7.5.4A 160 roots-per-iteration drift")
    runtime = runner["runtime_for_github_ablation"]
    if int(runtime["torch_threads"]) != TORCH_THREADS:
        raise ValueError("R7.5.4A GitHub torch-thread contract drift")
    if cost.get("schema") != "SPINCORE_R7_5_4_COST_TELEMETRY_SEMANTIC_FREEZE_V1":
        raise ValueError("R7.5.4A cost telemetry freeze missing")
    specs = postflop_candidate_specs(repo_root)
    if candidate not in specs:
        raise ValueError("unknown R7.5.4A candidate")
    if not execution_sha.strip():
        raise ValueError("immutable execution SHA is required")
    return {"runner": runner, "training": training, "preflight": preflight, "rep": rep, "v3": v3, "cost": cost}


def _new_runtime(repo_root: Path, *, solver, candidate: str, domain: str, seed: int):
    spec = postflop_candidate_specs(repo_root)[candidate]
    bundle = make_action_bundle(
        int(seed),
        domain=domain,
        selected_representation=SELECTED_REPRESENTATION,
        action_spec=spec,
        device="cpu",
        reservoir_capacity=RESERVOIR_CAPACITY,
        lr=LEARNING_RATE,
    )
    session = ActionDeepCFRSession(
        solver_library=solver,
        bundle=bundle,
        action_spec=spec,
        terminal_utility=icm_delta_utility(PAYOUT),
        device="cpu",
    )
    behavior = ActionUncertaintyDampedPolicyMixture(
        selected_representation=SELECTED_REPRESENTATION,
        device="cpu",
        epsilon_scale=EPSILON_SCALE,
        epsilon_cap=EPSILON_CAP,
    )
    session.collector.policy = behavior
    session.collector.rng = bundle.batch_rng
    state = {
        "schema": SCHEMA,
        "phase": PHASE,
        "root_level": ROOT_LEVEL,
        "candidate": candidate,
        "domain": domain,
        "training_seed": int(seed),
        "completed_iteration": 0,
        "global_root": 0,
        "scenario_counts": [0] * len(r74._scenario_cycle(domain)),
        "iteration_reports": [],
        "tree_collection_seconds_total": 0.0,
        "advantage_fit_seconds_total": 0.0,
        "policy_fit_seconds_total": 0.0,
    }
    return bundle, session, behavior, spec, state


def _restore_ensemble(bundle, behavior, extra: dict):
    rows = list(extra.get("behavior_ensemble") or [])
    if not rows:
        raise ValueError("resumed R7.5.4A checkpoint is missing four-member behavior ensemble")
    if len(rows) != ENSEMBLE_SIZE:
        raise ValueError("resumed R7.5.4A ensemble width mismatch")
    models = [bundle.advantage]
    first = rows[0]
    if int(first.get("member", -1)) != 0 or first.get("role") != "PRIMARY_AUTHORITATIVE_COUPLED_RNG":
        raise ValueError("invalid resumed primary ensemble descriptor")
    for member, row in enumerate(rows[1:], start=1):
        if int(row.get("member", -1)) != member:
            raise ValueError("resumed side-member order mismatch")
        expected_init, expected_batch = side_member_seeds(
            int(extra["worker_state"]["training_seed"]),
            int(extra["worker_state"]["completed_iteration"]),
            member,
        )
        if int(row.get("init_seed", -1)) != expected_init or int(row.get("batch_seed", -1)) != expected_batch:
            raise ValueError("resumed side-member seed provenance mismatch")
        _, model = make_advantage_action_model(
            SELECTED_REPRESENTATION,
            device="cpu",
            seed=expected_init,
        )
        model.load_state_dict(row["state_dict"])
        models.append(model)
    behavior.models = models
    _restore_behavior_stats(behavior, dict(extra.get("behavior_stats") or {}))


def _load_runtime(repo_root: Path, *, solver, checkpoint: Path, candidate: str, domain: str, seed: int, execution_sha: str):
    bundle, progress, spec, extra = load_action_checkpoint(checkpoint, repo_root=repo_root, device="cpu")
    state = dict(extra.get("worker_state") or {})
    if state.get("schema") != SCHEMA:
        raise ValueError("wrong R7.5.4A staged worker state schema")
    if state.get("candidate") != candidate or state.get("domain") != domain or int(state.get("training_seed", -1)) != int(seed):
        raise ValueError("R7.5.4A resumed state identity mismatch")
    if state.get("root_level") != ROOT_LEVEL or state.get("phase") != PHASE:
        raise ValueError("R7.5.4A resumed phase/root-level mismatch")
    if extra.get("execution_sha") != execution_sha:
        raise ValueError("R7.5.4A resumed checkpoint execution SHA mismatch")
    if progress.phase not in ("post_advantage_fit", "post_policy_fit"):
        raise ValueError("R7.5.4A checkpoint progress phase mismatch")
    if int(progress.iteration) != int(state["completed_iteration"]):
        raise ValueError("R7.5.4A checkpoint iteration mismatch")
    session = ActionDeepCFRSession(
        solver_library=solver,
        bundle=bundle,
        action_spec=spec,
        terminal_utility=icm_delta_utility(PAYOUT),
        device="cpu",
    )
    behavior = ActionUncertaintyDampedPolicyMixture(
        selected_representation=SELECTED_REPRESENTATION,
        device="cpu",
        epsilon_scale=EPSILON_SCALE,
        epsilon_cap=EPSILON_CAP,
    )
    session.collector.policy = behavior
    session.collector.rng = bundle.batch_rng
    _restore_ensemble(bundle, behavior, extra)
    return bundle, session, behavior, spec, state, progress


def _ensemble_payload(models, training_seed: int, iteration: int) -> list[dict]:
    rows = [
        {
            "member": 0,
            "role": "PRIMARY_AUTHORITATIVE_COUPLED_RNG",
            "init_seed": primary_reset_seed(training_seed, iteration),
        }
    ]
    for member in range(1, ENSEMBLE_SIZE):
        init_seed, batch_seed = side_member_seeds(training_seed, iteration, member)
        rows.append(
            {
                "member": member,
                "role": "SIDE_MEMBER_DOES_NOT_PERTURB_PRIMARY_RNG",
                "init_seed": init_seed,
                "batch_seed": batch_seed,
                "state_dict": models[member].state_dict(),
            }
        )
    return rows


def _run_iteration(*, bundle, session, behavior, state: dict, iteration: int, seed: int, domain: str) -> dict:
    expected = int(state["completed_iteration"]) + 1
    if int(iteration) != expected or not (1 <= int(iteration) <= ITERATIONS):
        raise ValueError(f"R7.5.4A staged iteration must advance exactly: expected {expected}, got {iteration}")
    scenarios = r74._scenario_cycle(domain)
    scenario_counts = list(state["scenario_counts"])
    global_root = int(state["global_root"])
    session.collector.reset_telemetry()
    roots_before = int(bundle.counters["roots"])
    nodes_before = int(bundle.counters["nodes"])
    advantage_seen_before = int(bundle.adv_mem.seen)
    strategy_seen_before = int(bundle.pol_mem.seen)

    tree_started = time.perf_counter()
    for _ in range(ROOTS_PER_ITERATION):
        scenario_index = global_root % len(scenarios)
        episode = scenarios[scenario_index]
        scenario_counts[scenario_index] += 1
        deck_seed = (int(seed) * 1_000_003 + global_root * 97 + int(iteration)) & ((1 << 64) - 1)
        session.collect_root(
            episode,
            iteration=int(iteration),
            exact_opponent_levels=EXACT_OPPONENT_LEVELS,
            deck_seed=int(deck_seed),
        )
        global_root += 1
    tree_seconds = time.perf_counter() - tree_started
    geometry = session.collector.telemetry_snapshot()

    fit_started = time.perf_counter()
    reset_seed = primary_reset_seed(seed, iteration)
    session.reset_advantage_network(init_seed=reset_seed, lr=LEARNING_RATE)
    session.train_advantage(steps=ADVANTAGE_STEPS, batch_size=BATCH_SIZE)
    primary_nrmse = audit_action_advantage_model(
        bundle.advantage,
        bundle.adv_mem.items,
        selected_representation=SELECTED_REPRESENTATION,
        sample_size=AUDIT_SIZE,
        seed=int(seed) ^ (int(iteration) * 0x45D9F3B),
        device="cpu",
    )
    models = [bundle.advantage]
    member_reports = [
        {
            "member": 0,
            "role": "PRIMARY_AUTHORITATIVE_COUPLED_RNG",
            "init_seed": int(reset_seed),
            "optimizer_steps": ADVANTAGE_STEPS,
            "final_weighted_nrmse": float(primary_nrmse),
        }
    ]
    for member in range(1, ENSEMBLE_SIZE):
        init_seed, batch_seed = side_member_seeds(seed, iteration, member)
        member_started = time.perf_counter()
        model, report = fit_independent_action_advantage_member(
            bundle.adv_mem.items,
            selected_representation=SELECTED_REPRESENTATION,
            init_seed=init_seed,
            batch_seed=batch_seed,
            steps=ADVANTAGE_STEPS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            device="cpu",
        )
        member_seconds = time.perf_counter() - member_started
        nrmse = audit_action_advantage_model(
            model,
            bundle.adv_mem.items,
            selected_representation=SELECTED_REPRESENTATION,
            sample_size=AUDIT_SIZE,
            seed=int(seed) ^ (int(iteration) * 0x13579B) ^ (int(member) * 0x2468AC),
            device="cpu",
        )
        report = dict(report)
        report.update(
            {
                "member": int(member),
                "role": "SIDE_MEMBER_DOES_NOT_PERTURB_PRIMARY_RNG",
                "optimizer_steps": ADVANTAGE_STEPS,
                "final_weighted_nrmse": float(nrmse),
                "fit_seconds": float(member_seconds),
            }
        )
        models.append(model)
        member_reports.append(report)
    advantage_fit_seconds = time.perf_counter() - fit_started
    behavior.models = models
    ensemble_nrmse = ensemble_action_advantage_nrmse(
        models,
        bundle.adv_mem.items,
        selected_representation=SELECTED_REPRESENTATION,
        sample_size=AUDIT_SIZE,
        seed=int(seed) ^ (int(iteration) * 0x5EEDBEEF),
        device="cpu",
    )

    roots_added = int(bundle.counters["roots"]) - roots_before
    nodes_added = int(bundle.counters["nodes"]) - nodes_before
    if roots_added != ROOTS_PER_ITERATION:
        raise RuntimeError("R7.5.4A worker root accounting drift")
    advantage_seen_added = int(bundle.adv_mem.seen) - advantage_seen_before
    strategy_seen_added = int(bundle.pol_mem.seen) - strategy_seen_before
    iteration_report = {
        "iteration": int(iteration),
        "roots_added": roots_added,
        "nodes_added": nodes_added,
        "tree_collection_seconds": float(tree_seconds),
        "tree_seconds_per_root": float(tree_seconds / roots_added),
        "advantage_fit_seconds": float(advantage_fit_seconds),
        "advantage_seen_added": advantage_seen_added,
        "strategy_seen_added": strategy_seen_added,
        "advantage_samples_per_root": float(advantage_seen_added / roots_added),
        "strategy_samples_per_root": float(strategy_seen_added / roots_added),
        "branch_geometry": geometry,
        "ensemble_weighted_nrmse": float(ensemble_nrmse),
        "ensemble_advantage_gate_pass": bool(ensemble_nrmse <= 0.75),
        "members": member_reports,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    state["completed_iteration"] = int(iteration)
    state["global_root"] = int(global_root)
    state["scenario_counts"] = scenario_counts
    state["tree_collection_seconds_total"] = float(state["tree_collection_seconds_total"]) + tree_seconds
    state["advantage_fit_seconds_total"] = float(state["advantage_fit_seconds_total"]) + advantage_fit_seconds
    state["iteration_reports"] = list(state["iteration_reports"]) + [iteration_report]
    return iteration_report


def _finalize(*, bundle, session, behavior, state: dict, seed: int, domain: str, candidate: str) -> dict:
    if int(state["completed_iteration"]) != ITERATIONS:
        raise ValueError("R7.5.4A cannot finalize before iteration 5")
    policy_started = time.perf_counter()
    losses = session.train_average_policy(steps=POLICY_STEPS, batch_size=BATCH_SIZE)
    policy_seconds = time.perf_counter() - policy_started
    if len(losses) != POLICY_STEPS:
        raise RuntimeError("R7.5.4A policy optimizer-step count drift")
    state["policy_fit_seconds_total"] = float(state["policy_fit_seconds_total"]) + policy_seconds
    policy_tv = audit_action_policy_model(
        bundle.policy,
        bundle.pol_mem.items,
        selected_representation=SELECTED_REPRESENTATION,
        sample_size=AUDIT_SIZE,
        seed=int(seed) ^ 0x2468ACE0,
        device="cpu",
    )
    ensemble_nrmse = ensemble_action_advantage_nrmse(
        behavior.models,
        bundle.adv_mem.items,
        selected_representation=SELECTED_REPRESENTATION,
        sample_size=AUDIT_SIZE,
        seed=int(seed) ^ 0x13572468,
        device="cpu",
    )
    roots = int(bundle.counters["roots"])
    nodes = int(bundle.counters["nodes"])
    total_compute = (
        float(state["tree_collection_seconds_total"])
        + float(state["advantage_fit_seconds_total"])
        + float(state["policy_fit_seconds_total"])
    )
    branch_visits = sum(int(row["branch_geometry"]["advantage_decision_visits"]) for row in state["iteration_reports"])
    effective_branches = sum(int(row["branch_geometry"]["effective_unique_aggressive_branches"]) for row in state["iteration_reports"])
    nominal_branches = sum(int(row["branch_geometry"]["nominal_aggressive_branches"]) for row in state["iteration_reports"])
    scenarios = r74._scenario_cycle(domain)
    return {
        "schema": FINAL_SCHEMA,
        "candidate": candidate,
        "domain": domain,
        "training_seed": int(seed),
        "selected_representation": SELECTED_REPRESENTATION,
        "root_level": ROOT_LEVEL,
        "iterations": ITERATIONS,
        "roots_per_iteration": ROOTS_PER_ITERATION,
        "roots": roots,
        "nodes": nodes,
        "nodes_per_root": float(nodes / roots),
        "tree_collection_seconds": float(state["tree_collection_seconds_total"]),
        "tree_seconds_per_root": float(state["tree_collection_seconds_total"] / roots),
        "seconds_per_root": float(state["tree_collection_seconds_total"] / roots),
        "full_training_compute_seconds": total_compute,
        "full_training_seconds_per_root": float(total_compute / roots),
        "advantage_fit_seconds": float(state["advantage_fit_seconds_total"]),
        "policy_fit_seconds": float(state["policy_fit_seconds_total"]),
        "advantage_samples": len(bundle.adv_mem.items),
        "advantage_seen": int(bundle.adv_mem.seen),
        "strategy_samples": len(bundle.pol_mem.items),
        "strategy_seen": int(bundle.pol_mem.seen),
        "advantage_samples_per_root": float(bundle.adv_mem.seen / roots),
        "strategy_samples_per_root": float(bundle.pol_mem.seen / roots),
        "advantage_decision_visits": branch_visits,
        "nominal_aggressive_branches_per_decision": float(nominal_branches / branch_visits) if branch_visits else 0.0,
        "effective_unique_aggressive_branches_per_decision": float(effective_branches / branch_visits) if branch_visits else 0.0,
        "peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in state["iteration_reports"]),
        "ensemble_advantage_weighted_nrmse": float(ensemble_nrmse),
        "policy_weighted_mean_tv": float(policy_tv),
        "advantage_gate_pass": bool(ensemble_nrmse <= 0.75),
        "policy_gate_pass": bool(policy_tv <= 0.12),
        "scenario_counts": [
            {"scenario": r74._scenario_descriptor(ep), "root_count": int(count)}
            for ep, count in zip(scenarios, state["scenario_counts"])
        ],
        "all_scenarios_exercised": all(int(count) > 0 for count in state["scenario_counts"]),
        "uncertainty_runtime_statistics": {
            **_behavior_stats(behavior),
            "mean_epsilon": float(behavior.epsilon_sum / behavior.calls) if behavior.calls else 0.0,
            "mean_disagreement": float(behavior.disagreement_sum / behavior.calls) if behavior.calls else 0.0,
        },
        "iteration_reports": list(state["iteration_reports"]),
        "average_policy_optimizer_steps": int(bundle.counters["policy_optimizer_steps"]),
        "primary_advantage_optimizer_steps": int(bundle.counters["adv_optimizer_steps"]),
        "strategic_selection_permitted_at_160": false,
        "production_training_authorized": false,
        "ready_for_tables": false,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Staged R7.5.4A 160 action-abstraction domain worker")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--domain", choices=("TRUE_HEADS_UP", "THREE_HANDED"), required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--target-iteration", type=int, choices=(1,2,3,4,5), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    _validate_repo_contract(
        repo_root,
        candidate=str(args.candidate),
        seed=int(args.training_seed),
        execution_sha=str(args.execution_sha),
    )
    if args.finalize and int(args.target_iteration) != ITERATIONS:
        raise SystemExit("--finalize is only legal with target iteration 5")
    if int(args.target_iteration) == ITERATIONS and not args.finalize:
        raise SystemExit("iteration 5 must finalize AveragePolicy in the same immutable stage")

    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("torch thread contract was not applied")
    solver = SolverLibrary(args.solver)

    if args.resume:
        bundle, session, behavior, spec, state, progress = _load_runtime(
            repo_root,
            solver=solver,
            checkpoint=args.resume,
            candidate=str(args.candidate),
            domain=str(args.domain),
            seed=int(args.training_seed),
            execution_sha=str(args.execution_sha),
        )
        if progress.phase == "post_policy_fit":
            raise SystemExit("cannot continue a finalized R7.5.4A seed")
    else:
        if int(args.target_iteration) != 1:
            raise SystemExit("a fresh R7.5.4A worker must start at iteration 1")
        bundle, session, behavior, spec, state = _new_runtime(
            repo_root,
            solver=solver,
            candidate=str(args.candidate),
            domain=str(args.domain),
            seed=int(args.training_seed),
        )

    wall_started = time.perf_counter()
    iteration_report = _run_iteration(
        bundle=bundle,
        session=session,
        behavior=behavior,
        state=state,
        iteration=int(args.target_iteration),
        seed=int(args.training_seed),
        domain=str(args.domain),
    )
    final_report = None
    progress_phase = "post_advantage_fit"
    if args.finalize:
        final_report = _finalize(
            bundle=bundle,
            session=session,
            behavior=behavior,
            state=state,
            seed=int(args.training_seed),
            domain=str(args.domain),
            candidate=str(args.candidate),
        )
        progress_phase = "post_policy_fit"
    stage_wall_seconds = time.perf_counter() - wall_started

    extra = {
        "execution_sha": str(args.execution_sha),
        "worker_state": state,
        "behavior_ensemble": _ensemble_payload(behavior.models, int(args.training_seed), int(args.target_iteration)),
        "behavior_stats": _behavior_stats(behavior),
        "stage_wall_seconds": float(stage_wall_seconds),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
        "final_report": final_report,
    }
    save_action_checkpoint(
        args.checkpoint_out,
        bundle,
        ActionProgress(
            iteration=int(args.target_iteration),
            phase=progress_phase,
            root_index=ROOTS_PER_ITERATION,
            advantage_optimizer_step=int(bundle.counters["adv_optimizer_steps"]),
            policy_optimizer_step=int(bundle.counters["policy_optimizer_steps"]),
        ),
        action_phase=PHASE,
        extra=extra,
    )

    payload = {
        "schema": SCHEMA,
        "execution_sha": str(args.execution_sha),
        "candidate": str(args.candidate),
        "domain": str(args.domain),
        "training_seed": int(args.training_seed),
        "target_iteration": int(args.target_iteration),
        "iteration_report": iteration_report,
        "stage_wall_seconds": float(stage_wall_seconds),
        "completed_iteration": int(state["completed_iteration"]),
        "finalized": bool(final_report is not None),
        "final_report": final_report,
        "production_training_authorized": false if False else False,
        "ready_for_tables": False,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
