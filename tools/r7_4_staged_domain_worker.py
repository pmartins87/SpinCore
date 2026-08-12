from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

import r7_4_stability_pilot_worker as mono
import run_r7_3_partial_exact_ensemble_paired as base
import run_r7_3_policy_mixture_uncertainty_damping as uncertainty
from run_r7_3_partial_exact_advantage_screen import PartialExactAdvantageCollector
from run_r7_3_replicated_640_candidate import _fit_policy
from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.r7 import (
    FROZEN_GATES,
    MidIterationProgress,
    audit_model_fit,
    cross_seed_policy_tv,
    load_checkpoint,
    save_checkpoint,
)
from spincore.r7_candidate_checkpoint import (
    pack_candidate_behavior,
    restore_candidate_behavior_models,
)
from spincore.solver import SolverLibrary


STAGE_SCHEMA = "SPINCORE_R7_4_STAGED_CHECKPOINT_V1"
SEED_REPORT_SCHEMA = "SPINCORE_R7_4_STAGED_SEED_REPORT_V1"
EQUIV_SCHEMA = "SPINCORE_R7_4_STAGED_RESUME_EQUIVALENCE_V1"
PAYOUT = mono.PAYOUT


def _behavior_stats(behavior) -> dict:
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


def _restore_behavior_stats(behavior, state: dict) -> None:
    for key, value in state.items():
        setattr(behavior, key, value)


def _runtime(bundle, behavior, solver, args):
    session = DeepCFRDomainSession(
        solver_library=solver,
        bundle=bundle,
        terminal_utility=icm_delta_utility(PAYOUT),
        device=args.device,
    )
    partial = PartialExactAdvantageCollector(
        policy=behavior,
        terminal_utility=session.terminal_utility,
        rng=bundle.batch_rng,
        advantage_memory=bundle.adv_mem,
        strategy_memory=bundle.pol_mem,
    )
    session.collector = partial
    return session, partial


def _new_state(*, seed: int, domain: str, solver, args):
    bundle = mono._make_bundle(seed, domain, args)
    behavior = uncertainty.UncertaintyDampedPolicyMixture(device=args.device)
    session, partial = _runtime(bundle, behavior, solver, args)
    return bundle, behavior, session, partial, {
        "schema": STAGE_SCHEMA,
        "domain": domain,
        "algorithm_seed": int(seed),
        "roots_per_iteration": int(args.roots_per_iteration),
        "completed_iteration": 0,
        "global_root": 0,
        "scenario_counts": [0] * len(mono._scenario_cycle(domain)),
        "checkpoints": [],
    }


def _pack_extra(*, freeze: dict, bundle, behavior, stage: dict) -> dict:
    if not behavior.models:
        raise ValueError("cannot checkpoint before the first fitted ensemble")
    candidate = pack_candidate_behavior(
        kind=str(freeze["behavior_kind"]),
        primary_model=bundle.advantage,
        current_models=list(behavior.models),
        previous_models=[],
        params=dict(freeze.get("params") or {}),
        fit_generation=int(stage["completed_iteration"]),
    )
    return {
        "semantic_freeze": freeze,
        "candidate_behavior": candidate,
        "r7_4_stage": dict(stage),
        "behavior_stats": _behavior_stats(behavior),
    }


def _save(path: Path, *, freeze: dict, bundle, behavior, stage: dict) -> None:
    save_checkpoint(
        path,
        bundle,
        MidIterationProgress(
            iteration=int(stage["completed_iteration"]),
            phase="post_advantage_fit",
            root_index=int(stage["roots_per_iteration"]),
        ),
        _pack_extra(freeze=freeze, bundle=bundle, behavior=behavior, stage=stage),
    )


def _load(path: Path, *, freeze: dict, solver, args, expected_domain: str, expected_seed: int):
    bundle, progress, extra = load_checkpoint(path, device=args.device)
    stage = dict(extra.get("r7_4_stage") or {})
    if stage.get("schema") != STAGE_SCHEMA:
        raise ValueError("wrong R7.4 staged checkpoint schema")
    if stage.get("domain") != expected_domain or int(stage.get("algorithm_seed", -1)) != int(expected_seed):
        raise ValueError("staged checkpoint domain/seed mismatch")
    if int(stage.get("roots_per_iteration", -1)) != int(args.roots_per_iteration):
        raise ValueError("staged checkpoint roots-per-iteration mismatch")
    if int(progress.iteration) != int(stage["completed_iteration"]) or progress.phase != "post_advantage_fit":
        raise ValueError("staged checkpoint progress mismatch")
    if bundle.domain != expected_domain or int(bundle.seed) != int(expected_seed):
        raise ValueError("base checkpoint domain/seed mismatch")
    current, previous, meta = restore_candidate_behavior_models(
        extra["candidate_behavior"],
        config=bundle.config,
        primary_model=bundle.advantage,
        device=args.device,
    )
    if previous:
        raise ValueError("uncertainty-damping staged checkpoint unexpectedly contains previous models")
    behavior = uncertainty.UncertaintyDampedPolicyMixture(device=args.device)
    behavior.models = list(current)
    _restore_behavior_stats(behavior, dict(extra.get("behavior_stats") or {}))
    session, partial = _runtime(bundle, behavior, solver, args)
    return bundle, behavior, session, partial, stage


def _run_one_iteration(
    *,
    seed: int,
    domain: str,
    iteration: int,
    bundle,
    behavior,
    session,
    partial,
    solver,
    args,
    ensemble_size: int,
    stage: dict,
):
    expected = int(stage["completed_iteration"]) + 1
    if int(iteration) != expected:
        raise ValueError(f"iteration must advance exactly one step: expected {expected}, got {iteration}")
    scenarios = mono._scenario_cycle(domain)
    live_by_scenario = [tuple(i for i, stack in enumerate(ep.stacks) if stack > 0) for ep in scenarios]
    scenario_counts = list(stage["scenario_counts"])
    global_root = int(stage["global_root"])

    for _root_index in range(int(args.roots_per_iteration)):
        scenario_index = global_root % len(scenarios)
        episode = scenarios[scenario_index]
        live = live_by_scenario[scenario_index]
        scenario_counts[scenario_index] += 1
        ds = (int(seed) * 1_000_003 + global_root * 97 + int(iteration)) & ((1 << 64) - 1)
        nodes = advantage_added = strategy_added = 0
        for traverser in live:
            root = solver.create(episode, int(ds))
            try:
                result = partial.collect_advantage_partial_exact(
                    root,
                    traverser=int(traverser),
                    iteration=int(iteration),
                    exact_opponent_levels=int(args.exact_opponent_levels),
                )
            finally:
                root.close()
            nodes += int(result.nodes)
            advantage_added += int(result.samples_added)
        for target_player in live:
            root = solver.create(episode, int(ds))
            try:
                strategy_added += int(
                    partial.collect_strategy_own_reach(
                        root,
                        target_player=int(target_player),
                        iteration=int(iteration),
                    )
                )
            finally:
                root.close()

        c = bundle.counters
        c["iteration"] = max(int(c["iteration"]), int(iteration))
        c["roots"] += 1
        c["nodes"] += int(nodes)
        c["advantage_samples"] += int(advantage_added)
        c["strategy_samples"] += int(strategy_added)
        global_root += 1

    memory_state = bundle.adv_mem.state_dict()
    primary_model, primary_report = base._fit_primary_member(
        session=session,
        bundle=bundle,
        seed=int(seed),
        iteration=int(iteration),
        args=args,
    )
    models = [primary_model]
    members = [primary_report]
    for member in range(1, int(ensemble_size)):
        model, report = base._train_member(
            memory_state=memory_state,
            algorithm_seed=int(seed),
            iteration=int(iteration),
            member=int(member),
            solver=solver,
            args=args,
        )
        report = dict(report)
        report["role"] = "SIDE_MEMBER_DOES_NOT_PERTURB_PRIMARY_RNG"
        models.append(model)
        members.append(report)
    behavior.models = models
    ensemble_nrmse = base._ensemble_nrmse(
        models,
        bundle.adv_mem,
        sample_size=int(args.audit_size),
        seed=int(seed) ^ (int(iteration) * 0x5EEDBEEF),
        device=args.device,
    )
    checkpoint = {
        "iteration": int(iteration),
        "roots": int(bundle.counters["roots"]),
        "nodes": int(bundle.counters["nodes"]),
        "advantage_samples": len(bundle.adv_mem.items),
        "strategy_samples": len(bundle.pol_mem.items),
        "ensemble_weighted_nrmse": float(ensemble_nrmse),
        "ensemble_frozen_fit_gate_pass": bool(
            ensemble_nrmse <= FROZEN_GATES["advantage_weighted_nrmse_max"]
        ),
        "members": members,
    }
    stage["completed_iteration"] = int(iteration)
    stage["global_root"] = int(global_root)
    stage["scenario_counts"] = scenario_counts
    stage["checkpoints"] = list(stage["checkpoints"]) + [checkpoint]
    print(json.dumps({"domain": domain, "seed": seed, "checkpoint": checkpoint}, sort_keys=True), flush=True)


def _finalize_seed(*, seed: int, domain: str, bundle, behavior, session, args, stage: dict) -> dict:
    if int(stage["completed_iteration"]) != int(args.iterations):
        raise ValueError("cannot finalize before all frozen iterations are complete")
    policy_progress = _fit_policy(
        bundle=bundle,
        session=session,
        seed=int(seed),
        device=args.device,
        chunk_steps=int(args.policy_chunk_steps),
        max_steps=int(args.policy_max_steps),
        fit_target=float(args.policy_fit_target),
        batch_size=int(args.batch_size),
        audit_size=int(args.audit_size),
    )
    policy_audit = audit_model_fit(
        bundle,
        sample_size=max(int(args.audit_size), 2048),
        seed=int(seed) ^ 0x2468ACE0,
        device=args.device,
    )
    final_ensemble_nrmse = base._ensemble_nrmse(
        behavior.models,
        bundle.adv_mem,
        sample_size=max(int(args.audit_size), 2048),
        seed=int(seed) ^ 0x13572468,
        device=args.device,
    )
    scenarios = mono._scenario_cycle(domain)
    scenario_counts = list(stage["scenario_counts"])
    runtime = mono._runtime_statistics_for_behavior(behavior, seed)
    fit = {
        "ensemble_advantage_weighted_nrmse": float(final_ensemble_nrmse),
        "policy_weighted_mean_tv": float(policy_audit["policy_weighted_mean_tv"]),
        "advantage_gate_pass": bool(
            final_ensemble_nrmse <= FROZEN_GATES["advantage_weighted_nrmse_max"]
        ),
        "policy_gate_pass": bool(
            float(policy_audit["policy_weighted_mean_tv"])
            <= FROZEN_GATES["policy_weighted_mean_tv_max"]
        ),
    }
    return {
        "schema": SEED_REPORT_SCHEMA,
        "algorithm_seed": int(seed),
        "domain": domain,
        "roots": int(bundle.counters["roots"]),
        "nodes": int(bundle.counters["nodes"]),
        "advantage_samples": len(bundle.adv_mem.items),
        "strategy_samples": len(bundle.pol_mem.items),
        "scenario_counts": [
            {"scenario": mono._scenario_descriptor(ep), "root_count": int(count)}
            for ep, count in zip(scenarios, scenario_counts)
        ],
        "all_scenarios_exercised": all(count > 0 for count in scenario_counts),
        "checkpoints": list(stage["checkpoints"]),
        "policy_progress": policy_progress,
        "uncertainty_runtime_statistics": runtime,
        "final_fit": fit,
        "staged_resume_used": True,
        "staged_checkpoint_schema": STAGE_SCHEMA,
    }


def _final_checkpoint(path: Path, *, freeze: dict, bundle, behavior, stage: dict, seed_report: dict) -> None:
    extra = _pack_extra(freeze=freeze, bundle=bundle, behavior=behavior, stage=stage)
    extra["r7_4_seed_report"] = seed_report
    save_checkpoint(
        path,
        bundle,
        MidIterationProgress(
            iteration=int(stage["completed_iteration"]),
            phase="post_policy_fit",
            root_index=int(stage["roots_per_iteration"]),
            policy_optimizer_step=int(bundle.counters.get("policy_optimizer_steps", 0)),
        ),
        extra,
    )


def _load_final(path: Path, *, device: str):
    bundle, progress, extra = load_checkpoint(path, device=device)
    if progress.phase != "post_policy_fit":
        raise ValueError("final staged checkpoint is not post-policy-fit")
    report = dict(extra.get("r7_4_seed_report") or {})
    if report.get("schema") != SEED_REPORT_SCHEMA:
        raise ValueError("missing staged seed report")
    return bundle, report


def _aggregate(*, freeze: dict, domain: str, roots_per_iteration: int, checkpoint_paths: list[Path], out: Path) -> int:
    args = mono._args_from_freeze(freeze, roots_per_iteration=roots_per_iteration, device="cpu")
    expected_seeds = mono._heldout_seeds(freeze)
    if len(checkpoint_paths) != 2:
        raise ValueError("aggregate requires exactly two final seed checkpoints")
    pairs = [_load_final(p, device=args.device) for p in checkpoint_paths]
    by_seed = {int(report["algorithm_seed"]): (bundle, report) for bundle, report in pairs}
    if set(by_seed) != set(expected_seeds):
        raise ValueError("final checkpoint seed set differs from frozen held-out seed set")
    bundles = [by_seed[s][0] for s in expected_seeds]
    reports = [by_seed[s][1] for s in expected_seeds]
    observations = base.diagnostic.shared_cross_seed_observations(
        bundles,
        per_seed=int(args.cross_seed_per_seed),
        seed=0x7400BEEF,
    )
    cross = cross_seed_policy_tv(
        bundles[0].policy, bundles[1].policy, observations, device=args.device
    )
    fit_pass = all(
        row["final_fit"]["advantage_gate_pass"] and row["final_fit"]["policy_gate_pass"]
        for row in reports
    )
    cross_pass = bool(
        float(cross["mean_tv"]) <= FROZEN_GATES["cross_seed_mean_tv_max"]
        and float(cross["p95_tv"]) <= FROZEN_GATES["cross_seed_p95_tv_max"]
    )
    scenario_pass = all(row["all_scenarios_exercised"] for row in reports)
    passed = bool(fit_pass and cross_pass and scenario_pass)
    payload = {
        "schema": mono.SCHEMA,
        "domain": domain,
        "behavior_semantic_id": freeze["behavior_semantic_id"],
        "ensemble_size": int(freeze["ensemble_size"]),
        "epsilon_scale": float(freeze["params"]["epsilon_scale"]),
        "epsilon_cap": float(freeze["params"]["epsilon_cap"]),
        "algorithm_seeds": expected_seeds,
        "seed_derivation": "SHA256('SpinCore|R7.4|heldout|index|' + frozen_evidence_sha256), first positive 31 bits; reject R7.3 seed collisions",
        "r7_3_selection_seeds_reused": False,
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "roots_per_seed": int(args.iterations * args.roots_per_iteration),
        "scenario_cycle_size": len(mono._scenario_cycle(domain)),
        "scenario_schedule": "DETERMINISTIC_GLOBAL_ROOT_MOD_SCENARIO_CYCLE",
        "deck_formula": "seed*1000003 + global_root*97 + iteration",
        "exact_opponent_levels": int(args.exact_opponent_levels),
        "extra_members_perturb_primary_rng": False,
        "per_seed": reports,
        "cross_seed_observation_count": len(observations),
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "frozen_gates": dict(FROZEN_GATES),
        "per_seed_fit_pass": bool(fit_pass),
        "cross_seed_pass": bool(cross_pass),
        "scenario_coverage_pass": bool(scenario_pass),
        "r7_4_domain_stability_pass": bool(passed),
        "acceptance_gate_changed": False,
        "staged_resume_used": True,
        "staged_checkpoint_schema": STAGE_SCHEMA,
        "ready_for_tables": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": payload["schema"],
        "domain": domain,
        "heldout_seeds": expected_seeds,
        "cross_seed": payload["cross_seed"],
        "per_seed_fit_pass": fit_pass,
        "scenario_coverage_pass": scenario_pass,
        "r7_4_domain_stability_pass": passed,
        "staged_resume_used": True,
    }, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def _eq(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


def _snapshot(bundle, behavior, stage: dict) -> dict:
    return {
        "counters": dict(bundle.counters),
        "adv_mem": bundle.adv_mem.state_dict(),
        "pol_mem": bundle.pol_mem.state_dict(),
        "batch_rng": bundle.batch_rng.getstate(),
        "torch_rng": torch.get_rng_state().clone().cpu(),
        "advantage": {k: v.detach().cpu().clone() for k, v in bundle.advantage.state_dict().items()},
        "policy": {k: v.detach().cpu().clone() for k, v in bundle.policy.state_dict().items()},
        "adv_opt": bundle.adv_opt.state_dict(),
        "pol_opt": bundle.pol_opt.state_dict(),
        "models": [
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            for model in behavior.models
        ],
        "behavior_stats": _behavior_stats(behavior),
        "stage": dict(stage),
    }


def _equivalence(*, freeze: dict, solver, out: Path, checkpoint: Path) -> int:
    # Mechanism regression only, not strategic evidence. It still executes the
    # real THREE_HANDED collection/fitting path on continuous and resumed arms.
    args = mono._args_from_freeze(freeze, roots_per_iteration=1, device="cpu")
    args.reservoir_capacity = min(int(args.reservoir_capacity), 4096)
    args.advantage_chunk_steps = min(int(args.advantage_chunk_steps), 8)
    args.advantage_max_steps_per_iteration = min(int(args.advantage_max_steps_per_iteration), 16)
    args.policy_chunk_steps = min(int(args.policy_chunk_steps), 8)
    args.policy_max_steps = min(int(args.policy_max_steps), 16)
    args.batch_size = min(int(args.batch_size), 16)
    args.audit_size = min(int(args.audit_size), 32)
    seed = mono._heldout_seeds(freeze)[0]
    ensemble_size = int(freeze["ensemble_size"])
    domain = "THREE_HANDED"

    b, beh, sess, part, stage = _new_state(seed=seed, domain=domain, solver=solver, args=args)
    _run_one_iteration(
        seed=seed, domain=domain, iteration=1, bundle=b, behavior=beh, session=sess,
        partial=part, solver=solver, args=args, ensemble_size=ensemble_size, stage=stage,
    )
    _save(checkpoint, freeze=freeze, bundle=b, behavior=beh, stage=stage)
    _run_one_iteration(
        seed=seed, domain=domain, iteration=2, bundle=b, behavior=beh, session=sess,
        partial=part, solver=solver, args=args, ensemble_size=ensemble_size, stage=stage,
    )
    continuous = _snapshot(b, beh, stage)

    rb, rbeh, rsess, rpart, rstage = _load(
        checkpoint, freeze=freeze, solver=solver, args=args,
        expected_domain=domain, expected_seed=seed,
    )
    _run_one_iteration(
        seed=seed, domain=domain, iteration=2, bundle=rb, behavior=rbeh, session=rsess,
        partial=rpart, solver=solver, args=args, ensemble_size=ensemble_size, stage=rstage,
    )
    resumed = _snapshot(rb, rbeh, rstage)
    checks = {key: _eq(continuous[key], resumed[key]) for key in continuous}
    payload = {
        "schema": EQUIV_SCHEMA,
        "domain": domain,
        "mechanism_only_not_strategic_evidence": True,
        "seed": int(seed),
        "roots_per_iteration": 1,
        "split_after_iteration": 1,
        "continued_through_iteration": 2,
        "checks": checks,
        "all_exact": all(checks.values()),
        "frozen_strategy_gate_changed": False,
        "ready_for_tables": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if payload["all_exact"] else 2


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic staged R7.4 domain worker")
    sub = ap.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--freeze", type=Path, required=True)
    common.add_argument("--solver", type=Path, required=True)
    common.add_argument("--domain", choices=mono.SUPPORTED_DOMAINS, required=True)
    common.add_argument("--roots-per-iteration", type=int, required=True)
    common.add_argument("--seed", type=int, required=True)

    p_adv = sub.add_parser("advance", parents=[common])
    p_adv.add_argument("--iteration", type=int, required=True)
    p_adv.add_argument("--checkpoint-in", type=Path)
    p_adv.add_argument("--checkpoint-out", type=Path, required=True)

    p_fin = sub.add_parser("finalize", parents=[common])
    p_fin.add_argument("--checkpoint-in", type=Path, required=True)
    p_fin.add_argument("--checkpoint-out", type=Path, required=True)
    p_fin.add_argument("--seed-report-out", type=Path, required=True)

    p_agg = sub.add_parser("aggregate")
    p_agg.add_argument("--freeze", type=Path, required=True)
    p_agg.add_argument("--domain", choices=mono.SUPPORTED_DOMAINS, required=True)
    p_agg.add_argument("--roots-per-iteration", type=int, required=True)
    p_agg.add_argument("--checkpoint", type=Path, action="append", required=True)
    p_agg.add_argument("--out", type=Path, required=True)

    p_eq = sub.add_parser("equivalence")
    p_eq.add_argument("--freeze", type=Path, required=True)
    p_eq.add_argument("--solver", type=Path, required=True)
    p_eq.add_argument("--out", type=Path, required=True)
    p_eq.add_argument("--checkpoint", type=Path, required=True)

    cli = ap.parse_args()
    freeze = json.loads(cli.freeze.read_text(encoding="utf-8"))
    if freeze.get("schema") != "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1":
        raise SystemExit("wrong semantic-freeze schema")
    if freeze.get("behavior_semantic_id") != "SPINCORE_R7_3_UNCERTAINTY_POLICY_MIXTURE_V1":
        raise SystemExit("staged R7.4 worker requires frozen uncertainty-damping semantic")
    uncertainty.EPSILON_SCALE = float(freeze["params"]["epsilon_scale"])
    uncertainty.EPSILON_CAP = float(freeze["params"]["epsilon_cap"])
    uncertainty.UncertaintyDampedPolicyMixture.instances = []
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))

    if cli.command == "aggregate":
        return _aggregate(
            freeze=freeze,
            domain=cli.domain,
            roots_per_iteration=int(cli.roots_per_iteration),
            checkpoint_paths=list(cli.checkpoint),
            out=cli.out,
        )

    solver = SolverLibrary(cli.solver)
    if cli.command == "equivalence":
        return _equivalence(
            freeze=freeze, solver=solver, out=cli.out, checkpoint=cli.checkpoint
        )

    heldout = mono._heldout_seeds(freeze)
    if int(cli.seed) not in heldout:
        raise SystemExit("seed is not one of the mechanically frozen R7.4 held-out seeds")
    args = mono._args_from_freeze(
        freeze, roots_per_iteration=int(cli.roots_per_iteration), device="cpu"
    )
    ensemble_size = int(freeze["ensemble_size"])

    if cli.command == "advance":
        if int(cli.iteration) == 1:
            if cli.checkpoint_in is not None:
                raise SystemExit("iteration 1 must not have checkpoint-in")
            bundle, behavior, session, partial, stage = _new_state(
                seed=int(cli.seed), domain=cli.domain, solver=solver, args=args
            )
        else:
            if cli.checkpoint_in is None:
                raise SystemExit("iteration >1 requires checkpoint-in")
            bundle, behavior, session, partial, stage = _load(
                cli.checkpoint_in,
                freeze=freeze,
                solver=solver,
                args=args,
                expected_domain=cli.domain,
                expected_seed=int(cli.seed),
            )
        _run_one_iteration(
            seed=int(cli.seed), domain=cli.domain, iteration=int(cli.iteration),
            bundle=bundle, behavior=behavior, session=session, partial=partial,
            solver=solver, args=args, ensemble_size=ensemble_size, stage=stage,
        )
        cli.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        _save(cli.checkpoint_out, freeze=freeze, bundle=bundle, behavior=behavior, stage=stage)
        return 0

    bundle, behavior, session, partial, stage = _load(
        cli.checkpoint_in,
        freeze=freeze,
        solver=solver,
        args=args,
        expected_domain=cli.domain,
        expected_seed=int(cli.seed),
    )
    report = _finalize_seed(
        seed=int(cli.seed), domain=cli.domain, bundle=bundle, behavior=behavior,
        session=session, args=args, stage=stage,
    )
    cli.seed_report_out.parent.mkdir(parents=True, exist_ok=True)
    cli.seed_report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cli.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    _final_checkpoint(
        cli.checkpoint_out, freeze=freeze, bundle=bundle, behavior=behavior,
        stage=stage, seed_report=report,
    )
    print(json.dumps({
        "schema": report["schema"],
        "seed": report["algorithm_seed"],
        "final_fit": report["final_fit"],
        "all_scenarios_exercised": report["all_scenarios_exercised"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
