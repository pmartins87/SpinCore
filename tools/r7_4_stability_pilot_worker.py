from __future__ import annotations

import argparse
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import torch

import run_r7_3_partial_exact_ensemble_paired as base
import run_r7_3_policy_mixture_uncertainty_damping as uncertainty
from run_r7_3_partial_exact_advantage_screen import PartialExactAdvantageCollector
from run_r7_3_replicated_640_candidate import _fit_policy
from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.r7 import FROZEN_GATES, audit_model_fit, cross_seed_policy_tv
from spincore.solver import Episode, SolverLibrary


SCHEMA = "SPINCORE_R7_4_HELDOUT_DOMAIN_STABILITY_V1"
PAYOUT = (0.5, 0.3, 0.2)
SUPPORTED_DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")


def _heldout_seeds(freeze: dict) -> list[int]:
    """Derive two seeds mechanically from immutable winner evidence.

    The selected R7.3 seeds must not be reused for the R7.4 generalization gate.
    Derivation from the already-frozen evidence SHA prevents post-result seed
    shopping while keeping the pilot fully reproducible.
    """
    evidence_hash = str(freeze["evidence_sha256"])
    selected = {int(x) for x in freeze.get("algorithm_seeds", [])}
    out: list[int] = []
    index = 0
    while len(out) < 2:
        digest = hashlib.sha256(f"SpinCore|R7.4|heldout|{index}|{evidence_hash}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") & 0x7FFFFFFF
        index += 1
        if seed <= 0 or seed in selected or seed in out:
            continue
        out.append(seed)
    return out


def _scenario_cycle(domain: str) -> tuple[Episode, ...]:
    if domain == "TRUE_HEADS_UP":
        episodes = []
        # Exercise symmetry plus materially unequal tournament stacks. Seat 0
        # remains dead so this is still the whole-hand true-HU domain.
        for stacks in ((0, 750, 750), (0, 500, 1000), (0, 1000, 500)):
            for dealer in (1, 2):
                episodes.append(Episode(1500, True, 0, 10, 20, stacks, dealer, (0,)))
        return tuple(episodes)
    if domain == "THREE_HANDED":
        episodes = []
        # Equal plus asymmetric stack profiles, with all dealer rotations. The
        # permutations are deterministic and sum to the same 1500-chip pool.
        profiles = (
            (500, 500, 500),
            (250, 500, 750),
            (250, 750, 500),
            (500, 250, 750),
            (750, 250, 500),
        )
        for stacks in profiles:
            for dealer in (0, 1, 2):
                episodes.append(Episode(1500, False, 0, 10, 20, stacks, dealer, ()))
        return tuple(episodes)
    raise ValueError(f"unsupported domain {domain!r}")


def _args_from_freeze(freeze: dict, *, roots_per_iteration: int, device: str) -> SimpleNamespace:
    ec = dict(freeze["execution_contract"])
    return SimpleNamespace(
        device=device,
        lr=float(ec["lr"]),
        reservoir_capacity=int(ec["reservoir_capacity"]),
        iterations=int(ec["iterations"]),
        roots_per_iteration=int(roots_per_iteration),
        exact_opponent_levels=int(ec["exact_opponent_levels"]),
        advantage_chunk_steps=int(ec["advantage_chunk_steps"]),
        advantage_max_steps_per_iteration=int(ec["advantage_max_steps_per_iteration"]),
        advantage_fit_target=float(ec["advantage_fit_target"]),
        policy_chunk_steps=int(ec["policy_chunk_steps"]),
        policy_max_steps=int(ec["policy_max_steps"]),
        policy_fit_target=float(ec["policy_fit_target"]),
        batch_size=int(ec["batch_size"]),
        audit_size=int(ec["audit_size"]),
        cross_seed_per_seed=int(ec["cross_seed_per_seed"]),
        seeds="",
    )


def _make_bundle(seed: int, domain: str, args) -> object:
    bundle = base.diagnostic.make_bundle(
        int(seed),
        device=args.device,
        reservoir_capacity=int(args.reservoir_capacity),
        lr=float(args.lr),
    )
    bundle.domain = domain
    return bundle


def _scenario_descriptor(ep: Episode) -> dict:
    return {
        "game_is_hu": bool(ep.game_is_hu),
        "stacks": list(ep.stacks),
        "dealer_id": int(ep.dealer_id),
        "dead_players": list(ep.dead_players),
    }


def _runtime_statistics_for_behavior(behavior: uncertainty.UncertaintyDampedPolicyMixture, seed: int) -> dict:
    """Snapshot diagnostics from this seed's own behavior instance only.

    The R7.3 implementation keeps a process-global instance registry for its
    report helper. A multi-domain R7.4 worker runs multiple seeds in one process,
    so relying on that registry can mislabel the later seed. Reading the local
    behavior object is semantically inert and makes provenance unambiguous.
    """
    calls = int(behavior.calls)
    return {
        "algorithm_seed": int(seed),
        "fitted_behavior_calls": calls,
        "mean_epsilon": float(behavior.epsilon_sum / calls) if calls else 0.0,
        "max_epsilon": float(behavior.epsilon_max),
        "mean_disagreement": float(behavior.disagreement_sum / calls) if calls else 0.0,
        "max_raw_epsilon_before_cap": float(behavior.raw_epsilon_max),
        "cap_hit_calls": int(behavior.cap_hit_calls),
        "cap_hit_fraction": float(behavior.cap_hit_calls / calls) if calls else 0.0,
        "epsilon_ge_0_10_fraction": float(behavior.epsilon_ge_010_calls / calls) if calls else 0.0,
        "epsilon_ge_0_25_fraction": float(behavior.epsilon_ge_025_calls / calls) if calls else 0.0,
    }


def run_seed(*, seed: int, domain: str, solver: SolverLibrary, args, epsilon_scale: float, epsilon_cap: float, ensemble_size: int):
    bundle = _make_bundle(seed, domain, args)
    session = DeepCFRDomainSession(
        solver_library=solver,
        bundle=bundle,
        terminal_utility=icm_delta_utility(PAYOUT),
        device=args.device,
    )
    behavior = uncertainty.UncertaintyDampedPolicyMixture(device=args.device)
    partial = PartialExactAdvantageCollector(
        policy=behavior,
        terminal_utility=session.terminal_utility,
        rng=bundle.batch_rng,
        advantage_memory=bundle.adv_mem,
        strategy_memory=bundle.pol_mem,
    )
    session.collector = partial

    scenarios = _scenario_cycle(domain)
    live_by_scenario = [tuple(i for i, stack in enumerate(ep.stacks) if stack > 0) for ep in scenarios]
    checkpoints = []
    scenario_counts = [0] * len(scenarios)
    global_root = 0

    for iteration in range(1, int(args.iterations) + 1):
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
        checkpoints.append({
            "iteration": int(iteration),
            "roots": int(bundle.counters["roots"]),
            "nodes": int(bundle.counters["nodes"]),
            "advantage_samples": len(bundle.adv_mem.items),
            "strategy_samples": len(bundle.pol_mem.items),
            "ensemble_weighted_nrmse": float(ensemble_nrmse),
            "ensemble_frozen_fit_gate_pass": bool(ensemble_nrmse <= FROZEN_GATES["advantage_weighted_nrmse_max"]),
            "members": members,
        })
        print(json.dumps({"domain": domain, "seed": seed, "checkpoint": checkpoints[-1]}, sort_keys=True), flush=True)

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
    runtime = _runtime_statistics_for_behavior(behavior, seed)
    fit = {
        "ensemble_advantage_weighted_nrmse": float(final_ensemble_nrmse),
        "policy_weighted_mean_tv": float(policy_audit["policy_weighted_mean_tv"]),
        "advantage_gate_pass": bool(final_ensemble_nrmse <= FROZEN_GATES["advantage_weighted_nrmse_max"]),
        "policy_gate_pass": bool(float(policy_audit["policy_weighted_mean_tv"]) <= FROZEN_GATES["policy_weighted_mean_tv_max"]),
    }
    return bundle, {
        "algorithm_seed": int(seed),
        "domain": domain,
        "roots": int(bundle.counters["roots"]),
        "nodes": int(bundle.counters["nodes"]),
        "advantage_samples": len(bundle.adv_mem.items),
        "strategy_samples": len(bundle.pol_mem.items),
        "scenario_counts": [
            {"scenario": _scenario_descriptor(ep), "root_count": int(count)}
            for ep, count in zip(scenarios, scenario_counts)
        ],
        "all_scenarios_exercised": all(count > 0 for count in scenario_counts),
        "checkpoints": checkpoints,
        "policy_progress": policy_progress,
        "uncertainty_runtime_statistics": runtime,
        "final_fit": fit,
    }


def main() -> int:
    global_scale = uncertainty.EPSILON_SCALE
    global_cap = uncertainty.EPSILON_CAP
    ap = argparse.ArgumentParser(description="Held-out R7.4 domain-generalization/stability pilot using the frozen R7.3 winner mechanism")
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--domain", choices=SUPPORTED_DOMAINS, required=True)
    ap.add_argument("--roots-per-iteration", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args_cli = ap.parse_args()

    freeze = json.loads(args_cli.freeze.read_text(encoding="utf-8"))
    if freeze.get("schema") != "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1":
        raise SystemExit("wrong semantic-freeze schema")
    if freeze.get("behavior_semantic_id") != "SPINCORE_R7_3_UNCERTAINTY_POLICY_MIXTURE_V1":
        raise SystemExit("R7.4 worker currently requires the selected uncertainty policy-mixture semantic")
    params = dict(freeze.get("params") or {})
    epsilon_scale = float(params["epsilon_scale"])
    epsilon_cap = float(params["epsilon_cap"])
    ensemble_size = int(freeze["ensemble_size"])
    if ensemble_size != 4:
        raise SystemExit("selected R7.4 frozen winner must currently be ensemble size 4")
    if int(args_cli.roots_per_iteration) <= 0:
        raise SystemExit("roots-per-iteration must be positive")

    uncertainty.EPSILON_SCALE = epsilon_scale
    uncertainty.EPSILON_CAP = epsilon_cap
    uncertainty.UncertaintyDampedPolicyMixture.instances = []
    heldout = _heldout_seeds(freeze)
    if any(seed in set(int(x) for x in freeze.get("algorithm_seeds", [])) for seed in heldout):
        raise SystemExit("held-out seed collision with selected R7.3 seeds")

    args = _args_from_freeze(freeze, roots_per_iteration=int(args_cli.roots_per_iteration), device="cpu")
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args_cli.solver)
    bundles = []
    reports = []
    for seed in heldout:
        bundle, report = run_seed(
            seed=seed,
            domain=str(args_cli.domain),
            solver=solver,
            args=args,
            epsilon_scale=epsilon_scale,
            epsilon_cap=epsilon_cap,
            ensemble_size=ensemble_size,
        )
        bundles.append(bundle)
        reports.append(report)

    observations = base.diagnostic.shared_cross_seed_observations(
        bundles,
        per_seed=int(args.cross_seed_per_seed),
        seed=0x7400BEEF,
    )
    cross = cross_seed_policy_tv(bundles[0].policy, bundles[1].policy, observations, device=args.device)
    fit_pass = all(row["final_fit"]["advantage_gate_pass"] and row["final_fit"]["policy_gate_pass"] for row in reports)
    cross_pass = bool(
        float(cross["mean_tv"]) <= FROZEN_GATES["cross_seed_mean_tv_max"]
        and float(cross["p95_tv"]) <= FROZEN_GATES["cross_seed_p95_tv_max"]
    )
    scenario_pass = all(row["all_scenarios_exercised"] for row in reports)
    passed = bool(fit_pass and cross_pass and scenario_pass)
    payload = {
        "schema": SCHEMA,
        "domain": str(args_cli.domain),
        "behavior_semantic_id": freeze["behavior_semantic_id"],
        "ensemble_size": ensemble_size,
        "epsilon_scale": epsilon_scale,
        "epsilon_cap": epsilon_cap,
        "algorithm_seeds": heldout,
        "seed_derivation": "SHA256('SpinCore|R7.4|heldout|index|' + frozen_evidence_sha256), first positive 31 bits; reject R7.3 seed collisions",
        "r7_3_selection_seeds_reused": False,
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "roots_per_seed": int(args.iterations * args.roots_per_iteration),
        "scenario_cycle_size": len(_scenario_cycle(str(args_cli.domain))),
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
        "ready_for_tables": False,
    }
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    args_cli.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "domain": payload["domain"],
        "heldout_seeds": heldout,
        "cross_seed": payload["cross_seed"],
        "per_seed_fit_pass": fit_pass,
        "scenario_coverage_pass": scenario_pass,
        "r7_4_domain_stability_pass": passed,
    }, indent=2, sort_keys=True), flush=True)
    uncertainty.EPSILON_SCALE = global_scale
    uncertainty.EPSILON_CAP = global_cap
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
