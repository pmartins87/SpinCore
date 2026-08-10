from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import run_r7_3_diagnostic as diagnostic
from run_r7_3_advantage_ensemble_end_to_end import (
    EnsembleAdvantagePolicy,
    _ensemble_nrmse,
    _train_member,
)
from run_r7_3_partial_exact_advantage_screen import PartialExactAdvantageCollector
from run_r7_3_replicated_640_candidate import _fit_policy, _fit_pass
from run_r7_3_variance_decomposition import _advantage_fit_nrmse, _finite

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.r7 import FROZEN_GATES, audit_model_fit, cross_seed_policy_tv
from spincore.solver import SolverLibrary


diagnostic.HISTORICAL_PARAMS_PER_NETWORK = 152_434
DEFAULT_SEEDS = (20260829, 20260807)
PAYOUT = (0.5, 0.3, 0.2)


def deck_seed(seed: int, iteration: int, root_index: int, roots_per_iteration: int) -> int:
    global_root = (int(iteration) - 1) * int(roots_per_iteration) + int(root_index)
    return (int(seed) * 1_000_003 + global_root * 97 + int(iteration)) & ((1 << 64) - 1)


def _fit_primary_member(*, session, bundle, seed: int, iteration: int, args):
    """Fit member zero through the authoritative coupled RNG stream.

    This is deliberately different from the earlier shared-deck ensemble screen.
    Member zero is the exact baseline training path: reset on the live bundle and
    consume bundle.batch_rng through session.train_advantage(). Extra ensemble
    members are side computations on a frozen memory clone and therefore do not
    perturb the recovered primary RNG stream used by later traversal/policy work.
    """
    reset_seed = (int(seed) ^ (int(iteration) * 0x9E3779B1)) & 0x7FFFFFFF
    session.reset_advantage_network(init_seed=reset_seed, lr=float(args.lr))
    local_steps = 0
    progress = []
    audit_seed = int(seed) ^ (int(iteration) * 0x45D9F3B)
    while local_steps < int(args.advantage_max_steps_per_iteration):
        steps = min(
            int(args.advantage_chunk_steps),
            int(args.advantage_max_steps_per_iteration) - local_steps,
        )
        session.train_advantage(steps=steps, batch_size=int(args.batch_size))
        local_steps += steps
        nrmse = _advantage_fit_nrmse(
            bundle,
            sample_size=int(args.audit_size),
            seed=audit_seed,
            device=args.device,
        )
        row = {
            "optimizer_steps": int(local_steps),
            "weighted_nrmse": float(nrmse),
            "frozen_gate_pass": _fit_pass(nrmse, "advantage_weighted_nrmse_max"),
            "fit_target_reached": _finite(nrmse)
            and float(nrmse) <= float(args.advantage_fit_target),
        }
        progress.append(row)
        if row["fit_target_reached"]:
            break
    return bundle.advantage, {
        "member": 0,
        "role": "PRIMARY_AUTHORITATIVE_COUPLED_RNG",
        "init_seed": int(reset_seed),
        "optimizer_steps": int(local_steps),
        "final_weighted_nrmse": float(progress[-1]["weighted_nrmse"]),
        "progress": progress,
    }


def run_seed(*, seed: int, ensemble_size: int, solver: SolverLibrary, args):
    bundle = diagnostic.make_bundle(
        int(seed),
        device=args.device,
        reservoir_capacity=int(args.reservoir_capacity),
        lr=float(args.lr),
    )
    session = DeepCFRDomainSession(
        solver_library=solver,
        bundle=bundle,
        terminal_utility=icm_delta_utility(PAYOUT),
        device=args.device,
    )

    behavior = EnsembleAdvantagePolicy(device=args.device)
    partial = PartialExactAdvantageCollector(
        policy=behavior,
        terminal_utility=session.terminal_utility,
        rng=bundle.batch_rng,
        advantage_memory=bundle.adv_mem,
        strategy_memory=bundle.pol_mem,
    )
    # Keep session.behavior as the native NeuralAdvantagePolicy so reset/train
    # semantics and counters remain authoritative for primary member zero.  Only
    # the collector's behavior policy is replaced by the ensemble wrapper.
    session.collector = partial

    episode = diagnostic.hu_episode()
    live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
    checkpoints = []

    for iteration in range(1, int(args.iterations) + 1):
        for root_index in range(int(args.roots_per_iteration)):
            ds = deck_seed(seed, iteration, root_index, int(args.roots_per_iteration))
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

        memory_state = bundle.adv_mem.state_dict()
        primary_model, primary_report = _fit_primary_member(
            session=session,
            bundle=bundle,
            seed=int(seed),
            iteration=int(iteration),
            args=args,
        )
        models = [primary_model]
        members = [primary_report]

        # Extra members intentionally do NOT consume bundle.batch_rng.  Thus the
        # future live CFR RNG state is the same one member-zero baseline training
        # would have left behind, while model averaging is isolated as a side
        # variance-control mechanism.
        for member in range(1, int(ensemble_size)):
            model, report = _train_member(
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
        ensemble_nrmse = _ensemble_nrmse(
            models,
            bundle.adv_mem,
            sample_size=int(args.audit_size),
            seed=int(seed) ^ (int(iteration) * 0x5EEDBEEF),
            device=args.device,
        )
        checkpoints.append(
            {
                "iteration": int(iteration),
                "roots": int(bundle.counters["roots"]),
                "nodes": int(bundle.counters["nodes"]),
                "advantage_samples": len(bundle.adv_mem.items),
                "advantage_seen": int(bundle.adv_mem.seen),
                "strategy_samples": len(bundle.pol_mem.items),
                "strategy_seen": int(bundle.pol_mem.seen),
                "primary_rng_adv_optimizer_steps_total": int(bundle.counters["adv_optimizer_steps"]),
                "ensemble_weighted_nrmse": float(ensemble_nrmse),
                "ensemble_frozen_fit_gate_pass": bool(
                    ensemble_nrmse <= FROZEN_GATES["advantage_weighted_nrmse_max"]
                ),
                "members": members,
            }
        )
        print(json.dumps({"seed": seed, "checkpoint": checkpoints[-1]}, sort_keys=True), flush=True)

    # Preserve the authoritative candidate contract here too: AveragePolicy
    # fitting continues from the same live bundle.batch_rng stream rather than
    # resetting to a diagnostic-only policy RNG.
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
    final_ensemble_nrmse = _ensemble_nrmse(
        behavior.models,
        bundle.adv_mem,
        sample_size=max(int(args.audit_size), 2048),
        seed=int(seed) ^ 0x13572468,
        device=args.device,
    )
    return bundle, {
        "algorithm_seed": int(seed),
        "ensemble_size": int(ensemble_size),
        "exact_opponent_levels": int(args.exact_opponent_levels),
        "roots": int(bundle.counters["roots"]),
        "nodes": int(bundle.counters["nodes"]),
        "advantage_samples": len(bundle.adv_mem.items),
        "advantage_seen": int(bundle.adv_mem.seen),
        "strategy_samples": len(bundle.pol_mem.items),
        "strategy_seen": int(bundle.pol_mem.seen),
        "primary_advantage_optimizer_steps": int(bundle.counters["adv_optimizer_steps"]),
        "policy_optimizer_steps": int(bundle.counters["policy_optimizer_steps"]),
        "checkpoints": checkpoints,
        "policy_progress": policy_progress,
        "final_fit": {
            "ensemble_advantage_weighted_nrmse": float(final_ensemble_nrmse),
            "policy_weighted_mean_tv": float(policy_audit["policy_weighted_mean_tv"]),
            "advantage_gate_pass": bool(
                final_ensemble_nrmse <= FROZEN_GATES["advantage_weighted_nrmse_max"]
            ),
            "policy_gate_pass": bool(
                float(policy_audit["policy_weighted_mean_tv"])
                <= FROZEN_GATES["policy_weighted_mean_tv_max"]
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Authoritative-deck/coupled-RNG partial-exact plus Advantage ensemble candidate"
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ensemble-size", type=int, choices=(1, 2, 4, 8), required=True)
    ap.add_argument("--exact-opponent-levels", type=int, default=2)
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-chunk-steps", type=int, default=256)
    ap.add_argument("--advantage-max-steps-per-iteration", type=int, default=4096)
    ap.add_argument("--advantage-fit-target", type=float, default=0.50)
    ap.add_argument("--policy-chunk-steps", type=int, default=256)
    ap.add_argument("--policy-max-steps", type=int, default=16384)
    ap.add_argument("--policy-fit-target", type=float, default=0.105)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=512)
    ap.add_argument("--cross-seed-per-seed", type=int, default=1024)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) != 2 or int(args.exact_opponent_levels) <= 0:
        raise SystemExit("requires exactly two seeds and positive exact-opponent level")
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()
    bundles = []
    reports = []
    for seed in seeds:
        bundle, report = run_seed(
            seed=int(seed), ensemble_size=int(args.ensemble_size), solver=solver, args=args
        )
        bundles.append(bundle)
        reports.append(report)

    observations = diagnostic.shared_cross_seed_observations(
        bundles, per_seed=int(args.cross_seed_per_seed), seed=0x715EED
    )
    cross = cross_seed_policy_tv(
        bundles[0].policy, bundles[1].policy, observations, device=args.device
    )
    fit_pass = all(
        row["final_fit"]["advantage_gate_pass"] and row["final_fit"]["policy_gate_pass"]
        for row in reports
    )
    cross_pass = bool(
        _finite(cross["mean_tv"])
        and _finite(cross["p95_tv"])
        and float(cross["mean_tv"]) <= FROZEN_GATES["cross_seed_mean_tv_max"]
        and float(cross["p95_tv"]) <= FROZEN_GATES["cross_seed_p95_tv_max"]
    )
    passed = bool(fit_pass and cross_pass)
    payload = {
        "schema": "SPINCORE_R7_3_PARTIAL_EXACT_ENSEMBLE_PAIRED_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "ensemble_size": int(args.ensemble_size),
        "exact_opponent_levels": int(args.exact_opponent_levels),
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "roots_per_seed": int(args.iterations * args.roots_per_iteration),
        "deck_semantics": "GENERATION2_AUTHORITATIVE_GLOBAL_ROOT_FORMULA_EXACT",
        "deck_formula": "seed*1000003 + global_root*97 + iteration",
        "rng_contract": "RECOVERED_SINGLE_COUPLED_BATCH_RNG_PRIMARY_MEMBER",
        "extra_members_perturb_primary_rng": False,
        "per_seed": reports,
        "cross_seed_observation_count": len(observations),
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "frozen_gates": dict(FROZEN_GATES),
        "per_seed_fit_pass": bool(fit_pass),
        "cross_seed_pass": bool(cross_pass),
        "r7_3_pass": bool(passed),
        "acceptance_gate_changed": False,
        "production_estimator_changed": False,
        "production_ensemble_changed": False,
        "promotion_note": (
            "Experimental paired candidate. Ensemble size 1 is the causal baseline and should "
            "reproduce the authoritative partial-exact level-2 screen at the same scale. Extra "
            "members are side fits that do not advance the live recovered batch RNG stream."
        ),
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if args.strict and not passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
