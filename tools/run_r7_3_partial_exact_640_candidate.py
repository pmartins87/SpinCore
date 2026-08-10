from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import run_r7_3_diagnostic as diagnostic
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


def run_seed(*, seed: int, solver: SolverLibrary, args):
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
    # Preserve the recovered single-RNG-state contract: partial-exact Advantage
    # sampling, own-reach strategy sampling and optimizer minibatches consume the
    # same persistent bundle.batch_rng object in execution order.
    partial = PartialExactAdvantageCollector(
        policy=session.behavior,
        terminal_utility=session.terminal_utility,
        rng=bundle.batch_rng,
        advantage_memory=bundle.adv_mem,
        strategy_memory=bundle.pol_mem,
    )
    session.collector.rng = bundle.batch_rng
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
                        session.collector.collect_strategy_own_reach(
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

        reset_seed = (int(seed) ^ (iteration * 0x9E3779B1)) & 0x7FFFFFFF
        session.reset_advantage_network(init_seed=reset_seed, lr=float(args.lr))
        local_steps = 0
        progress = []
        audit_seed = int(seed) ^ (iteration * 0x45D9F3B)
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
        checkpoints.append(
            {
                "iteration": int(iteration),
                "roots": int(bundle.counters["roots"]),
                "nodes": int(bundle.counters["nodes"]),
                "advantage_samples": len(bundle.adv_mem.items),
                "advantage_seen": int(bundle.adv_mem.seen),
                "strategy_samples": len(bundle.pol_mem.items),
                "strategy_seen": int(bundle.pol_mem.seen),
                "final_advantage_fit": progress[-1],
            }
        )
        print(json.dumps({"seed": seed, "checkpoint": checkpoints[-1]}, sort_keys=True), flush=True)

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
    final_fit = audit_model_fit(
        bundle,
        sample_size=max(int(args.audit_size), 2048),
        seed=int(seed) ^ 0x2468ACE0,
        device=args.device,
    )
    return bundle, {
        "algorithm_seed": int(seed),
        "roots": int(bundle.counters["roots"]),
        "nodes": int(bundle.counters["nodes"]),
        "advantage_samples": len(bundle.adv_mem.items),
        "advantage_seen": int(bundle.adv_mem.seen),
        "strategy_samples": len(bundle.pol_mem.items),
        "strategy_seen": int(bundle.pol_mem.seen),
        "advantage_optimizer_steps": int(bundle.counters["adv_optimizer_steps"]),
        "policy_optimizer_steps": int(bundle.counters["policy_optimizer_steps"]),
        "checkpoints": checkpoints,
        "policy_progress": policy_progress,
        "final_fit": {
            "advantage_weighted_nrmse": float(final_fit["advantage_weighted_nrmse"]),
            "policy_weighted_mean_tv": float(final_fit["policy_weighted_mean_tv"]),
            "advantage_gate_pass": _fit_pass(
                final_fit["advantage_weighted_nrmse"], "advantage_weighted_nrmse_max"
            ),
            "policy_gate_pass": _fit_pass(
                final_fit["policy_weighted_mean_tv"], "policy_weighted_mean_tv_max"
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="R7.3 deck-exact partial-opponent 640 candidate")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--exact-opponent-levels", type=int, required=True)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-chunk-steps", type=int, default=256)
    ap.add_argument("--advantage-max-steps-per-iteration", type=int, default=4096)
    ap.add_argument("--advantage-fit-target", type=float, default=0.50)
    ap.add_argument("--policy-chunk-steps", type=int, default=256)
    ap.add_argument("--policy-max-steps", type=int, default=32768)
    ap.add_argument("--policy-fit-target", type=float, default=0.105)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=1024)
    ap.add_argument("--cross-seed-per-seed", type=int, default=1024)
    ap.add_argument("--reservoir-capacity", type=int, default=400000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) != 2 or int(args.exact_opponent_levels) <= 0:
        raise SystemExit("exactly two seeds and a positive exact-opponent level are required")
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()
    bundles = []
    reports = []
    for seed in seeds:
        bundle, report = run_seed(seed=int(seed), solver=solver, args=args)
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
    cross_pass = (
        _finite(cross["mean_tv"])
        and _finite(cross["p95_tv"])
        and float(cross["mean_tv"]) <= FROZEN_GATES["cross_seed_mean_tv_max"]
        and float(cross["p95_tv"]) <= FROZEN_GATES["cross_seed_p95_tv_max"]
    )
    passed = bool(fit_pass and cross_pass)
    payload = {
        "schema": "SPINCORE_R7_3_PARTIAL_EXACT_640_CANDIDATE_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "exact_opponent_levels": int(args.exact_opponent_levels),
        "roots_per_seed": int(args.iterations * args.roots_per_iteration),
        "deck_semantics": "GENERATION2_AUTHORITATIVE_GLOBAL_ROOT_FORMULA_EXACT",
        "deck_formula": "seed*1000003 + global_root*97 + iteration",
        "rng_contract": "RECOVERED_SINGLE_COUPLED_BATCH_RNG",
        "reservoir_capacity": int(args.reservoir_capacity),
        "per_seed": reports,
        "cross_seed_observation_count": len(observations),
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "frozen_gates": dict(FROZEN_GATES),
        "per_seed_fit_pass": bool(fit_pass),
        "cross_seed_pass": bool(cross_pass),
        "r7_3_pass": bool(passed),
        "acceptance_gate_changed": False,
        "production_estimator_changed": True,
        "promotion_note": (
            "Experimental acceptance candidate only. Positive exact-opponent levels change the "
            "Advantage estimator and must be explicitly versioned and checkpoint/resume "
            "recertified before any production promotion."
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
