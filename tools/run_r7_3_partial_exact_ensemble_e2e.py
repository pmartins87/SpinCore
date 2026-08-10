from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.r7 import FROZEN_GATES, audit_model_fit, cross_seed_policy_tv
from spincore.solver import SolverLibrary

from run_r7_3_advantage_ensemble_end_to_end import (
    EnsembleAdvantagePolicy,
    _ensemble_nrmse,
    _train_member,
)
from run_r7_3_diagnostic import hu_episode, make_bundle, shared_cross_seed_observations
from run_r7_3_partial_exact_advantage_screen import PartialExactAdvantageCollector
from run_r7_3_path_replication_screen import ADV_RNG_XOR, STRATEGY_RNG_XOR, PAYOUT, _fit_average_policy


DEFAULT_SEEDS = (20260829, 20260807)
DEFAULT_SHARED_DECK_STREAM_SEED = 0xD3C5EED


def run_seed(*, seed: int, ensemble_size: int, solver: SolverLibrary, args):
    bundle = make_bundle(
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
        rng=random.Random(int(seed) ^ ADV_RNG_XOR),
        advantage_memory=bundle.adv_mem,
        strategy_memory=bundle.pol_mem,
    )
    session.behavior = behavior
    session.collector = partial
    advantage_rng = random.Random(int(seed) ^ ADV_RNG_XOR)
    strategy_rng = random.Random(int(seed) ^ STRATEGY_RNG_XOR)
    episode = hu_episode()
    live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
    checkpoints = []
    global_root = 0

    for iteration in range(1, int(args.iterations) + 1):
        for _ in range(int(args.roots_per_iteration)):
            ds = (
                int(args.deck_stream_seed) * 1_000_003 + global_root * 97 + iteration
            ) & ((1 << 64) - 1)
            nodes = advantage_added = strategy_added = 0
            partial.rng = advantage_rng
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

            partial.rng = strategy_rng
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

        state = bundle.adv_mem.state_dict()
        models = []
        members = []
        for member in range(int(ensemble_size)):
            model, report = _train_member(
                memory_state=state,
                algorithm_seed=int(seed),
                iteration=int(iteration),
                member=int(member),
                solver=solver,
                args=args,
            )
            models.append(model)
            members.append(report)
        behavior.models = models
        ensemble_nrmse = _ensemble_nrmse(
            models,
            bundle.adv_mem,
            sample_size=int(args.audit_size),
            seed=int(seed) ^ (iteration * 0x5EEDBEEF),
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
                "ensemble_weighted_nrmse": float(ensemble_nrmse),
                "ensemble_frozen_fit_gate_pass": bool(
                    ensemble_nrmse <= FROZEN_GATES["advantage_weighted_nrmse_max"]
                ),
                "members": members,
            }
        )
        print(json.dumps({"seed": seed, "checkpoint": checkpoints[-1]}, sort_keys=True), flush=True)

    bundle.batch_rng = random.Random(int(seed) ^ 0xA9E12C7)
    policy_progress, _ = _fit_average_policy(
        bundle=bundle,
        session=session,
        seed=int(seed),
        device=args.device,
        policy_chunk_steps=int(args.policy_chunk_steps),
        policy_max_steps=int(args.policy_max_steps),
        policy_fit_target=float(args.policy_fit_target),
        batch_size=int(args.batch_size),
        audit_size=int(args.audit_size),
    )
    policy_audit = audit_model_fit(
        bundle,
        sample_size=max(int(args.audit_size), 2048),
        seed=int(seed) ^ 0x2468ACE0,
        device=args.device,
    )
    ensemble_nrmse = _ensemble_nrmse(
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
        "checkpoints": checkpoints,
        "policy_progress": policy_progress,
        "final_fit": {
            "ensemble_advantage_weighted_nrmse": float(ensemble_nrmse),
            "policy_weighted_mean_tv": float(policy_audit["policy_weighted_mean_tv"]),
            "advantage_gate_pass": bool(ensemble_nrmse <= FROZEN_GATES["advantage_weighted_nrmse_max"]),
            "policy_gate_pass": bool(
                float(policy_audit["policy_weighted_mean_tv"])
                <= FROZEN_GATES["policy_weighted_mean_tv_max"]
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Partial-exact opponent sampling plus Advantage ensemble screen")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ensemble-size", type=int, choices=(1, 2, 4), required=True)
    ap.add_argument("--exact-opponent-levels", type=int, default=2)
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_SHARED_DECK_STREAM_SEED)
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
    observations = shared_cross_seed_observations(
        bundles, per_seed=int(args.cross_seed_per_seed), seed=0x715EED
    )
    cross = cross_seed_policy_tv(
        bundles[0].policy, bundles[1].policy, observations, device=args.device
    )
    fit_pass = all(
        x["final_fit"]["advantage_gate_pass"] and x["final_fit"]["policy_gate_pass"]
        for x in reports
    )
    payload = {
        "schema": "SPINCORE_R7_3_PARTIAL_EXACT_ENSEMBLE_E2E_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "ensemble_size": int(args.ensemble_size),
        "exact_opponent_levels": int(args.exact_opponent_levels),
        "algorithm_seeds": seeds,
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "per_seed": reports,
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "observation_count": len(observations),
        "per_seed_fit_pass": bool(fit_pass),
        "frozen_gates": dict(FROZEN_GATES),
        "interpretation_note": (
            "Diagnostic only. The strongest bounded path-variance mechanism (partial exact "
            "opponent expectation) is combined with raw-Advantage model ensembling before the "
            "unchanged hard regret-matching map. This factorial follow-up tests whether target "
            "variance and same-memory fit variance reductions are complementary end-to-end."
        ),
        "acceptance_gate_changed": False,
        "production_estimator_changed": False,
        "production_ensemble_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
