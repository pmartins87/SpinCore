from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.r7 import audit_model_fit, cross_seed_policy_tv
from spincore.solver import SolverLibrary

from run_r7_3_diagnostic import hu_episode, make_bundle, shared_cross_seed_observations
from run_r7_3_partial_exact_advantage_screen import PartialExactAdvantageCollector
from run_r7_3_path_replication_screen import (
    ADV_RNG_XOR,
    STRATEGY_RNG_XOR,
    PAYOUT,
    _fit_average_policy,
    _fit_pass,
    _safe_ratio,
)
from run_r7_3_variance_decomposition import _advantage_fit_nrmse, _finite


DEFAULT_SEEDS = (20260829, 20260807)
DEFAULT_SHARED_DECK_STREAM_SEED = 0xD3C5EED
COMMON_INIT_BASE = 0x51A6E001
COMMON_BATCH_BASE = 0xBA7C4001
INDEPENDENT_BATCH_XOR = 0xBA7C5EED
POLICY_BATCH_XOR = 0xA9E12C7


def _fit_advantage(session, bundle, *, seed: int, iteration: int, common_fit: bool, args):
    if common_fit:
        init_seed = (COMMON_INIT_BASE ^ (int(iteration) * 0x9E3779B1)) & 0x7FFFFFFF
        bundle.batch_rng = random.Random(COMMON_BATCH_BASE ^ (int(iteration) * 0x45D9F3B))
    else:
        init_seed = (int(seed) ^ (int(iteration) * 0x9E3779B1)) & 0x7FFFFFFF
        bundle.batch_rng = random.Random(
            int(seed) ^ INDEPENDENT_BATCH_XOR ^ (int(iteration) * 0x45D9F3B)
        )
    session.reset_advantage_network(init_seed=int(init_seed), lr=float(args.lr))
    local_steps = 0
    progress = []
    audit_seed = int(seed) ^ (int(iteration) * 0x13579B)
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
    return progress


def run_mode(*, common_fit: bool, seeds: list[int], solver: SolverLibrary, args):
    bundles = []
    reports = []
    episode = hu_episode()
    for seed in seeds:
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
        adv_rng = random.Random(int(seed) ^ ADV_RNG_XOR)
        strategy_rng = random.Random(int(seed) ^ STRATEGY_RNG_XOR)
        partial = PartialExactAdvantageCollector(
            policy=session.behavior,
            terminal_utility=session.terminal_utility,
            rng=adv_rng,
            advantage_memory=bundle.adv_mem,
            strategy_memory=bundle.pol_mem,
        )
        session.collector.rng = strategy_rng
        live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
        global_root = 0
        checkpoints = []
        for iteration in range(1, int(args.iterations) + 1):
            for _ in range(int(args.roots_per_iteration)):
                ds = (
                    int(args.deck_stream_seed) * 1_000_003 + global_root * 97 + iteration
                ) & ((1 << 64) - 1)
                nodes = advantage_added = strategy_added = 0
                for player in live:
                    root = solver.create(episode, int(ds))
                    try:
                        result = partial.collect_advantage_partial_exact(
                            root,
                            traverser=int(player),
                            iteration=int(iteration),
                            exact_opponent_levels=int(args.exact_opponent_levels),
                        )
                    finally:
                        root.close()
                    nodes += int(result.nodes)
                    advantage_added += int(result.samples_added)
                for player in live:
                    root = solver.create(episode, int(ds))
                    try:
                        strategy_added += int(
                            session.collector.collect_strategy_own_reach(
                                root,
                                target_player=int(player),
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

            progress = _fit_advantage(
                session,
                bundle,
                seed=int(seed),
                iteration=int(iteration),
                common_fit=bool(common_fit),
                args=args,
            )
            checkpoints.append(
                {
                    "iteration": int(iteration),
                    "roots": int(bundle.counters["roots"]),
                    "advantage_samples": len(bundle.adv_mem.items),
                    "advantage_seen": int(bundle.adv_mem.seen),
                    "strategy_samples": len(bundle.pol_mem.items),
                    "strategy_seen": int(bundle.pol_mem.seen),
                    "final_advantage_fit": progress[-1],
                }
            )

        # Keep final AveragePolicy training seed-specific in both modes.
        bundle.batch_rng = random.Random(int(seed) ^ POLICY_BATCH_XOR)
        policy_progress, final_fit = _fit_average_policy(
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
        reports.append(
            {
                "algorithm_seed": int(seed),
                "roots": int(bundle.counters["roots"]),
                "nodes": int(bundle.counters["nodes"]),
                "advantage_samples": len(bundle.adv_mem.items),
                "advantage_seen": int(bundle.adv_mem.seen),
                "strategy_samples": len(bundle.pol_mem.items),
                "strategy_seen": int(bundle.pol_mem.seen),
                "checkpoints": checkpoints,
                "policy_progress": policy_progress,
                "final_fit": final_fit,
            }
        )
        bundles.append(bundle)

    observations = shared_cross_seed_observations(
        bundles, per_seed=int(args.cross_seed_per_seed), seed=0x715EED
    )
    cross = cross_seed_policy_tv(
        bundles[0].policy, bundles[1].policy, observations, device=args.device
    )
    return {
        "common_advantage_fit_randomness": bool(common_fit),
        "per_seed": reports,
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "observation_count": len(observations),
        "all_fit_gates_pass": all(
            row["final_fit"]["advantage_gate_pass"]
            and row["final_fit"]["policy_gate_pass"]
            for row in reports
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Partial-exact plus common Advantage fit randomness screen")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_PARTIAL_EXACT_COMMON_FIT_SCREEN_256.json"))
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--exact-opponent-levels", type=int, default=2)
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
    independent = run_mode(common_fit=False, seeds=seeds, solver=solver, args=args)
    common = run_mode(common_fit=True, seeds=seeds, solver=solver, args=args)
    mean_ratio = _safe_ratio(
        float(common["cross_seed"]["mean_tv"]), float(independent["cross_seed"]["mean_tv"])
    )
    p95_ratio = _safe_ratio(
        float(common["cross_seed"]["p95_tv"]), float(independent["cross_seed"]["p95_tv"])
    )
    payload = {
        "schema": "SPINCORE_R7_3_PARTIAL_EXACT_COMMON_FIT_SCREEN_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "exact_opponent_levels": int(args.exact_opponent_levels),
        "independent_fit_randomness": independent,
        "common_fit_randomness": common,
        "summary": {
            "common_to_independent_mean_tv_ratio": float(mean_ratio),
            "common_to_independent_p95_tv_ratio": float(p95_ratio),
            "diagnosis": (
                "PARTIAL_EXACT_PLUS_COMMON_FIT_MATERIAL"
                if min(mean_ratio, p95_ratio) <= 0.85
                else "PARTIAL_EXACT_PLUS_COMMON_FIT_NOT_MATERIAL_AT_SCREEN_SCALE"
            ),
        },
        "interpretation_note": (
            "Diagnostic only. Both modes use the same partial-exact opponent estimator, shared "
            "deal schedule, seed-specific traversal/strategy randomness, and seed-specific final "
            "AveragePolicy training. Only AdvantageNet initialization and per-iteration minibatch "
            "RNG are made common across algorithm seeds in the common mode. This tests whether "
            "target-variance reduction and fit-randomness reduction combine constructively."
        ),
        "acceptance_gate_changed": False,
        "production_estimator_changed": False,
        "production_fit_rng_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
