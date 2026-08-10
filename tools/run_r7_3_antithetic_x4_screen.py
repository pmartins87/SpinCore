from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.r7 import cross_seed_policy_tv
from spincore.solver import SolverLibrary

from run_r7_3_diagnostic import hu_episode, make_bundle, shared_cross_seed_observations
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
X4 = 4


class ShiftedUniformRNG:
    """Marginally-uniform antithetic/rotated-lattice random stream.

    Every replicate uses the same underlying U_n sequence with a different
    quarter-turn offset. For each replicate, (U_n + offset) mod 1 is still
    exactly Uniform(0,1), so each complete external-sampling trajectory keeps
    the recovered estimator's marginal distribution. Correlation is introduced
    only across the four trajectories in a root/traverser group.
    """

    def __init__(self, seed: int, offset: float):
        self._rng = random.Random(int(seed))
        self._offset = float(offset)

    def random(self) -> float:
        return (self._rng.random() + self._offset) % 1.0


def _group_seed(algorithm_seed: int, iteration: int, global_root: int, traverser: int) -> int:
    x = int(algorithm_seed) ^ ADV_RNG_XOR
    x ^= int(iteration) * 0x9E3779B185EBCA87
    x ^= int(global_root) * 0xC2B2AE3D27D4EB4F
    x ^= int(traverser) * 0x165667B19E3779F9
    return x & ((1 << 64) - 1)


def _fit_advantage(session, bundle, *, seed, iteration, args):
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
    return progress


def run_mode(*, mode: str, seeds: list[int], solver: SolverLibrary, args):
    episode = hu_episode()
    bundles = []
    reports = []

    for algorithm_seed in seeds:
        bundle = make_bundle(
            int(algorithm_seed),
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
        independent_rng = random.Random(int(algorithm_seed) ^ ADV_RNG_XOR)
        strategy_rng = random.Random(int(algorithm_seed) ^ STRATEGY_RNG_XOR)
        live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
        global_root = 0
        checkpoints = []

        for iteration in range(1, int(args.iterations) + 1):
            for _ in range(int(args.roots_per_iteration)):
                deck_seed = (
                    int(args.deck_stream_seed) * 1_000_003 + global_root * 97 + iteration
                ) & ((1 << 64) - 1)
                nodes = advantage_added = strategy_added = 0

                for player in live:
                    if mode == "antithetic_x4":
                        rngs = [
                            ShiftedUniformRNG(
                                _group_seed(algorithm_seed, iteration, global_root, player),
                                replicate / X4,
                            )
                            for replicate in range(X4)
                        ]
                    elif mode == "independent_x4":
                        rngs = [independent_rng] * X4
                    else:
                        raise ValueError(mode)

                    for replicate in range(X4):
                        session.collector.rng = rngs[replicate]
                        root = solver.create(episode, int(deck_seed))
                        try:
                            result = session.collector.collect_advantage(
                                root,
                                traverser=int(player),
                                iteration=int(iteration),
                            )
                        finally:
                            root.close()
                        nodes += int(result.nodes)
                        advantage_added += int(result.samples_added)

                session.collector.rng = strategy_rng
                for player in live:
                    root = solver.create(episode, int(deck_seed))
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

                counters = bundle.counters
                counters["iteration"] = max(int(counters["iteration"]), int(iteration))
                counters["roots"] += 1
                counters["nodes"] += int(nodes)
                counters["advantage_samples"] += int(advantage_added)
                counters["strategy_samples"] += int(strategy_added)
                global_root += 1

            progress = _fit_advantage(
                session,
                bundle,
                seed=int(algorithm_seed),
                iteration=int(iteration),
                args=args,
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
                    "final_advantage_fit": progress[-1],
                }
            )

        policy_progress, final_fit = _fit_average_policy(
            bundle=bundle,
            session=session,
            seed=int(algorithm_seed),
            device=args.device,
            policy_chunk_steps=int(args.policy_chunk_steps),
            policy_max_steps=int(args.policy_max_steps),
            policy_fit_target=float(args.policy_fit_target),
            batch_size=int(args.batch_size),
            audit_size=int(args.audit_size),
        )
        reports.append(
            {
                "algorithm_seed": int(algorithm_seed),
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
        bundles,
        per_seed=int(args.cross_seed_per_seed),
        seed=0x715EED,
    )
    cross = cross_seed_policy_tv(
        bundles[0].policy,
        bundles[1].policy,
        observations,
        device=args.device,
    )
    return {
        "mode": mode,
        "advantage_replicates": X4,
        "per_seed": reports,
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "observation_count": len(observations),
        "all_fit_gates_pass": all(
            bool(row["final_fit"]["advantage_gate_pass"])
            and bool(row["final_fit"]["policy_gate_pass"])
            for row in reports
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="R7.3 antithetic versus independent Advantage x4")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_ANTITHETIC_X4_SCREEN_256.json"))
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
    if len(seeds) != 2:
        raise SystemExit("antithetic x4 screen requires exactly two seeds")
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()

    independent = run_mode(mode="independent_x4", seeds=seeds, solver=solver, args=args)
    antithetic = run_mode(mode="antithetic_x4", seeds=seeds, solver=solver, args=args)
    mean_ratio = _safe_ratio(
        float(antithetic["cross_seed"]["mean_tv"]),
        float(independent["cross_seed"]["mean_tv"]),
    )
    p95_ratio = _safe_ratio(
        float(antithetic["cross_seed"]["p95_tv"]),
        float(independent["cross_seed"]["p95_tv"]),
    )
    payload = {
        "schema": "SPINCORE_R7_3_ANTITHETIC_X4_SCREEN_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "independent_x4": independent,
        "antithetic_x4": antithetic,
        "summary": {
            "antithetic_to_independent_mean_tv_ratio": float(mean_ratio),
            "antithetic_to_independent_p95_tv_ratio": float(p95_ratio),
            "diagnosis": (
                "ANTITHETIC_ROTATED_LATTICE_X4_MATERIAL"
                if min(mean_ratio, p95_ratio) <= 0.85
                else "ANTITHETIC_ROTATED_LATTICE_X4_NOT_MATERIAL_AT_SCREEN_SCALE"
            ),
        },
        "interpretation_note": (
            "Diagnostic only. Antithetic x4 keeps every individual trajectory marginally "
            "distributed exactly as the recovered external-sampling estimator: replicate r uses "
            "the same underlying Uniform(0,1) sequence shifted by r/4 modulo 1. The four "
            "trajectories are therefore negatively/low-discrepancy correlated without changing "
            "their marginal policy sampling law. This tests variance reduction per four paths, "
            "not a gate relaxation or production promotion."
        ),
        "acceptance_gate_changed": False,
        "production_sampling_schedule_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
