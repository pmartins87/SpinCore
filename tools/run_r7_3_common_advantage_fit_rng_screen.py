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
    _collect_replicated_root,
    _fit_average_policy,
    _fit_pass,
    _safe_ratio,
)
from run_r7_3_variance_decomposition import _advantage_fit_nrmse, _finite


DEFAULT_SEEDS = (20260829, 20260807)
DEFAULT_SHARED_DECK_STREAM_SEED = 0xD3C5EED
COMMON_INIT_BASE = 0x51A6E001
COMMON_BATCH_BASE = 0xBA7C4001
SEPARATE_BATCH_XOR = 0xBA7C5EED
POLICY_BATCH_XOR = 0xA9E12C7


def _fit_advantage(
    *, session, bundle, algorithm_seed: int, iteration: int, common_fit_rng: bool, args
):
    if common_fit_rng:
        reset_seed = (COMMON_INIT_BASE ^ (int(iteration) * 0x9E3779B1)) & 0x7FFFFFFF
        bundle.batch_rng = random.Random(COMMON_BATCH_BASE ^ (int(iteration) * 0x45D9F3B))
    else:
        reset_seed = (int(algorithm_seed) ^ (int(iteration) * 0x9E3779B1)) & 0x7FFFFFFF
        bundle.batch_rng = random.Random(
            int(algorithm_seed) ^ SEPARATE_BATCH_XOR ^ (int(iteration) * 0x45D9F3B)
        )
    session.reset_advantage_network(init_seed=int(reset_seed), lr=float(args.lr))
    local_steps = 0
    progress = []
    audit_seed = int(algorithm_seed) ^ (int(iteration) * 0x13579B)
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


def run_mode(*, common_fit_rng: bool, seeds: list[int], solver: SolverLibrary, args):
    bundles = []
    reports = []
    episode = hu_episode()
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
        adv_rng = random.Random(int(algorithm_seed) ^ ADV_RNG_XOR)
        strategy_rng = random.Random(int(algorithm_seed) ^ STRATEGY_RNG_XOR)
        checkpoints = []
        global_root = 0
        for iteration in range(1, int(args.iterations) + 1):
            for _ in range(int(args.roots_per_iteration)):
                ds = (
                    int(args.deck_stream_seed) * 1_000_003 + global_root * 97 + iteration
                ) & ((1 << 64) - 1)
                _collect_replicated_root(
                    session=session,
                    solver=solver,
                    episode=episode,
                    deck_seed=ds,
                    iteration=iteration,
                    advantage_replicates=1,
                    strategy_replicates=1,
                    advantage_rng=adv_rng,
                    strategy_rng=strategy_rng,
                )
                global_root += 1
            progress = _fit_advantage(
                session=session,
                bundle=bundle,
                algorithm_seed=int(algorithm_seed),
                iteration=int(iteration),
                common_fit_rng=bool(common_fit_rng),
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

        # Isolate upstream Advantage-fit randomness from final policy fitting.
        bundle.batch_rng = random.Random(int(algorithm_seed) ^ POLICY_BATCH_XOR)
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
                "checkpoints": checkpoints,
                "policy_progress": policy_progress,
                "final_fit": final_fit,
                "roots": int(bundle.counters["roots"]),
                "nodes": int(bundle.counters["nodes"]),
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
        "common_advantage_init_and_batch_rng": bool(common_fit_rng),
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
    ap = argparse.ArgumentParser(description="R7.3 common Advantage neural-fit randomness screen")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_COMMON_ADVANTAGE_FIT_RNG_SCREEN_256.json"))
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
        raise SystemExit("requires exactly two algorithm seeds")
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()
    separate = run_mode(common_fit_rng=False, seeds=seeds, solver=solver, args=args)
    common = run_mode(common_fit_rng=True, seeds=seeds, solver=solver, args=args)
    mean_ratio = _safe_ratio(
        float(common["cross_seed"]["mean_tv"]), float(separate["cross_seed"]["mean_tv"])
    )
    p95_ratio = _safe_ratio(
        float(common["cross_seed"]["p95_tv"]), float(separate["cross_seed"]["p95_tv"])
    )
    payload = {
        "schema": "SPINCORE_R7_3_COMMON_ADVANTAGE_FIT_RNG_SCREEN_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "separate_fit_randomness": separate,
        "common_fit_randomness": common,
        "summary": {
            "common_to_separate_mean_tv_ratio": float(mean_ratio),
            "common_to_separate_p95_tv_ratio": float(p95_ratio),
            "diagnosis": (
                "COMMON_ADVANTAGE_FIT_RANDOMNESS_MATERIAL"
                if min(mean_ratio, p95_ratio) <= 0.85
                else "COMMON_ADVANTAGE_FIT_RANDOMNESS_NOT_MATERIAL_AT_SCREEN_SCALE"
            ),
        },
        "interpretation_note": (
            "Diagnostic only. Traversal and strategy sampling remain seed-specific and separate. "
            "The common mode uses the same AdvantageNet initialization seed and the same fresh "
            "per-iteration minibatch RNG sequence across algorithm seeds. Final AveragePolicy "
            "training is returned to seed-specific RNG, isolating upstream Advantage neural-fit "
            "randomness. No production RNG/checkpoint contract is changed."
        ),
        "acceptance_gate_changed": False,
        "production_training_rng_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
