from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.r7 import audit_model_fit, cross_seed_policy_tv
from spincore.solver import SolverLibrary

from run_r7_3_diagnostic import hu_episode, make_bundle, shared_cross_seed_observations
from run_r7_3_path_replication_screen import _fit_average_policy, _fit_pass, _safe_ratio
from run_r7_3_variance_decomposition import _advantage_fit_nrmse, _finite


DEFAULT_SEEDS = (20260829, 20260807)
DEFAULT_SHARED_DECK_STREAM_SEED = 0xD3C5EED
PAYOUT = (0.5, 0.3, 0.2)
ADV_RNG_XOR = 0x0A9D7A61
STRATEGY_RNG_XOR = 0x057A7E61
COMMON_ADV_BASE = 0xC0A771D5

MODE_SPECS = {
    "independent_1": (False, 1),
    "independent_4": (False, 4),
    "common_1": (True, 1),
    "common_4": (True, 4),
}


def _common_adv_seed(iteration: int, global_root: int, traverser: int, replicate: int) -> int:
    # Counter-based stream independent of algorithm seed and hidden cards.
    x = int(COMMON_ADV_BASE) & ((1 << 64) - 1)
    for value, mul in (
        (iteration, 0x9E3779B185EBCA87),
        (global_root, 0xC2B2AE3D27D4EB4F),
        (traverser, 0x165667B19E3779F9),
        (replicate, 0x85EBCA77C2B2AE63),
    ):
        x ^= (int(value) + 1) * mul
        x &= ((1 << 64) - 1)
        x ^= x >> 29
    return int(x)


def _memory_digest(items) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(item.observation)
        h.update(bytes(int(x) for x in item.legal))
        for value in item.target:
            h.update(float(value).hex().encode("ascii"))
            h.update(b";")
        h.update(float(item.weight).hex().encode("ascii"))
        h.update(str(int(item.iteration)).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def run_mode(
    *,
    mode_name: str,
    common_advantage_rng: bool,
    advantage_replicates: int,
    seeds: list[int],
    solver: SolverLibrary,
    device: str,
    deck_stream_seed: int,
    iterations: int,
    roots_per_iteration: int,
    advantage_chunk_steps: int,
    advantage_max_steps_per_iteration: int,
    advantage_fit_target: float,
    policy_chunk_steps: int,
    policy_max_steps: int,
    policy_fit_target: float,
    batch_size: int,
    audit_size: int,
    cross_seed_per_seed: int,
    reservoir_capacity: int,
    lr: float,
):
    episode = hu_episode()
    bundles = []
    reports = []

    for algorithm_seed in seeds:
        bundle = make_bundle(
            int(algorithm_seed),
            device=device,
            reservoir_capacity=int(reservoir_capacity),
            lr=float(lr),
        )
        session = DeepCFRDomainSession(
            solver_library=solver,
            bundle=bundle,
            terminal_utility=icm_delta_utility(PAYOUT),
            device=device,
        )
        independent_adv_rng = random.Random(int(algorithm_seed) ^ ADV_RNG_XOR)
        strategy_rng = random.Random(int(algorithm_seed) ^ STRATEGY_RNG_XOR)
        live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
        global_root = 0
        checkpoints = []

        for iteration in range(1, int(iterations) + 1):
            for _ in range(int(roots_per_iteration)):
                deck_seed = (
                    int(deck_stream_seed) * 1_000_003 + global_root * 97 + iteration
                ) & ((1 << 64) - 1)
                nodes = 0
                advantage_added = 0
                strategy_added = 0

                for player in live:
                    for replicate in range(int(advantage_replicates)):
                        if common_advantage_rng:
                            session.collector.rng = random.Random(
                                _common_adv_seed(iteration, global_root, player, replicate)
                            )
                        else:
                            session.collector.rng = independent_adv_rng
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

            iteration_adv_items = [
                item for item in bundle.adv_mem.items if int(item.iteration) == int(iteration)
            ]
            before_fit_digest = _memory_digest(iteration_adv_items)
            reset_seed = (int(algorithm_seed) ^ (iteration * 0x9E3779B1)) & 0x7FFFFFFF
            session.reset_advantage_network(init_seed=reset_seed, lr=lr)
            local_steps = 0
            progress = []
            audit_seed = int(algorithm_seed) ^ (iteration * 0x45D9F3B)
            while local_steps < int(advantage_max_steps_per_iteration):
                steps = min(
                    int(advantage_chunk_steps),
                    int(advantage_max_steps_per_iteration) - local_steps,
                )
                session.train_advantage(steps=steps, batch_size=int(batch_size))
                local_steps += steps
                nrmse = _advantage_fit_nrmse(
                    bundle,
                    sample_size=int(audit_size),
                    seed=audit_seed,
                    device=device,
                )
                row = {
                    "optimizer_steps": int(local_steps),
                    "weighted_nrmse": float(nrmse),
                    "frozen_gate_pass": _fit_pass(nrmse, "advantage_weighted_nrmse_max"),
                    "fit_target_reached": _finite(nrmse)
                    and float(nrmse) <= float(advantage_fit_target),
                }
                progress.append(row)
                if row["fit_target_reached"]:
                    break

            checkpoints.append(
                {
                    "iteration": int(iteration),
                    "roots": int(bundle.counters["roots"]),
                    "advantage_samples": len(bundle.adv_mem.items),
                    "advantage_seen": int(bundle.adv_mem.seen),
                    "strategy_samples": len(bundle.pol_mem.items),
                    "strategy_seen": int(bundle.pol_mem.seen),
                    "iteration_advantage_memory_digest_before_fit": before_fit_digest,
                    "final_advantage_fit": progress[-1],
                }
            )

        policy_progress, final_fit = _fit_average_policy(
            bundle=bundle,
            session=session,
            seed=int(algorithm_seed),
            device=device,
            policy_chunk_steps=int(policy_chunk_steps),
            policy_max_steps=int(policy_max_steps),
            policy_fit_target=float(policy_fit_target),
            batch_size=int(batch_size),
            audit_size=int(audit_size),
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
                "advantage_optimizer_steps": int(bundle.counters["adv_optimizer_steps"]),
                "policy_optimizer_steps": int(bundle.counters["policy_optimizer_steps"]),
                "checkpoints": checkpoints,
                "policy_progress": policy_progress,
                "final_fit": final_fit,
            }
        )
        bundles.append(bundle)

    observations = shared_cross_seed_observations(
        bundles,
        per_seed=int(cross_seed_per_seed),
        seed=0x715EED,
    )
    cross = cross_seed_policy_tv(
        bundles[0].policy,
        bundles[1].policy,
        observations,
        device=device,
    )
    iteration1_digest_equal = (
        reports[0]["checkpoints"][0]["iteration_advantage_memory_digest_before_fit"]
        == reports[1]["checkpoints"][0]["iteration_advantage_memory_digest_before_fit"]
    )
    if common_advantage_rng and not iteration1_digest_equal:
        raise RuntimeError(
            "common-path RNG invariant failed: iteration-1 uniform-policy Advantage memories differ"
        )

    return {
        "mode": mode_name,
        "common_advantage_rng": bool(common_advantage_rng),
        "advantage_replicates": int(advantage_replicates),
        "collection": reports,
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "cross_seed_observation_count": len(observations),
        "iteration1_common_memory_digest_equal": bool(iteration1_digest_equal),
        "all_fit_gates_pass": all(
            bool(row["final_fit"]["advantage_gate_pass"])
            and bool(row["final_fit"]["policy_gate_pass"])
            for row in reports
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="R7.3 common-random-number Advantage path screen")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_COMMON_PATH_RNG_SCREEN_256.json"),
    )
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--modes", default=",".join(MODE_SPECS))
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
        raise SystemExit("common-path screen requires exactly two algorithm seeds")
    modes = [x.strip() for x in str(args.modes).split(",") if x.strip()]
    if not modes or any(mode not in MODE_SPECS for mode in modes):
        raise SystemExit("invalid modes")
    if "independent_1" not in modes:
        raise SystemExit("independent_1 baseline required")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()
    reports = {}
    for mode in modes:
        common, reps = MODE_SPECS[mode]
        reports[mode] = run_mode(
            mode_name=mode,
            common_advantage_rng=common,
            advantage_replicates=reps,
            seeds=seeds,
            solver=solver,
            device=args.device,
            deck_stream_seed=int(args.deck_stream_seed),
            iterations=int(args.iterations),
            roots_per_iteration=int(args.roots_per_iteration),
            advantage_chunk_steps=int(args.advantage_chunk_steps),
            advantage_max_steps_per_iteration=int(args.advantage_max_steps_per_iteration),
            advantage_fit_target=float(args.advantage_fit_target),
            policy_chunk_steps=int(args.policy_chunk_steps),
            policy_max_steps=int(args.policy_max_steps),
            policy_fit_target=float(args.policy_fit_target),
            batch_size=int(args.batch_size),
            audit_size=int(args.audit_size),
            cross_seed_per_seed=int(args.cross_seed_per_seed),
            reservoir_capacity=int(args.reservoir_capacity),
            lr=float(args.lr),
        )

    baseline_mean = float(reports["independent_1"]["cross_seed"]["mean_tv"])
    baseline_p95 = float(reports["independent_1"]["cross_seed"]["p95_tv"])
    comparisons = {}
    for mode in modes:
        if mode == "independent_1":
            continue
        comparisons[mode] = {
            "mean_tv_ratio_to_independent_1": _safe_ratio(
                float(reports[mode]["cross_seed"]["mean_tv"]), baseline_mean
            ),
            "p95_tv_ratio_to_independent_1": _safe_ratio(
                float(reports[mode]["cross_seed"]["p95_tv"]), baseline_p95
            ),
        }

    common1_ratio = comparisons.get("common_1", {}).get("mean_tv_ratio_to_independent_1", 1.0)
    common4_ratio = comparisons.get("common_4", {}).get("mean_tv_ratio_to_independent_1", 1.0)
    payload = {
        "schema": "SPINCORE_R7_3_COMMON_PATH_RNG_SCREEN_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "modes": reports,
        "comparisons_to_independent_1": comparisons,
        "summary": {
            "independent_1_mean_tv": baseline_mean,
            "independent_1_p95_tv": baseline_p95,
            "common_1_mean_ratio": float(common1_ratio),
            "common_4_mean_ratio": float(common4_ratio),
            "diagnosis": (
                "COMMON_RANDOM_NUMBERS_MATERIAL_FOR_ADVANTAGE_PATH_STABILITY"
                if min(float(common1_ratio), float(common4_ratio)) <= 0.80
                else "COMMON_RANDOM_NUMBERS_NOT_MATERIAL_AT_SCREEN_SCALE"
            ),
        },
        "interpretation_note": (
            "Diagnostic only. Common modes derive the opponent-action RNG from iteration/root/"
            "traverser/replicate counters, excluding algorithm seed and hidden cards. This is a "
            "common-random-number variance-reduction experiment, not a frozen-gate change. Under "
            "the iteration-1 uniform policy and shared decks, common modes must produce byte-level "
            "identical Advantage memories across seeds; the workflow fails otherwise."
        ),
        "acceptance_gate_changed": False,
        "production_rng_contract_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
