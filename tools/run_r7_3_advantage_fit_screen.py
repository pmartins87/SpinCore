from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.r7 import FROZEN_GATES
from spincore.solver import SolverLibrary

from run_r7_3_diagnostic import hu_episode, make_bundle
from run_r7_3_support_overlap import _aggregate_strategy, _intersection_metrics, _make_keyer
from run_r7_3_variance_decomposition import _advantage_fit_nrmse, _finite


DEFAULT_SEEDS = (20260829, 20260807)
DEFAULT_SHARED_DECK_STREAM_SEED = 0xD3C5EED
PAYOUT = (0.5, 0.3, 0.2)
TRAVERSAL_RNG_XOR = 0x71A7E25D


def _iteration_items(items, iteration: int):
    return [item for item in items if int(item.iteration) == int(iteration)]


def _metrics(items_a, items_b) -> dict:
    out = {}
    for mode in ("raw", "poker_isomorphic"):
        keyer = _make_keyer(mode)
        agg_a = _aggregate_strategy(items_a, keyer)
        agg_b = _aggregate_strategy(items_b, keyer)
        out[mode] = _intersection_metrics(agg_a, agg_b)
    return out


def collect_mode(
    *,
    mode_name: str,
    seeds: list[int],
    deck_stream_seed: int,
    solver: SolverLibrary,
    device: str,
    iterations: int,
    roots_per_iteration: int,
    advantage_fit_target: float,
    advantage_max_steps_per_iteration: int,
    advantage_chunk_steps: int,
    batch_size: int,
    audit_size: int,
    reservoir_capacity: int,
    lr: float,
):
    bundles = []
    reports = []
    episode = hu_episode()

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
        # Remove the already-screened training/traversal RNG coupling from this
        # experiment. Traversal keeps a persistent algorithm-seed-specific RNG;
        # optimizer minibatches continue to use bundle.batch_rng.
        session.collector.rng = random.Random(int(algorithm_seed) ^ TRAVERSAL_RNG_XOR)

        global_root = 0
        checkpoints = []
        for iteration in range(1, int(iterations) + 1):
            for _ in range(int(roots_per_iteration)):
                deck_seed = (
                    int(deck_stream_seed) * 1_000_003 + global_root * 97 + iteration
                ) & ((1 << 64) - 1)
                session.collect_root(episode, iteration=iteration, deck_seed=deck_seed)
                global_root += 1

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
                    "frozen_gate_pass": _finite(nrmse)
                    and nrmse <= FROZEN_GATES["advantage_weighted_nrmse_max"],
                    "fit_target_reached": _finite(nrmse)
                    and nrmse <= float(advantage_fit_target),
                }
                progress.append(row)
                print(
                    json.dumps(
                        {
                            "mode": mode_name,
                            "seed": int(algorithm_seed),
                            "iteration": int(iteration),
                            "roots": int(bundle.counters["roots"]),
                            "advantage": row,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if row["fit_target_reached"]:
                    break

            checkpoints.append(
                {
                    "iteration": int(iteration),
                    "roots": int(bundle.counters["roots"]),
                    "advantage_progress": progress,
                    "final_iteration_fit": progress[-1],
                    "advantage_samples": len(bundle.adv_mem.items),
                    "strategy_samples": len(bundle.pol_mem.items),
                }
            )

        bundles.append(bundle)
        reports.append(
            {
                "algorithm_seed": int(algorithm_seed),
                "roots": int(bundle.counters["roots"]),
                "strategy_samples": len(bundle.pol_mem.items),
                "advantage_samples": len(bundle.adv_mem.items),
                "checkpoints": checkpoints,
            }
        )

    cumulative = _metrics(bundles[0].pol_mem.items, bundles[1].pol_mem.items)
    by_iteration = {}
    for iteration in range(1, int(iterations) + 1):
        by_iteration[str(iteration)] = _metrics(
            _iteration_items(bundles[0].pol_mem.items, iteration),
            _iteration_items(bundles[1].pol_mem.items, iteration),
        )

    return {
        "mode": mode_name,
        "advantage_fit_target": float(advantage_fit_target),
        "advantage_max_steps_per_iteration": int(advantage_max_steps_per_iteration),
        "collection": reports,
        "overlap_cumulative": cumulative,
        "overlap_by_iteration": by_iteration,
    }


def _safe_ratio(a: float, b: float) -> float:
    return float(a) / max(float(b), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "R7.3 screen for whether stronger AdvantageNet fitting reduces "
            "cross-seed CFR path/strategy-memory divergence"
        )
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_ADVANTAGE_FIT_SCREEN_256.json"),
    )
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_SHARED_DECK_STREAM_SEED)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-chunk-steps", type=int, default=256)
    ap.add_argument("--weak-target", type=float, default=0.70)
    ap.add_argument("--weak-max-steps", type=int, default=1024)
    ap.add_argument("--strong-target", type=float, default=0.50)
    ap.add_argument("--strong-max-steps", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=512)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) != 2:
        raise SystemExit("advantage-fit screen requires exactly two seeds")
    if args.iterations < 2:
        raise SystemExit("at least two iterations are required to observe fitted-advantage effects")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()

    weak = collect_mode(
        mode_name="weak_fit",
        seeds=seeds,
        deck_stream_seed=args.deck_stream_seed,
        solver=solver,
        device=args.device,
        iterations=args.iterations,
        roots_per_iteration=args.roots_per_iteration,
        advantage_fit_target=args.weak_target,
        advantage_max_steps_per_iteration=args.weak_max_steps,
        advantage_chunk_steps=args.advantage_chunk_steps,
        batch_size=args.batch_size,
        audit_size=args.audit_size,
        reservoir_capacity=args.reservoir_capacity,
        lr=args.lr,
    )
    strong = collect_mode(
        mode_name="strong_fit",
        seeds=seeds,
        deck_stream_seed=args.deck_stream_seed,
        solver=solver,
        device=args.device,
        iterations=args.iterations,
        roots_per_iteration=args.roots_per_iteration,
        advantage_fit_target=args.strong_target,
        advantage_max_steps_per_iteration=args.strong_max_steps,
        advantage_chunk_steps=args.advantage_chunk_steps,
        batch_size=args.batch_size,
        audit_size=args.audit_size,
        reservoir_capacity=args.reservoir_capacity,
        lr=args.lr,
    )

    weak_iter2 = weak["overlap_by_iteration"]["2"]["poker_isomorphic"]
    strong_iter2 = strong["overlap_by_iteration"]["2"]["poker_isomorphic"]
    weak_tv = float(weak_iter2["shared_target_weighted_mean_tv"])
    strong_tv = float(strong_iter2["shared_target_weighted_mean_tv"])
    weak_mass = 0.5 * (
        float(weak_iter2["lcfr_weight_coverage_A"])
        + float(weak_iter2["lcfr_weight_coverage_B"])
    )
    strong_mass = 0.5 * (
        float(strong_iter2["lcfr_weight_coverage_A"])
        + float(strong_iter2["lcfr_weight_coverage_B"])
    )
    weak_j = float(weak_iter2["jaccard"])
    strong_j = float(strong_iter2["jaccard"])

    tv_ratio = _safe_ratio(strong_tv, weak_tv) if math.isfinite(weak_tv) and math.isfinite(strong_tv) else math.inf
    mass_ratio = _safe_ratio(strong_mass, weak_mass)
    j_ratio = _safe_ratio(strong_j, weak_j)
    material = (tv_ratio <= 0.75) or (mass_ratio >= 1.25) or (j_ratio >= 1.25)

    payload = {
        "schema": "SPINCORE_R7_3_ADVANTAGE_FIT_SCREEN_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "roots_per_seed_per_mode": int(args.iterations * args.roots_per_iteration),
        "acceptance_gate_changed": False,
        "production_training_schedule_changed": False,
        "modes": {
            "weak_fit": weak,
            "strong_fit": strong,
        },
        "summary": {
            "weak_iteration2_poker_isomorphic_jaccard": weak_j,
            "strong_iteration2_poker_isomorphic_jaccard": strong_j,
            "strong_to_weak_jaccard_ratio": j_ratio,
            "weak_iteration2_mean_lcfr_weight_coverage": weak_mass,
            "strong_iteration2_mean_lcfr_weight_coverage": strong_mass,
            "strong_to_weak_weight_coverage_ratio": mass_ratio,
            "weak_iteration2_shared_target_weighted_mean_tv": weak_tv,
            "strong_iteration2_shared_target_weighted_mean_tv": strong_tv,
            "strong_to_weak_shared_target_tv_ratio": tv_ratio,
            "diagnosis": (
                "STRONGER_ADVANTAGE_FIT_MATERIAL_AT_SCREEN_SCALE"
                if material
                else "STRONGER_ADVANTAGE_FIT_NOT_DOMINANT_AT_SCREEN_SCALE"
            ),
        },
        "interpretation_note": (
            "Diagnostic only. Both modes use the same shared root-deck stream and "
            "algorithm-seed-specific traversal RNGs separated from optimizer minibatch RNG. "
            "Iteration 1 is collected before any fitted AdvantageNet exists, so differences "
            "between modes should begin only after advantage fitting can affect iteration 2."
        ),
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
