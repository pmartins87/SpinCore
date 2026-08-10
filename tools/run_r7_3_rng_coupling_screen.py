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


def collect_memory(
    *,
    algorithm_seed: int,
    deck_stream_seed: int,
    solver: SolverLibrary,
    device: str,
    iterations: int,
    roots_per_iteration: int,
    advantage_chunk_steps: int,
    advantage_max_steps_per_iteration: int,
    advantage_fit_target: float,
    batch_size: int,
    audit_size: int,
    reservoir_capacity: int,
    lr: float,
    split_traversal_rng: bool,
):
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

    # Current recovered semantics use bundle.batch_rng for both traversal/action
    # sampling and optimizer minibatch selection.  In the split diagnostic only,
    # traversal receives a dedicated persistent RNG while training keeps the
    # original bundle.batch_rng.  This changes no production/checkpoint contract.
    traversal_rng = None
    if split_traversal_rng:
        traversal_rng = random.Random(int(algorithm_seed) ^ TRAVERSAL_RNG_XOR)
        session.collector.rng = traversal_rng

    episode = hu_episode()
    global_root = 0
    checkpoints: list[dict] = []

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
        progress: list[dict] = []
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
            if row["fit_target_reached"]:
                break

        checkpoints.append(
            {
                "iteration": int(iteration),
                "roots": int(bundle.counters["roots"]),
                "advantage_progress": progress,
                "advantage_samples": len(bundle.adv_mem.items),
                "strategy_samples": len(bundle.pol_mem.items),
            }
        )

    return bundle, {
        "algorithm_seed": int(algorithm_seed),
        "deck_stream_seed": int(deck_stream_seed),
        "split_traversal_rng": bool(split_traversal_rng),
        "roots": int(bundle.counters["roots"]),
        "strategy_samples": len(bundle.pol_mem.items),
        "strategy_seen": int(bundle.pol_mem.seen),
        "advantage_samples": len(bundle.adv_mem.items),
        "advantage_seen": int(bundle.adv_mem.seen),
        "checkpoints": checkpoints,
    }


def mode_metrics(bundles) -> dict:
    out = {}
    for key_mode in ("raw", "poker_isomorphic"):
        keyer = _make_keyer(key_mode)
        agg_a = _aggregate_strategy(bundles[0].pol_mem.items, keyer)
        agg_b = _aggregate_strategy(bundles[1].pol_mem.items, keyer)
        out[key_mode] = _intersection_metrics(agg_a, agg_b)
    return out


def safe_ratio(a: float, b: float) -> float:
    return float(a) / max(float(b), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "R7.3 screening diagnostic for coupling between traversal/action RNG "
            "and optimizer minibatch RNG"
        )
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_RNG_COUPLING_SCREEN_256.json"),
    )
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_SHARED_DECK_STREAM_SEED)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-chunk-steps", type=int, default=256)
    ap.add_argument("--advantage-max-steps-per-iteration", type=int, default=1024)
    ap.add_argument("--advantage-fit-target", type=float, default=0.70)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=512)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) != 2:
        raise SystemExit("RNG coupling screen requires exactly two algorithm seeds")
    if args.iterations < 2:
        raise SystemExit("at least two iterations are required for training RNG to affect later traversal")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()

    mode_reports = {}
    for mode_name, split in (("coupled", False), ("split_traversal_training", True)):
        bundles = []
        collection = []
        for seed in seeds:
            bundle, report = collect_memory(
                algorithm_seed=seed,
                deck_stream_seed=args.deck_stream_seed,
                solver=solver,
                device=args.device,
                iterations=args.iterations,
                roots_per_iteration=args.roots_per_iteration,
                advantage_chunk_steps=args.advantage_chunk_steps,
                advantage_max_steps_per_iteration=args.advantage_max_steps_per_iteration,
                advantage_fit_target=args.advantage_fit_target,
                batch_size=args.batch_size,
                audit_size=args.audit_size,
                reservoir_capacity=args.reservoir_capacity,
                lr=args.lr,
                split_traversal_rng=split,
            )
            bundles.append(bundle)
            collection.append(report)
        metrics = mode_metrics(bundles)
        mode_reports[mode_name] = {
            "collection": collection,
            "overlap": metrics,
        }
        print(json.dumps({"mode": mode_name, "overlap": metrics}, sort_keys=True), flush=True)

    coupled = mode_reports["coupled"]["overlap"]["poker_isomorphic"]
    split = mode_reports["split_traversal_training"]["overlap"]["poker_isomorphic"]

    coupled_j = float(coupled["jaccard"])
    split_j = float(split["jaccard"])
    coupled_mass = 0.5 * (
        float(coupled["lcfr_weight_coverage_A"])
        + float(coupled["lcfr_weight_coverage_B"])
    )
    split_mass = 0.5 * (
        float(split["lcfr_weight_coverage_A"])
        + float(split["lcfr_weight_coverage_B"])
    )
    coupled_tv = float(coupled["shared_target_weighted_mean_tv"])
    split_tv = float(split["shared_target_weighted_mean_tv"])

    j_gain = safe_ratio(split_j, coupled_j)
    mass_gain = safe_ratio(split_mass, coupled_mass)
    tv_ratio = safe_ratio(split_tv, coupled_tv) if math.isfinite(coupled_tv) and math.isfinite(split_tv) else math.inf

    material = (j_gain >= 1.25) or (mass_gain >= 1.25) or (tv_ratio <= 0.75)
    diagnosis = (
        "TRAINING_TRAVERSAL_RNG_COUPLING_MATERIAL_AT_SCREEN_SCALE"
        if material
        else "TRAINING_TRAVERSAL_RNG_COUPLING_NOT_DOMINANT_AT_SCREEN_SCALE"
    )

    payload = {
        "schema": "SPINCORE_R7_3_RNG_COUPLING_SCREEN_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "roots_per_seed_per_mode": int(args.iterations * args.roots_per_iteration),
        "acceptance_gate_changed": False,
        "production_rng_contract_changed": False,
        "mode_reports": mode_reports,
        "summary": {
            "coupled_poker_isomorphic_jaccard": coupled_j,
            "split_poker_isomorphic_jaccard": split_j,
            "split_to_coupled_jaccard_ratio": j_gain,
            "coupled_mean_lcfr_weight_coverage": coupled_mass,
            "split_mean_lcfr_weight_coverage": split_mass,
            "split_to_coupled_weight_coverage_ratio": mass_gain,
            "coupled_shared_target_weighted_mean_tv": coupled_tv,
            "split_shared_target_weighted_mean_tv": split_tv,
            "split_to_coupled_shared_target_tv_ratio": tv_ratio,
            "diagnosis": diagnosis,
        },
        "interpretation_note": (
            "Diagnostic only. Current production/recovered semantics remain unchanged. "
            "The split mode gives traversal/action sampling its own persistent RNG while "
            "optimizer minibatch sampling retains bundle.batch_rng. Because deck seeds are "
            "explicit and common, changes in overlap/target TV isolate the effect of training "
            "RNG consumption feeding back into later traversal RNG state."
        ),
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
