from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.r7 import FROZEN_GATES, stratified_audit_indices
from spincore.solver import SolverLibrary
from spincore_nn import AveragePolicyNet

from run_r7_3_diagnostic import hu_episode, make_bundle
from run_r7_3_variance_decomposition import (
    _advantage_fit_nrmse,
    _finite,
    _tv,
    fit_policy,
)


DEFAULT_SEEDS = (20260829, 20260807)
PAYOUT = (0.5, 0.3, 0.2)
DEFAULT_SHARED_DECK_STREAM_SEED = 0xD3C5EED
INDEPENDENT_DECK_BASELINE_MEAN_TV = 0.469892218708992


def support_observations(items, *, sample_size: int, seed: int) -> list[bytes]:
    ids = stratified_audit_indices(len(items), int(sample_size), int(seed))
    return [items[i].observation for i in ids]


def collect_memory_shared_deck(
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
    episode = hu_episode()
    global_root = 0
    checkpoints: list[dict] = []

    for iteration in range(1, int(iterations) + 1):
        for _ in range(int(roots_per_iteration)):
            # Both algorithm seeds see the exact same hidden deal/future-board
            # stream for root k. Only model/reservoir/action-sampling randomness
            # differs. This is diagnostic common-random-number control; it does
            # not alter the production R7.3 acceptance gate.
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
            print(
                json.dumps(
                    {
                        "phase": "collect_shared_deck",
                        "algorithm_seed": int(algorithm_seed),
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
                "strategy_samples": len(bundle.pol_mem.items),
                "advantage_samples": len(bundle.adv_mem.items),
            }
        )

    return bundle, {
        "algorithm_seed": int(algorithm_seed),
        "deck_stream_seed": int(deck_stream_seed),
        "roots": int(bundle.counters["roots"]),
        "strategy_samples": len(bundle.pol_mem.items),
        "strategy_seen": int(bundle.pol_mem.seen),
        "advantage_samples": len(bundle.adv_mem.items),
        "advantage_seen": int(bundle.adv_mem.seen),
        "checkpoints": checkpoints,
    }


def comparisons_for_support(models: dict[tuple[int, int], AveragePolicyNet], observations, *, device: str):
    return {
        "same_memory_A_different_optimizer_seed": _tv(
            models[(0, 0)], models[(0, 1)], observations, device=device
        ),
        "same_memory_B_different_optimizer_seed": _tv(
            models[(1, 0)], models[(1, 1)], observations, device=device
        ),
        "different_memory_same_optimizer_seed_0": _tv(
            models[(0, 0)], models[(1, 0)], observations, device=device
        ),
        "different_memory_same_optimizer_seed_1": _tv(
            models[(0, 1)], models[(1, 1)], observations, device=device
        ),
    }


def mean_pair(block: dict, key_a: str, key_b: str) -> float:
    return 0.5 * (float(block[key_a]["mean_tv"]) + float(block[key_b]["mean_tv"]))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="R7.3 common-deck and support-conditioned instability diagnostic"
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_SHARED_DECK_SUPPORT_640.json"),
    )
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_SHARED_DECK_STREAM_SEED)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-chunk-steps", type=int, default=256)
    ap.add_argument("--advantage-max-steps-per-iteration", type=int, default=2048)
    ap.add_argument("--advantage-fit-target", type=float, default=0.70)
    ap.add_argument("--policy-chunk-steps", type=int, default=256)
    ap.add_argument("--policy-max-steps", type=int, default=8192)
    ap.add_argument("--policy-fit-target", type=float, default=0.105)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=1024)
    ap.add_argument("--support-sample-size", type=int, default=1024)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--independent-deck-baseline-mean-tv",
        type=float,
        default=INDEPENDENT_DECK_BASELINE_MEAN_TV,
    )
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) != 2:
        raise SystemExit("shared-deck diagnostic requires exactly two algorithm seeds")
    if args.iterations <= 0 or args.roots_per_iteration <= 0:
        raise SystemExit("positive collection schedule required")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()

    bundles = []
    collection = []
    for seed in seeds:
        bundle, report = collect_memory_shared_deck(
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
        )
        bundles.append(bundle)
        collection.append(report)

    support_a = support_observations(
        bundles[0].pol_mem.items,
        sample_size=args.support_sample_size,
        seed=0xA715EED,
    )
    support_b = support_observations(
        bundles[1].pol_mem.items,
        sample_size=args.support_sample_size,
        seed=0xB715EED,
    )
    union = support_a + support_b

    train_specs = (
        (0x12345, 0xABCDE),
        (0x6789A, 0xFEDCB),
    )
    models: dict[tuple[int, int], AveragePolicyNet] = {}
    policy_fits: dict[str, dict] = {}
    for memory_index, bundle in enumerate(bundles):
        for train_index, (init_seed, batch_seed) in enumerate(train_specs):
            model, report = fit_policy(
                bundle.pol_mem.items,
                config=bundle.config,
                init_seed=init_seed,
                batch_seed=batch_seed,
                device=args.device,
                lr=args.lr,
                chunk_steps=args.policy_chunk_steps,
                max_steps=args.policy_max_steps,
                fit_target=args.policy_fit_target,
                batch_size=args.batch_size,
                audit_size=args.audit_size,
                audit_seed=seeds[memory_index] ^ (train_index * 0x13579BDF),
            )
            models[(memory_index, train_index)] = model
            policy_fits[f"memory_{memory_index}_train_{train_index}"] = report

    by_support = {
        "support_A": comparisons_for_support(models, support_a, device=args.device),
        "support_B": comparisons_for_support(models, support_b, device=args.device),
        "union": comparisons_for_support(models, union, device=args.device),
    }

    own_support_within = 0.5 * (
        by_support["support_A"]["same_memory_A_different_optimizer_seed"]["mean_tv"]
        + by_support["support_B"]["same_memory_B_different_optimizer_seed"]["mean_tv"]
    )
    off_support_within = 0.5 * (
        by_support["support_B"]["same_memory_A_different_optimizer_seed"]["mean_tv"]
        + by_support["support_A"]["same_memory_B_different_optimizer_seed"]["mean_tv"]
    )
    across_shared_union = mean_pair(
        by_support["union"],
        "different_memory_same_optimizer_seed_0",
        "different_memory_same_optimizer_seed_1",
    )
    independent_baseline = float(args.independent_deck_baseline_mean_tv)
    shared_to_independent_ratio = across_shared_union / max(independent_baseline, 1e-12)
    off_to_own_ratio = off_support_within / max(own_support_within, 1e-12)

    all_policy_fit_pass = all(x["frozen_gate_pass"] for x in policy_fits.values())
    if not all_policy_fit_pass:
        chance_diagnosis = "POLICY_FIT_INSUFFICIENT_FOR_CHANCE_DECOMPOSITION"
    elif shared_to_independent_ratio <= 0.75:
        chance_diagnosis = "DECK_CHANCE_STREAM_MATERIAL_CONTRIBUTOR"
    elif shared_to_independent_ratio >= 0.90:
        chance_diagnosis = "DECK_CHANCE_STREAM_NOT_DOMINANT"
    else:
        chance_diagnosis = "DECK_CHANCE_STREAM_MIXED_CONTRIBUTOR"

    if off_to_own_ratio >= 1.5:
        support_diagnosis = "OFF_SUPPORT_POLICY_EXTRAPOLATION_MATERIAL"
    else:
        support_diagnosis = "OFF_SUPPORT_POLICY_EXTRAPOLATION_NOT_DOMINANT"

    payload = {
        "schema": "SPINCORE_R7_3_SHARED_DECK_SUPPORT_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "roots_per_seed": int(args.iterations * args.roots_per_iteration),
        "frozen_gates": dict(FROZEN_GATES),
        "acceptance_gate_changed": False,
        "collection": collection,
        "policy_fits": policy_fits,
        "support_observation_counts": {
            "support_A": len(support_a),
            "support_B": len(support_b),
            "union": len(union),
        },
        "comparisons_by_support": by_support,
        "reference_independent_deck_mean_tv": independent_baseline,
        "summary": {
            "all_policy_fit_gates_pass": bool(all_policy_fit_pass),
            "own_support_within_memory_mean_tv": float(own_support_within),
            "off_support_within_memory_mean_tv": float(off_support_within),
            "off_to_own_support_ratio": float(off_to_own_ratio),
            "shared_deck_across_memory_union_mean_tv": float(across_shared_union),
            "shared_to_independent_deck_ratio": float(shared_to_independent_ratio),
            "chance_diagnosis": chance_diagnosis,
            "support_diagnosis": support_diagnosis,
        },
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
