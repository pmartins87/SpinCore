from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.solver import SolverLibrary

from run_r7_3_diagnostic import hu_episode, make_bundle
from run_r7_3_support_overlap import _aggregate_strategy, _intersection_metrics, _make_keyer


DEFAULT_SEEDS = (20260829, 20260807)
DEFAULT_SHARED_DECK_STREAM_SEED = 0xD3C5EED
PAYOUT = (0.5, 0.3, 0.2)
STRATEGY_RNG_XOR = 0x0A11CE55


def _metrics(items_a, items_b):
    out = {}
    for mode in ("raw", "poker_isomorphic"):
        keyer = _make_keyer(mode)
        out[mode] = _intersection_metrics(
            _aggregate_strategy(items_a, keyer),
            _aggregate_strategy(items_b, keyer),
        )
    return out


def collect_uniform_strategy_memory(
    *,
    algorithm_seed: int,
    replicates: int,
    unique_roots: int,
    deck_stream_seed: int,
    solver: SolverLibrary,
    reservoir_capacity: int,
    device: str,
):
    bundle = make_bundle(
        int(algorithm_seed),
        device=device,
        reservoir_capacity=int(reservoir_capacity),
        lr=1e-3,
    )
    session = DeepCFRDomainSession(
        solver_library=solver,
        bundle=bundle,
        terminal_utility=icm_delta_utility(PAYOUT),
        device=device,
    )
    # No Advantage fitting is performed. DeepCFRDomainSession therefore keeps
    # behavior.ready=False, which is the exact zero-regret uniform bootstrap.
    if session.behavior.ready:
        raise RuntimeError("uniform bootstrap unexpectedly marked ready")
    session.collector.rng = random.Random(int(algorithm_seed) ^ STRATEGY_RNG_XOR)

    episode = hu_episode()
    live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
    samples_added = 0
    for root_index in range(int(unique_roots)):
        deck_seed = (
            int(deck_stream_seed) * 1_000_003 + root_index * 97 + 1
        ) & ((1 << 64) - 1)
        for player in live:
            for _ in range(int(replicates)):
                root = solver.create(episode, int(deck_seed))
                try:
                    samples_added += int(
                        session.collector.collect_strategy_own_reach(
                            root,
                            target_player=player,
                            iteration=1,
                        )
                    )
                finally:
                    root.close()

    return bundle.pol_mem, {
        "algorithm_seed": int(algorithm_seed),
        "replicates_per_target_player_per_unique_root": int(replicates),
        "unique_roots": int(unique_roots),
        "strategy_samples": len(bundle.pol_mem.items),
        "strategy_seen": int(bundle.pol_mem.seen),
        "samples_added": int(samples_added),
        "uniform_zero_regret_policy": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "R7.3 exact control for own-reach AveragePolicy support fragmentation "
            "under the identical zero-regret uniform behavior policy"
        )
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_OWN_REACH_SUPPORT_CURVE_256.json"),
    )
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--replicates", default="1,2,4,8")
    ap.add_argument("--unique-roots", type=int, default=256)
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_SHARED_DECK_STREAM_SEED)
    ap.add_argument("--reservoir-capacity", type=int, default=250000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) != 2:
        raise SystemExit("own-reach support curve requires exactly two algorithm seeds")
    reps = [int(x.strip()) for x in str(args.replicates).split(",") if x.strip()]
    if not reps or any(x <= 0 for x in reps):
        raise SystemExit("replicate counts must be positive")
    reps = sorted(set(reps))

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()

    rows = []
    for replicate_count in reps:
        memories = []
        collection = []
        for seed in seeds:
            memory, report = collect_uniform_strategy_memory(
                algorithm_seed=int(seed),
                replicates=int(replicate_count),
                unique_roots=int(args.unique_roots),
                deck_stream_seed=int(args.deck_stream_seed),
                solver=solver,
                reservoir_capacity=int(args.reservoir_capacity),
                device=args.device,
            )
            memories.append(memory)
            collection.append(report)
        metrics = _metrics(memories[0].items, memories[1].items)
        poker = metrics["poker_isomorphic"]
        target_tv = float(poker["shared_target_weighted_mean_tv"])
        if target_tv > 1e-12:
            raise RuntimeError(
                "uniform-policy shared targets must be identical; nonzero TV indicates a control failure"
            )
        rows.append(
            {
                "replicates": int(replicate_count),
                "collection": collection,
                "overlap": metrics,
                "summary": {
                    "poker_isomorphic_jaccard": float(poker["jaccard"]),
                    "mean_lcfr_weight_coverage": 0.5
                    * (
                        float(poker["lcfr_weight_coverage_A"])
                        + float(poker["lcfr_weight_coverage_B"])
                    ),
                    "shared_target_weighted_mean_tv": target_tv,
                    "shared_unique": int(poker["intersection_unique"]),
                    "union_unique": int(poker["union_unique"]),
                },
            }
        )
        print(json.dumps(rows[-1]["summary"], sort_keys=True), flush=True)

    base = rows[0]["summary"]
    last = rows[-1]["summary"]
    j_ratio = float(last["poker_isomorphic_jaccard"]) / max(
        float(base["poker_isomorphic_jaccard"]), 1e-12
    )
    coverage_ratio = float(last["mean_lcfr_weight_coverage"]) / max(
        float(base["mean_lcfr_weight_coverage"]), 1e-12
    )

    payload = {
        "schema": "SPINCORE_R7_3_OWN_REACH_SUPPORT_CURVE_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "unique_roots_per_seed": int(args.unique_roots),
        "replicate_counts": reps,
        "rows": rows,
        "summary": {
            "first_replicates": int(rows[0]["replicates"]),
            "last_replicates": int(rows[-1]["replicates"]),
            "last_to_first_jaccard_ratio": float(j_ratio),
            "last_to_first_weight_coverage_ratio": float(coverage_ratio),
            "diagnosis": (
                "OWN_REACH_SAMPLING_DENSITY_MATERIAL_FOR_SUPPORT"
                if j_ratio >= 1.25 or coverage_ratio >= 1.25
                else "OWN_REACH_SAMPLING_DENSITY_NOT_MATERIAL_FOR_SUPPORT"
            ),
        },
        "interpretation_note": (
            "Exact control: every run uses the same unique root-deck stream and the exact "
            "zero-regret uniform behavior policy. The only stochastic difference between the "
            "two algorithm seeds is own-reach action sampling. Shared strategy targets must "
            "therefore have TV exactly zero; only support coverage is being measured. This "
            "diagnostic does not modify production collection or acceptance gates."
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
