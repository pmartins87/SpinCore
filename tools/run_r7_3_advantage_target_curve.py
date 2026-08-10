from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility, regret_matching_policy
from spincore.solver import SolverLibrary

from run_r7_3_diagnostic import hu_episode, make_bundle
from run_r7_3_support_overlap import _make_keyer, _quantile


DEFAULT_SEEDS = (20260829, 20260807)
DEFAULT_SHARED_DECK_STREAM_SEED = 0xD3C5EED
PAYOUT = (0.5, 0.3, 0.2)
ADV_RNG_XOR = 0x0AD7A61


def _aggregate_advantage(items, keyer):
    out: dict[bytes, dict] = {}
    for sample in items:
        key = keyer(sample.observation)
        weight = float(sample.weight)
        row = out.get(key)
        if row is None:
            row = {
                "target_sum": [0.0] * len(sample.target),
                "weight": 0.0,
                "samples": 0,
                "legal": tuple(bool(x) for x in sample.legal),
            }
            out[key] = row
        elif row["legal"] != tuple(bool(x) for x in sample.legal):
            raise RuntimeError("same canonical observation produced inconsistent legal masks")
        row["weight"] += weight
        row["samples"] += 1
        for action, value in enumerate(sample.target):
            row["target_sum"][action] += weight * float(value)

    for row in out.values():
        denom = max(float(row["weight"]), 1e-12)
        row["target"] = [value / denom for value in row["target_sum"]]
        del row["target_sum"]
    return out


def _advantage_intersection_metrics(agg_a, agg_b):
    keys_a = set(agg_a)
    keys_b = set(agg_b)
    shared = keys_a & keys_b
    union = keys_a | keys_b

    total_weight_a = sum(float(row["weight"]) for row in agg_a.values())
    total_weight_b = sum(float(row["weight"]) for row in agg_b.values())
    shared_weight_a = sum(float(agg_a[key]["weight"]) for key in shared)
    shared_weight_b = sum(float(agg_b[key]["weight"]) for key in shared)

    pair_weights = []
    policy_tvs = []
    squared_error = 0.0
    target_energy = 0.0
    legal_coordinates = 0
    greedy_agreements = []

    for key in shared:
        a = agg_a[key]
        b = agg_b[key]
        if a["legal"] != b["legal"]:
            raise RuntimeError("shared key has inconsistent legal masks across memories")
        legal = tuple(i for i, yes in enumerate(a["legal"]) if yes)
        if not legal:
            raise RuntimeError("shared advantage sample has empty legal set")
        weight = min(float(a["weight"]), float(b["weight"]))
        pair_weights.append(weight)
        ta = a["target"]
        tb = b["target"]
        for action in legal:
            da = float(ta[action])
            db = float(tb[action])
            squared_error += weight * (da - db) ** 2
            target_energy += weight * 0.5 * (da * da + db * db)
            legal_coordinates += 1
        pa = regret_matching_policy(ta, legal)
        pb = regret_matching_policy(tb, legal)
        policy_tvs.append(0.5 * sum(abs(x - y) for x, y in zip(pa, pb)))
        greedy_a = max(legal, key=lambda action: float(ta[action]))
        greedy_b = max(legal, key=lambda action: float(tb[action]))
        greedy_agreements.append(1.0 if greedy_a == greedy_b else 0.0)

    pair_weight_total = sum(pair_weights)
    weighted_policy_tv = (
        sum(v * w for v, w in zip(policy_tvs, pair_weights)) / pair_weight_total
        if pair_weight_total > 0.0
        else math.inf
    )
    weighted_greedy_agreement = (
        sum(v * w for v, w in zip(greedy_agreements, pair_weights)) / pair_weight_total
        if pair_weight_total > 0.0
        else 0.0
    )
    relative_rmse = (
        math.sqrt(squared_error / max(target_energy, 1e-12))
        if pair_weight_total > 0.0
        else math.inf
    )

    return {
        "unique_A": len(keys_a),
        "unique_B": len(keys_b),
        "intersection_unique": len(shared),
        "union_unique": len(union),
        "jaccard": len(shared) / max(len(union), 1),
        "weight_coverage_A": shared_weight_a / max(total_weight_a, 1e-12),
        "weight_coverage_B": shared_weight_b / max(total_weight_b, 1e-12),
        "shared_target_relative_rmse": float(relative_rmse),
        "shared_regret_matching_weighted_mean_tv": float(weighted_policy_tv),
        "shared_regret_matching_p50_tv": _quantile(policy_tvs, 0.50),
        "shared_regret_matching_p95_tv": _quantile(policy_tvs, 0.95),
        "shared_weighted_greedy_action_agreement": float(weighted_greedy_agreement),
        "shared_legal_coordinate_count": int(legal_coordinates),
    }


def collect_uniform_advantage_memory(
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
    if session.behavior.ready:
        raise RuntimeError("uniform bootstrap unexpectedly marked ready")
    session.collector.rng = random.Random(int(algorithm_seed) ^ ADV_RNG_XOR)

    episode = hu_episode()
    live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
    nodes = 0
    samples_added = 0
    for root_index in range(int(unique_roots)):
        deck_seed = (
            int(deck_stream_seed) * 1_000_003 + root_index * 97 + 1
        ) & ((1 << 64) - 1)
        for player in live:
            for _ in range(int(replicates)):
                root = solver.create(episode, int(deck_seed))
                try:
                    result = session.collector.collect_advantage(
                        root,
                        traverser=player,
                        iteration=1,
                    )
                finally:
                    root.close()
                nodes += int(result.nodes)
                samples_added += int(result.samples_added)

    return bundle.adv_mem, {
        "algorithm_seed": int(algorithm_seed),
        "replicates_per_traverser_per_unique_root": int(replicates),
        "unique_roots": int(unique_roots),
        "advantage_samples": len(bundle.adv_mem.items),
        "advantage_seen": int(bundle.adv_mem.seen),
        "samples_added": int(samples_added),
        "nodes": int(nodes),
        "uniform_zero_regret_policy": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "R7.3 exact control for external-sampling Advantage target variance "
            "under the identical zero-regret uniform behavior policy"
        )
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_ADVANTAGE_TARGET_CURVE_256.json"),
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
        raise SystemExit("advantage target curve requires exactly two algorithm seeds")
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
            memory, report = collect_uniform_advantage_memory(
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

        overlap = {}
        for mode in ("raw", "poker_isomorphic"):
            keyer = _make_keyer(mode)
            overlap[mode] = _advantage_intersection_metrics(
                _aggregate_advantage(memories[0].items, keyer),
                _aggregate_advantage(memories[1].items, keyer),
            )
        poker = overlap["poker_isomorphic"]
        rows.append(
            {
                "replicates": int(replicate_count),
                "collection": collection,
                "overlap": overlap,
                "summary": {
                    "poker_isomorphic_jaccard": float(poker["jaccard"]),
                    "mean_weight_coverage": 0.5
                    * (float(poker["weight_coverage_A"]) + float(poker["weight_coverage_B"])),
                    "shared_target_relative_rmse": float(poker["shared_target_relative_rmse"]),
                    "shared_regret_matching_weighted_mean_tv": float(
                        poker["shared_regret_matching_weighted_mean_tv"]
                    ),
                    "shared_regret_matching_p95_tv": float(
                        poker["shared_regret_matching_p95_tv"]
                    ),
                    "shared_weighted_greedy_action_agreement": float(
                        poker["shared_weighted_greedy_action_agreement"]
                    ),
                },
            }
        )
        print(json.dumps(rows[-1]["summary"], sort_keys=True), flush=True)

    first = rows[0]["summary"]
    last = rows[-1]["summary"]
    tv_ratio = float(last["shared_regret_matching_weighted_mean_tv"]) / max(
        float(first["shared_regret_matching_weighted_mean_tv"]), 1e-12
    )
    rmse_ratio = float(last["shared_target_relative_rmse"]) / max(
        float(first["shared_target_relative_rmse"]), 1e-12
    )
    coverage_ratio = float(last["mean_weight_coverage"]) / max(
        float(first["mean_weight_coverage"]), 1e-12
    )

    payload = {
        "schema": "SPINCORE_R7_3_ADVANTAGE_TARGET_CURVE_V1",
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
            "last_to_first_regret_matching_tv_ratio": float(tv_ratio),
            "last_to_first_target_relative_rmse_ratio": float(rmse_ratio),
            "last_to_first_weight_coverage_ratio": float(coverage_ratio),
            "diagnosis": (
                "EXTERNAL_SAMPLING_ADVANTAGE_TARGET_VARIANCE_MATERIAL"
                if tv_ratio <= 0.75 or rmse_ratio <= 0.75 or coverage_ratio >= 1.25
                else "EXTERNAL_SAMPLING_ADVANTAGE_TARGET_VARIANCE_NOT_MATERIAL"
            ),
        },
        "interpretation_note": (
            "Exact bootstrap control: both algorithm seeds see the same unique deck stream and "
            "the exact uniform zero-regret policy; no neural fitting occurs. Any disagreement "
            "between aggregated Advantage targets on shared infosets therefore comes from "
            "external-sampling opponent-path variance. The regret-matching TV converts target "
            "noise into the policy-space quantity most relevant to subsequent CFR behavior. "
            "This diagnostic changes no production schedule or acceptance gate."
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
