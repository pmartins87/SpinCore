from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import torch

from spincore.solver import SolverLibrary
from spincore_nn.codec import decode_spnniv1

from run_r7_3_shared_deck_support import (
    DEFAULT_SHARED_DECK_STREAM_SEED,
    collect_memory_shared_deck,
)


DEFAULT_SEEDS = (20260829, 20260807)
SUIT_PERMUTATIONS = tuple(itertools.permutations(range(4)))
CARD_OFFSET = 8
CARD_COUNT = 7


def _remap_card_token(token: int, suit_perm: tuple[int, int, int, int]) -> int:
    if token == 0:
        return 0
    card_id = int(token) - 1
    rank_index = card_id // 4
    suit = card_id % 4
    return rank_index * 4 + int(suit_perm[suit]) + 1


def _canonical_card_observation(observation: bytes, *, structural: bool) -> bytes:
    decoded = decode_spnniv1(observation)
    visible = int(decoded.categorical[7])
    original = list(observation[CARD_OFFSET : CARD_OFFSET + CARD_COUNT])
    best: bytes | None = None

    for suit_perm in SUIT_PERMUTATIONS:
        cards = [_remap_card_token(token, suit_perm) for token in original]
        if structural:
            # Hold'em private cards are unordered. The three flop cards are also
            # unordered within the flop. Turn and river retain street order.
            holes = sorted(cards[:2])
            board = cards[2 : 2 + visible]
            flop_count = min(3, visible)
            flop = sorted(board[:flop_count])
            later = board[flop_count:]
            cards = holes + flop + later + [0] * (5 - visible)

        candidate = (
            observation[:CARD_OFFSET]
            + bytes(cards)
            + observation[CARD_OFFSET + CARD_COUNT :]
        )
        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise RuntimeError("no suit permutation generated")
    return best


def _raw_key(observation: bytes) -> bytes:
    return observation


def _make_keyer(mode: str):
    cache: dict[bytes, bytes] = {}

    def keyer(observation: bytes) -> bytes:
        if observation in cache:
            return cache[observation]
        if mode == "raw":
            key = _raw_key(observation)
        elif mode == "suit_isomorphic":
            key = _canonical_card_observation(observation, structural=False)
        elif mode == "poker_isomorphic":
            key = _canonical_card_observation(observation, structural=True)
        else:
            raise ValueError(mode)
        cache[observation] = key
        return key

    return keyer


def _aggregate_strategy(items, keyer):
    # Each StrategySample target is already a legal probability distribution.
    # LCFR weight is sample.weight (iteration). Aggregate repeated infosets to
    # the weighted average-policy target for that canonical support key.
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
            }
            out[key] = row
        row["weight"] += weight
        row["samples"] += 1
        for action, probability in enumerate(sample.target):
            row["target_sum"][action] += weight * float(probability)

    for row in out.values():
        denom = max(float(row["weight"]), 1e-12)
        row["target"] = [value / denom for value in row["target_sum"]]
        del row["target_sum"]
    return out


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.inf
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    position = (len(xs) - 1) * float(q)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return xs[lo]
    frac = position - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _tv(a: list[float], b: list[float]) -> float:
    return 0.5 * sum(abs(float(x) - float(y)) for x, y in zip(a, b))


def _intersection_metrics(agg_a: dict[bytes, dict], agg_b: dict[bytes, dict]) -> dict:
    keys_a = set(agg_a)
    keys_b = set(agg_b)
    shared = keys_a & keys_b
    union = keys_a | keys_b

    total_weight_a = sum(float(row["weight"]) for row in agg_a.values())
    total_weight_b = sum(float(row["weight"]) for row in agg_b.values())
    shared_weight_a = sum(float(agg_a[key]["weight"]) for key in shared)
    shared_weight_b = sum(float(agg_b[key]["weight"]) for key in shared)

    tvs: list[float] = []
    pair_weights: list[float] = []
    for key in shared:
        tvs.append(_tv(agg_a[key]["target"], agg_b[key]["target"]))
        pair_weights.append(min(float(agg_a[key]["weight"]), float(agg_b[key]["weight"])))

    pair_weight_total = sum(pair_weights)
    weighted_mean_tv = (
        sum(value * weight for value, weight in zip(tvs, pair_weights)) / pair_weight_total
        if pair_weight_total > 0.0
        else math.inf
    )

    by_street: dict[str, dict[str, float | int]] = {}
    streets = sorted({int(decode_spnniv1(key).categorical[1]) for key in shared})
    for street in streets:
        street_keys = [key for key in shared if int(decode_spnniv1(key).categorical[1]) == street]
        street_tvs = [_tv(agg_a[key]["target"], agg_b[key]["target"]) for key in street_keys]
        street_weights = [
            min(float(agg_a[key]["weight"]), float(agg_b[key]["weight"]))
            for key in street_keys
        ]
        denom = sum(street_weights)
        by_street[str(street)] = {
            "shared_unique": len(street_keys),
            "weighted_target_mean_tv": (
                sum(v * w for v, w in zip(street_tvs, street_weights)) / denom
                if denom > 0.0
                else math.inf
            ),
            "target_p50_tv": _quantile(street_tvs, 0.50),
            "target_p95_tv": _quantile(street_tvs, 0.95),
        }

    return {
        "unique_A": len(keys_a),
        "unique_B": len(keys_b),
        "intersection_unique": len(shared),
        "union_unique": len(union),
        "jaccard": len(shared) / max(len(union), 1),
        "unique_coverage_A": len(shared) / max(len(keys_a), 1),
        "unique_coverage_B": len(shared) / max(len(keys_b), 1),
        "lcfr_weight_coverage_A": shared_weight_a / max(total_weight_a, 1e-12),
        "lcfr_weight_coverage_B": shared_weight_b / max(total_weight_b, 1e-12),
        "shared_target_weighted_mean_tv": float(weighted_mean_tv),
        "shared_target_unweighted_mean_tv": (
            sum(tvs) / len(tvs) if tvs else math.inf
        ),
        "shared_target_p50_tv": _quantile(tvs, 0.50),
        "shared_target_p95_tv": _quantile(tvs, 0.95),
        "by_street": by_street,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "R7.3 shared-deck support-overlap diagnostic: exact support, suit "
            "isomorphism, and poker card-order/suit isomorphism"
        )
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_SUPPORT_OVERLAP_640.json"),
    )
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_SHARED_DECK_STREAM_SEED)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-chunk-steps", type=int, default=256)
    ap.add_argument("--advantage-max-steps-per-iteration", type=int, default=2048)
    ap.add_argument("--advantage-fit-target", type=float, default=0.70)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=1024)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) != 2:
        raise SystemExit("support-overlap diagnostic requires exactly two seeds")

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

    modes = ("raw", "suit_isomorphic", "poker_isomorphic")
    overlap = {}
    for mode in modes:
        keyer = _make_keyer(mode)
        agg_a = _aggregate_strategy(bundles[0].pol_mem.items, keyer)
        agg_b = _aggregate_strategy(bundles[1].pol_mem.items, keyer)
        overlap[mode] = _intersection_metrics(agg_a, agg_b)
        print(json.dumps({"mode": mode, "metrics": overlap[mode]}, sort_keys=True), flush=True)

    raw = overlap["raw"]
    poker_iso = overlap["poker_isomorphic"]
    raw_jaccard = float(raw["jaccard"])
    iso_jaccard = float(poker_iso["jaccard"])
    jaccard_gain = iso_jaccard / max(raw_jaccard, 1e-12)
    raw_mass = 0.5 * (
        float(raw["lcfr_weight_coverage_A"]) + float(raw["lcfr_weight_coverage_B"])
    )
    iso_mass = 0.5 * (
        float(poker_iso["lcfr_weight_coverage_A"])
        + float(poker_iso["lcfr_weight_coverage_B"])
    )

    payload = {
        "schema": "SPINCORE_R7_3_SUPPORT_OVERLAP_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "roots_per_seed": int(args.iterations * args.roots_per_iteration),
        "acceptance_gate_changed": False,
        "collection": collection,
        "overlap": overlap,
        "summary": {
            "raw_jaccard": raw_jaccard,
            "poker_isomorphic_jaccard": iso_jaccard,
            "poker_isomorphic_to_raw_jaccard_ratio": float(jaccard_gain),
            "raw_mean_lcfr_weight_coverage": float(raw_mass),
            "poker_isomorphic_mean_lcfr_weight_coverage": float(iso_mass),
            "raw_shared_target_weighted_mean_tv": float(raw["shared_target_weighted_mean_tv"]),
            "poker_isomorphic_shared_target_weighted_mean_tv": float(
                poker_iso["shared_target_weighted_mean_tv"]
            ),
        },
        "interpretation_note": (
            "This is a diagnostic, not an acceptance gate. poker_isomorphic keys "
            "identify observations equivalent under global suit relabeling, private-card "
            "order, and flop-card order while preserving turn/river street order."
        ),
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
