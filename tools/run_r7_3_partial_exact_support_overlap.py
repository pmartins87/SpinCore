from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from run_r7_3_partial_exact_640_candidate import run_seed
from run_r7_3_support_overlap import _aggregate_strategy, _make_keyer, _tv, _quantile
from spincore.solver import SolverLibrary
from spincore_nn.codec import decode_spnniv1


DEFAULT_SEEDS = (20260829, 20260807)


def _intersection_detail(agg_a, agg_b):
    shared = set(agg_a) & set(agg_b)
    rows = []
    for key in shared:
        a = agg_a[key]
        b = agg_b[key]
        decoded = decode_spnniv1(key)
        tv = _tv(a["target"], b["target"])
        pair_weight = min(float(a["weight"]), float(b["weight"]))
        rows.append(
            {
                "tv": float(tv),
                "weight": float(pair_weight),
                "street": int(decoded.categorical[1]),
                "legal_count": int(sum(int(x) for x in decoded.legal)),
                "history_len": int(decoded.history_len),
            }
        )
    return rows


def _weighted_mean(rows):
    denom = sum(float(x["weight"]) for x in rows)
    if denom <= 0.0:
        return math.inf
    return sum(float(x["tv"]) * float(x["weight"]) for x in rows) / denom


def _group(rows, field):
    out = {}
    values = sorted({int(x[field]) for x in rows})
    for value in values:
        subset = [x for x in rows if int(x[field]) == value]
        tvs = [float(x["tv"]) for x in subset]
        out[str(value)] = {
            "shared_unique": len(subset),
            "weighted_target_mean_tv": float(_weighted_mean(subset)),
            "target_unweighted_mean_tv": float(sum(tvs) / max(len(tvs), 1)),
            "target_p50_tv": float(_quantile(tvs, 0.50)),
            "target_p95_tv": float(_quantile(tvs, 0.95)),
            "pair_weight": float(sum(float(x["weight"]) for x in subset)),
        }
    return out


def _mode_metrics(items_a, items_b, mode):
    keyer = _make_keyer(mode)
    agg_a = _aggregate_strategy(items_a, keyer)
    agg_b = _aggregate_strategy(items_b, keyer)
    keys_a = set(agg_a)
    keys_b = set(agg_b)
    shared = keys_a & keys_b
    union = keys_a | keys_b
    total_weight_a = sum(float(x["weight"]) for x in agg_a.values())
    total_weight_b = sum(float(x["weight"]) for x in agg_b.values())
    shared_weight_a = sum(float(agg_a[k]["weight"]) for k in shared)
    shared_weight_b = sum(float(agg_b[k]["weight"]) for k in shared)
    rows = _intersection_detail(agg_a, agg_b)
    tvs = [float(x["tv"]) for x in rows]
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
        "shared_target_weighted_mean_tv": float(_weighted_mean(rows)),
        "shared_target_unweighted_mean_tv": float(sum(tvs) / max(len(tvs), 1)) if tvs else math.inf,
        "shared_target_p50_tv": float(_quantile(tvs, 0.50)),
        "shared_target_p95_tv": float(_quantile(tvs, 0.95)),
        "shared_target_max_tv": max(tvs) if tvs else math.inf,
        "by_street": _group(rows, "street"),
        "by_legal_action_count": _group(rows, "legal_count"),
        "by_history_len": _group(rows, "history_len"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Authoritative partial-exact strategy-memory support/target overlap forensic")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_PARTIAL_EXACT_SUPPORT_OVERLAP_256.json"))
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--exact-opponent-levels", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-chunk-steps", type=int, default=256)
    ap.add_argument("--advantage-max-steps-per-iteration", type=int, default=4096)
    ap.add_argument("--advantage-fit-target", type=float, default=0.50)
    ap.add_argument("--policy-chunk-steps", type=int, default=128)
    ap.add_argument("--policy-max-steps", type=int, default=128)
    ap.add_argument("--policy-fit-target", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=512)
    ap.add_argument("--cross-seed-per-seed", type=int, default=128)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) != 2:
        raise SystemExit("requires exactly two seeds")
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()
    bundles = []
    reports = []
    for seed in seeds:
        bundle, report = run_seed(seed=int(seed), solver=solver, args=args)
        bundles.append(bundle)
        reports.append(report)

    overlap = {
        mode: _mode_metrics(bundles[0].pol_mem.items, bundles[1].pol_mem.items, mode)
        for mode in ("raw", "suit_isomorphic", "poker_isomorphic")
    }
    raw = overlap["raw"]
    poker = overlap["poker_isomorphic"]
    payload = {
        "schema": "SPINCORE_R7_3_PARTIAL_EXACT_SUPPORT_OVERLAP_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "exact_opponent_levels": int(args.exact_opponent_levels),
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "roots_per_seed": int(args.iterations * args.roots_per_iteration),
        "deck_semantics": "GENERATION2_AUTHORITATIVE_GLOBAL_ROOT_FORMULA_EXACT",
        "deck_formula": "seed*1000003 + global_root*97 + iteration",
        "rng_contract": "RECOVERED_SINGLE_COUPLED_BATCH_RNG",
        "collection": reports,
        "overlap": overlap,
        "summary": {
            "raw_jaccard": float(raw["jaccard"]),
            "raw_mean_lcfr_weight_coverage": float(0.5 * (raw["lcfr_weight_coverage_A"] + raw["lcfr_weight_coverage_B"])),
            "raw_shared_target_weighted_mean_tv": float(raw["shared_target_weighted_mean_tv"]),
            "raw_shared_target_p95_tv": float(raw["shared_target_p95_tv"]),
            "poker_isomorphic_jaccard": float(poker["jaccard"]),
            "poker_isomorphic_mean_lcfr_weight_coverage": float(0.5 * (poker["lcfr_weight_coverage_A"] + poker["lcfr_weight_coverage_B"])),
            "poker_isomorphic_shared_target_weighted_mean_tv": float(poker["shared_target_weighted_mean_tv"]),
            "poker_isomorphic_shared_target_p95_tv": float(poker["shared_target_p95_tv"]),
        },
        "interpretation_note": (
            "Diagnostic only. This measures strategy-memory support and target disagreement after the "
            "authoritative paired partial-exact level-2 CFR collection. It answers whether bounded "
            "opponent enumeration actually stabilizes sigma targets on genuinely shared infosets, "
            "and decomposes the remaining tail by street, legal-action count and history length."
        ),
        "acceptance_gate_changed": False,
        "production_estimator_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
