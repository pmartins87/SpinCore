from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from run_r7_3_partial_exact_640_candidate import run_seed
from run_r7_3_partial_exact_support_overlap import _mode_metrics
from spincore.solver import SolverLibrary


DEFAULT_SEEDS = (20260829, 20260807)


def _filter_iteration(items, iteration: int):
    return [x for x in items if int(x.iteration) == int(iteration)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Authoritative partial-exact strategy support/target forensic split by CFR iteration")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_PARTIAL_EXACT_SUPPORT_BY_ITERATION_256.json"))
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--exact-opponent-levels", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-chunk-steps", type=int, default=256)
    ap.add_argument("--advantage-max-steps-per-iteration", type=int, default=4096)
    ap.add_argument("--advantage-fit-target", type=float, default=0.50)
    ap.add_argument("--policy-chunk-steps", type=int, default=8)
    ap.add_argument("--policy-max-steps", type=int, default=8)
    ap.add_argument("--policy-fit-target", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=512)
    ap.add_argument("--cross-seed-per-seed", type=int, default=16)
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

    modes = ("raw", "suit_isomorphic", "poker_isomorphic")
    pooled = {
        mode: _mode_metrics(bundles[0].pol_mem.items, bundles[1].pol_mem.items, mode)
        for mode in modes
    }
    by_iteration = {}
    for iteration in range(1, int(args.iterations) + 1):
        a = _filter_iteration(bundles[0].pol_mem.items, iteration)
        b = _filter_iteration(bundles[1].pol_mem.items, iteration)
        by_iteration[str(iteration)] = {
            "items_A": len(a),
            "items_B": len(b),
            "modes": {mode: _mode_metrics(a, b, mode) for mode in modes},
        }

    p1 = by_iteration.get("1", {}).get("modes", {}).get("poker_isomorphic", {})
    p2 = by_iteration.get("2", {}).get("modes", {}).get("poker_isomorphic", {})
    payload = {
        "schema": "SPINCORE_R7_3_PARTIAL_EXACT_SUPPORT_BY_ITERATION_V1",
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
        "pooled": pooled,
        "by_iteration": by_iteration,
        "summary": {
            "iteration1_poker_jaccard": p1.get("jaccard"),
            "iteration1_poker_shared_target_weighted_mean_tv": p1.get("shared_target_weighted_mean_tv"),
            "iteration1_poker_shared_target_p95_tv": p1.get("shared_target_p95_tv"),
            "iteration2_poker_jaccard": p2.get("jaccard"),
            "iteration2_poker_shared_target_weighted_mean_tv": p2.get("shared_target_weighted_mean_tv"),
            "iteration2_poker_shared_target_p95_tv": p2.get("shared_target_p95_tv"),
            "diagnosis": (
                "TARGET_DIVERGENCE_APPEARS_AFTER_FIRST_ADVANTAGE_FIT"
                if p1 and p2
                and float(p1.get("shared_target_weighted_mean_tv", 1.0)) <= 1e-9
                and float(p2.get("shared_target_weighted_mean_tv", 0.0)) >= 0.15
                else "ITERATION_SPLIT_REQUIRES_DETAILED_INTERPRETATION"
            ),
        },
        "interpretation_note": (
            "Diagnostic only. Iteration 1 starts from exact zero-regret uniform behavior in both seeds, "
            "so genuinely shared infosets should have identical strategy targets. Iteration 2 follows "
            "the first fitted Advantage behavior. Splitting the same authoritative partial-exact memory "
            "by sample.iteration distinguishes intrinsic strategy-target noise from feedback divergence "
            "created by the first Advantage fit and regret mapping."
        ),
        "acceptance_gate_changed": False,
        "production_algorithm_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
