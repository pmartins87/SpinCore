from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from spincore.solver import SolverLibrary

from run_r7_3_partial_exact_advantage_screen import (
    DEFAULT_SEEDS,
    DEFAULT_SHARED_DECK_STREAM_SEED,
    _safe_ratio,
    run_mode as run_partial_mode,
)
from run_r7_3_path_replication_screen import run_mode as run_authoritative_mode


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "R7.3 partial-exact opponent Advantage screen V2. The level-0 baseline "
            "is executed through the authoritative recovered collector rather than "
            "through the experimental subclass."
        )
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_PARTIAL_EXACT_ADVANTAGE_SCREEN_256_V2.json"),
    )
    ap.add_argument(
        "--reference",
        type=Path,
        default=Path("validation/R7_3_PATH_REPLICATION_SCREEN_256.json"),
    )
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--levels", default="1,2")
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
        raise SystemExit("partial-exact screen requires exactly two algorithm seeds")
    levels = sorted({int(x.strip()) for x in str(args.levels).split(",") if x.strip()})
    if not levels or any(level <= 0 for level in levels):
        raise SystemExit("V2 experimental levels must be positive; baseline is authoritative")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()

    # Level 0 is deliberately not reimplemented. This exact call path is the
    # recovered baseline used by the already-certified path-replication screen.
    _baseline_bundles, baseline_report = run_authoritative_mode(
        mode_name="authoritative_level0",
        advantage_replicates=1,
        strategy_replicates=1,
        seeds=seeds,
        deck_stream_seed=int(args.deck_stream_seed),
        solver=solver,
        device=args.device,
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
    baseline_cross = baseline_report["cross_seed_internal_corpus"]
    baseline_mean = float(baseline_cross["mean_tv"])
    baseline_p95 = float(baseline_cross["p95_tv"])
    baseline_nodes = sum(int(row["nodes"]) for row in baseline_report["collection"])

    persisted_reference = None
    if args.reference.exists():
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        ref_mean = float(reference["summary"]["baseline_cross_seed_mean_tv"])
        ref_p95 = float(reference["summary"]["baseline_cross_seed_p95_tv"])
        persisted_reference = {
            "mean_tv": ref_mean,
            "p95_tv": ref_p95,
            "current_authoritative_mean_abs_delta": abs(baseline_mean - ref_mean),
            "current_authoritative_p95_abs_delta": abs(baseline_p95 - ref_p95),
            "note": (
                "Informational comparison only. The experimental screen is anchored to the "
                "authoritative level-0 run executed in the same process and dependency image."
            ),
        }

    modes: dict[str, dict] = {"0_authoritative": baseline_report}
    comparisons: dict[str, dict] = {}
    for level in levels:
        report = run_partial_mode(
            exact_opponent_levels=int(level),
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
        modes[str(level)] = report
        comparisons[str(level)] = {
            "mean_tv_ratio_to_authoritative_level0": _safe_ratio(
                float(report["cross_seed"]["mean_tv"]), baseline_mean
            ),
            "p95_tv_ratio_to_authoritative_level0": _safe_ratio(
                float(report["cross_seed"]["p95_tv"]), baseline_p95
            ),
            "node_ratio_to_authoritative_level0": _safe_ratio(
                float(report["total_nodes"]), float(baseline_nodes)
            ),
        }

    best_level = min(
        levels,
        key=lambda level: float(modes[str(level)]["cross_seed"]["mean_tv"]),
    )
    best = modes[str(best_level)]
    best_mean_ratio = float(
        comparisons[str(best_level)]["mean_tv_ratio_to_authoritative_level0"]
    )
    best_p95_ratio = float(
        comparisons[str(best_level)]["p95_tv_ratio_to_authoritative_level0"]
    )

    payload = {
        "schema": "SPINCORE_R7_3_PARTIAL_EXACT_ADVANTAGE_SCREEN_V2",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "experimental_exact_opponent_levels": levels,
        "authoritative_level0": baseline_report,
        "persisted_level0_reference": persisted_reference,
        "experimental_modes": {str(level): modes[str(level)] for level in levels},
        "comparisons_to_authoritative_level0": comparisons,
        "summary": {
            "authoritative_level0_mean_tv": baseline_mean,
            "authoritative_level0_p95_tv": baseline_p95,
            "authoritative_level0_nodes": int(baseline_nodes),
            "best_partial_exact_level_by_mean_tv": int(best_level),
            "best_partial_exact_mean_tv": float(best["cross_seed"]["mean_tv"]),
            "best_partial_exact_p95_tv": float(best["cross_seed"]["p95_tv"]),
            "best_partial_exact_mean_ratio_to_baseline": best_mean_ratio,
            "best_partial_exact_p95_ratio_to_baseline": best_p95_ratio,
            "diagnosis": (
                "PARTIAL_EXACT_OPPONENT_EXPECTATION_MATERIAL"
                if best_mean_ratio <= 0.85 or best_p95_ratio <= 0.85
                else "PARTIAL_EXACT_OPPONENT_EXPECTATION_NOT_MATERIAL_AT_SCREEN_SCALE"
            ),
        },
        "interpretation_note": (
            "V2 eliminates a diagnostic-design ambiguity: level 0 is the authoritative recovered "
            "ExternalSamplingCollector itself, not an experimental reimplementation. Positive "
            "levels enumerate the next N opponent decisions and probability-weight downstream "
            "Advantage samples before resuming ordinary external sampling. The experiment changes "
            "no frozen gate and does not promote the estimator to production."
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
