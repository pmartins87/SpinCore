#!/usr/bin/env python3
from __future__ import annotations

"""Run the existing fixed-seed weak-baseline evaluator and also preserve row-level evidence.

This intentionally reuses the exact sampler/play/summarization helpers from
`evaluate_lean_strategy_quality.py`; the only added output is the deterministic
row table needed for a paired checkpoint-vs-checkpoint CI.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import evaluate_lean_strategy_quality as base  # noqa: E402
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler  # noqa: E402


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenarios", type=int, default=1000)
    p.add_argument("--workers", type=int, default=31)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--rows-report", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = _args()
    if args.scenarios <= 0 or args.workers <= 0:
        raise SystemExit("scenarios/workers must be positive")
    if not args.solver.is_file():
        raise SystemExit(f"solver not found: {args.solver}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    sampler = LegacyScenarioSampler(
        seed=args.seed ^ 0x5CE0A710,
        config=LegacyScenarioConfig(),
    )
    tasks = []
    domain_counts = {"THREE_HANDED": 0, "TRUE_HEADS_UP": 0}
    blind_counts: dict[str, int] = {}
    for index in range(args.scenarios):
        episode = sampler.sample_episode()
        domain = "TRUE_HEADS_UP" if episode.game_is_hu else "THREE_HANDED"
        domain_counts[domain] += 1
        blind = f"{episode.small_blind}/{episode.big_blind}"
        key = f"{domain}:{blind}"
        blind_counts[key] = blind_counts.get(key, 0) + 1
        deal_seed = base._mix64(args.seed, index, 0xD34A1)
        tasks.append((index, episode, deal_seed, int(args.seed)))

    rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(args.workers, args.scenarios),
        mp_context=ctx,
        initializer=base._init_worker,
        initargs=(str(args.solver.resolve()), str(args.checkpoint.resolve())),
    ) as pool:
        for chunk in pool.map(base._worker, tasks, chunksize=1):
            rows.extend(chunk)

    blocks = base._summarize(rows)
    report = {
        "schema": "SPINCORE_LEAN_STRATEGY_QUALITY_EVAL_V1",
        "checkpoint": str(args.checkpoint.resolve()),
        "scenarios": int(args.scenarios),
        "workers": int(min(args.workers, args.scenarios)),
        "seed": int(args.seed),
        "opponent_families": list(base.BASELINES),
        "hero_control": "UNIFORM_CONTROL",
        "domain_counts": domain_counts,
        "blind_counts": dict(sorted(blind_counts.items())),
        "method": {
            "scenario_sampler": "legacy empirical full 3H/HU blind-conditioned sampler",
            "deal_pairing": "same scenario and solver deal seed for SpinCore and uniform hero control",
            "seat_rotation": "evaluated hero rotated through every live seat",
            "primary_metric": "raw chip delta per hand; paired SpinCore-minus-uniform-control gain",
            "ci": "normal 95% CI over scenario-cluster means",
            "warning": "diagnostic versus fixed weak baselines; not an exploitability or GTO proof",
        },
        **blocks,
    }
    rows_report = {
        "schema": "SPINCORE_LEAN_STRATEGY_QUALITY_ROWS_V1",
        "checkpoint": str(args.checkpoint.resolve()),
        "scenarios": int(args.scenarios),
        "seed": int(args.seed),
        "rows": rows,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.rows_report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.rows_report.write_text(json.dumps(rows_report, separators=(",", ":")) + "\n", encoding="utf-8")
    base._print_summary(report)
    print(f"rows_report={args.rows_report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
