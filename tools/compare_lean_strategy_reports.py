#!/usr/bin/env python3
from __future__ import annotations

"""Summarize a fixed-seed SpinCore learning-curve evaluation across checkpoints."""

import argparse
import json
from pathlib import Path

BASELINES = ("UNIFORM_LEGAL", "PASSIVE_CALLER", "JAMMER")
DOMAINS = ("ALL", "THREE_HANDED", "TRUE_HEADS_UP")


def load(path: Path):
    data = json.loads(path.read_text())
    if data.get("schema") != "SPINCORE_LEAN_STRATEGY_QUALITY_EVAL_V1":
        raise SystemExit(f"wrong report schema: {path}")
    return data


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lt0", type=Path, required=True)
    p.add_argument("--lt1", type=Path, required=True)
    p.add_argument("--lt2a", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    reports = {"LT0_120K": load(args.lt0), "LT1_1P2M": load(args.lt1), "LT2A_1P8M": load(args.lt2a)}
    seeds = {int(r["seed"]) for r in reports.values()}
    scenarios = {int(r["scenarios"]) for r in reports.values()}
    if len(seeds) != 1 or len(scenarios) != 1:
        raise SystemExit("reports do not share the same seed/scenario count")

    rows = []
    for baseline in BASELINES:
        for domain in DOMAINS:
            row = {"opponent": baseline, "domain": domain, "checkpoints": {}}
            for label, report in reports.items():
                block = report["summary"][baseline][domain]
                cur = block["spincore_chip_ev"]
                gain = block["paired_gain_chip_ev"]
                row["checkpoints"][label] = {
                    "spincore_mean": cur["mean"],
                    "spincore_ci95_low": cur["ci95_low"],
                    "spincore_ci95_high": cur["ci95_high"],
                    "paired_gain_mean": gain["mean"],
                    "paired_gain_ci95_low": gain["ci95_low"],
                    "paired_gain_ci95_high": gain["ci95_high"],
                }
            row["delta_lt1_minus_lt0_spincore"] = row["checkpoints"]["LT1_1P2M"]["spincore_mean"] - row["checkpoints"]["LT0_120K"]["spincore_mean"]
            row["delta_lt2a_minus_lt1_spincore"] = row["checkpoints"]["LT2A_1P8M"]["spincore_mean"] - row["checkpoints"]["LT1_1P2M"]["spincore_mean"]
            row["delta_lt2a_minus_lt0_spincore"] = row["checkpoints"]["LT2A_1P8M"]["spincore_mean"] - row["checkpoints"]["LT0_120K"]["spincore_mean"]
            rows.append(row)

    result = {
        "schema": "SPINCORE_LEARNING_CURVE_WEAK_BASELINES_V1",
        "seed": seeds.pop(),
        "scenarios": scenarios.pop(),
        "note": "Checkpoint deltas compare fixed-seed report means. They are descriptive, not a paired CI for checkpoint-vs-checkpoint difference.",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    print("=== SpinCore weak-baseline learning curve ===")
    print(f"seed={result['seed']} scenarios={result['scenarios']}")
    for row in rows:
        if row["domain"] not in ("ALL", "TRUE_HEADS_UP"):
            continue
        print(f"--- {row['opponent']} / {row['domain']} ---")
        for label in ("LT0_120K", "LT1_1P2M", "LT2A_1P8M"):
            x = row["checkpoints"][label]
            print(f"{label}: cEV={x['spincore_mean']:+.3f} [{x['spincore_ci95_low']:+.3f},{x['spincore_ci95_high']:+.3f}] paired_vs_uniform={x['paired_gain_mean']:+.3f} [{x['paired_gain_ci95_low']:+.3f},{x['paired_gain_ci95_high']:+.3f}]")
        print(f"delta cEV LT1-LT0={row['delta_lt1_minus_lt0_spincore']:+.3f} LT2A-LT1={row['delta_lt2a_minus_lt1_spincore']:+.3f} LT2A-LT0={row['delta_lt2a_minus_lt0_spincore']:+.3f}")
    print(f"comparison_report={args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
