#!/usr/bin/env python3
from __future__ import annotations

"""Paired checkpoint-vs-checkpoint chip-EV delta using identical row keys."""

import argparse
import json
import math
from pathlib import Path
import statistics

BASELINES = ("UNIFORM_LEGAL", "PASSIVE_CALLER", "JAMMER")
DOMAINS = ("ALL", "THREE_HANDED", "TRUE_HEADS_UP")


def load_rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "SPINCORE_LEAN_STRATEGY_QUALITY_ROWS_V1":
        raise SystemExit(f"wrong rows schema: {path}")
    return data


def key(row):
    return (
        int(row["scenario"]),
        str(row["domain"]),
        str(row["blind"]),
        int(row["hero_seat"]),
        str(row["baseline"]),
    )


def mean_ci(values):
    vals = [float(x) for x in values]
    n = len(vals)
    if not vals:
        return {"n": 0, "mean": float("nan"), "sem": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(statistics.fmean(vals))
    sem = 0.0 if n == 1 else float(statistics.stdev(vals) / math.sqrt(n))
    half = 1.96 * sem
    return {"n": n, "mean": mean, "sem": sem, "ci95_low": mean - half, "ci95_high": mean + half}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--before-label", default="LT2A_1P8M")
    p.add_argument("--after-label", default="LT2B_4P5M")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    before = load_rows(args.before)
    after = load_rows(args.after)
    if int(before["seed"]) != int(after["seed"]) or int(before["scenarios"]) != int(after["scenarios"]):
        raise SystemExit("checkpoint row sets use different seed/scenario count")

    a = {key(row): row for row in before["rows"]}
    b = {key(row): row for row in after["rows"]}
    if set(a) != set(b):
        only_a = next(iter(set(a) - set(b)), None)
        only_b = next(iter(set(b) - set(a)), None)
        raise SystemExit(f"row-key mismatch only_before={only_a} only_after={only_b}")

    rows = []
    for baseline in BASELINES:
        for domain in DOMAINS:
            by_scenario = {}
            before_by_scenario = {}
            after_by_scenario = {}
            for k in sorted(a):
                scenario, row_domain, _blind, _seat, row_baseline = k
                if row_baseline != baseline:
                    continue
                if domain != "ALL" and row_domain != domain:
                    continue
                av = float(a[k]["spincore"])
                bv = float(b[k]["spincore"])
                by_scenario.setdefault(scenario, []).append(bv - av)
                before_by_scenario.setdefault(scenario, []).append(av)
                after_by_scenario.setdefault(scenario, []).append(bv)
            delta_clusters = [statistics.fmean(v) for _, v in sorted(by_scenario.items())]
            before_clusters = [statistics.fmean(v) for _, v in sorted(before_by_scenario.items())]
            after_clusters = [statistics.fmean(v) for _, v in sorted(after_by_scenario.items())]
            rows.append({
                "opponent": baseline,
                "domain": domain,
                "before": mean_ci(before_clusters),
                "after": mean_ci(after_clusters),
                "paired_checkpoint_delta": mean_ci(delta_clusters),
            })

    result = {
        "schema": "SPINCORE_CHECKPOINT_DELTA_V1",
        "before_label": args.before_label,
        "after_label": args.after_label,
        "seed": int(before["seed"]),
        "scenarios": int(before["scenarios"]),
        "pairing": "same scenario, deal, hero seat, opponent family and hero random stream; CI clustered by scenario",
        "warning": "weak-baseline checkpoint diagnostic; not exploitability/GTO proof",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== SpinCore paired checkpoint delta ===")
    print(f"before={args.before_label} after={args.after_label} seed={result['seed']} scenarios={result['scenarios']}")
    for row in rows:
        d = row["paired_checkpoint_delta"]
        print(
            f"{row['opponent']} / {row['domain']}: "
            f"delta={d['mean']:+.3f} chips/hand "
            f"CI95=[{d['ci95_low']:+.3f},{d['ci95_high']:+.3f}]"
        )
    print(f"checkpoint_delta_report={args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
