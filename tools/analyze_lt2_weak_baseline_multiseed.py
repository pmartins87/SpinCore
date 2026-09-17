#!/usr/bin/env python3
from __future__ import annotations

"""Analyze multi-seed Stage-A/Stage-B weak-baseline evaluations.

Primary question: does the Stage-B policy have positive mean chip EV against each
transparent weak opponent family in both 3H and HU?

The six primary claims (3 baselines x 2 domains) use a simultaneous family-wise
95% normal CI via Bonferroni. This avoids declaring PASS/FAIL from arbitrary
point thresholds. Stage-A and paired B-minus-A results are retained as secondary
diagnostics.
"""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist, fmean, stdev
from typing import Any

BASELINES = ("UNIFORM_LEGAL", "PASSIVE_CALLER", "JAMMER")
DOMAINS = ("THREE_HANDED", "TRUE_HEADS_UP")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["scenario"]),
        str(row["domain"]),
        str(row["blind"]),
        int(row["hero_seat"]),
        str(row["baseline"]),
    )


def _stats(values: list[float], z: float) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "sem": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "half_width": float("nan"),
        }
    mean = float(fmean(values))
    sd = 0.0 if n == 1 else float(stdev(values))
    sem = sd / math.sqrt(n) if n > 0 else float("nan")
    half = float(z * sem)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "half_width": half,
    }


def _cluster(records: list[dict[str, Any]], value_key: str, baseline: str, domain: str) -> list[float]:
    by_cluster: dict[tuple[int, int], list[float]] = {}
    for r in records:
        if r["baseline"] != baseline:
            continue
        if domain != "ALL" and r["domain"] != domain:
            continue
        k = (int(r["seed"]), int(r["scenario"]))
        by_cluster.setdefault(k, []).append(float(r[value_key]))
    return [float(fmean(v)) for v in by_cluster.values()]


def _per_seed(records: list[dict[str, Any]], value_key: str, baseline: str, domain: str) -> dict[str, float]:
    out: dict[str, float] = {}
    seeds = sorted({int(r["seed"]) for r in records})
    for seed in seeds:
        vals = _cluster([r for r in records if int(r["seed"]) == seed], value_key, baseline, domain)
        out[str(seed)] = float(fmean(vals)) if vals else float("nan")
    return out


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    a_files = sorted(run_dir.glob("seed_*_a_rows.json"))
    b_files = sorted(run_dir.glob("seed_*_b_rows.json"))
    if not a_files or len(a_files) != len(b_files):
        raise SystemExit("missing or unbalanced A/B row files")

    by_seed_a: dict[int, Path] = {}
    by_seed_b: dict[int, Path] = {}
    for p in a_files:
        seed = int(p.name.split("_")[1])
        by_seed_a[seed] = p
    for p in b_files:
        seed = int(p.name.split("_")[1])
        by_seed_b[seed] = p
    if set(by_seed_a) != set(by_seed_b):
        raise SystemExit("A/B seed sets differ")

    records: list[dict[str, Any]] = []
    scenarios_by_seed: dict[str, int] = {}
    for seed in sorted(by_seed_a):
        pa = _load(by_seed_a[seed])
        pb = _load(by_seed_b[seed])
        if pa.get("schema") != "SPINCORE_LEAN_STRATEGY_QUALITY_ROWS_V1" or pb.get("schema") != pa.get("schema"):
            raise SystemExit(f"wrong rows schema for seed {seed}")
        if int(pa.get("seed", -1)) != seed or int(pb.get("seed", -1)) != seed:
            raise SystemExit(f"seed mismatch for {seed}")
        if int(pa.get("scenarios", -1)) != int(pb.get("scenarios", -2)):
            raise SystemExit(f"scenario-count mismatch for seed {seed}")
        scenarios_by_seed[str(seed)] = int(pa["scenarios"])

        ma = {_key(r): r for r in pa.get("rows", [])}
        mb = {_key(r): r for r in pb.get("rows", [])}
        if set(ma) != set(mb):
            raise SystemExit(f"row-key mismatch for seed {seed}")
        for k in sorted(ma):
            ra = ma[k]
            rb = mb[k]
            if int(ra["uniform_control"]) != int(rb["uniform_control"]):
                raise SystemExit(f"uniform-control pairing drift for seed {seed}, key={k}")
            records.append(
                {
                    "seed": seed,
                    "scenario": int(ra["scenario"]),
                    "domain": str(ra["domain"]),
                    "blind": str(ra["blind"]),
                    "hero_seat": int(ra["hero_seat"]),
                    "baseline": str(ra["baseline"]),
                    "stage_a": float(ra["spincore"]),
                    "stage_b": float(rb["spincore"]),
                    "uniform_control": float(rb["uniform_control"]),
                    "b_minus_a": float(rb["spincore"] - ra["spincore"]),
                    "b_minus_uniform": float(rb["paired_gain"]),
                }
            )

    alpha_family = 0.05
    primary_claims = len(BASELINES) * len(DOMAINS)
    z95 = NormalDist().inv_cdf(0.975)
    zsim = NormalDist().inv_cdf(1.0 - alpha_family / (2.0 * primary_claims))

    primary: dict[str, Any] = {}
    secondary_all: dict[str, Any] = {}
    stage_delta: dict[str, Any] = {}
    all_primary_statuses: list[str] = []
    max_simultaneous_half_width = 0.0

    for baseline in BASELINES:
        primary[baseline] = {}
        secondary_all[baseline] = {}
        stage_delta[baseline] = {}
        for domain in DOMAINS:
            bvals = _cluster(records, "stage_b", baseline, domain)
            avals = _cluster(records, "stage_a", baseline, domain)
            dvals = _cluster(records, "b_minus_a", baseline, domain)
            bstat95 = _stats(bvals, z95)
            bstatsim = _stats(bvals, zsim)
            astat95 = _stats(avals, z95)
            dstat95 = _stats(dvals, z95)
            max_simultaneous_half_width = max(max_simultaneous_half_width, float(bstatsim["half_width"]))
            if float(bstatsim["ci_low"]) > 0.0:
                status = "POSITIVE"
            elif float(bstatsim["ci_high"]) < 0.0:
                status = "NEGATIVE"
            else:
                status = "UNRESOLVED"
            all_primary_statuses.append(status)
            primary[baseline][domain] = {
                "stage_b_raw_chip_ev_95": bstat95,
                "stage_b_raw_chip_ev_simultaneous_family95": bstatsim,
                "stage_a_raw_chip_ev_95": astat95,
                "paired_b_minus_a_95": dstat95,
                "stage_b_per_seed_means": _per_seed(records, "stage_b", baseline, domain),
                "b_minus_a_per_seed_means": _per_seed(records, "b_minus_a", baseline, domain),
                "status": status,
            }
            stage_delta[baseline][domain] = dstat95

        for value_key in ("stage_a", "stage_b", "b_minus_a", "b_minus_uniform"):
            secondary_all[baseline][value_key] = _stats(_cluster(records, value_key, baseline, "ALL"), z95)
        secondary_all[baseline]["stage_b_per_seed_means"] = _per_seed(records, "stage_b", baseline, "ALL")

    if all(s == "POSITIVE" for s in all_primary_statuses):
        overall_status = "ALL_SIX_WEAK_BASELINES_POSITIVE"
    elif any(s == "NEGATIVE" for s in all_primary_statuses):
        overall_status = "AT_LEAST_ONE_CONFIRMED_NEGATIVE"
    else:
        overall_status = "AT_LEAST_ONE_UNRESOLVED"

    report = {
        "schema": "SPINCORE_LT2_WEAK_BASELINE_MULTISEED_V1",
        "seeds": sorted(by_seed_a),
        "scenarios_by_seed": scenarios_by_seed,
        "total_scenarios_per_checkpoint": int(sum(scenarios_by_seed.values())),
        "paired_design": "Stage A and Stage B use identical scenario/deal/opponent/hero RNG streams within each seed",
        "primary_question": "Is Stage B raw chip EV > 0 against each weak baseline in both 3H and HU?",
        "primary_multiple_testing": {
            "claims": primary_claims,
            "family_alpha": alpha_family,
            "method": "Bonferroni simultaneous two-sided family-wise 95% CI",
            "z": zsim,
            "classification": "POSITIVE if simultaneous lower bound > 0; NEGATIVE if upper bound < 0; otherwise UNRESOLVED",
        },
        "precision": {
            "max_primary_simultaneous_half_width_chips_per_hand": max_simultaneous_half_width,
            "design_target_chips_per_hand": 5.0,
            "target_met": bool(max_simultaneous_half_width <= 5.0),
            "note": "5-chip target was chosen before this run to resolve the pilot's apparent ~8 to ~12 chip HU losses; it is a precision target, not a strength pass threshold.",
        },
        "overall_status": overall_status,
        "primary": primary,
        "secondary_all_domains": secondary_all,
        "warning": "This is a weak-opponent curriculum gate, not exploitability/GTO proof.",
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== SpinCore LT2 weak-baseline multi-seed review ===")
    print(f"seeds={report['seeds']} total_scenarios_per_checkpoint={report['total_scenarios_per_checkpoint']}")
    print(f"simultaneous_family95_z={zsim:.6f} max_half_width={max_simultaneous_half_width:.3f}")
    for baseline in BASELINES:
        for domain in DOMAINS:
            x = primary[baseline][domain]
            s = x["stage_b_raw_chip_ev_simultaneous_family95"]
            print(
                f"{baseline} {domain}: StageB={s['mean']:+.3f} "
                f"simulCI=[{s['ci_low']:+.3f},{s['ci_high']:+.3f}] status={x['status']}"
            )
    print(f"overall_status={overall_status}")
    print(f"precision_target_met={report['precision']['target_met']}")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
