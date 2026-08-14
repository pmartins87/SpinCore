from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

SCHEMA = "SPINCORE_R7_5_3C_PHASE1_DIAGNOSTIC_SUMMARY_V1"
CANDIDATES = (
    "H0_FIXED_V1",
    "H1_RELATIONAL_EXACT",
    "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY",
    "H3_HYBRID_EXACT_SEMANTIC",
)
DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")
SEEDS = (247244420, 1953786788, 2029202803)


def _mean(values):
    return float(sum(values) / len(values))


def _finite(value):
    return math.isfinite(float(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, action="append", required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    identities = set()
    for path in args.fit:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        report = dict(payload["report"])
        key = (str(report["candidate"]), str(report["domain"]), int(report["fit_seed"]))
        if key in identities:
            raise ValueError(f"duplicate fit identity {key}")
        identities.add(key)
        rows.append(report)

    expected = {(c, d, s) for c in CANDIDATES for d in DOMAINS for s in SEEDS}
    if identities != expected:
        missing = sorted(expected - identities)
        extra = sorted(identities - expected)
        raise ValueError(f"phase1 fit coverage mismatch missing={missing} extra={extra}")

    grouped = defaultdict(list)
    by_key = {}
    for row in rows:
        grouped[(row["candidate"], row["domain"])].append(row)
        by_key[(row["candidate"], row["domain"], int(row["fit_seed"]))] = row

    summary = {}
    for candidate in CANDIDATES:
        summary[candidate] = {}
        for domain in DOMAINS:
            group = sorted(grouped[(candidate, domain)], key=lambda x: int(x["fit_seed"]))
            adv = [float(x["heldout_advantage"]["weighted_nrmse"]) for x in group]
            tv = [float(x["heldout_policy"]["weighted_mean_tv"]) for x in group]
            adv_sps = [float(x["heldout_advantage"]["inference_samples_per_second"]) for x in group]
            pol_sps = [float(x["heldout_policy"]["inference_samples_per_second"]) for x in group]
            adv_step = [float(x["advantage_fit"]["seconds_per_step"]) for x in group]
            pol_step = [float(x["policy_fit"]["seconds_per_step"]) for x in group]
            if not all(_finite(v) for v in adv + tv + adv_sps + pol_sps + adv_step + pol_step):
                raise ValueError(f"non-finite diagnostic metric for {candidate}/{domain}")
            params = {int(x["parameter_count"]) for x in group}
            if len(params) != 1:
                raise ValueError(f"parameter count drift across seeds for {candidate}/{domain}")
            summary[candidate][domain] = {
                "parameter_count": params.pop(),
                "advantage_weighted_nrmse_mean": _mean(adv),
                "advantage_weighted_nrmse_worst": max(adv),
                "policy_weighted_mean_tv_mean": _mean(tv),
                "policy_weighted_mean_tv_worst": max(tv),
                "advantage_inference_samples_per_second_mean": _mean(adv_sps),
                "policy_inference_samples_per_second_mean": _mean(pol_sps),
                "advantage_optimizer_seconds_per_step_mean": _mean(adv_step),
                "policy_optimizer_seconds_per_step_mean": _mean(pol_step),
                "peak_rss_kib_max": max(int(x["peak_rss_kib"]) for x in group),
                "legacy_absolute_fit_pass_count": sum(
                    1 for x in group if bool(x["absolute_gates"]["fit_pass"])
                ),
                "seed_metrics": [
                    {
                        "fit_seed": int(x["fit_seed"]),
                        "advantage_weighted_nrmse": float(x["heldout_advantage"]["weighted_nrmse"]),
                        "policy_weighted_mean_tv": float(x["heldout_policy"]["weighted_mean_tv"]),
                        "advantage_inference_samples_per_second": float(x["heldout_advantage"]["inference_samples_per_second"]),
                        "policy_inference_samples_per_second": float(x["heldout_policy"]["inference_samples_per_second"]),
                    }
                    for x in group
                ],
            }

    paired_vs_h0 = {}
    for candidate in CANDIDATES[1:]:
        paired_vs_h0[candidate] = {}
        for domain in DOMAINS:
            deltas = []
            for seed in SEEDS:
                control = by_key[("H0_FIXED_V1", domain, seed)]
                current = by_key[(candidate, domain, seed)]
                deltas.append(
                    {
                        "fit_seed": seed,
                        "advantage_nrmse_delta_vs_h0": float(
                            current["heldout_advantage"]["weighted_nrmse"]
                            - control["heldout_advantage"]["weighted_nrmse"]
                        ),
                        "policy_tv_delta_vs_h0": float(
                            current["heldout_policy"]["weighted_mean_tv"]
                            - control["heldout_policy"]["weighted_mean_tv"]
                        ),
                    }
                )
            paired_vs_h0[candidate][domain] = {
                "per_seed": deltas,
                "advantage_nrmse_delta_mean": _mean(
                    [x["advantage_nrmse_delta_vs_h0"] for x in deltas]
                ),
                "policy_tv_delta_mean": _mean([x["policy_tv_delta_vs_h0"] for x in deltas]),
            }

    output = {
        "schema": SCHEMA,
        "execution_sha": args.execution_sha,
        "selection_authority": False,
        "diagnostic_only": True,
        "candidate_inference_generated_targets": False,
        "targets_origin": "frozen V1 behavior paired corpus from workflow run 31767822186",
        "coverage_complete": True,
        "summary": summary,
        "paired_vs_h0": paired_vs_h0,
        "interpretation": [
            "This summary may identify optimization difficulty, information loss and resource cost.",
            "It must not reject a symmetry-correct candidate solely for worse imitation of V1-generated targets.",
            "Candidate selection requires the precommitted candidate-specific end-to-end phase."
        ],
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
