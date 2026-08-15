from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

SCHEMA = "SPINCORE_R7_5_3D_V1_V2_CONVERGENCE_RESULT_V1"
FIT_SCHEMA = "SPINCORE_R7_5_3_CANDIDATE_FIT_V1"
PRECOMMIT_SCHEMA = "SPINCORE_R7_5_3D_V1_V2_CONVERGENCE_PRECOMMIT_V1"


def _mean(values):
    rows = [float(value) for value in values]
    if not rows or not all(math.isfinite(value) for value in rows):
        raise ValueError("empty/non-finite metric set")
    return sum(rows) / len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--fit-json", type=Path, action="append", required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    precommit = json.loads(
        (args.repo_root / "validation" / "R7_5_3D_V1_V2_CONVERGENCE_PRECOMMIT_20260815.json").read_text()
    )
    if precommit.get("schema") != PRECOMMIT_SCHEMA:
        raise RuntimeError("R7.5.3D precommit schema mismatch")
    if precommit.get("status") != "FROZEN_BEFORE_R7_5_3D_OUTPUTS":
        raise RuntimeError("R7.5.3D precommit is not frozen")
    if precommit.get("selection_authority") != "DIAGNOSTIC_ONLY":
        raise RuntimeError("R7.5.3D illegally has selection authority")
    if precommit.get("strategic_output_seen_before_freeze") is not False:
        raise RuntimeError("R7.5.3D freeze was not pre-output")

    expected_seeds = tuple(int(x) for x in precommit["fit_seeds"])
    expected_domains = tuple(str(x) for x in precommit["domains"])
    budget_by_steps = {}
    expected_cells = set()
    for candidate, cfg in precommit["candidates"].items():
        for budget in cfg["budgets"]:
            key = (candidate, int(budget["advantage_steps"]), int(budget["policy_steps"]))
            budget_by_steps[key] = str(budget["label"])
            for domain in expected_domains:
                for seed in expected_seeds:
                    expected_cells.add((candidate, domain, seed, str(budget["label"])))

    grouped = {}
    for path in args.fit_json:
        report = json.loads(path.read_text())
        if report.get("schema") != FIT_SCHEMA:
            raise RuntimeError(f"wrong fit schema: {path}")
        candidate = str(report["candidate"])
        domain = str(report["domain"])
        seed = int(report["fit_seed"])
        fit = report["fit_contract"]
        budget_key = (candidate, int(fit["advantage_steps"]), int(fit["policy_steps"]))
        if budget_key not in budget_by_steps:
            raise RuntimeError(f"unfrozen candidate/budget: {budget_key}")
        label = budget_by_steps[budget_key]
        cell = (candidate, domain, seed, label)
        if cell in grouped:
            raise RuntimeError(f"duplicate fit cell: {cell}")
        if int(report["split_seed"]) != int(precommit["split_seed"]):
            raise RuntimeError("split seed drift")
        if float(fit["learning_rate"]) != float(precommit["fit_contract"]["learning_rate"]):
            raise RuntimeError("learning-rate drift")
        if int(fit["batch_size"]) != int(precommit["fit_contract"]["batch_size"]):
            raise RuntimeError("batch-size drift")
        grouped[cell] = report

    actual_cells = set(grouped)
    if actual_cells != expected_cells:
        raise RuntimeError(
            f"fit inventory mismatch missing={sorted(expected_cells-actual_cells)} extra={sorted(actual_cells-expected_cells)}"
        )

    summaries = defaultdict(dict)
    corpus_identity = {}
    for candidate, cfg in precommit["candidates"].items():
        for budget in cfg["budgets"]:
            label = str(budget["label"])
            for domain in expected_domains:
                rows = [grouped[(candidate, domain, seed, label)] for seed in expected_seeds]
                identities = [row["corpus"] for row in rows]
                if any(identity != identities[0] for identity in identities[1:]):
                    raise RuntimeError(f"corpus identity drift across seeds: {candidate}/{domain}/{label}")
                corpus_identity.setdefault(domain, identities[0])
                if corpus_identity[domain] != identities[0]:
                    raise RuntimeError(f"corpus identity drift across candidate/budget: {domain}")
                summaries[candidate][f"{domain}/{label}"] = {
                    "advantage_weighted_nrmse_mean": _mean(row["heldout_advantage"]["weighted_nrmse"] for row in rows),
                    "policy_weighted_mean_tv_mean": _mean(row["heldout_policy"]["weighted_mean_tv"] for row in rows),
                    "advantage_seconds_per_step_mean": _mean(row["advantage_fit"]["seconds_per_step"] for row in rows),
                    "policy_seconds_per_step_mean": _mean(row["policy_fit"]["seconds_per_step"] for row in rows),
                    "advantage_elapsed_seconds_mean": _mean(row["advantage_fit"]["elapsed_seconds"] for row in rows),
                    "policy_elapsed_seconds_mean": _mean(row["policy_fit"]["elapsed_seconds"] for row in rows),
                    "advantage_last100_loss_mean": _mean(row["advantage_fit"]["mean_last_100_loss"] for row in rows),
                    "policy_last100_loss_mean": _mean(row["policy_fit"]["mean_last_100_loss"] for row in rows),
                    "peak_rss_kib_mean": _mean(row["peak_rss_kib"] for row in rows),
                    "parameter_count": int(rows[0]["parameter_count"]),
                    "fit_seed_rows": [
                        {
                            "fit_seed": int(row["fit_seed"]),
                            "advantage_weighted_nrmse": float(row["heldout_advantage"]["weighted_nrmse"]),
                            "policy_weighted_mean_tv": float(row["heldout_policy"]["weighted_mean_tv"]),
                            "advantage_seconds_per_step": float(row["advantage_fit"]["seconds_per_step"]),
                            "policy_seconds_per_step": float(row["policy_fit"]["seconds_per_step"]),
                        }
                        for row in rows
                    ],
                }

    adv_band = float(precommit["equivalence_bands_against_V1_1x"]["advantage_weighted_nrmse_absolute"])
    pol_band = float(precommit["equivalence_bands_against_V1_1x"]["policy_weighted_mean_tv_absolute"])
    within_all = True
    improvement_cells = 0
    comparison = {}
    for domain in expected_domains:
        v1 = summaries["C0_V1_FROZEN_CONTROL"][f"{domain}/1x"]
        v21 = summaries["C1_V2_NO_FLOP_TOKEN"][f"{domain}/1x"]
        v22 = summaries["C1_V2_NO_FLOP_TOKEN"][f"{domain}/2x"]
        adv_within = v22["advantage_weighted_nrmse_mean"] <= v1["advantage_weighted_nrmse_mean"] + adv_band
        pol_within = v22["policy_weighted_mean_tv_mean"] <= v1["policy_weighted_mean_tv_mean"] + pol_band
        within_all = within_all and adv_within and pol_within
        adv_improved = v22["advantage_weighted_nrmse_mean"] < v21["advantage_weighted_nrmse_mean"]
        pol_improved = v22["policy_weighted_mean_tv_mean"] < v21["policy_weighted_mean_tv_mean"]
        improvement_cells += int(adv_improved) + int(pol_improved)
        comparison[domain] = {
            "V1_1x_advantage": v1["advantage_weighted_nrmse_mean"],
            "V2_1x_advantage": v21["advantage_weighted_nrmse_mean"],
            "V2_2x_advantage": v22["advantage_weighted_nrmse_mean"],
            "V2_2x_advantage_within_V1_band": adv_within,
            "V2_1x_to_2x_advantage_improved": adv_improved,
            "V1_1x_policy_tv": v1["policy_weighted_mean_tv_mean"],
            "V2_1x_policy_tv": v21["policy_weighted_mean_tv_mean"],
            "V2_2x_policy_tv": v22["policy_weighted_mean_tv_mean"],
            "V2_2x_policy_within_V1_band": pol_within,
            "V2_1x_to_2x_policy_improved": pol_improved,
        }

    if within_all and improvement_cells >= 3:
        classification = "UNDERTRAINING_SUPPORTED"
    elif not within_all:
        classification = "GAP_PERSISTS_AT_2X"
    else:
        classification = "INCONCLUSIVE"

    payload = {
        "schema": SCHEMA,
        "execution_sha": str(args.execution_sha),
        "precommit_schema": PRECOMMIT_SCHEMA,
        "selection_authority": "DIAGNOSTIC_ONLY",
        "classification": classification,
        "classification_scope": "V1-generated target convergence only; no production representation authority",
        "improvement_cells_V2_1x_to_2x": int(improvement_cells),
        "all_V2_2x_cells_within_V1_1x_equivalence_bands": bool(within_all),
        "comparison": comparison,
        "summaries": dict(summaries),
        "corpus_identity": corpus_identity,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
