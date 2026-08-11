from __future__ import annotations

import argparse
import json
from pathlib import Path

import summarize_r7_3_durability_matrix as base


SUPPLEMENTAL = {
    "size8_temporal_w50": "validation/R7_3_POLICY_MIXTURE_SIZE8_TEMPORAL_W50_320.json",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize the base R7.3 durability matrix plus promoted compositions")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_DURABILITY_EXTENDED_SUMMARY.json"))
    args = ap.parse_args()

    expected = dict(base.EXPECTED)
    expected.update(SUPPLEMENTAL)

    pending: dict[str, dict[str, str]] = {}
    rows: list[dict] = []
    for label, rel in expected.items():
        payload, error = base._read(args.repo_root / rel)
        if error:
            pending[label] = {"source": rel, "status": error}
            continue
        rows.append(base._row(label, rel, payload))

    if pending:
        print(json.dumps({
            "schema": "SPINCORE_R7_3_DURABILITY_EXTENDED_SUMMARY_V1",
            "complete": False,
            "pending": pending,
            "completed_labels": [row["label"] for row in rows],
            "expected_candidate_rows": len(expected),
        }, indent=2, sort_keys=True))
        return 3

    baseline_row = dict(base.BASELINE)
    baseline_row.update({
        "p50_tv": 0.24680466949939728,
        "max_tv": 0.9055466651916504,
        "r7_3_pass": False,
        "mean_ratio_to_size4_no_damping": 1.0,
        "p95_ratio_to_size4_no_damping": 1.0,
        "schema": "SPINCORE_R7_3_PARTIAL_EXACT_POLICY_MIXTURE_PAIRED_V1",
        "automatic_promotion_eligible": True,
    })

    fit_eligible = [r for r in rows if r["per_seed_fit_pass"]]
    both_improved = [
        r for r in fit_eligible
        if r["mean_tv"] < base.BASELINE["mean_tv"] and r["p95_tv"] < base.BASELINE["p95_tv"]
    ]
    conservative = [
        r for r in both_improved
        if r["label"] not in base.NON_PROMOTABLE_CONTROLS
    ]
    gate_pass = [
        r for r in fit_eligible
        if r["mean_tv"] <= 0.15 and r["p95_tv"] <= 0.35 and r["r7_3_pass"]
    ]

    def best(seq):
        return min(seq, key=lambda r: (r["p95_tv"], r["mean_tv"])) if seq else None

    payload = {
        "schema": "SPINCORE_R7_3_DURABILITY_EXTENDED_SUMMARY_V1",
        "complete": True,
        "expected_candidate_rows": len(expected),
        "rows_including_baseline": len(rows) + 1,
        "base_matrix_schema": "SPINCORE_R7_3_DURABILITY_MATRIX_SUMMARY_V4",
        "supplemental_labels": sorted(SUPPLEMENTAL),
        "frozen_gates": {
            "advantage_weighted_nrmse_max": 0.75,
            "policy_weighted_mean_tv_max": 0.12,
            "cross_seed_mean_tv_max": 0.15,
            "cross_seed_p95_tv_max": 0.35,
        },
        "five_iteration_size4_no_damping_baseline": baseline_row,
        "rows": [baseline_row] + rows,
        "best_fit_eligible_by_p95": best(fit_eligible),
        "best_conservative_both_improved": best(conservative),
        "full_cross_seed_gate_pass_rows": gate_pass,
        "promotion_shortlist": sorted(conservative, key=lambda r: (r["p95_tv"], r["mean_tv"])),
        "interpretation_note": (
            "Extended evidence consolidation keeps the original 15-candidate base matrix intact and adds "
            "explicitly promoted compositions. Ranking is not promotion. Any changed behavior semantics "
            "still require freeze/versioning, fresh-process reproducibility and deterministic checkpoint/"
            "resume recertification before acceptance-scale execution."
        ),
        "acceptance_gate_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
