from __future__ import annotations

import argparse
import json
from pathlib import Path


BASELINE = {
    "label": "size4_no_damping",
    "mean_tv": 0.2665907144546509,
    "p95_tv": 0.5670017600059509,
    "per_seed_fit_pass": True,
    "source": "validation/R7_3_PARTIAL_EXACT_POLICY_MIXTURE_COMPOUNDING_320.json",
}

EXPECTED = {
    "size4_tremble_e15_decay50": "validation/R7_3_POLICY_MIXTURE_DECAY_TREMBLE_e15_320.json",
    "size4_tremble_e30_decay50": "validation/R7_3_POLICY_MIXTURE_DECAY_TREMBLE_e30_320.json",
    "size4_tremble_e45_decay50": "validation/R7_3_POLICY_MIXTURE_DECAY_TREMBLE_e45_320.json",
    "size1_no_damping": "validation/R7_3_SIZE1_TREMBLE_baseline_e0_320.json",
    "size1_tremble_e30_decay50": "validation/R7_3_SIZE1_TREMBLE_tremble_e30_320.json",
    "size4_temporal_w50": "validation/R7_3_POLICY_MIXTURE_TEMPORAL_BLEND_w50_320.json",
    "size4_temporal_w75": "validation/R7_3_POLICY_MIXTURE_TEMPORAL_BLEND_w75_320.json",
    "size4_first_transition_e30": "validation/R7_3_POLICY_MIXTURE_FIRST_TRANSITION_TREMBLE_E30_320.json",
    "size4_uncertainty_s05": "validation/R7_3_POLICY_MIXTURE_UNCERTAINTY_DAMPING_s05_320.json",
    "size4_uncertainty_s10": "validation/R7_3_POLICY_MIXTURE_UNCERTAINTY_DAMPING_s10_320.json",
    "size4_regret_floor_e05": "validation/R7_3_POLICY_MIXTURE_REGRET_FLOOR_e05_320.json",
    "size4_regret_floor_e10": "validation/R7_3_POLICY_MIXTURE_REGRET_FLOOR_e10_320.json",
    "direct_behavior_control": "validation/R7_3_DIRECT_BEHAVIOR_COMPOUNDING_320.json",
    "direct_behavior_aggregated_regret": "validation/R7_3_DIRECT_BEHAVIOR_AGGREGATED_REGRET_320.json",
}

NON_PROMOTABLE_CONTROLS = {
    "direct_behavior_control",
    "direct_behavior_aggregated_regret",
}


def _read(path: Path):
    if not path.exists():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid_json:{exc}"
    if payload.get("runner_failed_before_report"):
        return None, "runner_failed_before_report"
    if "cross_seed" not in payload:
        return None, "cross_seed_missing"
    return payload, None


def _row(label: str, path: str, payload: dict):
    cross = payload["cross_seed"]
    mean = float(cross["mean_tv"])
    p95 = float(cross["p95_tv"])
    fit = bool(payload.get("per_seed_fit_pass", False))
    return {
        "label": label,
        "source": path,
        "schema": payload.get("schema"),
        "mean_tv": mean,
        "p50_tv": float(cross.get("p50_tv", float("nan"))),
        "p95_tv": p95,
        "max_tv": float(cross.get("max_tv", float("nan"))),
        "per_seed_fit_pass": fit,
        "r7_3_pass": bool(payload.get("r7_3_pass", False)),
        "mean_ratio_to_size4_no_damping": mean / BASELINE["mean_tv"],
        "p95_ratio_to_size4_no_damping": p95 / BASELINE["p95_tv"],
        "theoretical_equivalence_claimed": payload.get("theoretical_equivalence_claimed"),
        "production_policy_mapping_changed": payload.get("production_policy_mapping_changed"),
        "automatic_promotion_eligible": label not in NON_PROMOTABLE_CONTROLS,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize completed R7.3 five-iteration durability matrix")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_DURABILITY_MATRIX_SUMMARY.json"))
    args = ap.parse_args()

    pending = {}
    rows = []
    for label, rel in EXPECTED.items():
        payload, error = _read(args.repo_root / rel)
        if error:
            pending[label] = {"source": rel, "status": error}
            continue
        rows.append(_row(label, rel, payload))

    if pending:
        print(json.dumps({
            "schema": "SPINCORE_R7_3_DURABILITY_MATRIX_SUMMARY_V3",
            "complete": False,
            "pending": pending,
            "completed_labels": [row["label"] for row in rows],
        }, indent=2, sort_keys=True))
        return 3

    baseline_row = dict(BASELINE)
    baseline_row.update({
        "p50_tv": 0.24680466949939728,
        "max_tv": 0.9055466651916504,
        "r7_3_pass": False,
        "mean_ratio_to_size4_no_damping": 1.0,
        "p95_ratio_to_size4_no_damping": 1.0,
        "schema": "SPINCORE_R7_3_PARTIAL_EXACT_POLICY_MIXTURE_PAIRED_V1",
        "automatic_promotion_eligible": True,
    })
    all_rows = [baseline_row] + rows
    fit_eligible = [r for r in rows if r["per_seed_fit_pass"]]
    durable_both_improved = [
        r for r in fit_eligible
        if r["mean_tv"] < BASELINE["mean_tv"] and r["p95_tv"] < BASELINE["p95_tv"]
    ]
    best_by_p95 = min(fit_eligible, key=lambda r: (r["p95_tv"], r["mean_tv"])) if fit_eligible else None
    best_durable = min(
        durable_both_improved,
        key=lambda r: (r["p95_tv"], r["mean_tv"]),
    ) if durable_both_improved else None
    conservative = [
        r for r in durable_both_improved
        if r["label"] not in NON_PROMOTABLE_CONTROLS
    ]
    conservative_best = min(
        conservative,
        key=lambda r: (r["p95_tv"], r["mean_tv"]),
    ) if conservative else None

    payload = {
        "schema": "SPINCORE_R7_3_DURABILITY_MATRIX_SUMMARY_V3",
        "complete": True,
        "frozen_gates": {
            "advantage_weighted_nrmse_max": 0.75,
            "policy_weighted_mean_tv_max": 0.12,
            "cross_seed_mean_tv_max": 0.15,
            "cross_seed_p95_tv_max": 0.35,
        },
        "five_iteration_size4_no_damping_baseline": baseline_row,
        "rows": all_rows,
        "best_fit_eligible_by_p95": best_by_p95,
        "best_both_improved": best_durable,
        "best_conservative_both_improved": conservative_best,
        "promotion_shortlist": conservative,
        "non_promotable_controls": sorted(NON_PROMOTABLE_CONTROLS),
        "interpretation_note": (
            "Automatic evidence consolidation only. A row is not promoted merely by ranking first. "
            "Any new behavior semantics still require explicit algorithm versioning, strategic audit, "
            "and deterministic checkpoint/resume recertification before an acceptance-scale run. "
            "Direct Behavior variants remain causal/smoothing controls and are excluded from automatic "
            "conservative promotion because theoretical equivalence is not established."
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
