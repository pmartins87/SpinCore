#!/usr/bin/env python3
from __future__ import annotations

"""Descriptive root-cause diagnosis of the closed post-15315 K=12 stability FAIL.

This runner is strictly non-promotional. It consumes only already-created
internal stability artifacts, performs no training, opens no external-strength
benchmark, changes no frozen threshold/seed/member set, and cannot change the
recorded K=12 FAIL. Its sole purpose is hypothesis generation for a future,
separately preregistered research branch.

PROJECT_CONTRACT_IDS:
VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,VALID-038,
SAFE-001,SAFE-010,SAFE-011,ART-001
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import evaluate_post15315_rebuild_stability as stability

SCHEMA = "SPINCORE_POST15315_K12_INSTABILITY_DIAGNOSTIC_V1"
GATE_SCHEMA = "SPINCORE_POST15315_K12_STABILITY_GATE_V1"
ACTION_NAMES = tuple(stability.ACTION_NAMES)
FOLD = int(stability.FOLD)
ALL_IN = int(stability.ALL_IN)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def p95(x: np.ndarray) -> float:
    return float(np.quantile(x, 0.95)) if x.size else float("nan")


def tv_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(a - b).sum(axis=-1)


def abs_action_delta(a: np.ndarray, b: np.ndarray, mask: np.ndarray, action: int) -> dict:
    d = np.abs(a[mask, action] - b[mask, action])
    return {
        "count": int(mask.sum()),
        "mean_probability_delta": float(d.mean()) if d.size else None,
        "p95_probability_delta": p95(d) if d.size else None,
    }


def subset_tv(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict:
    d = tv_rows(a[mask], b[mask])
    return {
        "count": int(mask.sum()),
        "mean_tv": float(d.mean()) if d.size else None,
        "p95_tv": p95(d) if d.size else None,
    }


def top_action_deltas(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> list[dict]:
    if mask is None:
        aa, bb = a, b
    else:
        aa, bb = a[mask], b[mask]
    if aa.shape[0] == 0:
        return []
    d = np.abs(aa.mean(axis=0) - bb.mean(axis=0))
    order = np.argsort(-d)
    return [
        {
            "action": ACTION_NAMES[int(i)],
            "mean_probability_delta": float(d[int(i)]),
            "mean_probability_a": float(aa[:, int(i)].mean()),
            "mean_probability_b": float(bb[:, int(i)].mean()),
        }
        for i in order
        if float(d[int(i)]) > 0.0
    ]


def member_rows(probs: np.ndarray, ensemble: np.ndarray, surfaces: dict[str, np.ndarray]) -> list[dict]:
    rows: list[dict] = []
    for i in range(probs.shape[0]):
        tv = tv_rows(probs[i], ensemble)
        row = {
            "member": i + 1,
            "overall_mean_tv_to_k12_ensemble": float(tv.mean()),
            "overall_p95_tv_to_k12_ensemble": p95(tv),
            "argmax_disagreement_vs_k12_ensemble": float(
                np.mean(np.argmax(probs[i], axis=1) != np.argmax(ensemble, axis=1))
            ),
            "preflop_72o_allin": abs_action_delta(
                probs[i], ensemble, surfaces["surface_72o"], ALL_IN
            ),
            "trips_plus_fold": abs_action_delta(
                probs[i], ensemble, surfaces["surface_trips_fold"], FOLD
            ),
            "high_card_no_draw_allin": abs_action_delta(
                probs[i], ensemble, surfaces["surface_high_card_no_draw"], ALL_IN
            ),
        }
        rows.append(row)
    return rows


def action_instability(probs: np.ndarray, ensemble: np.ndarray) -> list[dict]:
    abs_dev = np.abs(probs - ensemble[None, :, :])
    state_member_mean = abs_dev.mean(axis=0)
    totals = state_member_mean.sum(axis=0)
    denom = float(totals.sum())
    rows = []
    for action in range(probs.shape[2]):
        x = state_member_mean[:, action]
        rows.append(
            {
                "action": ACTION_NAMES[action],
                "mean_abs_member_probability_deviation": float(x.mean()),
                "statewise_p95_abs_member_probability_deviation": p95(x),
                "share_of_total_absolute_dispersion": (
                    float(totals[action] / denom) if denom > 0.0 else 0.0
                ),
            }
        )
    return sorted(rows, key=lambda r: r["share_of_total_absolute_dispersion"], reverse=True)


def concentration(probs: np.ndarray, ensemble: np.ndarray) -> dict:
    # Per-state average member TV to the K=12 ensemble.
    x = tv_rows(probs, ensemble[None, :, :]).mean(axis=0)
    order = np.sort(x)[::-1]
    total = float(order.sum())
    out = {
        "mean": float(x.mean()),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(x.max()),
    }
    for pct in (1, 5, 10):
        n = max(1, int(np.ceil(len(order) * pct / 100.0)))
        out[f"top_{pct}pct_share_of_total_dispersion"] = (
            float(order[:n].sum() / total) if total > 0.0 else 0.0
        )
    return out


def pairwise_rows(probs: np.ndarray) -> list[dict]:
    rows = []
    for i, j in itertools.combinations(range(probs.shape[0]), 2):
        tv = tv_rows(probs[i], probs[j])
        rows.append(
            {
                "member_a": i + 1,
                "member_b": j + 1,
                "mean_tv": float(tv.mean()),
                "p95_tv": p95(tv),
                "argmax_disagreement": float(
                    np.mean(np.argmax(probs[i], axis=1) != np.argmax(probs[j], axis=1))
                ),
            }
        )
    return sorted(rows, key=lambda r: r["mean_tv"], reverse=True)


def split_rows(probs: np.ndarray, surfaces: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for index, (left, right) in enumerate(stability._balanced_splits(12), 1):
        a = probs[np.asarray(left)].mean(axis=0)
        b = probs[np.asarray(right)].mean(axis=0)
        tv = tv_rows(a, b)
        trips_mask = surfaces["surface_trips_fold"]
        trips_delta = np.abs(a[trips_mask, FOLD] - b[trips_mask, FOLD])
        rows.append(
            {
                "split_index": index,
                "left_members": [x + 1 for x in left],
                "right_members": [x + 1 for x in right],
                "broad_mean_tv": float(tv.mean()),
                "broad_p95_tv": p95(tv),
                "trips_fold_mean_delta": float(trips_delta.mean()),
                "trips_fold_p95_delta": p95(trips_delta),
                "top_broad_action_mean_deltas": top_action_deltas(a, b)[:5],
                "top_trips_action_mean_deltas": top_action_deltas(a, b, trips_mask)[:5],
            }
        )
    return rows


def validate_gate(k12_dir: Path) -> tuple[dict, Path, Path]:
    gate_path = k12_dir / "post15315_k12_stability_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("schema") != GATE_SCHEMA or gate.get("status") != "PASS":
        raise RuntimeError("K=12 stability gate is not mechanically PASS")
    if not bool(gate.get("mechanically_clean")):
        raise RuntimeError("K=12 stability gate is not mechanically clean")
    if str(gate.get("k12_stability_result")) != "FAIL":
        raise RuntimeError("diagnostic is permitted only for recorded K=12 FAIL")
    if bool(gate.get("aggregate_quality_authorized")):
        raise RuntimeError("failed K=12 unexpectedly authorizes aggregate quality")
    if bool(gate.get("strength_benchmark_performed")):
        raise RuntimeError("K=12 gate unexpectedly opened external strength")

    bank_path = k12_dir / "post15315_k12_stability_bank.npz"
    report_path = k12_dir / "post15315_k12_stability_report.json"
    if sha256(bank_path) != str(gate.get("bank_sha256")):
        raise RuntimeError("K=12 bank SHA mismatch")
    if sha256(report_path) != str(gate.get("stability_report_sha256")):
        raise RuntimeError("K=12 report SHA mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if bool(report.get("rebuild_stability_pass")):
        raise RuntimeError("K=12 report does not preserve scientific FAIL")
    return gate, bank_path, report_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k12-dir", type=Path, required=True)
    ap.add_argument("--k8-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    k12_dir = args.k12_dir.resolve(strict=True)
    k8_dir = args.k8_dir.resolve(strict=True)
    gate, bank_path, report_path = validate_gate(k12_dir)

    with np.load(bank_path, allow_pickle=False) as z:
        probs = np.asarray(z["probs"], dtype=np.float32)
        legal = np.asarray(z["legal"], dtype=np.bool_)
        surfaces = {
            "surface_72o": np.asarray(z["surface_72o"], dtype=np.bool_),
            "surface_trips_fold": np.asarray(z["surface_trips_fold"], dtype=np.bool_),
            "surface_high_card_no_draw": np.asarray(z["surface_high_card_no_draw"], dtype=np.bool_),
        }
    stability._validate_input(probs, legal, surfaces)
    if probs.shape[0] != 12:
        raise RuntimeError(f"expected K=12 bank, got {probs.shape}")

    k8_bank = k8_dir / "post15315_k8_stability_bank.npz"
    with np.load(k8_bank, allow_pickle=False) as z:
        probs8 = np.asarray(z["probs"], dtype=np.float32)
        legal8 = np.asarray(z["legal"], dtype=np.bool_)
        s72_8 = np.asarray(z["surface_72o"], dtype=np.bool_)
        trips8 = np.asarray(z["surface_trips_fold"], dtype=np.bool_)
        high8 = np.asarray(z["surface_high_card_no_draw"], dtype=np.bool_)

    if probs8.shape[0] != 8 or probs8.shape[1:] != probs.shape[1:]:
        raise RuntimeError("K=8/K=12 common-bank shape mismatch")
    if not np.array_equal(legal8, legal):
        raise RuntimeError("K=8/K=12 legal-mask mismatch")
    if not np.array_equal(s72_8, surfaces["surface_72o"]):
        raise RuntimeError("K=8/K=12 72o surface mismatch")
    if not np.array_equal(trips8, surfaces["surface_trips_fold"]):
        raise RuntimeError("K=8/K=12 trips surface mismatch")
    if not np.array_equal(high8, surfaces["surface_high_card_no_draw"]):
        raise RuntimeError("K=8/K=12 high-card surface mismatch")
    max_first8_diff = float(np.max(np.abs(probs8 - probs[:8])))
    if max_first8_diff > 1e-6:
        raise RuntimeError(
            f"K=8 member predictions do not reproduce inside K=12 bank: maxdiff={max_first8_diff}"
        )

    ensemble12 = probs.mean(axis=0)
    ensemble8 = probs8.mean(axis=0)
    extension_mean = probs[8:12].mean(axis=0)

    splits = split_rows(probs, surfaces)
    worst_broad = sorted(splits, key=lambda r: r["broad_mean_tv"], reverse=True)[:10]
    worst_trips = sorted(splits, key=lambda r: r["trips_fold_mean_delta"], reverse=True)[:10]

    output = {
        "schema": SCHEMA,
        "status": "PASS",
        "purpose": "descriptive_root_cause_hypothesis_generation_only",
        "frozen_k12_failure_is_immutable": True,
        "promotion_decision_performed": False,
        "training_performed": False,
        "member_selection_performed": False,
        "member_weighting_performed": False,
        "threshold_change_performed": False,
        "seed_change_performed": False,
        "external_strength_benchmark_performed": False,
        "k12_gate_sha256": sha256(k12_dir / "post15315_k12_stability_gate.json"),
        "k12_bank_sha256": sha256(bank_path),
        "k12_report_sha256": sha256(report_path),
        "k8_bank_sha256": sha256(k8_bank),
        "state_count": int(probs.shape[1]),
        "first8_reproduction_max_abs_probability_difference": max_first8_diff,
        "surface_counts": {name: int(mask.sum()) for name, mask in surfaces.items()},
        "member_to_k12_ensemble": member_rows(probs, ensemble12, surfaces),
        "action_dispersion": action_instability(probs, ensemble12),
        "state_dispersion_concentration": concentration(probs, ensemble12),
        "top_15_pairwise_disagreements": pairwise_rows(probs)[:15],
        "worst_10_balanced_splits_by_broad_mean_tv": worst_broad,
        "worst_10_balanced_splits_by_trips_fold_mean_delta": worst_trips,
        "k8_to_k12_ensemble_shift": {
            "overall": subset_tv(ensemble8, ensemble12, np.ones(probs.shape[1], dtype=np.bool_)),
            "preflop_72o_allin": abs_action_delta(
                ensemble8, ensemble12, surfaces["surface_72o"], ALL_IN
            ),
            "trips_plus_fold": abs_action_delta(
                ensemble8, ensemble12, surfaces["surface_trips_fold"], FOLD
            ),
            "high_card_no_draw_allin": abs_action_delta(
                ensemble8, ensemble12, surfaces["surface_high_card_no_draw"], ALL_IN
            ),
            "top_action_mean_probability_shifts": top_action_deltas(ensemble8, ensemble12),
        },
        "original8_vs_extension4": {
            "overall": subset_tv(ensemble8, extension_mean, np.ones(probs.shape[1], dtype=np.bool_)),
            "preflop_72o_allin": abs_action_delta(
                ensemble8, extension_mean, surfaces["surface_72o"], ALL_IN
            ),
            "trips_plus_fold": abs_action_delta(
                ensemble8, extension_mean, surfaces["surface_trips_fold"], FOLD
            ),
            "high_card_no_draw_allin": abs_action_delta(
                ensemble8, extension_mean, surfaces["surface_high_card_no_draw"], ALL_IN
            ),
            "top_action_mean_probability_differences": top_action_deltas(
                ensemble8, extension_mean
            ),
        },
        "interpretation_rule": (
            "These rows may identify where rebuild variance concentrates and generate hypotheses "
            "for a new preregistered research branch. They may not be used to drop/weight members, "
            "relax thresholds, replace seeds, promote K=12, or open external strength evidence."
        ),
    }
    atomic_json(output, args.out.resolve())
    print("POST15315_K12_INSTABILITY_DIAGNOSTIC_PASS", flush=True)
    print(f"report={args.out.resolve()}", flush=True)
    print("STOP: descriptive evidence only; K=12 scientific FAIL remains immutable.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
