#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate frozen post-15315 rebuild-seed stability from common-state predictions.

Input NPZ contract:
  probs: [K,N,7] final legal action-probability vectors, after member-local
         fullpool/specialist routing.
  legal: [N,7] bool legality mask.
  surface_72o: [N] bool, preflop 72o with ALL_IN legal.
  surface_trips_fold: [N] bool, postflop trips-or-better with FOLD legal.
  surface_high_card_no_draw: [N] bool, postflop high-card/no-immediate-draw
                             with ALL_IN legal.

The evaluator performs no training, model selection, or external strength
benchmark. It implements the thresholds frozen before teacher 15315 is observed.
"""

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "contracts" / "post15315_multirebuild_preregistration.json"

ACTION_NAMES = (
    "FOLD",
    "CHECK_CALL",
    "POT_33",
    "POT_50",
    "POT_75",
    "POT_100",
    "ALL_IN",
)
FOLD = 0
ALL_IN = 6
EPS = 1e-8


def tv_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(a - b).sum(axis=-1)


def _safe_p95(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, 0.95))


def _balanced_splits(k: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    if k == 8:
        lefts = [
            (0, *tail)
            for tail in itertools.combinations(range(1, 8), 3)
        ]
    elif k == 12:
        all_lefts = [
            (0, *tail)
            for tail in itertools.combinations(range(1, 12), 5)
        ]
        rng = random.Random(153159011)
        chosen = sorted(rng.sample(range(len(all_lefts)), 100))
        lefts = [all_lefts[i] for i in chosen]
    else:
        raise ValueError(f"unsupported rebuild count K={k}; expected 8 or 12")

    universe = set(range(k))
    return [
        (tuple(left), tuple(sorted(universe - set(left))))
        for left in lefts
    ]


def _validate_input(
    probs: np.ndarray,
    legal: np.ndarray,
    surfaces: dict[str, np.ndarray],
) -> None:
    if probs.ndim != 3:
        raise ValueError(f"probs must be [K,N,A], got {probs.shape}")
    k, n, a = probs.shape
    if k not in (8, 12):
        raise ValueError(f"K must be 8 or 12, got {k}")
    if a != len(ACTION_NAMES):
        raise ValueError(f"action dimension must be 7, got {a}")
    if legal.shape != (n, a):
        raise ValueError(f"legal shape mismatch: {legal.shape} vs {(n, a)}")
    if not np.isfinite(probs).all():
        raise ValueError("nonfinite probability")
    if float(probs.min()) < -EPS:
        raise ValueError("negative probability")
    sums = probs.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("probability rows do not sum to one")
    illegal_mass = np.where(legal[None, :, :], 0.0, probs)
    if float(np.max(illegal_mass)) > 1e-7:
        raise ValueError("member probability mass on illegal action")
    if not np.all(legal.any(axis=1)):
        raise ValueError("state with no legal action")
    for name, mask in surfaces.items():
        if mask.shape != (n,):
            raise ValueError(f"{name} shape mismatch: {mask.shape} vs {(n,)}")


def _pairwise_rows(probs: np.ndarray) -> list[dict]:
    rows = []
    for i, j in itertools.combinations(range(probs.shape[0]), 2):
        tv = tv_rows(probs[i], probs[j])
        rows.append(
            {
                "member_a": i + 1,
                "member_b": j + 1,
                "mean_tv": float(tv.mean()),
                "median_tv": float(np.median(tv)),
                "p95_tv": _safe_p95(tv),
                "argmax_disagreement": float(
                    np.mean(np.argmax(probs[i], axis=1) != np.argmax(probs[j], axis=1))
                ),
            }
        )
    return rows


def _surface_delta(
    pa: np.ndarray,
    pb: np.ndarray,
    mask: np.ndarray,
    action: int,
) -> dict:
    d = np.abs(pa[mask, action] - pb[mask, action])
    return {
        "count": int(mask.sum()),
        "mean_probability_delta": float(d.mean()) if d.size else None,
        "statewise_p95_probability_delta": _safe_p95(d) if d.size else None,
        "mean_probability_a": float(pa[mask, action].mean()) if d.size else None,
        "mean_probability_b": float(pb[mask, action].mean()) if d.size else None,
    }


def _split_rows(
    probs: np.ndarray,
    surfaces: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for index, (left, right) in enumerate(_balanced_splits(probs.shape[0]), 1):
        a = probs[np.asarray(left)].mean(axis=0)
        b = probs[np.asarray(right)].mean(axis=0)
        tv = tv_rows(a, b)
        rows.append(
            {
                "split_index": index,
                "left_members": [x + 1 for x in left],
                "right_members": [x + 1 for x in right],
                "broad_mean_tv": float(tv.mean()),
                "broad_statewise_p95_tv": _safe_p95(tv),
                "broad_argmax_disagreement": float(
                    np.mean(np.argmax(a, axis=1) != np.argmax(b, axis=1))
                ),
                "surface_72o_allin": _surface_delta(
                    a, b, surfaces["surface_72o"], ALL_IN
                ),
                "surface_trips_plus_fold": _surface_delta(
                    a, b, surfaces["surface_trips_fold"], FOLD
                ),
                "surface_high_card_no_draw_allin": _surface_delta(
                    a, b, surfaces["surface_high_card_no_draw"], ALL_IN
                ),
            }
        )
    return rows


def _loo_rows(probs: np.ndarray) -> list[dict]:
    full = probs.mean(axis=0)
    rows = []
    for i in range(probs.shape[0]):
        keep = np.delete(probs, i, axis=0).mean(axis=0)
        tv = tv_rows(full, keep)
        rows.append(
            {
                "left_out_member": i + 1,
                "mean_tv": float(tv.mean()),
                "p95_tv": _safe_p95(tv),
                "argmax_disagreement": float(
                    np.mean(np.argmax(full, axis=1) != np.argmax(keep, axis=1))
                ),
            }
        )
    return rows


def _worst(rows: Iterable[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows]
    return max(vals) if vals else float("nan")


def _worst_surface(rows: Iterable[dict], surface: str, key: str) -> float:
    vals = [row[surface][key] for row in rows]
    vals = [float(x) for x in vals if x is not None]
    return max(vals) if vals else float("nan")


def evaluate(
    probs: np.ndarray,
    legal: np.ndarray,
    surface_72o: np.ndarray,
    surface_trips_fold: np.ndarray,
    surface_high_card_no_draw: np.ndarray,
    manifest: dict,
) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    legal = np.asarray(legal, dtype=bool)
    surfaces = {
        "surface_72o": np.asarray(surface_72o, dtype=bool),
        "surface_trips_fold": np.asarray(surface_trips_fold, dtype=bool),
        "surface_high_card_no_draw": np.asarray(surface_high_card_no_draw, dtype=bool),
    }
    _validate_input(probs, legal, surfaces)

    coverage_min = manifest["shared_fresh_evaluation"]["stability_state_bank"][
        "required_surfaces"
    ]
    coverage = {
        "preflop_72o_allin_opportunities": int(surfaces["surface_72o"].sum()),
        "postflop_trips_plus_fold_legal": int(surfaces["surface_trips_fold"].sum()),
        "postflop_high_card_no_immediate_draw_allin_legal": int(
            surfaces["surface_high_card_no_draw"].sum()
        ),
    }
    coverage_ok = {
        key: int(coverage[key]) >= int(coverage_min[key])
        for key in coverage_min
    }

    pairs = _pairwise_rows(probs)
    splits = _split_rows(probs, surfaces)
    loo = _loo_rows(probs)
    t = manifest["rebuild_stability_gate"]["thresholds"]

    summary = {
        "broad_worst_split_mean_tv": _worst(splits, "broad_mean_tv"),
        "broad_worst_split_statewise_p95_tv": _worst(
            splits, "broad_statewise_p95_tv"
        ),
        "leave_one_out_worst_mean_tv": _worst(loo, "mean_tv"),
        "preflop_72o_allin_worst_split_mean_probability_delta": _worst_surface(
            splits, "surface_72o_allin", "mean_probability_delta"
        ),
        "preflop_72o_allin_worst_split_statewise_p95_probability_delta":
            _worst_surface(
                splits,
                "surface_72o_allin",
                "statewise_p95_probability_delta",
            ),
        "trips_plus_fold_worst_split_mean_probability_delta": _worst_surface(
            splits, "surface_trips_plus_fold", "mean_probability_delta"
        ),
        "trips_plus_fold_worst_split_statewise_p95_probability_delta":
            _worst_surface(
                splits,
                "surface_trips_plus_fold",
                "statewise_p95_probability_delta",
            ),
        "high_card_no_draw_allin_worst_split_mean_probability_delta":
            _worst_surface(
                splits,
                "surface_high_card_no_draw_allin",
                "mean_probability_delta",
            ),
        "high_card_no_draw_allin_worst_split_statewise_p95_probability_delta":
            _worst_surface(
                splits,
                "surface_high_card_no_draw_allin",
                "statewise_p95_probability_delta",
            ),
    }
    criteria = {
        metric: bool(
            np.isfinite(summary[metric])
            and summary[metric] <= float(t[metric + "_max"])
        )
        for metric in summary
    }
    criteria["surface_coverage_minima"] = bool(all(coverage_ok.values()))

    ensemble = probs.mean(axis=0)
    member_argmax = np.argmax(probs, axis=2)
    modal = []
    for state in range(member_argmax.shape[1]):
        counts = np.bincount(member_argmax[:, state], minlength=len(ACTION_NAMES))
        modal.append(int(np.max(counts)))
    modal = np.asarray(modal)

    return {
        "schema": "SPINCORE_POST15315_REBUILD_STABILITY_RESULT_V1",
        "protocol_id": manifest["protocol_id"],
        "rebuild_count": int(probs.shape[0]),
        "state_count": int(probs.shape[1]),
        "action_names": list(ACTION_NAMES),
        "coverage": coverage,
        "coverage_minima": coverage_min,
        "coverage_ok": coverage_ok,
        "pairwise_member_dispersion": pairs,
        "balanced_half_splits": splits,
        "leave_one_out": loo,
        "ensemble_summary": {
            "mean_member_agreement_with_modal_argmax": float(
                np.mean(modal / float(probs.shape[0]))
            ),
            "ensemble_nonfinite": bool(not np.isfinite(ensemble).all()),
        },
        "stability_summary": summary,
        "precommitted_criteria": criteria,
        "rebuild_stability_pass": bool(all(criteria.values())),
        "interpretation": (
            "This gate measures rebuild-seed stability only. PASS does not imply "
            "strategic strength or authorize promotion; quality, safety, DC0 and "
            "fresh external-strength gates remain separate."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with np.load(args.input, allow_pickle=False) as z:
        required = {
            "probs",
            "legal",
            "surface_72o",
            "surface_trips_fold",
            "surface_high_card_no_draw",
        }
        missing = sorted(required - set(z.files))
        if missing:
            raise SystemExit(f"missing NPZ arrays: {missing}")
        report = evaluate(
            z["probs"],
            z["legal"],
            z["surface_72o"],
            z["surface_trips_fold"],
            z["surface_high_card_no_draw"],
            manifest,
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("POST15315_REBUILD_STABILITY_" + (
        "PASS" if report["rebuild_stability_pass"] else "FAIL"
    ))
    print(f"report={args.report.resolve()}")
    return 0 if report["rebuild_stability_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
