from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import evaluate_post15315_rebuild_stability as stability

MANIFEST = ROOT / "contracts" / "post15315_multirebuild_preregistration.json"


def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def synthetic_inputs(k=8, n=6000):
    legal = np.ones((n, 7), dtype=bool)
    base = np.asarray([0.15, 0.30, 0.10, 0.10, 0.10, 0.10, 0.15], dtype=np.float64)
    probs = np.broadcast_to(base, (k, n, 7)).copy()
    s72 = np.zeros(n, dtype=bool)
    s72[:1600] = True
    trips = np.zeros(n, dtype=bool)
    trips[1600:2900] = True
    hc = np.zeros(n, dtype=bool)
    hc[1000:6000] = True
    return probs, legal, s72, trips, hc


def test_identical_eight_member_rebuilds_pass_frozen_stability_gate():
    args = synthetic_inputs()
    report = stability.evaluate(*args, manifest())
    assert report["rebuild_count"] == 8
    assert len(report["pairwise_member_dispersion"]) == 28
    assert len(report["balanced_half_splits"]) == 35
    assert len(report["leave_one_out"]) == 8
    assert report["rebuild_stability_pass"] is True
    assert all(report["coverage_ok"].values())


def test_material_member_split_fails_without_member_dropping():
    probs, legal, s72, trips, hc = synthetic_inputs()
    # Four members choose a strongly Fold-heavy policy and four choose a strongly
    # ALL_IN-heavy policy. The full ensemble may look moderate, but the frozen
    # split-half gate must expose the rebuild instability.
    probs[:4, :, :] = 0.0
    probs[:4, :, 0] = 0.80
    probs[:4, :, 1] = 0.20
    probs[4:, :, :] = 0.0
    probs[4:, :, 1] = 0.20
    probs[4:, :, 6] = 0.80

    report = stability.evaluate(probs, legal, s72, trips, hc, manifest())
    assert report["rebuild_stability_pass"] is False
    assert report["precommitted_criteria"]["broad_worst_split_mean_tv"] is False
    assert (
        report["stability_summary"]["broad_worst_split_mean_tv"]
        > manifest()["rebuild_stability_gate"]["thresholds"][
            "broad_worst_split_mean_tv_max"
        ]
    )


def test_surface_coverage_is_a_hard_gate_not_a_posthoc_warning():
    probs, legal, s72, trips, hc = synthetic_inputs()
    s72[:] = False
    s72[:100] = True
    report = stability.evaluate(probs, legal, s72, trips, hc, manifest())
    assert report["coverage_ok"]["preflop_72o_allin_opportunities"] is False
    assert report["precommitted_criteria"]["surface_coverage_minima"] is False
    assert report["rebuild_stability_pass"] is False


def test_k12_extension_uses_exactly_100_deterministic_balanced_splits():
    probs, legal, s72, trips, hc = synthetic_inputs(k=12)
    a = stability.evaluate(probs, legal, s72, trips, hc, manifest())
    b = stability.evaluate(probs, legal, s72, trips, hc, manifest())
    assert len(a["balanced_half_splits"]) == 100
    assert a["balanced_half_splits"] == b["balanced_half_splits"]
    assert all(len(row["left_members"]) == 6 for row in a["balanced_half_splits"])
    assert all(len(row["right_members"]) == 6 for row in a["balanced_half_splits"])
