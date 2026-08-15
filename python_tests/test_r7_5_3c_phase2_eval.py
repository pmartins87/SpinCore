from __future__ import annotations

import math

import numpy as np
import pytest

from spincore.r7_5_representation_v3_phase2_eval import (
    BLOCKED,
    DOMAIN_CONFLICT,
    H2,
    H3,
    INCONCLUSIVE,
    aligned_series_mean,
    bootstrap_mean_ci,
    classify_local_deviation_ci,
    classify_pairwise_crossplay_ci,
    combine_domain_directions,
    cross_seed_policy_stability,
    equal_group_stratified_bootstrap_mean_ci,
    paired_two_seed_representation_difference,
    resolve_frozen_winner,
    summarize_distribution,
    total_variation,
    validate_sentinel_vectors,
)


def test_total_variation_and_distribution_summary():
    assert total_variation((1.0, 0.0), (0.0, 1.0)) == pytest.approx(1.0)
    assert total_variation((0.5, 0.5), (0.5, 0.5)) == pytest.approx(0.0)
    summary = summarize_distribution((0.0, 0.5, 1.0))
    assert summary["mean"] == pytest.approx(0.5)
    assert summary["p50"] == pytest.approx(0.5)
    assert summary["p95"] == pytest.approx(0.95)
    assert summary["max"] == pytest.approx(1.0)


def test_cross_seed_policy_stability_recomputes_frozen_gate():
    left = [(0.7, 0.3), (0.1, 0.9)]
    right = [(0.65, 0.35), (0.15, 0.85)]
    report = cross_seed_policy_stability(left, right)
    assert report["count"] == 2
    assert report["mean"] == pytest.approx(0.05)
    assert report["gate_pass"] is True


def test_paired_two_seed_difference_averages_seeds_before_state_bootstrap():
    h2 = [(0.10, 0.20, 0.30), (0.30, 0.20, 0.10)]
    h3 = [(0.05, 0.10, 0.15), (0.25, 0.10, 0.05)]
    assert aligned_series_mean(h2) == pytest.approx((0.20, 0.20, 0.20))
    diff = paired_two_seed_representation_difference(h2_seed_values=h2, h3_seed_values=h3)
    assert diff == pytest.approx((-0.05, -0.10, -0.10))


def test_bootstrap_is_deterministic_for_same_precommitted_seed_key():
    values = tuple(np.linspace(-0.01, 0.02, 257))
    a = bootstrap_mean_ci(values, seed_parts=("SpinCore", "PHASE2", "localdev", "HU"))
    b = bootstrap_mean_ci(values, seed_parts=("SpinCore", "PHASE2", "localdev", "HU"))
    assert a == b
    assert a["unit_count"] == 257
    assert a["ci_low"] <= a["estimate"] <= a["ci_high"]


def test_equal_group_bootstrap_does_not_weight_by_group_size():
    groups = {
        "HU": (1.0, 1.0),
        "THREE_HANDED": tuple([3.0] * 10),
    }
    report = equal_group_stratified_bootstrap_mean_ci(
        groups,
        seed_parts=("SpinCore", "PHASE2", "pairwise"),
    )
    assert report["estimate"] == pytest.approx(2.0)
    assert report["group_means"] == pytest.approx({"HU": 1.0, "THREE_HANDED": 3.0})


def test_material_direction_classifiers_require_ci_beyond_floor():
    assert classify_local_deviation_ci(-0.003, -0.001) == H3
    assert classify_local_deviation_ci(0.001, 0.004) == H2
    assert classify_local_deviation_ci(-0.002, 0.0005) == INCONCLUSIVE
    assert classify_pairwise_crossplay_ci(0.001, 0.004) == H3
    assert classify_pairwise_crossplay_ci(-0.004, -0.001) == H2
    assert classify_pairwise_crossplay_ci(-0.0005, 0.002) == INCONCLUSIVE


def test_domain_direction_conflict_blocks_pooling_away_disagreement():
    assert combine_domain_directions({"TRUE_HEADS_UP": H3, "THREE_HANDED": H2}) == DOMAIN_CONFLICT
    assert combine_domain_directions({"TRUE_HEADS_UP": H3, "THREE_HANDED": INCONCLUSIVE}) == H3
    assert combine_domain_directions({"TRUE_HEADS_UP": INCONCLUSIVE, "THREE_HANDED": INCONCLUSIVE}) == INCONCLUSIVE


def test_sentinels_enforce_finite_illegal_zero_normalized_and_noncollapse():
    report = validate_sentinel_vectors(
        probabilities=[
            (0.5, 0.5, 0.0),
            (0.8, 0.2, 0.0),
            (0.0, 0.25, 0.75),
        ],
        legal_sets=[(0, 1), (0, 1), (1, 2)],
        logits=[
            (0.0, 0.0, -1.0e9),
            (1.0, 0.0, -1.0e9),
            (-1.0e9, 0.0, 1.0),
        ],
    )
    assert report["gate_pass"] is True
    assert report["distinct_multi_action_vectors"] is True

    bad = validate_sentinel_vectors(
        probabilities=[(0.5, 0.5, 0.0), (0.5, 0.5, 0.0)],
        legal_sets=[(0, 1), (0, 1)],
        logits=[(0.0, 0.0, -1.0e9), (0.0, 0.0, -1.0e9)],
    )
    assert bad["gate_pass"] is False
    assert "POLICY_COLLAPSE_OR_NO_DISTINCT_MULTI_ACTION_VECTOR" in bad["failures"]


def test_frozen_winner_resolution_exact_logic():
    assert resolve_frozen_winner(
        h2_hard_gate_pass=True,
        h3_hard_gate_pass=True,
        local_deviation_direction=H3,
        pairwise_crossplay_direction=INCONCLUSIVE,
    )["winner"] == H3

    assert resolve_frozen_winner(
        h2_hard_gate_pass=True,
        h3_hard_gate_pass=True,
        local_deviation_direction=H2,
        pairwise_crossplay_direction=H3,
    )["status"] == BLOCKED

    tiebreak = resolve_frozen_winner(
        h2_hard_gate_pass=True,
        h3_hard_gate_pass=True,
        local_deviation_direction=INCONCLUSIVE,
        pairwise_crossplay_direction=INCONCLUSIVE,
    )
    assert tiebreak["winner"] == H2
    assert "TIEBREAK" in tiebreak["reason"]

    guarded = resolve_frozen_winner(
        h2_hard_gate_pass=True,
        h3_hard_gate_pass=False,
        local_deviation_direction=H3,
        pairwise_crossplay_direction=INCONCLUSIVE,
    )
    assert guarded["status"] == BLOCKED
    assert guarded["winner"] is None


def test_nonfinite_policy_or_invalid_ci_rejected():
    with pytest.raises(ValueError):
        total_variation((math.nan, 0.0), (0.5, 0.5))
    with pytest.raises(ValueError):
        classify_local_deviation_ci(0.1, -0.1)
