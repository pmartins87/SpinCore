from __future__ import annotations

from r7_5_arch_reset_v1plus_phase2b4_downstream_street_localization import (
    REFERENCE_COMMON_ROOT_SIGMA_TV,
    decision_from_pooled,
)


def _metric(tv: float) -> dict:
    return {
        "pair_count": 240,
        "root_action_value_mean_abs_diff": {"count": 240, "mean": 0.01, "p50": 0.01, "p95": 0.02, "max": 0.03},
        "target_mean_abs_diff": {"count": 240, "mean": 0.01, "p50": 0.01, "p95": 0.02, "max": 0.03},
        "legal_sign_disagreement_fraction": {"count": 240, "mean": 0.1, "p50": 0.0, "p95": 0.5, "max": 1.0},
        "regret_matching_policy_tv": {"count": 240, "mean": float(tv), "p50": float(tv), "p95": float(tv), "max": float(tv)},
        "dominant_legal_action_mismatch_rate": 0.1,
    }


def _pooled(river: float, turn: float, flop: float, preflop: float) -> dict:
    return {
        "NATIVE_CONTINUATION": _metric(REFERENCE_COMMON_ROOT_SIGMA_TV),
        "COMMON_FROM_RIVER": _metric(river),
        "COMMON_FROM_TURN": _metric(turn),
        "COMMON_FROM_FLOP": _metric(flop),
        "COMMON_FROM_PREFLOP": _metric(preflop),
    }


def test_postflop_dominant() -> None:
    d = decision_from_pooled(_pooled(0.28, 0.24, 0.20, 0.18))
    assert d["classification"] == "POSTFLOP_FEEDBACK_DOMINANT"
    assert d["high_residual_after_full_policy_commonization"] is True
    assert d["next_route"] == "DIAGNOSE_RESIDUAL_AFTER_FULL_POLICY_COMMONIZATION"


def test_mixed_pref_post() -> None:
    d = decision_from_pooled(_pooled(0.29, 0.24, 0.20, 0.08))
    assert d["classification"] == "PREFLOP_AND_POSTFLOP_FEEDBACK_MIXED"
    assert d["preflop_increment_material"] is True
    assert d["high_residual_after_full_policy_commonization"] is False


def test_preflop_dominant() -> None:
    d = decision_from_pooled(_pooled(0.32, 0.31, 0.30, 0.08))
    assert d["classification"] == "PREFLOP_DOWNSTREAM_FEEDBACK_DOMINANT"
    assert d["preflop_increment_material"] is True


def test_weak() -> None:
    d = decision_from_pooled(_pooled(0.32, 0.31, 0.30, 0.29))
    assert d["classification"] == "DEPTH_LOCALIZATION_WEAK_OR_UNRESOLVED"
    assert d["high_residual_after_full_policy_commonization"] is True


def test_reference_frozen() -> None:
    assert abs(REFERENCE_COMMON_ROOT_SIGMA_TV - 0.32770276958712846) < 1e-15


if __name__ == "__main__":
    test_postflop_dominant()
    test_mixed_pref_post()
    test_preflop_dominant()
    test_weak()
    test_reference_frozen()
    print("R7.5 architecture-reset Phase2B4 street-localization synthetic tests PASS")
