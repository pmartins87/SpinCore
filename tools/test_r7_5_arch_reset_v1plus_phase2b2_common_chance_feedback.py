from __future__ import annotations

from r7_5_arch_reset_v1plus_phase2b2_common_chance_feedback import (
    REFERENCE_CHANCE_K1_TV,
    decision_from_pooled,
    paired_k_metrics,
)


def _summary(mean: float) -> dict:
    return {
        "pair_count": 1,
        "target_mean_abs_diff": {"count": 1, "mean": 0.1, "p50": 0.1, "p95": 0.1, "max": 0.1},
        "legal_sign_disagreement_fraction": {"count": 1, "mean": 0.1, "p50": 0.1, "p95": 0.1, "max": 0.1},
        "regret_matching_policy_tv": {"count": 1, "mean": float(mean), "p50": float(mean), "p95": float(mean), "max": float(mean)},
        "dominant_legal_action_mismatch_rate": 0.0,
    }


def _pooled(common: float, independent: float) -> dict:
    return {
        "COMMON_TRAVERSAL_RNG": {f"K{k}": _summary(common) for k in (1, 2, 4, 8, 16)},
        "INDEPENDENT_TRAVERSAL_RNG": {f"K{k}": _summary(independent) for k in (1, 2, 4, 8, 16)},
    }


def test_paired_k_metrics() -> None:
    legal = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    a = []
    b = []
    for i in range(16):
        a.append((1.0 + i, -1.0 - i, 0, 0, 0, 0, 0, 0, 0, 0))
        b.append((0.5 + i, -0.5 - i, 0, 0, 0, 0, 0, 0, 0, 0))
    k1 = paired_k_metrics(a, b, legal, 1)
    k4 = paired_k_metrics(a, b, legal, 4)
    k16 = paired_k_metrics(a, b, legal, 16)
    assert len(k1) == 16
    assert len(k4) == 4
    assert len(k16) == 1
    assert all(abs(row["target_mean_abs_diff"] - 0.5) < 1e-12 for row in k1)
    assert all(row["legal_sign_disagreement_fraction"] == 0.0 for row in k1)
    assert all(abs(row["regret_matching_policy_tv"]) < 1e-12 for row in k1)


def test_decision_shared_support_pass() -> None:
    decision = decision_from_pooled(_pooled(common=0.20, independent=0.25))
    assert decision["shared_support_gate_pass"] is True
    assert decision["classification"] == "COMMON_CHANCE_SUPPORT_MATERIALLY_SUPPORTED"
    assert decision["next_route"] == "PRECOMMIT_SMALL_SHARED_CHANCE_SUPPORT_TRAINING_PILOT"
    assert decision["small_training_pilot_precommit_allowed"] is True


def test_decision_feedback_dominant() -> None:
    decision = decision_from_pooled(_pooled(common=0.44, independent=0.45))
    assert decision["shared_support_gate_pass"] is False
    assert decision["classification"] == "BEHAVIOR_FEEDBACK_REMAINS_DOMINANT_ON_COMMON_CHANCE"
    assert decision["small_training_pilot_precommit_allowed"] is False


def test_decision_mixed() -> None:
    # Large reduction, but still above the absolute <=0.35 shared-support gate.
    decision = decision_from_pooled(_pooled(common=0.37, independent=0.36))
    assert decision["shared_support_gate_pass"] is False
    assert decision["classification"] == "MIXED_CHANCE_SUPPORT_AND_FEEDBACK"
    assert decision["next_route"] == "LOCALIZE_REMAINING_FEEDBACK_BEFORE_TRAINING"


def test_reference_frozen() -> None:
    assert abs(REFERENCE_CHANCE_K1_TV - 0.5153716032136447) < 1e-15


if __name__ == "__main__":
    test_paired_k_metrics()
    test_decision_shared_support_pass()
    test_decision_feedback_dominant()
    test_decision_mixed()
    test_reference_frozen()
    print("R7.5 architecture-reset Phase2B2 common-chance synthetic tests PASS")
