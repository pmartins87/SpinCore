from __future__ import annotations

from r7_5_arch_reset_v1plus_phase2b3_root_feedback_decomposition import (
    REFERENCE_NATIVE_TV,
    _mean_policy,
    _mean_values,
    _target,
    _target_pair_metrics,
    decision_from_pooled,
)


def _summary(mean: float) -> dict:
    return {"count": 1, "mean": float(mean), "p50": float(mean), "p95": float(mean), "max": float(mean)}


def _mode(tv: float) -> dict:
    return {
        "pair_count": 1,
        "target_mean_abs_diff": _summary(0.1),
        "legal_sign_disagreement_fraction": _summary(0.1),
        "regret_matching_policy_tv": _summary(tv),
        "dominant_legal_action_mismatch_rate": 0.0,
    }


def _pooled(common_sigma: float, common_values: float) -> dict:
    return {
        "NATIVE": _mode(REFERENCE_NATIVE_TV),
        "COMMON_ROOT_SIGMA": _mode(common_sigma),
        "COMMON_ACTION_VALUES": _mode(common_values),
        "root_sigma_tv": _summary(0.4),
        "root_action_value_mean_abs_diff": _summary(0.2),
        "crossed_root_sigma_step_tv": _summary(0.2),
        "crossed_downstream_value_step_tv": _summary(0.2),
    }


def test_target_centering() -> None:
    legal = (0, 1, 2, 9)
    values = (1.0, 2.0, 3.0, 0, 0, 0, 0, 0, 0, 4.0)
    sigma = (0.1, 0.2, 0.3, 0, 0, 0, 0, 0, 0, 0.4)
    target = _target(values, sigma, legal)
    baseline = 3.0
    assert abs(target[0] + 2.0) < 1e-12
    assert abs(target[1] + 1.0) < 1e-12
    assert abs(target[2] - 0.0) < 1e-12
    assert abs(target[9] - 1.0) < 1e-12
    assert abs(sum(sigma[s] * target[s] for s in legal)) < 1e-12
    assert baseline == sum(sigma[s] * values[s] for s in legal)


def test_counterfactual_helpers() -> None:
    legal = (0, 1, 2, 9)
    a = (0.1, 0.2, 0.3, 0, 0, 0, 0, 0, 0, 0.4)
    b = (0.4, 0.3, 0.2, 0, 0, 0, 0, 0, 0, 0.1)
    bar = _mean_policy(a, b, legal)
    assert abs(sum(bar[s] for s in legal) - 1.0) < 1e-12
    va = tuple(float(i) for i in range(10))
    vb = tuple(float(10 - i) for i in range(10))
    vm = _mean_values(va, vb)
    assert all(abs(value - 5.0) < 1e-12 for value in vm)
    metrics = _target_pair_metrics(_target(va, a, legal), _target(vb, b, legal), legal)
    assert 0.0 <= metrics["regret_matching_policy_tv"] <= 1.0 + 1e-12
    assert 0.0 <= metrics["legal_sign_disagreement_fraction"] <= 1.0


def test_decision_root_dominant() -> None:
    d = decision_from_pooled(_pooled(common_sigma=0.20, common_values=0.36))
    assert d["classification"] == "ROOT_BASELINE_DOMINANT"
    assert d["root_sigma_removal_effect"]["material"] is True
    assert d["downstream_value_removal_effect"]["material"] is False


def test_decision_downstream_dominant() -> None:
    d = decision_from_pooled(_pooled(common_sigma=0.36, common_values=0.20))
    assert d["classification"] == "DOWNSTREAM_CONTINUATION_DOMINANT"


def test_decision_mixed() -> None:
    d = decision_from_pooled(_pooled(common_sigma=0.20, common_values=0.20))
    assert d["classification"] == "MIXED_ROOT_AND_DOWNSTREAM_FEEDBACK"
    assert d["training_pilot_precommit_allowed"] is False


def test_decision_unresolved() -> None:
    d = decision_from_pooled(_pooled(common_sigma=0.36, common_values=0.36))
    assert d["classification"] == "NONLINEAR_INTERACTION_OR_UNRESOLVED"


if __name__ == "__main__":
    test_target_centering()
    test_counterfactual_helpers()
    test_decision_root_dominant()
    test_decision_downstream_dominant()
    test_decision_mixed()
    test_decision_unresolved()
    print("R7.5 architecture-reset Phase2B3 root-feedback synthetic tests PASS")
