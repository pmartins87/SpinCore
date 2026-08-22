from __future__ import annotations

from collections import Counter

import r7_5_arch_reset_v1plus_advantage_forensic as forensic


def test_counter_tv() -> None:
    same = forensic._counter_tv(Counter({"A": 2, "B": 2}), Counter({"A": 5, "B": 5}))
    assert abs(float(same["tv"])) < 1e-15
    disjoint = forensic._counter_tv(Counter({"A": 4}), Counter({"B": 7}))
    assert abs(float(disjoint["tv"]) - 1.0) < 1e-15


def test_jaccard() -> None:
    a = {b"a", b"b", b"c"}
    b = {b"b", b"c", b"d"}
    row = forensic._jaccard(a, b)
    assert row["intersection"] == 2
    assert row["union"] == 4
    assert abs(float(row["jaccard"]) - 0.5) < 1e-15


def test_target_policy_and_shared_group_comparison() -> None:
    mask = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    left = forensic._new_coarse_aggregate(mask)
    right = forensic._new_coarse_aggregate(mask)
    forensic._add_coarse_aggregate(left, (1.0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0), 1.0)
    forensic._add_coarse_aggregate(right, (-1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0), 1.0)
    key = (b"x" * 32, mask)
    result = forensic._shared_group_comparison({key: left}, {key: right}, include_top=False)
    assert result["shared_groups"] == 1
    assert abs(float(result["target_derived_regret_matching_policy_tv"]["mean"]) - 1.0) < 1e-15
    assert abs(float(result["positive_sign_disagreement_fraction"]["mean"]) - 1.0) < 1e-15


def test_target_policy_uniform_when_no_positive_advantage() -> None:
    mask = (1, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    policy = forensic._target_policy((-2.0, 0.0, -1.0, 0, 0, 0, 0, 0, 0, 0), mask)
    assert abs(policy[0] - 0.5) < 1e-15
    assert abs(policy[2] - 0.5) < 1e-15
    assert abs(sum(policy) - 1.0) < 1e-15


def test_frozen_contract_constants() -> None:
    assert forensic.DOMAIN == "THREE_HANDED"
    assert forensic.REPRESENTATION.endswith("EXACT_STRUCTURED_HISTORY_FINAL")
    assert forensic.EXPECTED_STAGE_INDEX == 12
    assert forensic.EXPECTED_ROOTS == 768
    assert forensic.EXPECTED_ADV_CAPACITY == 100_000
    assert forensic.POLICY_COUNT == 1024
    assert "exact_observation" in forensic.PROJECTION_NAMES
    assert "geometry_plus_v1_like_history" in forensic.PROJECTION_NAMES


def main() -> int:
    test_counter_tv()
    test_jaccard()
    test_target_policy_and_shared_group_comparison()
    test_target_policy_uniform_when_no_positive_advantage()
    test_frozen_contract_constants()
    print("R7.5 architecture-reset Advantage forensic synthetic tests PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
