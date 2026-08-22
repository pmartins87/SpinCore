from __future__ import annotations

import r7_5_arch_reset_v1plus_phase2b0_policy_algebra_screen as screen


def test_identical_members_match_control() -> None:
    legal = (0, 1, 2)
    row = (0.2, -0.1, 0.1, 0, 0, 0, 0, 0, 0, 0)
    control, candidate, stats = screen.raw_mean_then_regret_match_same_epsilon([row, row, row, row], legal)
    assert max(abs(a - b) for a, b in zip(control, candidate)) < 1e-15
    assert abs(float(stats["epsilon"])) < 1e-15


def test_raw_mean_is_applied_before_regret_matching() -> None:
    legal = (0, 1)
    rows = [
        (100.0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0),
        (-1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0),
        (-1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0),
        (-1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0),
    ]
    control, candidate, stats = screen.raw_mean_then_regret_match_same_epsilon(rows, legal)
    # RM-then-mean treats the four members as 1 vote for slot 0 and 3 votes for
    # slot 1 before damping.  Raw-mean-then-RM preserves the much larger raw
    # magnitude of the first member before the nonlinearity, so the two paths
    # must differ while sharing the exact same epsilon scalar.
    assert screen._policy_tv(control, candidate) > 0.1
    assert stats["raw_mean_positive_legal_count"] == 2
    assert candidate[0] > candidate[1]
    assert abs(sum(control) - 1.0) < 1e-15
    assert abs(sum(candidate) - 1.0) < 1e-15


def test_zero_raw_mean_falls_back_to_uniform() -> None:
    legal = (0, 1)
    rows = [
        (1.0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0),
        (-1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0),
        (1.0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0),
        (-1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0),
    ]
    _control, candidate, stats = screen.raw_mean_then_regret_match_same_epsilon(rows, legal)
    assert abs(candidate[0] - 0.5) < 1e-15
    assert abs(candidate[1] - 0.5) < 1e-15
    assert stats["raw_mean_positive_legal_count"] == 0


def test_same_epsilon_preserves_legal_mass() -> None:
    legal = (0, 2, 9)
    rows = [
        (0.2, 0, -0.1, 0, 0, 0, 0, 0, 0, 0.3),
        (-0.1, 0, 0.4, 0, 0, 0, 0, 0, 0, 0.1),
        (0.1, 0, 0.2, 0, 0, 0, 0, 0, 0, -0.1),
        (0.0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.2),
    ]
    control, candidate, stats = screen.raw_mean_then_regret_match_same_epsilon(rows, legal)
    assert 0.0 <= float(stats["epsilon"]) <= screen.EPSILON_CAP
    assert abs(sum(candidate) - 1.0) < 1e-15
    assert all(candidate[index] == 0.0 for index in range(10) if index not in legal)
    assert abs(sum(control) - 1.0) < 1e-15


def test_frozen_constants() -> None:
    assert screen.DOMAIN == "THREE_HANDED"
    assert screen.REPRESENTATION == "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL"
    assert screen.SOURCE_EXECUTION_SHA == "4bfa55d69029cd69536fa6dbfcadd162719cb887"
    assert screen.EXPECTED_ROOTS == 768
    assert screen.EXPECTED_STAGE_INDEX == 12
    assert screen.ENSEMBLE_SIZE == 4
    assert screen.POLICY_COUNT == 1024
    assert screen.EPSILON_SCALE == 1.75
    assert screen.EPSILON_CAP == 0.5


def main() -> int:
    test_identical_members_match_control()
    test_raw_mean_is_applied_before_regret_matching()
    test_zero_raw_mean_falls_back_to_uniform()
    test_same_epsilon_preserves_legal_mass()
    test_frozen_constants()
    print("R7.5 architecture-reset Phase2B0 policy-algebra synthetic tests PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
