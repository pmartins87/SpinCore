from __future__ import annotations

import math

import r7_5_arch_reset_v1plus_phase2c0_structural_reach_factorization as c0


def main() -> int:
    hands = c0._ordered_hands((0, 1))
    assert len(hands) == 2450
    assert len(set(hands)) == 2450
    assert all(a != b and a not in (0, 1) and b not in (0, 1) for a, b in hands)

    ones = [1.0] * len(hands)
    stats = c0._joint_stats(hands, ones, ones)
    expected = 50 * 49 * 48 * 47
    assert stats["positive_joint_assignments"] == expected
    assert abs(stats["normalizer"] - float(expected)) <= 1e-9
    assert abs(stats["effective_joint_support"] - float(expected)) <= 1e-6

    half = [1.0 if i % 2 == 0 else 0.0 for i in range(len(hands))]
    sparse = c0._joint_stats(hands, half, half)
    assert sparse["normalizer"] > 0.0
    assert 0.0 < sparse["effective_joint_support"] <= sparse["positive_joint_assignments"]

    assert c0._mix64(1, 2, 3) == c0._mix64(1, 2, 3)
    assert c0._mix64(1, 2, 3) != c0._mix64(1, 2, 4)
    assert c0.ANCHORS_PER_REGION_PER_EVAL == 2
    assert c0.FACTORIZATION_CHECKS == 128
    assert c0.FILLER_CHECKS_PER_SEAT == 32
    assert c0.TOL == 1e-12
    assert c0.B16_RESULT_SHA256 == "3b5e71c3cc92ed530589877f6790333b1f94b579bb39e7c687082787693d958c"

    print("R7.5 architecture-reset Phase2C0 structural reach-factorization synthetic tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
