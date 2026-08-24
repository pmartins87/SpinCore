from __future__ import annotations

import math

import r7_5_arch_reset_v1plus_phase2c0_structural_reach_factorization as c0
import r7_5_arch_reset_v1plus_phase2c1_exact_range_reach_solver_prototype as c1


def main() -> int:
    assert c1.MAX_WORKERS == 16
    assert c1.DIRECT_CHECKS_PER_OPPONENT == 128
    assert c1.TOL == 1e-12
    assert c1.MAX_TABLE_POLICY_EVALUATIONS == 4902
    assert c1.MAX_RAW_REACH_BYTES == 39200
    assert c1.C0_RESULT_SHA256 == "55e83be4fd8776e0fcdb63e7d4400ed05aff8c48213898ad8f1abe3713a35876"

    hands = c0._ordered_hands((0, 1))
    assert len(hands) == 2450
    ones = [1.0] * len(hands)
    stats = c0._joint_stats(hands, ones, ones)
    # With two fixed actor cards, ordered opponent A has 50*49 choices and,
    # after card removal, opponent B has 48*47 choices.
    exact_joint = 50 * 49 * 48 * 47
    assert stats["normalizer"] == float(exact_joint)
    assert stats["positive_joint_assignments"] == exact_joint
    assert abs(stats["effective_joint_support"] - float(exact_joint)) <= 1e-9

    assert c1._reach_storage_bytes(2, 2450) == 39200
    assert c1._relative_error(10.0, 10.0) == 0.0
    assert abs(c1._relative_error(10.0 + 1e-10, 10.0) - 1e-11) <= 1e-15
    assert math.isinf(c1._relative_error(math.inf, 1.0))

    digest_a = c1._table_sha256([0.0, 1.0, 2.0])
    digest_b = c1._table_sha256([0.0, 1.0, 2.0])
    digest_c = c1._table_sha256([0.0, 1.0, 3.0])
    assert digest_a == digest_b
    assert digest_a != digest_c

    ref = {
        "rows": [
            {
                "behavior_seed": 1,
                "evaluation_seed": 2,
                "state_index": 3,
                "region": "PREFLOP_CONTINUATION_1",
                "normalizer": 1.0,
            }
        ]
    }
    task = {
        "behavior_seed": 1,
        "evaluation_seed": 2,
        "state_index": 3,
        "region": "PREFLOP_CONTINUATION_1",
    }
    assert c1._reference_row(ref, task)["normalizer"] == 1.0

    print("R7.5 architecture-reset Phase2C1 exact range/reach synthetic tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
