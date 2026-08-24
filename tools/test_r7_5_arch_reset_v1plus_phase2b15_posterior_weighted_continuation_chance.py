from __future__ import annotations

import math

import r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance as b15


def _targets():
    rows = []
    for r in range(b15.K):
        row = [0.0] * 10
        row[0] = float(r)
        row[1] = float(2 * r)
        rows.append(tuple(row))
    return rows


def main() -> int:
    assert b15.K == 64
    assert b15.BLOCKS == 2
    assert b15.ANCHORS_PER_REGION_PER_EVAL == 16
    assert b15.REGIONS == ("PREFLOP_CONTINUATION_1", "PREFLOP_CONTINUATION_2PLUS")
    assert b15.TARGET_ITERATION == 3
    assert b15.MAX_WORKERS == 30

    rows = _targets()
    arithmetic = b15._mean_targets(rows)
    weighted, stats = b15._self_normalized_mean(rows, [0.0] * b15.K)
    for a, w in zip(arithmetic, weighted):
        assert abs(a - w) <= 1e-12
    assert abs(stats["ess"] - 64.0) <= 1e-12
    assert abs(stats["max_normalized_weight"] - 1.0 / 64.0) <= 1e-12
    assert stats["zero_weight_count"] == 0

    concentrated = [-math.inf] * b15.K
    concentrated[7] = 0.0
    estimate, stats = b15._self_normalized_mean(rows, concentrated)
    assert abs(stats["ess"] - 1.0) <= 1e-12
    assert abs(stats["max_normalized_weight"] - 1.0) <= 1e-12
    assert stats["zero_weight_count"] == 63
    assert abs(estimate[0] - 7.0) <= 1e-12
    assert abs(estimate[1] - 14.0) <= 1e-12

    legal = (0, 1, 2)
    left = (1.0, 0.0, -1.0, 0, 0, 0, 0, 0, 0, 0)
    right = (0.0, 1.0, -1.0, 0, 0, 0, 0, 0, 0, 0)
    assert abs(b15._policy_tv(
        b15.regret_matching_policy(left, legal),
        b15.regret_matching_policy(right, legal),
    ) - 1.0) <= 1e-12
    assert abs(b15._sign_disagreement(left, right, legal) - (2.0 / 3.0)) <= 1e-12
    assert b15._dominant_mismatch(left, right, legal) == 1

    # Seed construction is deterministic, block-specific and independent of a
    # behavior-seed argument because no behavior seed is accepted.
    assert b15._chance_seeds(2029384436, 10, 0, 3) == b15._chance_seeds(2029384436, 10, 0, 3)
    assert b15._chance_seeds(2029384436, 10, 0, 3) != b15._chance_seeds(2029384436, 10, 1, 3)
    assert b15._traversal_seed(2029384436, 10) == b15._traversal_seed(2029384436, 10)

    # Decision hierarchy: healthy, material synthetic summary passes.
    pooled = {
        "ess": {"p50": 32.0, "p10": 20.0},
        "max_normalized_weight": {"p95": 0.10},
        "posterior_shift_tv": {"mean": 0.08},
        "tv_absolute_improvement": 0.05,
        "tv_relative_improvement": 0.20,
        "sign_absolute_improvement": 0.05,
        "sign_relative_improvement": 0.20,
        "tail_relative_improvement": 0.20,
    }
    behavior = {
        "1342191342": {"tv_absolute_improvement": 0.04},
        "1801739323": {"tv_absolute_improvement": 0.03},
    }
    regions = {
        "PREFLOP_CONTINUATION_1": {"tv_absolute_improvement": 0.02},
        "PREFLOP_CONTINUATION_2PLUS": {"tv_absolute_improvement": 0.03},
    }
    pair_count = len(b15.TRAINING_SEEDS) * len(b15.EVALUATION_SEEDS) * len(b15.REGIONS) * b15.ANCHORS_PER_REGION_PER_EVAL
    pairs = [{"posterior_tv": 0.2}] * pair_count
    decision = b15._decision(pairs, pooled, behavior, regions)
    assert decision["screen_pass"] is True
    assert decision["classification"] == "POSTERIOR_WEIGHTED_CONTINUATION_ESTIMATOR_SUPPORTED"
    assert decision["training_authorized"] is False
    assert decision["production_training_authorized"] is False
    assert decision["ready_for_tables"] is False

    # Weight degeneracy wins before scientific interpretation.
    bad = dict(pooled)
    bad["ess"] = {"p50": 10.0, "p10": 4.0}
    decision = b15._decision(pairs, bad, behavior, regions)
    assert decision["screen_pass"] is False
    assert decision["classification"] == "POSTERIOR_IMPORTANCE_WEIGHT_DEGENERACY"

    print("R7.5 architecture-reset Phase2B15 posterior-weighted continuation synthetic tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
