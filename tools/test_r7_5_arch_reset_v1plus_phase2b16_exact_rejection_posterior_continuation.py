from __future__ import annotations

import math

import r7_5_arch_reset_v1plus_phase2b16_exact_rejection_posterior_continuation as b16


def main() -> int:
    assert b16.K == 64
    assert b16.BLOCKS == 2
    assert b16.MAX_WORKERS == 30
    assert b16.MAX_PROPOSALS_PER_TASK == 50000
    assert b16.DIRECT_TV_MAX == 0.24

    # Deterministic namespaces and open-unit uniforms.
    a = b16._proposal_seeds(2029384436, 11, 0, 3)
    b = b16._proposal_seeds(2029384436, 11, 0, 3)
    c = b16._proposal_seeds(2029384436, 11, 1, 3)
    assert a == b and a != c
    u = b16._open_unit(a[2])
    assert 0.0 < u < 1.0
    assert u == b16._open_unit(a[2])

    # Exact rejection rule: likelihood one always accepts; zero never accepts.
    assert b16._accept_log_likelihood(0.0, 0.999999999999)
    assert not b16._accept_log_likelihood(-math.inf, 1e-12)
    assert b16._accept_log_likelihood(math.log(0.5), 0.49)
    assert not b16._accept_log_likelihood(math.log(0.5), 0.51)

    # Mean target contract.
    rows = []
    for i in range(b16.K):
        row = [0.0] * 10
        row[0] = float(i)
        row[3] = float(2 * i)
        rows.append(tuple(row))
    mean = b16._mean_targets(rows)
    assert abs(mean[0] - 31.5) <= 1e-12
    assert abs(mean[3] - 63.0) <= 1e-12

    # Synthetic direct-posterior summary that clears all frozen gates.
    direct = {
        "count": 128,
        "valid_count": 128,
        "proposal_cap_hits": 0,
        "direct_tv": {"mean": 0.20},
        "direct_sign_disagreement_mean": 0.20,
        "direct_tail_rate_tv_ge_035": 0.25,
        "direct_dominant_mismatch_rate": 0.25,
    }
    by_behavior = {
        "1342191342": {"direct_tv": {"mean": 0.22}},
        "1801739323": {"direct_tv": {"mean": 0.19}},
    }
    by_region = {
        "PREFLOP_CONTINUATION_1": {"direct_tv": {"mean": 0.24}},
        "PREFLOP_CONTINUATION_2PLUS": {"direct_tv": {"mean": 0.25}},
    }
    b15_result = {
        "pooled": {
            "posterior_tv": {"mean": 0.3157316176926827},
            "posterior_sign_disagreement_mean": 0.25846354166666663,
        },
        "by_behavior_seed": {
            "1342191342": {"posterior_tv": {"mean": 0.34598231772639293}},
            "1801739323": {"posterior_tv": {"mean": 0.28548091765897243}},
        },
        "by_region": {
            "PREFLOP_CONTINUATION_1": {"posterior_tv": {"mean": 0.30315391026090316}},
            "PREFLOP_CONTINUATION_2PLUS": {"posterior_tv": {"mean": 0.3283093251244622}},
        },
    }
    decision = b16._decision(direct, by_behavior, by_region, b15_result)
    assert decision["screen_pass"] is True
    assert decision["classification"] == "EXACT_REJECTION_POSTERIOR_CONTINUATION_SUPPORTED"
    assert decision["training_authorized"] is False

    # A still-unstable exact posterior closes estimator-level tuning.
    bad = dict(direct)
    bad["direct_tv"] = {"mean": 0.29}
    decision = b16._decision(bad, by_behavior, by_region, b15_result)
    assert decision["screen_pass"] is False
    assert decision["classification"] == "EXACT_POSTERIOR_STILL_TOO_UNSTABLE_CLOSE_ESTIMATOR_REPAIR_PATH"

    # Proposal cap takes precedence over scientific interpretation.
    cap = {"count": 128, "valid_count": 127, "proposal_cap_hits": 1}
    decision = b16._decision(cap, by_behavior, by_region, b15_result)
    assert decision["classification"] == "EXACT_POSTERIOR_REJECTION_COMPUTE_INFEASIBLE"

    print("R7.5 architecture-reset Phase2B16 exact-rejection synthetic tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
