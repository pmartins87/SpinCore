from __future__ import annotations

from r7_5_arch_reset_v1plus_phase2b5_preflop_feedback_stabilization import (
    ARMS,
    CONTROL_ARM,
    REFERENCE_CONTROL_TV,
    REFERENCE_ORACLE_COMMON_PREFLOP_TV,
    _mix_uniform,
    _nonforced_preflop_count,
    decision_from_results,
)


def _obs(events):
    raw = bytearray(120)
    raw[:8] = b"SPNNIV3\x00"
    raw[116:120] = int(len(events)).to_bytes(4, "little")
    for actor_rel, street, action_type, forced in events:
        event = bytearray(20)
        event[0] = int(actor_rel)
        event[1] = int(street)
        event[2] = int(action_type)
        event[3] = int(forced)
        raw.extend(event)
    return bytes(raw)


def _mode(tv, mismatch=0.20):
    summary = {"count": 1, "mean": float(tv), "p50": float(tv), "p95": float(tv), "max": float(tv)}
    return {
        "pair_count": 1,
        "root_action_value_mean_abs_diff": summary,
        "target_mean_abs_diff": summary,
        "legal_sign_disagreement_fraction": summary,
        "regret_matching_policy_tv": summary,
        "dominant_legal_action_mismatch_rate": float(mismatch),
    }


def _synthetic(floor010_tv=0.20, floor025_tv=0.19, floor050_tv=0.18):
    tv = {
        CONTROL_ARM: REFERENCE_CONTROL_TV,
        "DEPTH_COMMON_GE_1": REFERENCE_ORACLE_COMMON_PREFLOP_TV,
        "DEPTH_COMMON_GE_2": 0.12,
        "DEPTH_COMMON_GE_3": 0.19,
        "DEPTH_COMMON_GE_4": 0.25,
        "DEPTH_COMMON_GE_5": 0.28,
        "DEPTH_COMMON_GE_6": 0.30,
        "UNIFORM_FLOOR_010": floor010_tv,
        "UNIFORM_FLOOR_025": floor025_tv,
        "UNIFORM_FLOOR_050": floor050_tv,
        "UNIFORM_FLOOR_075": 0.17,
        "UNIFORM_FLOOR_100": 0.06,
    }
    pooled = {arm: _mode(tv[arm]) for arm in ARMS}
    per_scenario = {}
    for i in range(15):
        per_scenario[str(i)] = {arm: _mode(tv[arm]) for arm in ARMS}
    return pooled, per_scenario


def main():
    observation = _obs(
        [
            (1, 0, 1, 1),
            (2, 0, 2, 1),
            (0, 0, 3, 0),
            (1, 1, 4, 0),
            (2, 0, 5, 0),
        ]
    )
    assert _nonforced_preflop_count(observation) == 2

    legal = (0, 1, 4, 9)
    policy = (0.2, 0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.4)
    mixed0 = _mix_uniform(policy, legal, 0.0)
    assert all(abs(a - b) < 1e-15 for a, b in zip(mixed0, policy))
    mixed1 = _mix_uniform(policy, legal, 1.0)
    assert all(abs(mixed1[s] - 0.25) < 1e-15 for s in legal)
    assert abs(sum(mixed1) - 1.0) < 1e-15

    pooled, per_scenario = _synthetic()
    decision = decision_from_results(pooled, per_scenario)
    assert decision["classification"] == "MILD_PREFLOP_DAMPING_CANDIDATE"
    assert decision["selected_mild_candidate"] == "UNIFORM_FLOOR_010"
    assert decision["small_training_pilot_precommit_allowed"] is True
    assert decision["largest_positive_depth_increment"] in {
        "DELTA1", "DELTA2", "DELTA3", "DELTA4", "DELTA5", "DEEPER_THAN_6"
    }

    pooled, per_scenario = _synthetic(floor010_tv=0.29, floor025_tv=0.27, floor050_tv=0.20)
    decision = decision_from_results(pooled, per_scenario)
    assert decision["classification"] == "HEAVY_DAMPING_REQUIRED_NO_PILOT"
    assert decision["small_training_pilot_precommit_allowed"] is False

    print("R7.5 architecture-reset Phase2B5 preflop stabilization synthetic tests PASS")


if __name__ == "__main__":
    main()
