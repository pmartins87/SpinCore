from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))

import diagnose_3h_teacher_raw_advantages_17315 as rawdiag


def test_raw_mean_can_create_extreme_action_without_member_majority():
    members=[]
    for _ in range(7):
        members.append({"FOLD":-1.0,"CHECK_CALL":1.0,"ALL_IN":-1.0})
    members.append({"FOLD":-1.0,"CHECK_CALL":-8.0,"ALL_IN":8.0})
    detail={
        "selected_slot_name":"ALL_IN",
        "legal_slots":[0,1,9],
        "legal_names":["FOLD","CHECK_CALL","ALL_IN"],
        "mean_raw_advantages":{
            "FOLD":-1.0,
            "CHECK_CALL":-0.125,
            "ALL_IN":0.125,
        },
        "member_raw_advantages":members,
        "sigma":{"FOLD":0.0,"CHECK_CALL":0.0,"ALL_IN":1.0},
    }
    m=rawdiag._member_policy_mechanism(detail)
    assert m["ensemble_selected_probability"]==1.0
    assert m["member_selected_policy_argmax_count"]==1
    assert m["ensemble_selected_is_sole_positive_mean_raw"] is True
    assert m["extreme_with_no_member_majority"] is True
    assert m["raw_mean_amplification_ge_25pp"] is True
    assert m["member_policy_mixture_selected_probability"] < 0.25


def test_consensus_extreme_is_not_labeled_no_member_majority():
    members=[
        {"FOLD":-1.0,"CHECK_CALL":-0.1,"ALL_IN":1.0}
        for _ in range(8)
    ]
    detail={
        "selected_slot_name":"ALL_IN",
        "legal_slots":[0,1,9],
        "legal_names":["FOLD","CHECK_CALL","ALL_IN"],
        "mean_raw_advantages":{"FOLD":-1.0,"CHECK_CALL":-0.1,"ALL_IN":1.0},
        "member_raw_advantages":members,
        "sigma":{"FOLD":0.0,"CHECK_CALL":0.0,"ALL_IN":1.0},
    }
    m=rawdiag._member_policy_mechanism(detail)
    assert m["member_selected_policy_argmax_count"]==8
    assert m["extreme_with_no_member_majority"] is False
    assert m["member_policy_mixture_selected_probability"]==1.0
    assert abs(m["ensemble_minus_member_mixture_probability"]) < 1e-12


def test_mechanism_summary_keeps_flag_categories_separate():
    base={
        "ensemble_selected_probability":1.0,
        "member_policy_mixture_selected_probability":0.125,
        "ensemble_minus_member_mixture_probability":0.875,
        "member_selected_positive_count":1,
        "member_selected_policy_argmax_count":1,
        "ensemble_selected_is_sole_positive_mean_raw":True,
        "ensemble_extreme_probability_ge_80pct":True,
        "raw_mean_amplification_ge_25pp":True,
        "extreme_with_no_member_majority":True,
    }
    rows=[
        {"flag_code":"POSTFLOP_DEEP_HIGH_CARD_JAM","mechanism":dict(base)},
        {"flag_code":"PREFLOP_DEEP_72O_JAM","mechanism":dict(base)},
    ]
    s=rawdiag._mechanism_summary(rows)
    assert set(s)=={
        "POSTFLOP_DEEP_HIGH_CARD_JAM",
        "PREFLOP_DEEP_72O_JAM",
    }
    assert s["POSTFLOP_DEEP_HIGH_CARD_JAM"]["count"]==1
    assert s["PREFLOP_DEEP_72O_JAM"]["extreme_with_no_member_majority_count"]==1
