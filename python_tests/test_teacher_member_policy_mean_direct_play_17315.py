from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))

import validate_teacher_member_policy_mean_17315 as gate


def _summary(mean,ge80=0,opps=100,flags=None):
    return {
        "flag_counts":flags or {},
        "opportunity_policy":{
            "POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW":{
                "opportunities":opps,
                "ge_80pct_count":ge80,
                "mean_target_probability":0.1,
            },
            "PREFLOP_DEEP_72O_JAM":{
                "opportunities":10,
                "mean_target_probability":0.33,
            },
            "POSTFLOP_TRIPS_PLUS_FOLD":{
                "opportunities":10,
                "mean_target_probability":0.10,
            },
        },
    }


def test_direct_play_gate_passes_noninferior_safe_candidate():
    A=_summary(0.0,ge80=4,opps=1000)
    B=_summary(0.0,ge80=1,opps=1000)
    paired={"ci95_low":-0.25}
    d=gate.adjudicate(A,B,paired)
    assert d["status"]=="PASS"
    assert d["preferred_aggregation"]=="B_MEMBER_POLICY_MEAN"
    assert d["production_change_authorized"] is False
    assert d["long_training_unblocked"] is False
    assert d["next_gate"]=="FALLBACK_MODEL_SIGN_ERROR_GATE"


def test_direct_play_gate_rejects_ev_regression():
    A=_summary(0.0,ge80=0,opps=1000)
    B=_summary(0.0,ge80=0,opps=1000)
    d=gate.adjudicate(A,B,{"ci95_low":-0.51})
    assert d["status"]=="FAIL"
    assert d["checks"]["ev_noninferiority"] is False


def test_direct_play_gate_rejects_critical_absurdity():
    A=_summary(0.0,ge80=0,opps=1000)
    B=_summary(0.0,ge80=0,opps=1000,flags={"PREFLOP_AA_FOLD":1})
    d=gate.adjudicate(A,B,{"ci95_low":0.0})
    assert d["status"]=="FAIL"
    assert d["checks"]["zero_aa_fold"] is False


def test_direct_play_gate_rejects_high_card_extreme_rate():
    A=_summary(0.0,ge80=4,opps=1000)
    B=_summary(0.0,ge80=3,opps=1000)
    d=gate.adjudicate(A,B,{"ci95_low":0.0})
    assert d["status"]=="FAIL"
    assert d["checks"]["high_card_no_draw_ge80"] is False
