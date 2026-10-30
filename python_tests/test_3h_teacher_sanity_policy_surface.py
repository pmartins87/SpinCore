from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_teacher_sanity_17315 as audit
from spincore.deepcrusher_benchmark import DecisionTrace


def cid(rank:int,suit:int)->int:
    return (rank-2)*4+suit


def trace(*,hole,board=(),stacks=(400,400,0),policy_detail=None):
    return DecisionTrace(
        scenario_index=1,
        domain="THREE_HANDED",
        blind="10/20",
        lineup=("SPINCORE","DEEPCRUSHER","DEEPCRUSHER"),
        lineup_index=0,
        decision_index=7,
        actor=0,
        policy_id="SPINCORE",
        street=0 if not board else (1 if len(board)==3 else 2 if len(board)==4 else 3),
        visible_board_count=len(board),
        pot=100,
        current_bet=20,
        to_call=20,
        min_raise_to=40,
        max_raise_to=400,
        stacks=stacks,
        street_commitments=(20,20,20),
        total_commitments=(20,20,20),
        hole_cards=tuple(hole),
        board=tuple(board),
        action_type=1,
        amount_to=20,
        policy_detail=policy_detail,
    )


def legal_detail(**probs):
    rows=[]
    slots={"FOLD":0,"CHECK_CALL":1,"ALL_IN":9}
    for name,p in probs.items():
        rows.append({"slot":slots[name],"name":name,"probability":float(p)})
    return {
        "selected_slot_name":"CHECK_CALL",
        "legal_actions":rows,
    }


def test_no_draw_high_card_opportunity_is_separate_surface():
    t=trace(
        hole=(cid(10,0),cid(9,1)),
        board=(cid(13,2),cid(5,3),cid(4,1)),
        policy_detail=legal_detail(FOLD=.1,CHECK_CALL=.6,ALL_IN=.3),
    )
    codes=audit._opportunity_codes(t)
    assert "POSTFLOP_DEEP_HIGH_CARD_JAM" in codes
    assert "POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW" in codes
    assert audit._policy_probability(t,"ALL_IN")==0.3


def test_draw_high_card_does_not_enter_no_draw_surface():
    t=trace(
        hole=(cid(8,1),cid(6,1)),
        board=(cid(14,3),cid(11,1),cid(9,1)),
        policy_detail=legal_detail(FOLD=.1,CHECK_CALL=.5,ALL_IN=.4),
    )
    codes=audit._opportunity_codes(t)
    assert "POSTFLOP_DEEP_HIGH_CARD_JAM" in codes
    assert "POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW" not in codes


def test_top_pair_fold_probability_surface_is_recorded_even_if_not_sampled():
    t=trace(
        hole=(cid(14,0),cid(8,1)),
        board=(cid(14,2),cid(13,0),cid(9,1)),
        policy_detail=legal_detail(FOLD=.37,CHECK_CALL=.63),
    )
    codes=audit._opportunity_codes(t)
    assert "POSTFLOP_TOP_PAIR_FOLD" in codes
    assert audit._policy_probability(t,"FOLD")==0.37


def test_opportunity_policy_row_preserves_scenario_id():
    t=trace(
        hole=(cid(10,0),cid(9,1)),
        board=(cid(13,2),cid(5,3),cid(4,1)),
        policy_detail=legal_detail(FOLD=.1,CHECK_CALL=.6,ALL_IN=.3),
    )
    row=audit._opportunity_policy_row(
        t,
        "POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW",
        0.3,
    )
    assert row["scenario"]==1
    assert row["lineup_index"]==0
    assert row["decision_index"]==7
