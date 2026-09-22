from spincore.deepcrusher_benchmark import DecisionTrace
from spincore.decision_sanity import evaluate_visible_hand, preflop_class, sanity_flags


def cid(rank: int, suit: int) -> int:
    return (rank - 2) * 4 + suit


def trace(*, hole, board=(), action_type=0, blind="10/20", stacks=(400,400,0), actor=0):
    return DecisionTrace(
        scenario_index=1,
        domain="TRUE_HEADS_UP",
        blind=blind,
        lineup=("SPINCORE","DEEPCRUSHER","DEAD"),
        lineup_index=0,
        decision_index=0,
        actor=actor,
        policy_id="SPINCORE",
        street=0 if not board else (1 if len(board)==3 else 2 if len(board)==4 else 3),
        visible_board_count=len(board),
        pot=60,
        current_bet=20,
        to_call=20,
        min_raise_to=40,
        max_raise_to=400,
        stacks=stacks,
        street_commitments=(20,20,0),
        total_commitments=(20,20,0),
        hole_cards=tuple(hole),
        board=tuple(board),
        action_type=action_type,
        amount_to=0,
    )


def test_preflop_class_and_aa_fold_flag():
    aa=(cid(14,0),cid(14,1))
    assert preflop_class(aa)=="AA"
    flags=sanity_flags(trace(hole=aa,action_type=0))
    assert [x.code for x in flags]==["PREFLOP_AA_FOLD"]


def test_72o_jam_only_flagged_at_nontrivial_depth():
    hand=(cid(7,0),cid(2,1))
    assert preflop_class(hand)=="72o"
    deep=sanity_flags(trace(hole=hand,action_type=5,stacks=(400,400,0)))
    shallow=sanity_flags(trace(hole=hand,action_type=5,stacks=(100,100,0)))
    assert [x.code for x in deep]==["PREFLOP_DEEP_72O_JAM"]
    assert shallow==()


def test_top_pair_fold_is_review_not_automatic_error():
    hole=(cid(14,0),cid(8,1))
    board=(cid(14,2),cid(13,0),cid(9,1))
    flags=sanity_flags(trace(hole=hole,board=board,action_type=0))
    assert [x.code for x in flags]==["POSTFLOP_TOP_PAIR_FOLD"]
    assert flags[0].severity=="REVIEW"
    assert flags[0].context["top_pair_kicker"]==8


def test_trips_and_full_house_fold_escalation():
    trips_hole=(cid(7,0),cid(7,1))
    trips_board=(cid(7,2),cid(13,0),cid(2,0))
    flags=sanity_flags(trace(hole=trips_hole,board=trips_board,action_type=0))
    assert [x.code for x in flags]==["POSTFLOP_TRIPS_PLUS_FOLD"]

    fh_hole=(cid(7,0),cid(7,1))
    fh_board=(cid(7,2),cid(13,0),cid(13,1))
    category,_=evaluate_visible_hand(fh_hole,fh_board)
    assert category==6
    flags=sanity_flags(trace(hole=fh_hole,board=fh_board,action_type=0))
    assert [x.code for x in flags]==["POSTFLOP_MONSTER_FOLD"]
    assert flags[0].severity=="CRITICAL"
