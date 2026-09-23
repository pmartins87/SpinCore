from types import SimpleNamespace

from spincore.deepcrusher_action import openppl_history_origin, translate_openppl_decision
from spincore.openppl_program import DirectAction, ReturnValue


def snap(**overrides):
    data = dict(
        legal_fold=True,
        legal_check=False,
        legal_call=True,
        legal_bet=False,
        legal_raise=True,
        legal_all_in=True,
        min_raise_to=80,
        max_raise_to=300,
        current_bet=40,
        pot=100,
        to_call=20,
        street_commitments=(20, 40, 0),
    )
    data.update(overrides)
    return SimpleNamespace(**data)


def test_positive_return_is_final_betsize_in_big_blinds():
    action = translate_openppl_decision(
        ReturnValue(5.0),
        public=snap(),
        actor=0,
        big_blind_chips=20,
    )
    assert (action.action_type, action.amount_to) == (4, 100)


def test_raise_to_uses_final_wager_and_clamps_to_legal_minmax():
    low = translate_openppl_decision(
        DirectAction("RaiseTo", 3.0, "bb_expression"),
        public=snap(),
        actor=0,
        big_blind_chips=20,
    )
    high = translate_openppl_decision(
        DirectAction("RaiseTo", 100.0, "bb_expression"),
        public=snap(),
        actor=0,
        big_blind_chips=20,
    )
    assert (low.action_type, low.amount_to) == (4, 80)
    assert (high.action_type, high.amount_to) == (4, 300)


def test_raise_by_bb_adds_to_ncall_target():
    action = translate_openppl_decision(
        DirectAction("RaiseBy", 2.5, "bb_expression"),
        public=snap(),
        actor=0,
        big_blind_chips=20,
    )
    # ncall target = hero current 20 + call 20 = 40; +2.5bb=50 => 90.
    assert (action.action_type, action.amount_to) == (4, 90)


def test_raise_by_percent_matches_openholdem_pot_after_call_formula():
    action = translate_openppl_decision(
        DirectAction("RaiseBy", 0.50, "pot_fraction"),
        public=snap(),
        actor=0,
        big_blind_chips=20,
    )
    # hero bet 20 + call 20 + 50% * (pot 100 + call 20) = 100.
    assert (action.action_type, action.amount_to) == (4, 100)


def test_fixed_halfpot_code_and_small_negative_encoding_match():
    fixed = translate_openppl_decision(
        ReturnValue(-1000005),
        public=snap(),
        actor=0,
        big_blind_chips=20,
    )
    encoded = translate_openppl_decision(
        ReturnValue(-0.50),
        public=snap(),
        actor=0,
        big_blind_chips=20,
    )
    assert fixed == encoded
    assert (fixed.action_type, fixed.amount_to) == (4, 100)


def test_betmax_is_allin_and_zero_is_checkfold():
    jam = translate_openppl_decision(
        DirectAction("BetMax"),
        public=snap(),
        actor=0,
        big_blind_chips=20,
    )
    check = translate_openppl_decision(
        ReturnValue(0),
        public=snap(
            legal_fold=False,
            legal_check=True,
            legal_call=False,
            legal_bet=True,
            legal_raise=False,
        ),
        actor=0,
        big_blind_chips=20,
    )
    assert (jam.action_type, jam.amount_to) == (5, 0)
    assert (check.action_type, check.amount_to) == (1, 0)


def test_unavailable_aggression_uses_openppl_passive_backup_chain():
    action = translate_openppl_decision(
        DirectAction("BetHalfPot"),
        public=snap(
            legal_raise=False,
            legal_bet=False,
            legal_all_in=False,
            legal_call=True,
        ),
        actor=0,
        big_blind_chips=20,
    )
    assert (action.action_type, action.amount_to) == (2, 0)


def test_literal_fold_on_free_action_degrades_to_check():
    action = translate_openppl_decision(
        DirectAction("Fold"),
        public=snap(
            legal_fold=False,
            legal_check=True,
            legal_call=False,
            legal_bet=True,
            legal_raise=False,
        ),
        actor=0,
        big_blind_chips=20,
    )
    assert (action.action_type, action.amount_to) == (1, 0)


def test_history_origin_distinguishes_raise_button_from_betsize():
    raise_min_decision = DirectAction("RaiseMin")
    raise_min_action = translate_openppl_decision(
        raise_min_decision,
        public=snap(),
        actor=0,
        big_blind_chips=20,
    )
    assert openppl_history_origin(raise_min_decision, raise_min_action) == "raise"

    sized_decision = ReturnValue(5.0)
    sized_action = translate_openppl_decision(
        sized_decision,
        public=snap(),
        actor=0,
        big_blind_chips=20,
    )
    assert openppl_history_origin(sized_decision, sized_action) == "betsize"


def test_history_origin_tracks_executed_passive_backup():
    decision = DirectAction("RaiseMin")
    action = translate_openppl_decision(
        decision,
        public=snap(
            legal_raise=False,
            legal_bet=False,
            legal_all_in=False,
            legal_call=True,
        ),
        actor=0,
        big_blind_chips=20,
    )
    assert action.action_type == 2
    assert openppl_history_origin(decision, action) == "call"
