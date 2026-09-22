from __future__ import annotations

from spincore.deepcrusher_history_symbols import DeepCrusherHistorySymbols
from spincore.deepcrusher_state import (
    ACTION_BET_TO,
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_RAISE_TO,
    DeepCrusherStateView,
    PublicActionEvent,
    STREET_FLOP,
    STREET_PREFLOP,
    STREET_TURN,
)


def ev(actor, street, action, *, forced=False, paid=0.0, commit=0.0, before=0.0, after=0.0):
    return PublicActionEvent(
        actor_rel=actor,
        street=street,
        action_type=action,
        forced=forced,
        paid_bb=paid,
        resulting_commitment_bb=commit,
        pot_before_bb=before,
        pot_after_bb=after,
    )


def view(*, street=STREET_TURN, history=(), stacks=(20.0, 18.0, 14.0)):
    return DeepCrusherStateView(
        domain=0,
        street=street,
        dealer_rel=0,
        small_blind_rel=1,
        big_blind_rel=2,
        live_count=3,
        visible_board=4 if street == STREET_TURN else 3 if street == STREET_FLOP else 0,
        statuses=(0, 0, 0),
        ranks=(14, 13, 12, 9, 3, 2 if street == STREET_TURN else 0, 0),
        same_suit=(0,) * 21,
        pot_bb=16.0,
        to_call_bb=0.0,
        current_bet_bb=0.0,
        stacks_bb=stacks,
        street_commitments_bb=(0.0, 0.0, 0.0),
        total_commitments_bb=(8.0, 10.0, 10.0),
        small_blind_bb=0.5,
        blind_index=0,
        min_raise_to_bb=2.0,
        max_raise_to_bb=20.0,
        primitive_legal=(False, True, False, True, False, True),
        history=tuple(history),
    )


def test_per_street_hero_counters_prevaction_and_previousround_aliases():
    history = (
        ev(1, STREET_PREFLOP, ACTION_BET_TO, forced=True, paid=.5, commit=.5),
        ev(2, STREET_PREFLOP, ACTION_BET_TO, forced=True, paid=1, commit=1),
        ev(0, STREET_PREFLOP, ACTION_RAISE_TO, paid=2, commit=2),
        ev(1, STREET_PREFLOP, ACTION_CALL, paid=1.5, commit=2),
        ev(2, STREET_PREFLOP, ACTION_RAISE_TO, paid=5, commit=6),
        ev(0, STREET_PREFLOP, ACTION_CALL, paid=4, commit=6),
        ev(1, STREET_FLOP, ACTION_CHECK, commit=0),
        ev(0, STREET_FLOP, ACTION_BET_TO, paid=3, commit=3),
        ev(1, STREET_FLOP, ACTION_RAISE_TO, paid=8, commit=8),
        ev(0, STREET_FLOP, ACTION_CALL, paid=5, commit=8),
        ev(1, STREET_TURN, ACTION_CHECK, commit=0),
        ev(0, STREET_TURN, ACTION_CHECK, commit=0),
    )
    s = DeepCrusherHistorySymbols(view(history=history))

    assert s("didbetsizeround_preflop") == 1
    assert s("didraisround_preflop") == 0
    assert s("didcallround_preflop") == 1
    assert s("didbetsizeround_previousround") == 1
    assert s("didcallround_previousround") == 1
    assert s("nbetsround_preflop") == 6
    assert s("nbetsround_previousround") == 8
    assert s("prevaction") == 0
    assert s("BotRaisedBeforeFlop") == 1
    assert s("BotCalledOnFlop") == 1
    assert s("BotCheckedOnTurn") == 1
    assert s("lastraised_previousround") == 1
    # No aggression yet on turn: OpenHoldem keeps the prior raischair.
    assert s("raischair") == 1
    assert s("StackSize_raischair") == 18


def test_current_orbit_callers_and_raisers_are_reconstructed_from_transcript():
    history = (
        ev(1, STREET_FLOP, ACTION_BET_TO, paid=3, commit=3),
        ev(2, STREET_FLOP, ACTION_CALL, paid=3, commit=3),
    )
    s = DeepCrusherHistorySymbols(view(street=STREET_FLOP, history=history))
    assert s("nopponentstruelyraising") == 1
    assert s("nopponentscalling") == 1
    assert s("firstcallerchair") == 2
    assert s("lastcallerchair") == 2
    assert s("raischair") == 1
    assert s("RaisesSinceLastPlay") == 1
    assert s("CallsSinceLastPlay") == 1
    assert int(s("callbits_flop")) == (1 << 2)


def test_call_before_later_raise_is_not_current_caller():
    history = (
        ev(1, STREET_PREFLOP, ACTION_BET_TO, forced=True, paid=.5, commit=.5),
        ev(2, STREET_PREFLOP, ACTION_BET_TO, forced=True, paid=1, commit=1),
        ev(1, STREET_PREFLOP, ACTION_CALL, paid=.5, commit=1),
        ev(2, STREET_PREFLOP, ACTION_RAISE_TO, paid=3, commit=4),
    )
    s = DeepCrusherHistorySymbols(view(street=STREET_PREFLOP, history=history))
    assert s("nopponentscalling") == 0
    # Historical callbits remains accumulated even though the call is stale now.
    assert int(s("callbits_preflop")) == (1 << 1)
    assert s("raischair") == 2


def test_hero_action_resets_since_last_play_window():
    history = (
        ev(1, STREET_FLOP, ACTION_BET_TO, paid=2, commit=2),
        ev(0, STREET_FLOP, ACTION_CALL, paid=2, commit=2),
        ev(1, STREET_FLOP, ACTION_RAISE_TO, paid=4, commit=6),
    )
    s = DeepCrusherHistorySymbols(view(street=STREET_FLOP, history=history))
    assert s("CallsSinceLastPlay") == 0
    assert s("RaisesSinceLastPlay") == 1
    assert s("BotsActionsOnThisRound") == 1


def test_raises_before_flop_tracks_opponents_not_hero():
    history = (
        ev(0, STREET_PREFLOP, ACTION_RAISE_TO, paid=2, commit=2),
        ev(1, STREET_PREFLOP, ACTION_CALL, paid=2, commit=2),
    )
    s = DeepCrusherHistorySymbols(view(street=STREET_PREFLOP, history=history))
    assert s("RaisesBeforeFlop") == 0
    assert s("BotRaisedBeforeFlop") == 1
    assert s("Raises") == 0


def test_unknown_history_symbol_fails_closed():
    s = DeepCrusherHistorySymbols(view(history=()))
    try:
        s("not_a_history_symbol")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown history symbol must fail closed")
