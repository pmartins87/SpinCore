from __future__ import annotations

import pytest

from spincore.deepcrusher_native_symbols import (
    BENCHMARK_ENVIRONMENT_PROFILE_ID,
    DeepCrusherPrimitiveSymbols,
    OPENHOLDEM_UNDEFINED,
    UnknownDeepCrusherNativeSymbol,
    frozen_benchmark_environment,
)
from spincore.deepcrusher_state import (
    ACTION_ALL_IN,
    DeepCrusherStateView,
    PublicActionEvent,
    STREET_FLOP,
)


def _view(*, statuses=(0, 0, 2), stacks=(9.0, 7.0, 0.0), street_commit=(1.5, 3.0, 0.0)):
    return DeepCrusherStateView(
        domain=1,
        street=STREET_FLOP,
        dealer_rel=1,
        small_blind_rel=1,
        big_blind_rel=0,
        live_count=2,
        visible_board=3,
        statuses=statuses,
        ranks=(14, 13, 12, 7, 2, 0, 0),
        same_suit=(0,) * 21,
        pot_bb=5.5,
        to_call_bb=1.5,
        current_bet_bb=3.0,
        stacks_bb=stacks,
        street_commitments_bb=street_commit,
        total_commitments_bb=(2.5, 4.0, 0.0),
        small_blind_bb=0.5,
        blind_index=3,
        min_raise_to_bb=4.5,
        max_raise_to_bb=10.5,
        primitive_legal=(True, False, True, False, True, True),
        history=(
            PublicActionEvent(
                actor_rel=1,
                street=STREET_FLOP,
                action_type=ACTION_ALL_IN,
                forced=False,
                paid_bb=2.0,
                resulting_commitment_bb=3.0,
                pot_before_bb=3.5,
                pot_after_bb=5.5,
            ),
        ),
    )


def test_direct_bb_normalized_chip_and_position_symbols():
    s = DeepCrusherPrimitiveSymbols(_view())
    assert s("IsFlop") == 1
    assert s("IsPreflop") == 0
    assert s("betround") == 2
    assert s("bblind") == 1
    assert s("sblind") == pytest.approx(0.5)
    assert s("AmountToCall") == pytest.approx(1.5)
    assert s("DollarsToCall") == pytest.approx(1.5)
    assert s("PotSize") == pytest.approx(5.5)
    assert s("pot") == pytest.approx(5.5)
    assert s("StackSize") == pytest.approx(9.0)
    assert s("balance") == pytest.approx(9.0)
    assert s("currentbet") == pytest.approx(1.5)
    assert s("BetSize") == pytest.approx(3.0)
    assert s("userchair") == 0
    assert s("dealerchair") == 1
    assert s("smallblindchair") == 1
    assert s("bigblindchair") == 0
    assert s("InButton") == 0
    assert s("InSmallBlind") == 0
    assert s("InBigBlind") == 1


def test_true_hu_absent_slot_is_not_counted_as_allin_opponent():
    s = DeepCrusherPrimitiveSymbols(_view())
    assert s("headsupchair") == 1
    assert s("nplayersplaying") == 2
    assert s("nopponentsallin") == 0
    assert s("OpponentIsAllin") == 0
    assert s("balance_headsupchair") == pytest.approx(7.0)
    assert s("currentbet_headsupchair") == pytest.approx(3.0)


def test_real_opponent_allin_is_counted_but_absent_slot_is_not():
    s = DeepCrusherPrimitiveSymbols(
        _view(statuses=(0, 2, 2), stacks=(9.0, 0.0, 0.0), street_commit=(1.5, 3.0, 0.0))
    )
    # rel=1 has a nonzero commitment and therefore is a real all-in player.
    assert s("headsupchair") == 1
    assert s("nplayersplaying") == 2
    assert s("nopponentsallin") == 1
    assert s("OpponentIsAllin") == 1


def test_environment_override_is_explicit_and_case_insensitive():
    s = DeepCrusherPrimitiveSymbols(_view(), environment={"network$ggpoker": 1})
    assert s("NETWORK$GGPOKER") == 1


def test_unknown_symbol_fails_closed():
    s = DeepCrusherPrimitiveSymbols(_view())
    with pytest.raises(UnknownDeepCrusherNativeSymbol):
        s("HaveTopPair")


def test_frozen_benchmark_environment_matches_no_pt_ggpoker_contract():
    env = frozen_benchmark_environment([
        "network$ggpoker",
        "network$ipoker",
        "chair$SomeVillain",
        "log$Diagnostic",
        "pt_hands_headsupchair",
        "colourcode_headsupchair",
        "prwin",
        "prtie",
    ])
    assert BENCHMARK_ENVIRONMENT_PROFILE_ID == "GGPoker_NoPT_NoNotes_V1"
    assert env["network$ggpoker"] == 1
    assert env["network$ipoker"] == 0
    assert env["chair$SomeVillain"] == OPENHOLDEM_UNDEFINED == -1
    assert env["log$Diagnostic"] == 1
    assert env["pt_hands_headsupchair"] == -1
    assert env["colourcode_headsupchair"] == 0
    assert "prwin" not in env
    assert "prtie" not in env
