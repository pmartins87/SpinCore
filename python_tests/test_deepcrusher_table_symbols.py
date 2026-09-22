from __future__ import annotations

from spincore.deepcrusher_state import (
    DOMAIN_THREE_HANDED,
    DOMAIN_TRUE_HEADS_UP,
    DeepCrusherStateView,
    STREET_PREFLOP,
)
from spincore.deepcrusher_table_symbols import DeepCrusherTableSymbols


def _view(
    *,
    domain=DOMAIN_THREE_HANDED,
    dealer=2,
    sb=0,
    bb=1,
    statuses=(0, 0, 0),
    stacks=(20.0, 18.0, 25.0),
    street_commit=(0.5, 1.0, 0.0),
):
    return DeepCrusherStateView(
        domain=domain,
        street=STREET_PREFLOP,
        dealer_rel=dealer,
        small_blind_rel=sb,
        big_blind_rel=bb,
        live_count=sum(1 for x in statuses if x != 1),
        visible_board=0,
        statuses=statuses,
        ranks=(14, 13, 0, 0, 0, 0, 0),
        same_suit=(0,) * 21,
        pot_bb=1.5,
        to_call_bb=0.5,
        current_bet_bb=max(street_commit),
        stacks_bb=stacks,
        street_commitments_bb=street_commit,
        total_commitments_bb=street_commit,
        small_blind_bb=0.5,
        blind_index=0,
        min_raise_to_bb=2.0,
        max_raise_to_bb=20.5,
        primitive_legal=(True, False, True, False, True, True),
        history=(),
    )


def test_frozen_holdem_and_pokerval_constants():
    s = DeepCrusherTableSymbols(_view())
    assert s("isnl") == 1
    assert s("isfl") == 0
    assert s("ispl") == 0
    assert s("isomaha") == 0
    assert s("istournament") == 1
    assert s("ismyturn") == 1
    assert s("isfinaltable") == 1
    assert s("sitename$openholdem") == 1
    assert s("hearts") == 0
    assert s("diamonds") == 1
    assert s("clubs") == 2
    assert s("spades") == 3
    assert s("onepair") == 0x01000000
    assert s("twopair") == 0x02000000
    assert s("threeofakind") == 0x04000000


def test_three_handed_deal_and_bet_positions():
    # hero SB, BB=rel1, dealer=rel2
    s = DeepCrusherTableSymbols(_view())
    assert s("dealposition") == 1
    assert s("betposition") == 1
    assert s("buttonchair") == 2
    assert s("utgchair") == 2
    assert s("cutoffchair") == -1
    assert s("mp1chair") == -1

    # hero dealer/button: dealposition 3
    d = DeepCrusherTableSymbols(_view(dealer=0, sb=1, bb=2))
    assert d("dealposition") == 3
    assert d("betposition") == 3


def test_folded_player_is_skipped_by_betposition_but_not_dealt():
    # hero dealer, rel1 SB folded, rel2 BB remains.
    s = DeepCrusherTableSymbols(
        _view(dealer=0, sb=1, bb=2, statuses=(0, 1, 0))
    )
    assert s("nplayersdealt") == 3
    assert s("nplayersplaying") == 2
    assert int(s("playersdealtbits")) == 0b111
    assert int(s("playersplayingbits")) == 0b101
    # Scan after dealer: folded SB skipped, BB first, hero second.
    assert s("betposition") == 2


def test_true_hu_compaction_and_positions():
    v = _view(
        domain=DOMAIN_TRUE_HEADS_UP,
        dealer=0,
        sb=0,
        bb=1,
        statuses=(0, 0, 2),
        stacks=(12.0, 15.0, 0.0),
        street_commit=(0.5, 1.0, 0.0),
    )
    s = DeepCrusherTableSymbols(v)
    assert s("nplayersdealt") == 2
    assert s("dealposition") == 2  # dealer/SB in HU
    assert s("betposition") == 2
    assert int(s("playersdealtbits")) == 0b011
    assert int(s("opponentsplayingbits")) == 0b010
    assert s("utgchair") == -1


def test_currentbet_ncallbets_and_bigstack_multiplexer_values():
    s = DeepCrusherTableSymbols(_view())
    assert s("currentbet0") == 0.5
    assert s("currentbet1") == 1.0
    assert s("currentbet2") == 0.0
    assert s("currentbet7") == 0.0
    assert s("ncurrentbets") == 0.5
    assert s("ncallbets") == 1.0

    # rel2 has the largest playing-opponent stack.
    assert s("bigstackchair") == 2
    assert s("balance_bigstackchair") == 25.0
    assert s("currentbet_bigstackchair") == 0.0


def test_real_allin_is_in_playing_bits_and_allin_bits():
    v = _view(
        statuses=(0, 2, 0),
        stacks=(20.0, 0.0, 25.0),
        street_commit=(0.5, 5.0, 5.0),
    )
    s = DeepCrusherTableSymbols(v)
    assert int(s("playersplayingbits")) == 0b111
    assert int(s("playersallinbits")) == 0b010
    assert s("nopponentsallin") == 1
