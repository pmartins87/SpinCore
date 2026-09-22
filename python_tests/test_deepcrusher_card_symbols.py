from __future__ import annotations

from spincore.deepcrusher_card_symbols import (
    CATEGORY_FULL_HOUSE,
    CATEGORY_STRAIGHT,
    DeepCrusherCardSymbols,
    evaluate_cards,
    pokerval,
)
from spincore.deepcrusher_state import DeepCrusherStateView, STREET_FLOP, STREET_RIVER


def _view(
    *,
    ranks=(14, 13, 12, 11, 2, 0, 0),
    suits=(0, 1, 0, 2, 3, -1, -1),
    street=STREET_FLOP,
):
    same = []
    for i in range(7):
        for j in range(i + 1, 7):
            same.append(
                int(
                    ranks[i] > 0
                    and ranks[j] > 0
                    and suits[i] >= 0
                    and suits[i] == suits[j]
                )
            )
    return DeepCrusherStateView(
        domain=1,
        street=street,
        dealer_rel=0,
        small_blind_rel=0,
        big_blind_rel=1,
        live_count=2,
        visible_board=sum(1 for x in ranks[2:] if x > 0),
        statuses=(0, 0, 2),
        ranks=tuple(ranks),
        same_suit=tuple(same),
        pot_bb=4.0,
        to_call_bb=1.0,
        current_bet_bb=2.0,
        stacks_bb=(18.0, 18.0, 0.0),
        street_commitments_bb=(1.0, 2.0, 0.0),
        total_commitments_bb=(2.0, 3.0, 0.0),
        small_blind_bb=0.5,
        blind_index=0,
        min_raise_to_bb=3.0,
        max_raise_to_bb=19.0,
        primitive_legal=(True, False, True, False, True, True),
        history=(),
        exact_suits=tuple(suits),
    )


def test_rank_bits_hi_lo_and_rank_only_board_expressions():
    s = DeepCrusherCardSymbols(_view())
    assert int(s("rankbitsplayer")) & (1 << 14)
    assert int(s("rankbitsplayer")) & (1 << 13)
    assert int(s("rankbitsplayer")) & (1 << 1)  # low-Ace compatibility bit
    assert s("rankhiplayer") == 14
    assert s("rankloplayer") == 13
    assert s("rankhicommon") == 12
    assert s("ranklocommon") == 2
    assert s("board$QJ") == 1
    assert s("board$QJ2") == 1
    assert s("board$QQ") == 0
    assert s("board$A") == 0


def test_specific_suit_and_suited_expressions_use_openholdem_suit_ids():
    # h=0,d=1,c=2,s=3. Hero AhKd; board Qh Jc 2s.
    s = DeepCrusherCardSymbols(_view())
    assert s("hand$Ah") == 1
    assert s("hand$Ad") == 0
    assert s("hand$AKsuited") == 0
    assert s("board$Qh") == 1
    assert s("board$Qs") == 0

    suited_view = _view(
        ranks=(14, 13, 12, 11, 2, 0, 0),
        suits=(0, 0, 1, 1, 3, -1, -1),
    )
    suited = DeepCrusherCardSymbols(suited_view)
    assert suited("hand$AKsuited") == 1
    assert suited("hand$AhKh") == 1


def test_openholdem_dominant_suit_tie_order_is_preserved():
    # Board Ah Kd Qc => all suits tied at 1; OpenHoldem chooses clubs (2).
    v = _view(
        ranks=(7, 6, 14, 13, 12, 0, 0),
        suits=(3, 3, 0, 1, 2, -1, -1),
    )
    s = DeepCrusherCardSymbols(v)
    assert s("nsuitedcommon") == 1
    assert s("tsuitcommon") == 2


def test_straight_metrics_match_openholdem_window_logic():
    v = _view(
        ranks=(14, 5, 4, 3, 9, 0, 0),
        suits=(0, 1, 2, 3, 0, -1, -1),
    )
    s = DeepCrusherCardSymbols(v)
    # A543 + hero/board combined is one rank short of wheel.
    assert s("nstraight") == 4
    assert s("nstraightfill") == 1
    assert s("nstraightcommon") >= 1
    assert s("nstraightfillcommon") >= 2


def test_best_hand_categories_and_pokerval_encoding():
    # Hero 7h7d, board 7cKsKd2h2d => best full house 777KK.
    v = _view(
        ranks=(7, 7, 7, 13, 13, 2, 2),
        suits=(0, 1, 2, 3, 2, 0, 1),
        street=STREET_RIVER,
    )
    s = DeepCrusherCardSymbols(v)
    assert s("isfullhouse") == 1
    assert s("isthreeofakind") == 0
    expected = 0x20000000 + (7 << 16) + (7 << 12) + (7 << 8) + (13 << 4) + 13
    assert int(s("pokerval")) == expected
    assert int(s("npcbits")) == 2


def test_wheel_straight_uses_low_ace_nibble_one():
    value = evaluate_cards(((14, 0), (5, 1), (4, 2), (3, 3), (2, 0)))
    assert value.category == CATEGORY_STRAIGHT
    assert value.nibbles == (5, 4, 3, 2, 1)
    assert pokerval(value) == (
        0x08000000 + (5 << 16) + (4 << 12) + (3 << 8) + (2 << 4) + 1
    )


def test_board_only_partial_pokerval_is_comparable_on_flop():
    # Board Q Q 2 => one pair Q with 2 kicker in the first available kicker slot.
    v = _view(
        ranks=(14, 13, 12, 12, 2, 0, 0),
        suits=(0, 1, 0, 1, 2, -1, -1),
    )
    s = DeepCrusherCardSymbols(v)
    expected = 0x01000000 + (12 << 16) + (12 << 12) + (2 << 8)
    assert int(s("pokervalcommon")) == expected


def test_pocket_pair_symbol_is_not_same_as_best_hand_pair_category():
    # Pocket 88 with A K Q board: ispair true even though best current hand is pair 88.
    v = _view(
        ranks=(8, 8, 14, 13, 12, 0, 0),
        suits=(0, 1, 2, 3, 0, -1, -1),
    )
    s = DeepCrusherCardSymbols(v)
    assert s("ispair") == 1
    assert s("isonepair") == 1
