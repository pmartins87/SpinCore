from __future__ import annotations

from itertools import permutations

import pytest

from spincore.card_symmetry_v1 import canonicalize_v1_cards

RANKS = "23456789TJQKA"
SUITS = "shdc"


def tok(card: str) -> int:
    rank = RANKS.index(card[0].upper())
    suit = SUITS.index(card[1].lower())
    return rank * 4 + suit + 1


def rename_suits(cards: tuple[int, ...], permutation: tuple[int, int, int, int]) -> tuple[int, ...]:
    out = []
    for token in cards:
        if token == 0:
            out.append(0)
            continue
        card_id = token - 1
        rank, suit = divmod(card_id, 4)
        out.append(rank * 4 + permutation[suit] + 1)
    return tuple(out)


def suit_equalities(cards: tuple[int, ...]) -> tuple[tuple[bool, ...], ...]:
    visible = [token for token in cards if token]
    suits = [(token - 1) % 4 for token in visible]
    return tuple(tuple(left == right for right in suits) for left in suits)


def test_private_card_order_is_invariant() -> None:
    a = (tok("As"), tok("Kh"), 0, 0, 0, 0, 0)
    b = (tok("Kh"), tok("As"), 0, 0, 0, 0, 0)
    assert canonicalize_v1_cards(a) == canonicalize_v1_cards(b)


def test_all_flop_permutations_are_invariant() -> None:
    hero = (tok("Qs"), tok("Jh"))
    flop = (tok("Qh"), tok("8s"), tok("2h"))
    expected = canonicalize_v1_cards(hero + flop + (0, 0))
    for order in permutations(flop):
        assert canonicalize_v1_cards(hero + tuple(order) + (0, 0)) == expected


def test_all_24_global_suit_renamings_are_invariant() -> None:
    cards = (
        tok("Qs"), tok("Jh"),
        tok("Qh"), tok("8s"), tok("2h"),
        tok("7c"), tok("Ad"),
    )
    expected = canonicalize_v1_cards(cards)
    for permutation in permutations(range(4)):
        assert canonicalize_v1_cards(rename_suits(cards, permutation)) == expected


def test_turn_and_river_timeline_is_not_quotiented() -> None:
    a = (
        tok("As"), tok("Kh"),
        tok("Qh"), tok("8s"), tok("2h"),
        tok("7c"), tok("Ad"),
    )
    b = a[:5] + (a[6], a[5])
    assert canonicalize_v1_cards(a) != canonicalize_v1_cards(b)


def test_private_and_public_roles_are_not_quotiented() -> None:
    a = (
        tok("As"), tok("Kh"),
        tok("Qh"), tok("8s"), tok("2h"),
        0, 0,
    )
    b = (
        tok("Qh"), tok("Kh"),
        tok("As"), tok("8s"), tok("2h"),
        0, 0,
    )
    assert canonicalize_v1_cards(a) != canonicalize_v1_cards(b)


def test_global_suit_relationships_are_preserved() -> None:
    cards = (
        tok("Ah"), tok("Kh"),
        tok("Qh"), tok("Jh"), tok("2c"),
        tok("7c"), 0,
    )
    canonical = canonicalize_v1_cards(cards)
    assert suit_equalities(canonical) == suit_equalities(tuple(sorted(cards[:2])) + tuple(sorted(cards[2:5])) + cards[5:])


def test_preflop_padding_is_preserved() -> None:
    canonical = canonicalize_v1_cards((tok("Ah"), tok("Kh"), 0, 0, 0, 0, 0))
    assert canonical[2:] == (0, 0, 0, 0, 0)


def test_invalid_visibility_patterns_fail_closed() -> None:
    with pytest.raises(ValueError):
        canonicalize_v1_cards((tok("As"), tok("Kh"), tok("Qh"), 0, 0, 0, 0))
    with pytest.raises(ValueError):
        canonicalize_v1_cards((tok("As"), tok("Kh"), 0, 0, 0, tok("7c"), 0))
    with pytest.raises(ValueError):
        canonicalize_v1_cards((tok("As"), tok("Kh"), 0, 0, 0, 0, tok("Ad")))


def test_duplicate_visible_cards_fail_closed() -> None:
    with pytest.raises(ValueError):
        canonicalize_v1_cards((tok("As"), tok("As"), 0, 0, 0, 0, 0))
