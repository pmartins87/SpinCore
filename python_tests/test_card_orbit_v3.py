from __future__ import annotations

from itertools import combinations, permutations

from spincore_nn.card_orbit_v3 import (
    canonical_card_orbit_key_v3,
    canonical_flop_orbit_key_from_physical_cards,
)
from spincore_nn.codec_v3 import DecodedInputV3

Card = tuple[int, int]
DECK: tuple[Card, ...] = tuple((rank, suit) for rank in range(2, 15) for suit in range(4))


def _relations(cards: list[Card | None]) -> tuple[int, ...]:
    out: list[int] = []
    for left in range(7):
        for right in range(left + 1, 7):
            a, b = cards[left], cards[right]
            out.append(int(a is not None and b is not None and a[1] == b[1]))
    return tuple(out)


def _item(hole: tuple[Card, Card], board: tuple[Card, ...]) -> DecodedInputV3:
    street = {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]
    cards: list[Card | None] = [hole[0], hole[1], None, None, None, None, None]
    for index, card in enumerate(board):
        cards[index + 2] = card
    return DecodedInputV3(
        categorical=(0, street, 0, 1, 2, 3, len(board), 0, 0, 0),
        rank_tokens=tuple(card[0] if card is not None else 0 for card in cards),
        same_suit=_relations(cards),
        numeric=(
            6.0, 0.0, 0.0,
            20.0, 20.0, 20.0,
            0.0, 0.0, 0.0,
            2.0, 2.0, 2.0,
            0.5, 0.0, 2.0, 40.0,
        ),
        primitive_legal=(1, 1, 1, 1, 1, 1),
        history=(),
    )


def test_card_orbit_is_invariant_to_only_intended_order_and_suit_symmetries() -> None:
    hole = ((14, 0), (13, 1))
    flop = ((12, 0), (8, 1), (2, 2))
    baseline = canonical_card_orbit_key_v3(_item(hole, flop))

    for suit_perm in permutations(range(4)):
        h = tuple((rank, suit_perm[suit]) for rank, suit in hole)
        f = tuple((rank, suit_perm[suit]) for rank, suit in flop)
        assert canonical_card_orbit_key_v3(_item(h, f)) == baseline

    assert canonical_card_orbit_key_v3(_item((hole[1], hole[0]), flop)) == baseline
    for permuted_flop in permutations(flop):
        assert canonical_card_orbit_key_v3(_item(hole, tuple(permuted_flop))) == baseline

    # Turn and river are temporal roles, not simultaneous-card permutations.
    turn_river = _item(hole, flop + ((7, 3), (6, 0)))
    swapped = _item(hole, flop + ((6, 0), (7, 3)))
    assert canonical_card_orbit_key_v3(turn_river) != canonical_card_orbit_key_v3(swapped)

    # Two-tone and monotone flops must not alias.
    two_tone = _item(hole, ((12, 0), (11, 0), (2, 2)))
    monotone = _item(((14, 1), (13, 2)), ((12, 0), (11, 0), (2, 0)))
    assert canonical_card_orbit_key_v3(two_tone) != canonical_card_orbit_key_v3(monotone)


def test_all_22100_physical_flops_reduce_to_exactly_1755_suit_isomorphism_classes() -> None:
    keys = {
        canonical_flop_orbit_key_from_physical_cards(tuple(flop))
        for flop in combinations(DECK, 3)
    }
    assert len(keys) == 1755
