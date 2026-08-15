from __future__ import annotations

import random

from spincore_nn.codec_v3 import DecodedInputV3
from spincore_nn.semantics_v3 import derive_objective_semantics_v3

Card = tuple[int, int]
DECK: tuple[Card, ...] = tuple((rank, suit) for rank in range(2, 15) for suit in range(4))
STRAIGHT_WINDOWS = (
    frozenset((14, 2, 3, 4, 5)),
    frozenset((2, 3, 4, 5, 6)),
    frozenset((3, 4, 5, 6, 7)),
    frozenset((4, 5, 6, 7, 8)),
    frozenset((5, 6, 7, 8, 9)),
    frozenset((6, 7, 8, 9, 10)),
    frozenset((7, 8, 9, 10, 11)),
    frozenset((8, 9, 10, 11, 12)),
    frozenset((9, 10, 11, 12, 13)),
    frozenset((10, 11, 12, 13, 14)),
)


def _has_straight(cards: tuple[Card, ...]) -> bool:
    ranks = {rank for rank, _ in cards}
    return any(window <= ranks for window in STRAIGHT_WINDOWS)


def _has_flush(cards: tuple[Card, ...]) -> bool:
    counts = [0, 0, 0, 0]
    for _, suit in cards:
        counts[suit] += 1
    return max(counts, default=0) >= 5


def _relations(cards: list[Card | None]) -> tuple[int, ...]:
    out: list[int] = []
    for left in range(7):
        for right in range(left + 1, 7):
            a, b = cards[left], cards[right]
            out.append(int(a is not None and b is not None and a[1] == b[1]))
    return tuple(out)


def _spnniv3(hole: tuple[Card, Card], board: tuple[Card, ...]) -> DecodedInputV3:
    street = {3: 1, 4: 2}[len(board)]
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


def _physical_completion_reference(hole: tuple[Card, Card], board: tuple[Card, ...]) -> dict[str, object]:
    visible = tuple(hole) + tuple(board)
    unseen = tuple(card for card in DECK if card not in set(visible))

    already_straight = _has_straight(visible)
    board_already_straight = _has_straight(board)
    already_flush = _has_flush(visible)
    board_already_flush = _has_flush(board)

    straight = tuple(
        card for card in unseen
        if not already_straight and _has_straight(visible + (card,))
    )
    board_straight = tuple(
        card for card in unseen
        if not board_already_straight and _has_straight(board + (card,))
    )
    flush = tuple(
        card for card in unseen
        if not already_flush and _has_flush(visible + (card,))
    )
    board_flush = tuple(
        card for card in unseen
        if not board_already_flush and _has_flush(board + (card,))
    )

    hero_adds_straight = int(bool(straight) and set(straight) != set(board_straight))
    hero_adds_flush = int(bool(flush) and set(flush) != set(board_flush))

    flush_highest_hero_rank = 0
    flush_higher_unseen_count = 0
    if flush:
        suit_counts = [0, 0, 0, 0]
        for _, suit in visible:
            suit_counts[suit] += 1
        for suit, count in enumerate(suit_counts):
            if count != 4:
                continue
            hero_ranks = [rank for rank, hero_suit in hole if hero_suit == suit]
            if not hero_ranks:
                continue
            high = max(hero_ranks)
            higher = sum(1 for rank in range(high + 1, 15) if (rank, suit) in unseen)
            if high > flush_highest_hero_rank:
                flush_highest_hero_rank = high
                flush_higher_unseen_count = higher

    return {
        "straight_completion_card_count": len(straight),
        "straight_completion_distinct_rank_count": len({rank for rank, _ in straight}),
        "board_straight_completion_card_count": len(board_straight),
        "hero_adds_to_straight_draw": hero_adds_straight,
        "flush_completion_card_count": len(flush),
        "board_flush_completion_card_count": len(board_flush),
        "hero_adds_to_flush_draw": hero_adds_flush,
        "flush_draw_highest_hero_rank": flush_highest_hero_rank,
        "flush_draw_higher_unseen_count": flush_higher_unseen_count,
    }


def test_spnniv3_out_counts_match_physical_deck_enumeration() -> None:
    rng = random.Random(0x753C2026)
    # Both flop and turn are sampled. 512 states already compare >20k physical
    # one-card continuations while keeping the ordinary main regression cheap.
    for sample in range(512):
        deck = list(DECK)
        rng.shuffle(deck)
        hole = (deck[0], deck[1])
        board_len = 3 if sample % 2 == 0 else 4
        board = tuple(deck[2 : 2 + board_len])
        sem = derive_objective_semantics_v3(_spnniv3(hole, board))
        reference = _physical_completion_reference(hole, board)
        for field, expected in reference.items():
            actual = getattr(sem, field)
            assert actual == expected, (
                sample,
                field,
                actual,
                expected,
                hole,
                board,
            )
