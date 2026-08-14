from __future__ import annotations

from dataclasses import replace
from itertools import permutations
from typing import Iterable

from spincore_nn.codec import DecodedInput

NUM_SUITS = 4
NUM_CARD_TOKENS = 7


def _decode_token(token: int) -> tuple[int, int]:
    """Return zero-based (rank, suit) for an SPNNIV1 card token."""
    value = int(token)
    if value < 1 or value > 52:
        raise ValueError(f"card token must be in [1, 52], got {value}")
    card_id = value - 1
    return card_id // NUM_SUITS, card_id % NUM_SUITS


def _encode_token(rank: int, suit: int) -> int:
    if not 0 <= int(rank) < 13 or not 0 <= int(suit) < NUM_SUITS:
        raise ValueError("invalid rank/suit")
    return int(rank) * NUM_SUITS + int(suit) + 1


def _permute_token(token: int, suit_permutation: tuple[int, int, int, int]) -> int:
    if int(token) == 0:
        return 0
    rank, suit = _decode_token(int(token))
    return _encode_token(rank, suit_permutation[suit])


def _validate_layout(cards: tuple[int, ...]) -> None:
    if len(cards) != NUM_CARD_TOKENS:
        raise ValueError(f"SPNNIV1 requires {NUM_CARD_TOKENS} card slots")
    if any(int(token) < 0 or int(token) > 52 for token in cards):
        raise ValueError("card tokens must lie in [0, 52]")
    if cards[0] == 0 or cards[1] == 0:
        raise ValueError("both private-card slots must be visible")

    flop = cards[2:5]
    flop_count = sum(int(token) != 0 for token in flop)
    if flop_count not in (0, 3):
        raise ValueError("flop must expose either zero or exactly three cards")
    if cards[5] != 0 and flop_count != 3:
        raise ValueError("turn cannot be visible before the flop")
    if cards[6] != 0 and cards[5] == 0:
        raise ValueError("river cannot be visible before the turn")

    visible = [int(token) for token in cards if int(token) != 0]
    if len(visible) != len(set(visible)):
        raise ValueError("duplicate visible cards are invalid")


def canonicalize_v1_cards(cards: Iterable[int]) -> tuple[int, ...]:
    """Canonicalize only true lossless card symmetries of SPNNIV1.

    Preserved roles/timeline:
      [private, private | flop, flop, flop | turn | river]

    Quotiented symmetries:
      * order of the two private cards;
      * order of the three simultaneously revealed flop cards;
      * one global renaming of the four suits across every visible card.

    Turn and river are deliberately never reordered, and private/public roles are
    never exchanged.  Enumerating all 24 suit permutations makes the quotient
    definition explicit and independent of absolute suit names.
    """
    original = tuple(int(token) for token in cards)
    _validate_layout(original)

    best: tuple[int, ...] | None = None
    for permutation in permutations(range(NUM_SUITS)):
        suit_permutation = tuple(int(value) for value in permutation)
        hero = sorted(_permute_token(token, suit_permutation) for token in original[:2])

        if original[2] == 0:
            flop = [0, 0, 0]
        else:
            flop = sorted(_permute_token(token, suit_permutation) for token in original[2:5])

        candidate = tuple(
            hero
            + flop
            + [
                _permute_token(original[5], suit_permutation),
                _permute_token(original[6], suit_permutation),
            ]
        )
        if best is None or candidate < best:
            best = candidate

    if best is None:  # pragma: no cover - permutations(range(4)) is non-empty.
        raise RuntimeError("failed to canonicalize card tuple")
    return best


def canonicalize_v1_input(item: DecodedInput) -> DecodedInput:
    """Return an otherwise byte-for-byte semantic V1 input with canonical cards."""
    return replace(item, cards=canonicalize_v1_cards(item.cards))
