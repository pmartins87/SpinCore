from __future__ import annotations

"""OpenHoldem-compatible card and hand symbols for the DeepCrusher R8 oracle.

The implementation is intentionally independent from SpinCore's learned policy.
It projects the current actor's observable cards into the legacy OpenHoldem
symbols used by the pinned OpenPPL library. Unknown/non-card symbols are left to
other providers.

Primary parity references:
- OpenHoldem/CSymbolEngineCards.cpp
- OpenHoldem/CSymbolEnginePokerval.cpp
- OpenHoldem/CSymbolEngineOpenPPLHandAndBoardExpression.cpp
from the pinned pmartins87/myoh_private OpenHoldem tree.
"""

from dataclasses import dataclass
from itertools import combinations
import re
from typing import Iterable

from spincore.deepcrusher_state import DeepCrusherStateView


CATEGORY_HIGH = 0
CATEGORY_PAIR = 1
CATEGORY_TWO_PAIR = 2
CATEGORY_TRIPS = 3
CATEGORY_STRAIGHT = 4
CATEGORY_FLUSH = 5
CATEGORY_FULL_HOUSE = 6
CATEGORY_QUADS = 7
CATEGORY_STRAIGHT_FLUSH = 8

POKERVAL_BASE = {
    CATEGORY_HIGH: 0x00000000,
    CATEGORY_PAIR: 0x01000000,
    CATEGORY_TWO_PAIR: 0x02000000,
    CATEGORY_TRIPS: 0x04000000,
    CATEGORY_STRAIGHT: 0x08000000,
    CATEGORY_FLUSH: 0x10000000,
    CATEGORY_FULL_HOUSE: 0x20000000,
    CATEGORY_QUADS: 0x40000000,
    CATEGORY_STRAIGHT_FLUSH: 0x80000000,
}

RANK_FROM_CHAR = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}
SUIT_FROM_CHAR = {"h": 0, "d": 1, "c": 2, "s": 3}
_CARD_EXPR = re.compile(r"^(hand|board)\$(.+)$", re.I)


class UnknownDeepCrusherCardSymbol(KeyError):
    pass


@dataclass(frozen=True, order=True)
class HandValue:
    category: int
    # OpenHoldem-rank ordering for comparison, e.g. pair=(pair,k1,k2,k3),
    # straight=(high,), full-house=(trips,pair).
    tie: tuple[int, ...]
    # Five OpenHoldem rank nibbles used in pokerval. Missing partial-hand
    # positions are zero; a wheel uses the historical low-Ace nibble 1.
    nibbles: tuple[int, int, int, int, int]
    # Dominant/flush suit for flush and straight-flush, else -1.
    flush_suit: int = -1


def _rankbits(ranks: Iterable[int]) -> int:
    bits = 0
    for raw in ranks:
        rank = int(raw)
        if rank <= 0:
            continue
        if rank < 2 or rank > 14:
            raise ValueError(f"invalid OpenHoldem rank: {rank}")
        bits |= 1 << rank
        if rank == 14:
            bits |= 1 << 1
    return bits


def _rank_hi(bits: int) -> int:
    for rank in range(14, 1, -1):
        if bits & (1 << rank):
            return rank
    return 0


def _rank_lo(bits: int) -> int:
    for rank in range(2, 15):
        if bits & (1 << rank):
            return rank
    return 0


def _straight_high(ranks: Iterable[int]) -> int:
    present = set(int(x) for x in ranks if int(x) > 0)
    for high in range(14, 4, -1):
        if all(rank in present for rank in range(high - 4, high + 1)):
            return high
    if {14, 5, 4, 3, 2}.issubset(present):
        return 5
    return 0


def _straight_nibbles(high: int) -> tuple[int, int, int, int, int]:
    if high == 5:
        return (5, 4, 3, 2, 1)
    return tuple(high - offset for offset in range(5))  # type: ignore[return-value]


def _five_value(cards: tuple[tuple[int, int], ...]) -> HandValue:
    if len(cards) != 5:
        raise ValueError("_five_value requires five cards")
    ranks = [rank for rank, _ in cards]
    suits = [suit for _, suit in cards]
    counts = {rank: ranks.count(rank) for rank in set(ranks)}
    flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)

    if flush and straight_high:
        nibbles = _straight_nibbles(straight_high)
        return HandValue(
            CATEGORY_STRAIGHT_FLUSH,
            (straight_high,),
            nibbles,
            suits[0],
        )

    quads = sorted((rank for rank, n in counts.items() if n == 4), reverse=True)
    if quads:
        quad = quads[0]
        kicker = max(rank for rank in ranks if rank != quad)
        return HandValue(
            CATEGORY_QUADS,
            (quad, kicker),
            (quad, quad, quad, quad, kicker),
        )

    trips = sorted((rank for rank, n in counts.items() if n == 3), reverse=True)
    pairs = sorted((rank for rank, n in counts.items() if n >= 2), reverse=True)
    if trips:
        pair_candidates = [rank for rank in pairs if rank != trips[0]]
        if pair_candidates:
            trip, pair = trips[0], pair_candidates[0]
            return HandValue(
                CATEGORY_FULL_HOUSE,
                (trip, pair),
                (trip, trip, trip, pair, pair),
            )

    if flush:
        ordered = tuple(sorted(ranks, reverse=True))
        return HandValue(CATEGORY_FLUSH, ordered, ordered, suits[0])

    if straight_high:
        nibbles = _straight_nibbles(straight_high)
        return HandValue(CATEGORY_STRAIGHT, (straight_high,), nibbles)

    if trips:
        trip = trips[0]
        kickers = tuple(sorted((rank for rank in ranks if rank != trip), reverse=True))
        return HandValue(
            CATEGORY_TRIPS,
            (trip, *kickers),
            (trip, trip, trip, kickers[0], kickers[1]),
        )

    exact_pairs = sorted((rank for rank, n in counts.items() if n == 2), reverse=True)
    if len(exact_pairs) >= 2:
        top, bottom = exact_pairs[:2]
        kicker = max(rank for rank in ranks if rank not in (top, bottom))
        return HandValue(
            CATEGORY_TWO_PAIR,
            (top, bottom, kicker),
            (top, top, bottom, bottom, kicker),
        )
    if len(exact_pairs) == 1:
        pair = exact_pairs[0]
        kickers = tuple(sorted((rank for rank in ranks if rank != pair), reverse=True))
        return HandValue(
            CATEGORY_PAIR,
            (pair, *kickers),
            (pair, pair, kickers[0], kickers[1], kickers[2]),
        )

    ordered = tuple(sorted(ranks, reverse=True))
    return HandValue(CATEGORY_HIGH, ordered, ordered)


def _partial_value(cards: tuple[tuple[int, int], ...]) -> HandValue:
    """Hand_EVAL_N-compatible ordering for fewer than five cards."""
    if not cards:
        return HandValue(CATEGORY_HIGH, (), (0, 0, 0, 0, 0))
    ranks = [rank for rank, _ in cards]
    counts = {rank: ranks.count(rank) for rank in set(ranks)}

    quads = sorted((rank for rank, n in counts.items() if n == 4), reverse=True)
    if quads:
        quad = quads[0]
        return HandValue(CATEGORY_QUADS, (quad,), (quad, quad, quad, quad, 0))

    trips = sorted((rank for rank, n in counts.items() if n == 3), reverse=True)
    pairs = sorted((rank for rank, n in counts.items() if n == 2), reverse=True)
    if trips:
        trip = trips[0]
        kickers = tuple(sorted((rank for rank in ranks if rank != trip), reverse=True))
        padded = list(kickers[:2]) + [0, 0]
        return HandValue(
            CATEGORY_TRIPS,
            (trip, *kickers),
            (trip, trip, trip, padded[0], padded[1]),
        )
    if len(pairs) >= 2:
        top, bottom = pairs[:2]
        return HandValue(
            CATEGORY_TWO_PAIR,
            (top, bottom),
            (top, top, bottom, bottom, 0),
        )
    if len(pairs) == 1:
        pair = pairs[0]
        kickers = tuple(sorted((rank for rank in ranks if rank != pair), reverse=True))
        padded = list(kickers[:3]) + [0, 0, 0]
        return HandValue(
            CATEGORY_PAIR,
            (pair, *kickers),
            (pair, pair, padded[0], padded[1], padded[2]),
        )
    ordered = tuple(sorted(ranks, reverse=True))
    padded = list(ordered[:5]) + [0, 0, 0, 0, 0]
    return HandValue(
        CATEGORY_HIGH,
        ordered,
        tuple(padded[:5]),  # type: ignore[arg-type]
    )


def evaluate_cards(cards: Iterable[tuple[int, int]]) -> HandValue:
    items = tuple((int(rank), int(suit)) for rank, suit in cards)
    if len(items) <= 4:
        return _partial_value(items)
    if len(items) > 7:
        raise ValueError("Hold'em evaluator supports at most seven cards")
    return max(
        (_five_value(tuple(combo)) for combo in combinations(items, 5)),
        key=lambda value: (value.category, value.tie),
    )


def pokerval(value: HandValue) -> int:
    out = int(POKERVAL_BASE[value.category])
    for shift, rank in zip((16, 12, 8, 4, 0), value.nibbles):
        out += int(rank) << shift
    return out


def _dominant_suit(cards: Iterable[tuple[int, int]]) -> tuple[int, int]:
    counts = {suit: 0 for suit in range(4)}
    for _, suit in cards:
        if suit >= 0:
            counts[int(suit)] += 1
    # OpenHoldem CSymbolEngineCards::GetDominantSuit tie order:
    # clubs, diamonds, hearts, spades.
    for suit in (2, 1, 0, 3):
        if counts[suit] == max(counts.values(), default=0):
            return suit, counts[suit]
    return 0, 0


def _straight_metrics(ranks: Iterable[int]) -> tuple[int, int]:
    bits = _rankbits(ranks)
    best_connected = 0
    best_fill = 5
    for shift in range(10, 0, -1):
        window = (bits >> shift) & 0x1F
        if window == 0x1F:
            connected = 5
        elif (window & 0x1E) == 0x1E or (window & 0x0F) == 0x0F:
            connected = 4
        elif (
            (window & 0x1C) == 0x1C
            or (window & 0x0E) == 0x0E
            or (window & 0x07) == 0x07
        ):
            connected = 3
        elif (
            (window & 0x18) == 0x18
            or (window & 0x0C) == 0x0C
            or (window & 0x06) == 0x06
            or (window & 0x03) == 0x03
        ):
            connected = 2
        else:
            connected = 1
        best_connected = max(best_connected, connected)
        best_fill = min(best_fill, 5 - window.bit_count())
    return best_connected, best_fill


def _rank_multiplicity(ranks: Iterable[int]) -> tuple[int, int]:
    values = [int(x) for x in ranks if int(x) > 0]
    if not values:
        return 0, 0
    counts = {rank: values.count(rank) for rank in set(values)}
    maximum = max(counts.values())
    top_rank = max(rank for rank, count in counts.items() if count == maximum)
    return maximum, top_rank


def _card_expression_tokens(text: str) -> tuple[list[int], list[tuple[int, int]], bool]:
    body = text
    suited = body.lower().endswith("suited")
    if suited:
        body = body[:-6]
    ranks: list[int] = []
    exact: list[tuple[int, int]] = []
    index = 0
    while index < len(body):
        char = body[index].upper()
        if char in RANK_FROM_CHAR:
            rank = RANK_FROM_CHAR[char]
            ranks.append(rank)
            if index + 1 < len(body) and body[index + 1].lower() in SUIT_FROM_CHAR:
                exact.append((rank, SUIT_FROM_CHAR[body[index + 1].lower()]))
                index += 2
                continue
        index += 1
    if not ranks:
        raise UnknownDeepCrusherCardSymbol(text)
    return ranks, exact, suited


@dataclass(frozen=True)
class DeepCrusherCardSymbols:
    view: DeepCrusherStateView

    @staticmethod
    def fixed_symbols() -> frozenset[str]:
        return frozenset({
            "rankbits", "rankbitsplayer", "rankbitscommon", "rankbitspoker",
            "rankhi", "rankhicommon", "rankhiplayer", "rankhipoker",
            "ranklo", "ranklocommon", "rankloplayer", "ranklopoker",
            "srankbits", "srankbitsplayer", "srankbitscommon",
            "ispair", "issuited", "isconnector",
            "nsuited", "nsuitedcommon", "tsuit", "tsuitcommon",
            "nstraight", "nstraightcommon", "nstraightfill", "nstraightfillcommon",
            "nranked", "nrankedcommon", "trank", "trankcommon",
            "ncommoncardsknown",
            "pokerval", "pokervalplayer", "pokervalcommon",
            "pcbits", "npcbits",
            "ishicard", "isonepair", "istwopair", "isthreeofakind",
            "isstraight", "isflush", "isfullhouse", "isfourofakind",
            "isstraightflush", "isroyalflush",
        })

    @classmethod
    def supports(cls, name: str) -> bool:
        low = str(name).lower()
        return (
            low in {x.lower() for x in cls.fixed_symbols()}
            or low.startswith("hand$")
            or low.startswith("board$")
            or re.fullmatch(r"\$\$(?:pr|ps)[01]", low) is not None
            or re.fullmatch(r"\$\$(?:cr|cs)[0-4]", low) is not None
        )

    def _slots(self) -> tuple[tuple[int, int], ...]:
        suits = self.view.openholdem_suits
        return tuple(
            (int(rank), int(suits[index]))
            for index, rank in enumerate(self.view.ranks)
            if int(rank) > 0
        )

    def _hole(self) -> tuple[tuple[int, int], tuple[int, int]]:
        suits = self.view.openholdem_suits
        return (
            (int(self.view.ranks[0]), int(suits[0])),
            (int(self.view.ranks[1]), int(suits[1])),
        )

    def _board(self) -> tuple[tuple[int, int], ...]:
        suits = self.view.openholdem_suits
        return tuple(
            (int(self.view.ranks[index]), int(suits[index]))
            for index in range(2, 7)
            if int(self.view.ranks[index]) > 0
        )

    def _value(self) -> HandValue:
        return evaluate_cards(self._slots())

    def _player_value(self) -> HandValue:
        return evaluate_cards(self._hole())

    def _common_value(self) -> HandValue:
        return evaluate_cards(self._board())

    def _pcbits(self) -> int:
        value = self._value()
        hole = self._hole()
        hole_ranks = (hole[0][0], hole[1][0])
        bits = 0
        nibbles = value.nibbles

        if value.category in (CATEGORY_STRAIGHT, CATEGORY_STRAIGHT_FLUSH):
            for index, rank in enumerate(nibbles):
                wanted = 14 if rank == 1 else rank
                for hole_rank, hole_suit in hole:
                    if hole_rank != wanted:
                        continue
                    if value.category == CATEGORY_STRAIGHT_FLUSH and hole_suit != value.flush_suit:
                        continue
                    bits |= 1 << (4 - index)
                    break
            return bits

        if value.category == CATEGORY_FLUSH:
            for index, rank in enumerate(nibbles):
                if any(
                    hole_rank == rank and hole_suit == value.flush_suit
                    for hole_rank, hole_suit in hole
                ):
                    bits |= 1 << (4 - index)
            return bits

        if value.category == CATEGORY_QUADS:
            quad, kicker = value.tie[0], value.tie[1] if len(value.tie) > 1 else 0
            matches = sum(1 for rank in hole_ranks if rank == quad)
            if matches >= 1:
                bits |= 1 << 4
            if matches >= 2:
                bits |= 1 << 3
            if kicker and any(rank == kicker for rank in hole_ranks):
                bits |= 1 << 0
            return bits

        if value.category == CATEGORY_FULL_HOUSE:
            trip, pair = value.tie[:2]
            trip_matches = sum(1 for rank in hole_ranks if rank == trip)
            pair_matches = sum(1 for rank in hole_ranks if rank == pair)
            if trip_matches >= 1:
                bits |= 1 << 4
            if trip_matches >= 2:
                bits |= 1 << 3
            if pair_matches >= 1:
                bits |= 1 << 1
            if pair_matches >= 2:
                bits |= 1 << 0
            return bits

        if value.category == CATEGORY_TRIPS:
            trip = value.tie[0]
            trip_matches = sum(1 for rank in hole_ranks if rank == trip)
            if trip_matches >= 1:
                bits |= 1 << 4
            if trip_matches >= 2:
                bits |= 1 << 3
            # Preserve OpenHoldem's historical implementation: both kicker
            # tests map to bit 1, so at most one kicker bit is counted.
            for kicker in value.tie[1:3]:
                if any(rank == kicker for rank in hole_ranks):
                    bits |= 1 << 1
            return bits

        if value.category == CATEGORY_TWO_PAIR:
            top, bottom = value.tie[:2]
            top_matches = sum(1 for rank in hole_ranks if rank == top)
            bottom_matches = sum(1 for rank in hole_ranks if rank == bottom)
            if top_matches >= 1:
                bits |= 1 << 4
            if top_matches >= 2:
                bits |= 1 << 3
            if bottom_matches >= 1:
                bits |= 1 << 2
            if bottom_matches >= 2:
                bits |= 1 << 1
            if len(value.tie) >= 3 and any(rank == value.tie[2] for rank in hole_ranks):
                bits |= 1 << 0
            return bits

        if value.category == CATEGORY_PAIR:
            pair = value.tie[0]
            pair_matches = sum(1 for rank in hole_ranks if rank == pair)
            if pair_matches >= 1:
                bits |= 1 << 4
            if pair_matches >= 2:
                bits |= 1 << 3
            for slot, kicker in zip((2, 1, 0), value.tie[1:4]):
                if any(rank == kicker for rank in hole_ranks):
                    bits |= 1 << slot
            return bits

        for index, rank in enumerate(nibbles):
            if rank and any(hole_rank == rank for hole_rank in hole_ranks):
                bits |= 1 << (4 - index)
        return bits

    def _rankbitspoker(self) -> int:
        value = pokerval(self._value())
        bits = 0
        for shift in (16, 12, 8, 4, 0):
            bits |= 1 << ((value >> shift) & 0xF)
        if bits & (1 << 14):
            bits |= 1 << 1
        return bits

    def _suited_rankbits(self, cards: tuple[tuple[int, int], ...]) -> int:
        suit, _ = _dominant_suit(cards)
        return _rankbits(rank for rank, card_suit in cards if card_suit == suit)

    def _card_expression(self, name: str) -> float:
        match = _CARD_EXPR.fullmatch(name)
        if match is None:
            raise UnknownDeepCrusherCardSymbol(name)
        source = self._hole() if match.group(1).lower() == "hand" else self._board()
        wanted_ranks, exact_cards, suited = _card_expression_tokens(match.group(2))

        available_counts: dict[int, int] = {}
        for rank, _ in source:
            available_counts[rank] = available_counts.get(rank, 0) + 1
        wanted_counts: dict[int, int] = {}
        for rank in wanted_ranks:
            wanted_counts[rank] = wanted_counts.get(rank, 0) + 1
        if any(available_counts.get(rank, 0) < count for rank, count in wanted_counts.items()):
            return 0.0

        if suited:
            if match.group(1).lower() == "hand":
                if len(source) < 2 or source[0][1] != source[1][1]:
                    return 0.0
            else:
                search_bits = _rankbits(wanted_ranks)
                dominant_bits = self._suited_rankbits(source)
                if (search_bits & dominant_bits) != search_bits:
                    return 0.0

        for exact in exact_cards:
            if exact not in source:
                return 0.0
        return 1.0

    def resolve(self, name: str) -> float:
        low = str(name).lower()
        if low.startswith(("hand$", "board$")):
            return self._card_expression(name)

        hole = self._hole()
        board = self._board()
        all_cards = hole + board
        hole_ranks = tuple(rank for rank, _ in hole)
        board_ranks = tuple(rank for rank, _ in board)
        all_ranks = tuple(rank for rank, _ in all_cards)

        rankbits_player = _rankbits(hole_ranks)
        rankbits_common = _rankbits(board_ranks)
        rankbits_all = rankbits_player | rankbits_common

        if low == "rankbits":
            return float(rankbits_all)
        if low == "rankbitsplayer":
            return float(rankbits_player)
        if low == "rankbitscommon":
            return float(rankbits_common)
        if low == "rankbitspoker":
            return float(self._rankbitspoker())

        if low == "rankhi":
            return float(_rank_hi(rankbits_all))
        if low == "rankhicommon":
            return float(_rank_hi(rankbits_common))
        if low == "rankhiplayer":
            return float(_rank_hi(rankbits_player))
        if low == "rankhipoker":
            return float(_rank_hi(self._rankbitspoker()))
        if low == "ranklo":
            return float(_rank_lo(rankbits_all))
        if low == "ranklocommon":
            return float(_rank_lo(rankbits_common))
        if low == "rankloplayer":
            return float(_rank_lo(rankbits_player))
        if low == "ranklopoker":
            return float(_rank_lo(self._rankbitspoker()))

        if low == "srankbits":
            return float(self._suited_rankbits(all_cards))
        if low == "srankbitsplayer":
            return float(self._suited_rankbits(hole))
        if low == "srankbitscommon":
            return float(self._suited_rankbits(board))

        if low == "ispair":
            return float(hole_ranks[0] == hole_ranks[1])
        if low == "issuited":
            return float(hole[0][1] == hole[1][1])
        if low == "isconnector":
            return float(abs(hole_ranks[0] - hole_ranks[1]) == 1)

        if low in ("nsuited", "tsuit"):
            suit, count = _dominant_suit(all_cards)
            return float(count if low == "nsuited" else suit)
        if low in ("nsuitedcommon", "tsuitcommon"):
            suit, count = _dominant_suit(board)
            return float(count if low == "nsuitedcommon" else suit)

        if low in ("nstraight", "nstraightfill"):
            connected, fill = _straight_metrics(all_ranks)
            return float(connected if low == "nstraight" else fill)
        if low in ("nstraightcommon", "nstraightfillcommon"):
            if not board_ranks:
                connected, fill = 0, 5
            else:
                connected, fill = _straight_metrics(board_ranks)
            return float(connected if low == "nstraightcommon" else fill)

        if low in ("nranked", "trank"):
            count, rank = _rank_multiplicity(all_ranks)
            return float(count if low == "nranked" else rank)
        if low in ("nrankedcommon", "trankcommon"):
            count, rank = _rank_multiplicity(board_ranks)
            return float(count if low == "nrankedcommon" else rank)
        if low == "ncommoncardsknown":
            return float(len(board))

        value = self._value()
        if low == "pokerval":
            return float(pokerval(value))
        if low == "pokervalplayer":
            return float(pokerval(self._player_value()))
        if low == "pokervalcommon":
            return float(pokerval(self._common_value()))
        if low == "pcbits":
            return float(self._pcbits())
        if low == "npcbits":
            return float(self._pcbits().bit_count())

        category_symbols = {
            "ishicard": CATEGORY_HIGH,
            "isonepair": CATEGORY_PAIR,
            "istwopair": CATEGORY_TWO_PAIR,
            "isthreeofakind": CATEGORY_TRIPS,
            "isstraight": CATEGORY_STRAIGHT,
            "isflush": CATEGORY_FLUSH,
            "isfullhouse": CATEGORY_FULL_HOUSE,
            "isfourofakind": CATEGORY_QUADS,
            "isstraightflush": CATEGORY_STRAIGHT_FLUSH,
        }
        if low in category_symbols:
            return float(value.category == category_symbols[low])
        if low == "isroyalflush":
            return float(
                value.category == CATEGORY_STRAIGHT_FLUSH
                and value.tie
                and value.tie[0] == 14
            )

        # Technical card symbols used by the pinned OpenPPL library.
        match = re.fullmatch(r"\$\$(pr|ps)([01])", low)
        if match:
            index = int(match.group(2))
            return float(hole[index][0] if match.group(1) == "pr" else hole[index][1])
        match = re.fullmatch(r"\$\$(cr|cs)([0-4])", low)
        if match:
            index = int(match.group(2))
            if index >= len(board):
                return -1.0
            return float(board[index][0] if match.group(1) == "cr" else board[index][1])

        raise UnknownDeepCrusherCardSymbol(name)

    def __call__(self, name: str) -> float:
        return self.resolve(name)
