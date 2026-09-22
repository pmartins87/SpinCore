from __future__ import annotations

"""Heuristic strategic sanity audit for offline SpinCore benchmark decisions.

These flags are diagnostics, not poker-theory verdicts.  They intentionally
surface decisions that deserve human inspection (for example AA folds, deep
72o jams, or strong postflop folds) while preserving the full betting context.
"""

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable

from spincore.deepcrusher_benchmark import DecisionTrace


CATEGORY_NAMES = {
    0: "HIGH_CARD",
    1: "PAIR",
    2: "TWO_PAIR",
    3: "TRIPS",
    4: "STRAIGHT",
    5: "FLUSH",
    6: "FULL_HOUSE",
    7: "QUADS",
    8: "STRAIGHT_FLUSH",
}


def card_rank(card_id: int) -> int:
    value = int(card_id)
    if value < 0 or value >= 52:
        raise ValueError(f"card id outside 0..51: {value}")
    return 2 + value // 4


def card_suit(card_id: int) -> int:
    value = int(card_id)
    if value < 0 or value >= 52:
        raise ValueError(f"card id outside 0..51: {value}")
    return value % 4


def card_text(card_id: int) -> str:
    ranks = "23456789TJQKA"
    suits = "shdc"
    return ranks[card_rank(card_id) - 2] + suits[card_suit(card_id)]


def _straight_high(ranks: Iterable[int]) -> int:
    values = set(int(x) for x in ranks)
    for hi in range(14, 4, -1):
        if all(r in values for r in range(hi - 4, hi + 1)):
            return hi
    if {14, 5, 4, 3, 2}.issubset(values):
        return 5
    return 0


def _evaluate_five(cards: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    ranks = [card_rank(c) for c in cards]
    suits = [card_suit(c) for c in cards]
    counts = {r: ranks.count(r) for r in set(ranks)}
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
    flush = len(set(suits)) == 1
    straight = _straight_high(ranks)

    if flush and straight:
        return (8, (straight,))
    if groups[0][0] == 4:
        quad = groups[0][1]
        kicker = max(r for r in ranks if r != quad)
        return (7, (quad, kicker))
    trips = sorted((r for r, n in counts.items() if n == 3), reverse=True)
    pairs = sorted((r for r, n in counts.items() if n >= 2), reverse=True)
    if trips:
        pair_candidates = [r for r in pairs if r != trips[0]]
        if pair_candidates:
            return (6, (trips[0], pair_candidates[0]))
    if flush:
        return (5, tuple(sorted(ranks, reverse=True)))
    if straight:
        return (4, (straight,))
    if trips:
        kickers = sorted((r for r in ranks if r != trips[0]), reverse=True)[:2]
        return (3, (trips[0], *kickers))
    exact_pairs = sorted((r for r, n in counts.items() if n == 2), reverse=True)
    if len(exact_pairs) >= 2:
        p1, p2 = exact_pairs[:2]
        kicker = max(r for r in ranks if r not in (p1, p2))
        return (2, (p1, p2, kicker))
    if len(exact_pairs) == 1:
        p = exact_pairs[0]
        kickers = sorted((r for r in ranks if r != p), reverse=True)[:3]
        return (1, (p, *kickers))
    return (0, tuple(sorted(ranks, reverse=True)))


def evaluate_visible_hand(hole: tuple[int, int], board: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    cards = tuple(hole) + tuple(board)
    if len(cards) < 5 or len(cards) > 7:
        raise ValueError("postflop visible hand requires 5..7 cards")
    return max(_evaluate_five(tuple(combo)) for combo in combinations(cards, 5))


def preflop_class(hole: tuple[int, int]) -> str:
    r1, r2 = card_rank(hole[0]), card_rank(hole[1])
    hi, lo = max(r1, r2), min(r1, r2)
    names = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T"}
    a = names.get(hi, str(hi))
    b = names.get(lo, str(lo))
    if hi == lo:
        return a + b
    return a + b + ("s" if card_suit(hole[0]) == card_suit(hole[1]) else "o")


def _bb_from_blind(blind: str) -> int:
    try:
        return int(str(blind).split("/", 1)[1])
    except Exception as exc:
        raise ValueError(f"invalid blind label: {blind!r}") from exc


def effective_stack_bb(trace: DecisionTrace) -> float:
    bb = _bb_from_blind(trace.blind)
    if bb <= 0:
        return 0.0
    hero = int(trace.stacks[trace.actor])
    opponents = [
        int(trace.stacks[s])
        for s in range(3)
        if s != trace.actor and trace.lineup[s] != "DEAD"
    ]
    if not opponents:
        return 0.0
    return min(hero, max(opponents)) / float(bb)


@dataclass(frozen=True)
class SanityFlag:
    code: str
    severity: str
    reason: str
    context: dict[str, object]


def _base_context(trace: DecisionTrace) -> dict[str, object]:
    hole = trace.hole_cards
    out: dict[str, object] = {
        "scenario_index": trace.scenario_index,
        "domain": trace.domain,
        "blind": trace.blind,
        "lineup_index": trace.lineup_index,
        "decision_index": trace.decision_index,
        "actor": trace.actor,
        "policy_id": trace.policy_id,
        "street": trace.street,
        "pot": trace.pot,
        "to_call": trace.to_call,
        "effective_stack_bb": effective_stack_bb(trace),
        "action_type": trace.action_type,
        "action_name": trace.action_name,
        "amount_to": trace.amount_to,
        "hole": [card_text(x) for x in hole] if hole is not None else None,
        "preflop_class": preflop_class(hole) if hole is not None else None,
        "board": [card_text(x) for x in trace.board],
    }
    if trace.pot > 0:
        out["to_call_over_pot"] = trace.to_call / float(trace.pot)
    return out


def sanity_flags(trace: DecisionTrace) -> tuple[SanityFlag, ...]:
    if trace.policy_id != "SPINCORE" or trace.hole_cards is None:
        return ()

    flags: list[SanityFlag] = []
    hole = trace.hole_cards
    cls = preflop_class(hole)
    ctx = _base_context(trace)

    # Preflop: these are high-value diagnostics, but 72o jams are only surfaced
    # at non-trivial depth because very short-stack shove ranges can be extremely wide.
    if trace.visible_board_count == 0:
        if trace.action_name == "FOLD" and cls == "AA":
            flags.append(SanityFlag(
                "PREFLOP_AA_FOLD",
                "CRITICAL",
                "SpinCore folded pocket aces preflop.",
                ctx,
            ))
        if (
            trace.action_name == "ALL_IN"
            and cls == "72o"
            and effective_stack_bb(trace) >= 10.0
        ):
            flags.append(SanityFlag(
                "PREFLOP_DEEP_72O_JAM",
                "HIGH_REVIEW",
                "SpinCore jammed 72o with at least 10bb effective.",
                ctx,
            ))
        return tuple(flags)

    category, kickers = evaluate_visible_hand(hole, trace.board)
    ctx = dict(ctx)
    ctx["made_hand_category"] = CATEGORY_NAMES[category]
    ctx["hand_rank_tiebreak"] = list(kickers)

    board_ranks = [card_rank(x) for x in trace.board]
    hole_ranks = [card_rank(x) for x in hole]
    just_top_pair = (
        category == 1
        and bool(board_ranks)
        and max(board_ranks) in hole_ranks
        and board_ranks.count(max(board_ranks)) == 1
    )
    if just_top_pair:
        matching = max(board_ranks)
        other = [r for r in hole_ranks if r != matching]
        ctx["top_pair_rank"] = matching
        ctx["top_pair_kicker"] = max(other) if other else matching

    if trace.action_name == "FOLD":
        if category >= 6:
            flags.append(SanityFlag(
                "POSTFLOP_MONSTER_FOLD",
                "CRITICAL",
                "SpinCore folded a full house or stronger made hand.",
                ctx,
            ))
        elif category >= 3:
            flags.append(SanityFlag(
                "POSTFLOP_TRIPS_PLUS_FOLD",
                "HIGH_REVIEW",
                "SpinCore folded trips or a stronger made hand; inspect betting context.",
                ctx,
            ))
        elif just_top_pair:
            flags.append(SanityFlag(
                "POSTFLOP_TOP_PAIR_FOLD",
                "REVIEW",
                "SpinCore folded top pair; this can be correct, so inspect price, kicker and board.",
                ctx,
            ))

    if (
        trace.action_name == "ALL_IN"
        and category == 0
        and effective_stack_bb(trace) >= 10.0
    ):
        flags.append(SanityFlag(
            "POSTFLOP_DEEP_HIGH_CARD_JAM",
            "REVIEW",
            "SpinCore jammed with only high-card showdown value at at least 10bb effective; draws are not yet excluded.",
            ctx,
        ))

    return tuple(flags)


def flag_to_dict(flag: SanityFlag) -> dict[str, object]:
    return asdict(flag)
