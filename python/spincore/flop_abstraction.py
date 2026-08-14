from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "shdc"
RANK_FROM_CHAR = {c: i + 2 for i, c in enumerate(RANK_CHARS)}
SUIT_FROM_CHAR = {c: i for i, c in enumerate(SUIT_CHARS)}


@dataclass(frozen=True, order=True)
class RSCard:
    rank: int
    suit: int


def decode_spin_card_id(card_id: int) -> RSCard:
    """Decode the authoritative SpinCore Card::id() layout.

    SpinCore uses id = (rank - 2) * 4 + suit.
    """
    card_id = int(card_id)
    if card_id < 0 or card_id >= 52:
        raise ValueError(f"card id out of range: {card_id}")
    return RSCard(rank=2 + card_id // 4, suit=card_id % 4)


def encode_spin_card_id(card: RSCard) -> int:
    if card.rank < 2 or card.rank > 14 or card.suit < 0 or card.suit > 3:
        raise ValueError(f"invalid card: {card}")
    return (card.rank - 2) * 4 + card.suit


def parse_card(text: str) -> RSCard:
    text = str(text).strip()
    if len(text) != 2 or text[0] not in RANK_FROM_CHAR or text[1] not in SUIT_FROM_CHAR:
        raise ValueError(f"invalid card string: {text!r}")
    return RSCard(RANK_FROM_CHAR[text[0]], SUIT_FROM_CHAR[text[1]])


def card_text(card: RSCard) -> str:
    if card.rank < 2 or card.rank > 14 or card.suit < 0 or card.suit > 3:
        raise ValueError(f"invalid card: {card}")
    return RANK_CHARS[card.rank - 2] + SUIT_CHARS[card.suit]


def parse_flop_text(text: str) -> tuple[RSCard, RSCard, RSCard]:
    compact = "".join(str(text).split())
    if len(compact) != 6:
        raise ValueError(f"invalid flop string: {text!r}")
    cards = (parse_card(compact[0:2]), parse_card(compact[2:4]), parse_card(compact[4:6]))
    if len(set(cards)) != 3:
        raise ValueError(f"duplicate card in flop: {text!r}")
    return cards


def _normalize_three_cards(cards: Iterable[int | RSCard]) -> tuple[RSCard, RSCard, RSCard]:
    out: list[RSCard] = []
    for card in cards:
        out.append(card if isinstance(card, RSCard) else decode_spin_card_id(int(card)))
    if len(out) != 3:
        raise ValueError(f"flop must contain exactly 3 cards, got {len(out)}")
    if len(set(out)) != 3:
        raise ValueError("flop contains duplicate cards")
    return out[0], out[1], out[2]


def suit_isomorphic_key(cards: Iterable[int | RSCard]) -> tuple[tuple[int, int], ...]:
    """Exact flop identity modulo a global permutation of the four suits.

    This is lossless for Hold'em flop strategy with respect to absolute suit names:
    it removes only S/H/D/C naming symmetry.  The key is also invariant to flop-card
    order because a flop is an unordered three-card set.
    """
    flop = _normalize_three_cards(cards)
    best: tuple[tuple[int, int], ...] | None = None
    for perm in itertools.permutations(range(4)):
        candidate = tuple(sorted((c.rank, perm[c.suit]) for c in flop))
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


def suit_isomorphic_text(cards: Iterable[int | RSCard]) -> str:
    key = suit_isomorphic_key(cards)
    return "".join(card_text(RSCard(rank=r, suit=s)) for r, s in key)


def _straight_windows() -> tuple[frozenset[int], ...]:
    windows: list[frozenset[int]] = [frozenset({14, 2, 3, 4, 5})]
    for low in range(2, 11):
        windows.append(frozenset(range(low, low + 5)))
    return tuple(windows)


STRAIGHT_WINDOWS = _straight_windows()


def is_connected_by_two_added_cards(ranks: Sequence[int]) -> bool:
    """User-proposed C/D definition for an unpaired flop.

    Connected means the three distinct flop ranks can all belong to at least one
    five-rank straight, i.e. exactly two additional ranks can complete a straight.
    """
    distinct = frozenset(int(r) for r in ranks)
    if len(distinct) != 3:
        return False
    return any(distinct.issubset(window) for window in STRAIGHT_WINDOWS)


def flop53_key(cards: Iterable[int | RSCard]) -> str:
    """Reproduce the 53-class proposal exactly as a diagnostic/control taxonomy.

    Four S/N rank-band flags:
      A | KQJ | T987 | 65432
    Suit flag:
      R = rainbow, F = at least two cards share a suit
    Shape flag:
      P = paired/trips (takes precedence), C = connected, D = disconnected
    """
    flop = _normalize_three_cards(cards)
    ranks = [c.rank for c in flop]
    suits = [c.suit for c in flop]

    rank_flags = (
        any(r == 14 for r in ranks),
        any(r in (13, 12, 11) for r in ranks),
        any(r in (10, 9, 8, 7) for r in ranks),
        any(r in (6, 5, 4, 3, 2) for r in ranks),
    )
    prefix = "".join("S" if flag else "N" for flag in rank_flags)
    suit_flag = "R" if len(set(suits)) == 3 else "F"

    if len(set(ranks)) < 3:
        shape = "P"
    else:
        shape = "C" if is_connected_by_two_added_cards(ranks) else "D"
    return prefix + suit_flag + shape


def all_physical_flops() -> Iterable[tuple[int, int, int]]:
    return itertools.combinations(range(52), 3)


def reference_counts() -> dict[str, object]:
    physical = 0
    iso: Counter[tuple[tuple[int, int], ...]] = Counter()
    c53: Counter[str] = Counter()
    for flop in all_physical_flops():
        physical += 1
        iso[suit_isomorphic_key(flop)] += 1
        c53[flop53_key(flop)] += 1

    return {
        "physical_flops": physical,
        "suit_isomorphic_classes": len(iso),
        "flop53_classes": len(c53),
        "suit_isomorphic_class_size_min": min(iso.values()),
        "suit_isomorphic_class_size_max": max(iso.values()),
        "flop53_class_size_min": min(c53.values()),
        "flop53_class_size_max": max(c53.values()),
        "flop53_counts": dict(sorted(c53.items())),
    }


def audit_legacy_184_summary(path: str | Path) -> dict[str, object]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"centroide", "rotulo", "quantidade_flops"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError(f"legacy 184 CSV missing required columns: {sorted(required)}")

    sizes = [int(row["quantidade_flops"]) for row in rows]
    labels = {str(row["rotulo"]) for row in rows}
    centroids = [str(row["centroide"]) for row in rows]
    return {
        "rows": len(rows),
        "unique_centroids": len(set(centroids)),
        "unique_labels": len(labels),
        "physical_flops_claimed": sum(sizes),
        "bucket_size_min": min(sizes),
        "bucket_size_max": max(sizes),
        "bucket_size_mean": sum(sizes) / len(sizes),
        "centroids_parseable": all(_flop_text_parseable(x) for x in centroids),
        "summary_shape_pass": len(rows) == 184 and len(set(centroids)) == 184 and sum(sizes) == 22100,
    }


def _flop_text_parseable(text: str) -> bool:
    try:
        parse_flop_text(text)
        return True
    except ValueError:
        return False


def _unordered_physical_key(cards: Iterable[RSCard]) -> tuple[int, int, int]:
    return tuple(sorted(encode_spin_card_id(c) for c in cards))  # type: ignore[return-value]


def audit_legacy_184_mapping(path: str | Path) -> dict[str, object]:
    """Audit the historical 184Flops.json if/when it is recovered.

    The legacy DLL tried all six flop-card order permutations when looking up a
    mapping entry, so this audit normalizes JSON keys to unordered physical flops.
    It does not assume one particular key order in the file.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("legacy mapping JSON must be an object")

    normalized: dict[tuple[int, int, int], str] = {}
    duplicate_conflicts: list[str] = []
    invalid_rows: list[str] = []
    representative_texts: set[str] = set()

    for key, value in raw.items():
        try:
            physical = _unordered_physical_key(parse_flop_text(str(key)))
            representative = "".join(card_text(c) for c in parse_flop_text(str(value)))
        except ValueError:
            invalid_rows.append(str(key))
            continue
        representative_texts.add(representative)
        prior = normalized.get(physical)
        if prior is not None and prior != representative:
            duplicate_conflicts.append(str(key))
        normalized[physical] = representative

    all_physical = set(all_physical_flops())
    mapped_physical = set(normalized)
    missing = all_physical - mapped_physical
    extra = mapped_physical - all_physical

    # Absolute suit names must not split an exact suit-isomorphic flop class into
    # strategically different representatives.  Representative suit names are
    # canonicalized before comparison because the historical file may choose a
    # different physical suit spelling for equivalent representatives.
    reps_by_iso: defaultdict[tuple[tuple[int, int], ...], set[tuple[tuple[int, int], ...]]] = defaultdict(set)
    for physical, representative in normalized.items():
        reps_by_iso[suit_isomorphic_key(physical)].add(suit_isomorphic_key(parse_flop_text(representative)))
    suit_iso_splits = sum(1 for reps in reps_by_iso.values() if len(reps) > 1)

    canonical_reps = {
        suit_isomorphic_key(parse_flop_text(rep))
        for rep in representative_texts
    }

    return {
        "json_rows": len(raw),
        "normalized_physical_flops": len(normalized),
        "unique_representative_texts": len(representative_texts),
        "unique_representatives_mod_suit": len(canonical_reps),
        "invalid_rows": len(invalid_rows),
        "duplicate_conflicts": len(duplicate_conflicts),
        "missing_physical_flops": len(missing),
        "extra_physical_flops": len(extra),
        "suit_isomorphic_input_classes_split": suit_iso_splits,
        "complete_22100_pass": len(normalized) == 22100 and not missing and not extra,
        "suit_permutation_invariance_pass": suit_iso_splits == 0,
    }
