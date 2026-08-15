from __future__ import annotations

from itertools import permutations

from spincore_nn.codec_v3 import DecodedInputV3

# Only true simultaneous-card symmetries are quotiented here. Turn and river
# keep their temporal roles and are never permuted.
_HOLE_PERMS = tuple(permutations((0, 1)))
_FLOP_PERMS = tuple(permutations((2, 3, 4)))
_UPPER = tuple((left, right) for left in range(7) for right in range(left + 1, 7))


def canonical_card_orbit_key_v3(item: DecodedInputV3) -> tuple[int, ...]:
    """Exact card-state orbit representative for SPNNIV3.

    The SPNNIV3 carrier already removes physical suit names by storing only the
    pairwise same-suit equivalence relation. This function removes the remaining
    strategically meaningless ordering of Hero's two private cards and the three
    simultaneously dealt flop cards.

    The returned key is lossless modulo exactly those symmetries: it contains all
    seven rank slots followed by all 21 pairwise same-suit bits after choosing the
    lexicographically minimal member of the 2! x 3! orbit. Turn and river remain
    distinct ordered slots.
    """
    ranks = tuple(int(value) for value in item.rank_tokens)
    matrix = item.same_suit_matrix()
    if len(ranks) != 7 or len(matrix) != 7:
        raise ValueError("unexpected SPNNIV3 card dimensions")

    best: tuple[int, ...] | None = None
    for hole in _HOLE_PERMS:
        for flop in _FLOP_PERMS:
            source = (hole[0], hole[1], flop[0], flop[1], flop[2], 5, 6)
            candidate_ranks = tuple(ranks[index] for index in source)
            candidate_suits = tuple(
                int(matrix[source[left]][source[right]])
                for left, right in _UPPER
            )
            candidate = candidate_ranks + candidate_suits
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise AssertionError("empty SPNNIV3 card orbit")
    return best


def canonical_flop_orbit_key_from_physical_cards(
    flop: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> tuple[int, ...]:
    """Reference flop-only orbit key used by exhaustive invariance regression.

    Physical suit labels are discarded; only rank and equality relations remain.
    """
    if len(set(flop)) != 3:
        raise ValueError("flop cards must be physically distinct")
    best: tuple[int, ...] | None = None
    for perm in permutations(flop):
        ranks = tuple(int(card[0]) for card in perm)
        relations = (
            int(perm[0][1] == perm[1][1]),
            int(perm[0][1] == perm[2][1]),
            int(perm[1][1] == perm[2][1]),
        )
        candidate = ranks + relations
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise AssertionError("empty flop orbit")
    return best
