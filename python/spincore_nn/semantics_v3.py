from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from spincore_nn.codec_v3 import DecodedInputV3

# Hand categories match the C++ HandCategory ordering.
HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8

_STRAIGHT_WINDOWS = (
    (14, 2, 3, 4, 5),
    (2, 3, 4, 5, 6),
    (3, 4, 5, 6, 7),
    (4, 5, 6, 7, 8),
    (5, 6, 7, 8, 9),
    (6, 7, 8, 9, 10),
    (7, 8, 9, 10, 11),
    (8, 9, 10, 11, 12),
    (9, 10, 11, 12, 13),
    (10, 11, 12, 13, 14),
)


@dataclass(frozen=True, order=True)
class _FiveRank:
    category: int
    kickers: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class ObjectiveSemanticsV3:
    # Board texture, all label-free.
    board_distinct_ranks: int
    board_max_rank_multiplicity: int
    board_pair_rank_count: int
    board_trip_rank_count: int
    board_quad_rank_count: int
    board_distinct_suit_classes: int
    board_max_suit_count: int
    board_high_rank: int
    board_low_rank: int
    board_rank_span: int
    board_broadway_count: int
    board_max_straight_window_occupancy: int
    board_straight_windows_3plus: int
    board_straight_windows_4plus: int
    board_has_straight: int
    board_has_flush: int

    # Exact made hand / objective Hero contribution.
    made_category: int
    board_only_category: int
    best_hand_hole_cards_min: int
    best_hand_hole_cards_max: int
    pocket_pair: int
    hole_rank_match_count: int
    hole_board_multiplicity_low: int
    hole_board_multiplicity_high: int
    overcard_count: int
    highest_matched_board_rank_tier: int

    # Objective one-card draw geometry. No Good/Mid/Weak labels.
    already_has_straight: int
    already_has_flush: int
    straight_completion_card_count: int
    straight_completion_distinct_rank_count: int
    board_straight_completion_card_count: int
    hero_adds_to_straight_draw: int
    flush_completion_card_count: int
    board_flush_completion_card_count: int
    hero_adds_to_flush_draw: int
    flush_draw_highest_hero_rank: int
    flush_draw_higher_unseen_count: int
    backdoor_straight: int
    backdoor_flush: int
    combo_draw: int
    pair_plus_draw: int

    # Objective public lineage/history summary.
    preflop_limper_count: int
    preflop_aggression_count: int
    preflop_pot_family: int  # 0 unopened, 1 limp, 2 SRP, 3 3BP, 4 4BP+
    preflop_first_aggressor_rel_plus1: int
    preflop_last_aggressor_rel_plus1: int
    hero_called_last_preflop_aggression: int
    current_street_aggression_count: int
    current_street_first_aggressor_rel_plus1: int
    current_street_last_aggressor_rel_plus1: int
    lineage_aggressor_checked_current_street: int
    prior_street_checked_through: int

    # Pairwise stack geometry; opponent slots keep actor-relative position.
    opponent1_present: int
    opponent2_present: int
    opponent1_contesting: int
    opponent2_contesting: int
    opponent1_actionable: int
    opponent2_actionable: int
    opponent1_effective_remaining_bb: float
    opponent2_effective_remaining_bb: float
    opponent1_pairwise_spr: float
    opponent2_pairwise_spr: float
    opponent1_effective_total_cap_bb: float
    opponent2_effective_total_cap_bb: float
    opponent1_commitment_gap_bb: float
    opponent2_commitment_gap_bb: float

    def categorical(self) -> tuple[int, ...]:
        return (
            self.board_distinct_ranks,
            self.board_max_rank_multiplicity,
            self.board_pair_rank_count,
            self.board_trip_rank_count,
            self.board_quad_rank_count,
            self.board_distinct_suit_classes,
            self.board_max_suit_count,
            self.board_high_rank,
            self.board_low_rank,
            self.board_rank_span,
            self.board_broadway_count,
            self.board_max_straight_window_occupancy,
            self.board_straight_windows_3plus,
            self.board_straight_windows_4plus,
            self.board_has_straight,
            self.board_has_flush,
            self.made_category,
            self.board_only_category,
            self.best_hand_hole_cards_min,
            self.best_hand_hole_cards_max,
            self.pocket_pair,
            self.hole_rank_match_count,
            self.hole_board_multiplicity_low,
            self.hole_board_multiplicity_high,
            self.overcard_count,
            self.highest_matched_board_rank_tier,
            self.already_has_straight,
            self.already_has_flush,
            self.straight_completion_card_count,
            self.straight_completion_distinct_rank_count,
            self.board_straight_completion_card_count,
            self.hero_adds_to_straight_draw,
            self.flush_completion_card_count,
            self.board_flush_completion_card_count,
            self.hero_adds_to_flush_draw,
            self.flush_draw_highest_hero_rank,
            self.flush_draw_higher_unseen_count,
            self.backdoor_straight,
            self.backdoor_flush,
            self.combo_draw,
            self.pair_plus_draw,
            self.preflop_limper_count,
            self.preflop_aggression_count,
            self.preflop_pot_family,
            self.preflop_first_aggressor_rel_plus1,
            self.preflop_last_aggressor_rel_plus1,
            self.hero_called_last_preflop_aggression,
            self.current_street_aggression_count,
            self.current_street_first_aggressor_rel_plus1,
            self.current_street_last_aggressor_rel_plus1,
            self.lineage_aggressor_checked_current_street,
            self.prior_street_checked_through,
            self.opponent1_present,
            self.opponent2_present,
            self.opponent1_contesting,
            self.opponent2_contesting,
            self.opponent1_actionable,
            self.opponent2_actionable,
        )

    def numeric(self) -> tuple[float, ...]:
        return (
            self.opponent1_effective_remaining_bb,
            self.opponent2_effective_remaining_bb,
            self.opponent1_pairwise_spr,
            self.opponent2_pairwise_spr,
            self.opponent1_effective_total_cap_bb,
            self.opponent2_effective_total_cap_bb,
            self.opponent1_commitment_gap_bb,
            self.opponent2_commitment_gap_bb,
        )


def _canonical_cards(item: DecodedInputV3) -> tuple[tuple[int, int, int], ...]:
    """Return visible `(rank, canonical_suit_class, role_slot)` cards.

    Suit class ids are internal only and assigned by equivalence relation. They
    are never exposed as semantic categories, so their arbitrary names cannot
    leak into the network.
    """
    matrix = item.same_suit_matrix()
    visible = item.visible_mask
    suit_class = [-1] * 7
    next_class = 0
    for index in range(7):
        if not visible[index]:
            continue
        prior = next((j for j in range(index) if visible[j] and matrix[index][j]), None)
        if prior is None:
            suit_class[index] = next_class
            next_class += 1
        else:
            suit_class[index] = suit_class[prior]
    return tuple(
        (int(item.rank_tokens[index]), int(suit_class[index]), index)
        for index in range(7)
        if visible[index]
    )


def _evaluate_five(cards: tuple[tuple[int, int, int], ...]) -> _FiveRank:
    if len(cards) != 5:
        raise ValueError("_evaluate_five requires exactly five cards")
    ranks = sorted((rank for rank, _, _ in cards), reverse=True)
    suits = [suit for _, suit, _ in cards]
    counts: dict[int, int] = {}
    for rank in ranks:
        counts[rank] = counts.get(rank, 0) + 1
    unique = sorted(counts, reverse=True)
    straight_high = 0
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            straight_high = unique[0]
        elif unique == [14, 5, 4, 3, 2]:
            straight_high = 5
    flush = len(set(suits)) == 1
    if straight_high and flush:
        return _FiveRank(STRAIGHT_FLUSH, (straight_high, 0, 0, 0, 0))
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    if groups[0][0] == 4:
        quad = groups[0][1]
        kicker = max(rank for rank in ranks if rank != quad)
        return _FiveRank(QUADS, (quad, kicker, 0, 0, 0))
    trips = sorted((rank for rank, count in counts.items() if count >= 3), reverse=True)
    pairs = sorted((rank for rank, count in counts.items() if count >= 2), reverse=True)
    if trips:
        trip = trips[0]
        pair_candidates = [rank for rank in pairs if rank != trip]
        if pair_candidates:
            return _FiveRank(FULL_HOUSE, (trip, pair_candidates[0], 0, 0, 0))
    if flush:
        return _FiveRank(FLUSH, tuple(ranks))
    if straight_high:
        return _FiveRank(STRAIGHT, (straight_high, 0, 0, 0, 0))
    if trips:
        trip = trips[0]
        kickers = sorted((rank for rank in ranks if rank != trip), reverse=True)[:2]
        return _FiveRank(TRIPS, (trip, *kickers, 0, 0))
    exact_pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if len(exact_pairs) >= 2:
        hi, lo = exact_pairs[:2]
        kicker = max(rank for rank in ranks if rank not in (hi, lo))
        return _FiveRank(TWO_PAIR, (hi, lo, kicker, 0, 0))
    if len(exact_pairs) == 1:
        pair = exact_pairs[0]
        kickers = sorted((rank for rank in ranks if rank != pair), reverse=True)[:3]
        return _FiveRank(PAIR, (pair, *kickers, 0))
    return _FiveRank(HIGH_CARD, tuple(ranks))


def _best_rank(cards: tuple[tuple[int, int, int], ...]) -> _FiveRank:
    if len(cards) < 5:
        return _FiveRank(HIGH_CARD, (0, 0, 0, 0, 0))
    return max(_evaluate_five(tuple(combo)) for combo in combinations(cards, 5))


def _best_hole_contribution(cards: tuple[tuple[int, int, int], ...]) -> tuple[int, int]:
    if len(cards) < 5:
        return 0, 0
    ranked = []
    best: _FiveRank | None = None
    for combo in combinations(cards, 5):
        rank = _evaluate_five(tuple(combo))
        holes = sum(1 for _, _, slot in combo if slot < 2)
        if best is None or rank > best:
            best = rank
            ranked = [holes]
        elif rank == best:
            ranked.append(holes)
    return min(ranked), max(ranked)


def _rank_counts(cards: tuple[tuple[int, int, int], ...]) -> dict[int, int]:
    out: dict[int, int] = {}
    for rank, _, _ in cards:
        out[rank] = out.get(rank, 0) + 1
    return out


def _suit_counts(cards: tuple[tuple[int, int, int], ...]) -> dict[int, int]:
    out: dict[int, int] = {}
    for _, suit, _ in cards:
        out[suit] = out.get(suit, 0) + 1
    return out


def _straight_summary(rank_counts: dict[int, int]) -> tuple[int, int, int, bool]:
    occupancies = [sum(1 for rank in window if rank_counts.get(rank, 0) > 0) for window in _STRAIGHT_WINDOWS]
    maximum = max(occupancies, default=0)
    return maximum, sum(x >= 3 for x in occupancies), sum(x >= 4 for x in occupancies), maximum >= 5


def _has_straight(cards: tuple[tuple[int, int, int], ...]) -> bool:
    return _straight_summary(_rank_counts(cards))[3]


def _has_flush(cards: tuple[tuple[int, int, int], ...]) -> bool:
    return max(_suit_counts(cards).values(), default=0) >= 5


def _deck_unseen(cards: tuple[tuple[int, int, int], ...]) -> tuple[tuple[int, int, int], ...]:
    seen = {(rank, suit) for rank, suit, _ in cards}
    return tuple((rank, suit, 7) for rank in range(2, 15) for suit in range(4) if (rank, suit) not in seen)


def _one_card_completions(
    cards: tuple[tuple[int, int, int], ...],
    board: tuple[tuple[int, int, int], ...],
) -> tuple[int, int, int, int, int, int, int, int]:
    if len(board) >= 5:
        return 0, 0, 0, 0, 0, 0, 0, 0
    unseen = _deck_unseen(cards)
    has_straight = _has_straight(cards)
    board_has_straight = _has_straight(board)
    has_flush = _has_flush(cards)
    board_has_flush = _has_flush(board)

    straight_cards: list[tuple[int, int, int]] = []
    board_straight_cards: list[tuple[int, int, int]] = []
    flush_cards: list[tuple[int, int, int]] = []
    board_flush_cards: list[tuple[int, int, int]] = []
    for card in unseen:
        if not has_straight and _has_straight(cards + (card,)):
            straight_cards.append(card)
        if not board_has_straight and _has_straight(board + (card,)):
            board_straight_cards.append(card)
        if not has_flush and _has_flush(cards + (card,)):
            flush_cards.append(card)
        if not board_has_flush and _has_flush(board + (card,)):
            board_flush_cards.append(card)

    straight_rank_count = len({rank for rank, _, _ in straight_cards})
    hero_adds_straight = int(bool(straight_cards) and set((r, s) for r, s, _ in straight_cards) != set((r, s) for r, s, _ in board_straight_cards))
    hero_adds_flush = int(bool(flush_cards) and set((r, s) for r, s, _ in flush_cards) != set((r, s) for r, s, _ in board_flush_cards))

    flush_highest = 0
    flush_higher_unseen = 0
    if flush_cards:
        combined_suits = _suit_counts(cards)
        candidate_suits = [suit for suit, count in combined_suits.items() if count == 4]
        for suit in candidate_suits:
            hero_ranks = [rank for rank, card_suit, slot in cards if slot < 2 and card_suit == suit]
            if not hero_ranks:
                continue
            high = max(hero_ranks)
            higher = sum(
                1
                for rank in range(high + 1, 15)
                if (rank, suit) not in {(r, s) for r, s, _ in cards}
            )
            if high > flush_highest:
                flush_highest = high
                flush_higher_unseen = higher

    return (
        len(straight_cards),
        straight_rank_count,
        len(board_straight_cards),
        hero_adds_straight,
        len(flush_cards),
        len(board_flush_cards),
        hero_adds_flush,
        (flush_highest << 8) | flush_higher_unseen,
    )


def _backdoors(cards: tuple[tuple[int, int, int], ...], board: tuple[tuple[int, int, int], ...]) -> tuple[int, int]:
    if len(board) != 3:
        return 0, 0
    combined_counts = _rank_counts(cards)
    board_counts = _rank_counts(board)
    backdoor_straight = 0
    if not _has_straight(cards):
        for window in _STRAIGHT_WINDOWS:
            combined_occupied = {rank for rank in window if combined_counts.get(rank, 0)}
            board_occupied = {rank for rank in window if board_counts.get(rank, 0)}
            if len(combined_occupied) == 3 and len(combined_occupied - board_occupied) > 0:
                backdoor_straight = 1
                break
    combined_suits = _suit_counts(cards)
    board_suits = _suit_counts(board)
    backdoor_flush = 0
    if not _has_flush(cards):
        for suit, count in combined_suits.items():
            hero_has = any(slot < 2 and card_suit == suit for _, card_suit, slot in cards)
            if count == 3 and hero_has and board_suits.get(suit, 0) < 3:
                backdoor_flush = 1
                break
    return backdoor_straight, backdoor_flush


def _public_lineage(item: DecodedInputV3) -> tuple[int, ...]:
    # Reconstruct public betting commitment state from exact structured events.
    preflop_limpers: set[int] = set()
    preflop_aggressors: list[int] = []
    hero_called_last_preflop = 0
    street_aggressors: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    street_checks: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}
    street_seen_voluntary: dict[int, bool] = {0: False, 1: False, 2: False, 3: False}
    current_bet = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    commitments = {(street, rel): 0.0 for street in range(4) for rel in range(3)}

    for event in item.history:
        actor, street, action_type, forced = event.categorical
        resulting = float(event.numeric[1])
        prior_high = current_bet[street]
        prior_actor = commitments[(street, actor)]
        if forced:
            commitments[(street, actor)] = resulting
            current_bet[street] = max(current_bet[street], resulting)
            continue
        street_seen_voluntary[street] = True
        aggressive = resulting > prior_high + 1e-6 and action_type in (3, 4, 5)
        call_like = resulting > prior_actor + 1e-6 and not aggressive
        if aggressive:
            street_aggressors[street].append(actor)
            current_bet[street] = max(current_bet[street], resulting)
            if street == 0:
                preflop_aggressors.append(actor)
                hero_called_last_preflop = 0
        elif action_type == 1:  # check
            street_checks[street].add(actor)
        if street == 0 and call_like:
            if not preflop_aggressors:
                preflop_limpers.add(actor)
            elif actor == 0:
                hero_called_last_preflop = 1
        commitments[(street, actor)] = resulting

    raises = len(preflop_aggressors)
    if raises == 0:
        family = 1 if preflop_limpers else 0
    elif raises == 1:
        family = 2
    elif raises == 2:
        family = 3
    else:
        family = 4

    street = int(item.categorical[1])
    current_aggr = street_aggressors[street]
    lineage_aggressor = preflop_aggressors[-1] if preflop_aggressors else None
    if street > 1 and street_aggressors[street - 1]:
        lineage_aggressor = street_aggressors[street - 1][-1]
    elif street > 2 and street_aggressors[street - 2]:
        lineage_aggressor = street_aggressors[street - 2][-1]

    lineage_checked = int(lineage_aggressor is not None and lineage_aggressor in street_checks[street])
    prior_checked_through = 0
    if street >= 2:
        prior = street - 1
        prior_checked_through = int(street_seen_voluntary[prior] and not street_aggressors[prior])

    return (
        len(preflop_limpers),
        raises,
        family,
        (preflop_aggressors[0] + 1) if preflop_aggressors else 0,
        (preflop_aggressors[-1] + 1) if preflop_aggressors else 0,
        hero_called_last_preflop,
        len(current_aggr),
        (current_aggr[0] + 1) if current_aggr else 0,
        (current_aggr[-1] + 1) if current_aggr else 0,
        lineage_checked,
        prior_checked_through,
    )


def _pairwise_stack(item: DecodedInputV3) -> tuple[tuple[int, ...], tuple[float, ...]]:
    domain = int(item.categorical[0])
    statuses = tuple(int(x) for x in item.categorical[7:10])
    pot = float(item.numeric[0])
    hero_stack = float(item.numeric[3])
    hero_total = float(item.numeric[9])
    cats: list[int] = []
    nums: list[float] = []
    per_opp = []
    for rel in (1, 2):
        absent = domain == 1 and rel == 2
        status = statuses[rel]
        stack = float(item.numeric[3 + rel])
        total = float(item.numeric[9 + rel])
        present = int(not absent)
        contesting = int(not absent and status != 1)
        actionable = int(not absent and status == 0 and stack > 0.0)
        effective = min(hero_stack, stack) if actionable else 0.0
        spr = effective / pot if actionable and pot > 0 else 0.0
        cap = min(hero_stack + hero_total, stack + total) if contesting else 0.0
        gap = total - hero_total if present else 0.0
        per_opp.append((present, contesting, actionable, effective, spr, cap, gap))
    for value_index in range(3):
        cats.extend([int(per_opp[0][value_index]), int(per_opp[1][value_index])])
    for value_index in range(3, 7):
        nums.extend([float(per_opp[0][value_index]), float(per_opp[1][value_index])])
    return tuple(cats), tuple(nums)


def derive_objective_semantics_v3(item: DecodedInputV3) -> ObjectiveSemanticsV3:
    cards = _canonical_cards(item)
    board = tuple(card for card in cards if card[2] >= 2)
    holes = tuple(card for card in cards if card[2] < 2)

    board_counts = _rank_counts(board)
    board_suits = _suit_counts(board)
    board_ranks = sorted(board_counts, reverse=True)
    board_max_occ, board_3plus, board_4plus, board_straight = _straight_summary(board_counts)
    board_pair_ranks = sum(count >= 2 for count in board_counts.values())
    board_trip_ranks = sum(count >= 3 for count in board_counts.values())
    board_quad_ranks = sum(count >= 4 for count in board_counts.values())

    made = _best_rank(cards).category if len(cards) >= 5 else HIGH_CARD
    board_only = _best_rank(board).category if len(board) >= 5 else HIGH_CARD
    hole_min, hole_max = _best_hole_contribution(cards)

    pocket = int(holes[0][0] == holes[1][0])
    multiplicities = sorted(board_counts.get(rank, 0) for rank, _, _ in holes)
    hole_match_count = sum(value > 0 for value in multiplicities)
    board_high = max(board_ranks, default=0)
    overcards = sum(rank > board_high for rank, _, _ in holes) if board else 0
    matched_ranks = sorted({rank for rank, _, _ in holes if board_counts.get(rank, 0)}, reverse=True)
    matched_tier = 0
    if matched_ranks:
        distinct_desc = sorted(board_counts, reverse=True)
        matched_tier = min(distinct_desc.index(rank) + 1 for rank in matched_ranks)

    (
        straight_cards,
        straight_ranks,
        board_straight_cards,
        hero_straight,
        flush_cards,
        board_flush_cards,
        hero_flush,
        flush_packed,
    ) = _one_card_completions(cards, board)
    flush_highest = flush_packed >> 8
    flush_higher = flush_packed & 0xFF
    backdoor_straight, backdoor_flush = _backdoors(cards, board)
    combo = int(straight_cards > 0 and flush_cards > 0)
    pair_plus = int(made >= PAIR and (straight_cards > 0 or flush_cards > 0))

    lineage = _public_lineage(item)
    stack_cat, stack_num = _pairwise_stack(item)

    return ObjectiveSemanticsV3(
        board_distinct_ranks=len(board_counts),
        board_max_rank_multiplicity=max(board_counts.values(), default=0),
        board_pair_rank_count=board_pair_ranks,
        board_trip_rank_count=board_trip_ranks,
        board_quad_rank_count=board_quad_ranks,
        board_distinct_suit_classes=len(board_suits),
        board_max_suit_count=max(board_suits.values(), default=0),
        board_high_rank=max(board_ranks, default=0),
        board_low_rank=min(board_ranks, default=0),
        board_rank_span=(max(board_ranks) - min(board_ranks)) if board_ranks else 0,
        board_broadway_count=sum(rank >= 10 for rank, _, _ in board),
        board_max_straight_window_occupancy=board_max_occ,
        board_straight_windows_3plus=board_3plus,
        board_straight_windows_4plus=board_4plus,
        board_has_straight=int(board_straight),
        board_has_flush=int(max(board_suits.values(), default=0) >= 5),
        made_category=made,
        board_only_category=board_only,
        best_hand_hole_cards_min=hole_min,
        best_hand_hole_cards_max=hole_max,
        pocket_pair=pocket,
        hole_rank_match_count=hole_match_count,
        hole_board_multiplicity_low=multiplicities[0],
        hole_board_multiplicity_high=multiplicities[1],
        overcard_count=overcards,
        highest_matched_board_rank_tier=matched_tier,
        already_has_straight=int(_has_straight(cards)),
        already_has_flush=int(_has_flush(cards)),
        straight_completion_card_count=straight_cards,
        straight_completion_distinct_rank_count=straight_ranks,
        board_straight_completion_card_count=board_straight_cards,
        hero_adds_to_straight_draw=hero_straight,
        flush_completion_card_count=flush_cards,
        board_flush_completion_card_count=board_flush_cards,
        hero_adds_to_flush_draw=hero_flush,
        flush_draw_highest_hero_rank=flush_highest,
        flush_draw_higher_unseen_count=flush_higher,
        backdoor_straight=backdoor_straight,
        backdoor_flush=backdoor_flush,
        combo_draw=combo,
        pair_plus_draw=pair_plus,
        preflop_limper_count=lineage[0],
        preflop_aggression_count=lineage[1],
        preflop_pot_family=lineage[2],
        preflop_first_aggressor_rel_plus1=lineage[3],
        preflop_last_aggressor_rel_plus1=lineage[4],
        hero_called_last_preflop_aggression=lineage[5],
        current_street_aggression_count=lineage[6],
        current_street_first_aggressor_rel_plus1=lineage[7],
        current_street_last_aggressor_rel_plus1=lineage[8],
        lineage_aggressor_checked_current_street=lineage[9],
        prior_street_checked_through=lineage[10],
        opponent1_present=stack_cat[0],
        opponent2_present=stack_cat[1],
        opponent1_contesting=stack_cat[2],
        opponent2_contesting=stack_cat[3],
        opponent1_actionable=stack_cat[4],
        opponent2_actionable=stack_cat[5],
        opponent1_effective_remaining_bb=stack_num[0],
        opponent2_effective_remaining_bb=stack_num[1],
        opponent1_pairwise_spr=stack_num[2],
        opponent2_pairwise_spr=stack_num[3],
        opponent1_effective_total_cap_bb=stack_num[4],
        opponent2_effective_total_cap_bb=stack_num[5],
        opponent1_commitment_gap_bb=stack_num[6],
        opponent2_commitment_gap_bb=stack_num[7],
    )
