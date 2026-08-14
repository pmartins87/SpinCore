from __future__ import annotations

from itertools import permutations

from spincore_nn.codec_v3 import DecodedHistoryEventV3, DecodedInputV3
from spincore_nn.semantics_v3 import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    derive_objective_semantics_v3,
)

Card = tuple[int, int]


def _relations(cards: list[Card | None]) -> tuple[int, ...]:
    values: list[int] = []
    for left in range(7):
        for right in range(left + 1, 7):
            a, b = cards[left], cards[right]
            values.append(int(a is not None and b is not None and a[1] == b[1]))
    return tuple(values)


def _event(
    actor: int,
    street: int,
    action_type: int,
    *,
    forced: int = 0,
    paid: float = 0.0,
    resulting: float = 0.0,
    pot_before: float = 0.0,
    pot_after: float = 0.0,
) -> DecodedHistoryEventV3:
    return DecodedHistoryEventV3(
        (actor, street, action_type, forced),
        (paid, resulting, pot_before, pot_after),
    )


def _item(
    hole: tuple[Card, Card],
    board: tuple[Card, ...],
    *,
    domain: int = 0,
    stacks: tuple[float, float, float] = (20.0, 15.0, 10.0),
    totals: tuple[float, float, float] = (2.0, 2.0, 2.0),
    statuses: tuple[int, int, int] | None = None,
    pot: float = 6.0,
    history: tuple[DecodedHistoryEventV3, ...] = (),
) -> DecodedInputV3:
    if len(board) not in (0, 3, 4, 5):
        raise ValueError("board must be preflop/flop/turn/river length")
    street = {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]
    cards: list[Card | None] = [hole[0], hole[1], None, None, None, None, None]
    for index, card in enumerate(board):
        cards[index + 2] = card
    ranks = tuple(card[0] if card else 0 for card in cards)
    if statuses is None:
        statuses = (0, 0, 2) if domain == 1 else (0, 0, 0)
    live = 2 if domain == 1 else 3
    numeric = (
        pot, 0.0, 0.0,
        *stacks,
        0.0, 0.0, 0.0,
        *totals,
        0.5, 0.0, 2.0, 30.0,
    )
    return DecodedInputV3(
        categorical=(domain, street, 0, 1 if domain == 0 else 0, 2 if domain == 0 else 1, live, len(board), *statuses),
        rank_tokens=ranks,
        same_suit=_relations(cards),
        numeric=tuple(float(x) for x in numeric),
        primitive_legal=(1, 1, 1, 1, 1, 1),
        history=history,
    )


def test_all_nine_made_hand_categories() -> None:
    cases = [
        (HIGH_CARD, ((14, 0), (13, 1)), ((9, 2), (7, 3), (2, 0))),
        (PAIR, ((14, 0), (13, 1)), ((14, 2), (7, 3), (2, 0))),
        (TWO_PAIR, ((14, 0), (13, 1)), ((14, 2), (13, 3), (2, 0))),
        (TRIPS, ((14, 0), (14, 1)), ((14, 2), (7, 3), (2, 0))),
        (STRAIGHT, ((8, 0), (7, 1)), ((6, 2), (5, 3), (4, 0))),
        (FLUSH, ((14, 0), (13, 0)), ((12, 0), (7, 0), (2, 0))),
        (FULL_HOUSE, ((14, 0), (14, 1)), ((14, 2), (13, 0), (13, 1))),
        (QUADS, ((14, 0), (14, 1)), ((14, 2), (14, 3), (13, 0))),
        (STRAIGHT_FLUSH, ((8, 0), (7, 0)), ((6, 0), (5, 0), (4, 0))),
    ]
    for expected, hole, board in cases:
        actual = derive_objective_semantics_v3(_item(hole, board)).made_category
        assert actual == expected, (expected, actual, hole, board)


def test_best_five_hole_contribution_is_objective() -> None:
    board_broadway = ((14, 0), (13, 1), (12, 2), (11, 3), (10, 0))
    board_plays = derive_objective_semantics_v3(_item(((2, 1), (2, 2)), board_broadway))
    assert board_plays.made_category == STRAIGHT
    assert board_plays.board_only_category == STRAIGHT
    assert (board_plays.best_hand_hole_cards_min, board_plays.best_hand_hole_cards_max) == (0, 0)

    # Same A/K ranks can replace their board counterparts without changing the
    # best straight: min=0 and max=2 captures the tie objectively.
    tied = derive_objective_semantics_v3(_item(((14, 3), (13, 2)), board_broadway))
    assert (tied.best_hand_hole_cards_min, tied.best_hand_hole_cards_max) == (0, 2)

    one_hole = derive_objective_semantics_v3(
        _item(((10, 0), (2, 1)), ((14, 1), (13, 2), (12, 3), (11, 0), (3, 1)))
    )
    assert one_hole.made_category == STRAIGHT
    assert (one_hole.best_hand_hole_cards_min, one_hole.best_hand_hole_cards_max) == (1, 1)

    two_hole = derive_objective_semantics_v3(
        _item(((8, 0), (7, 1)), ((6, 2), (5, 3), (4, 0), (13, 1), (2, 2)))
    )
    assert two_hole.made_category == STRAIGHT
    assert (two_hole.best_hand_hole_cards_min, two_hole.best_hand_hole_cards_max) == (2, 2)


def test_wheel_and_open_ended_draw_geometry() -> None:
    wheel_edge = derive_objective_semantics_v3(
        _item(((4, 2), (5, 3)), ((2, 0), (3, 1), (9, 2)))
    )
    assert not wheel_edge.already_has_straight
    assert wheel_edge.straight_completion_card_count == 8
    assert wheel_edge.straight_completion_distinct_rank_count == 2
    assert wheel_edge.board_straight_completion_card_count == 0
    assert wheel_edge.hero_adds_to_straight_draw == 1

    oesd = derive_objective_semantics_v3(
        _item(((8, 1), (7, 2)), ((6, 3), (5, 0), (2, 1)))
    )
    assert oesd.straight_completion_card_count == 8
    assert oesd.straight_completion_distinct_rank_count == 2
    assert oesd.hero_adds_to_straight_draw == 1

    gutshot = derive_objective_semantics_v3(
        _item(((14, 0), (13, 1)), ((12, 2), (11, 3), (2, 0)))
    )
    assert gutshot.straight_completion_card_count == 4
    assert gutshot.straight_completion_distinct_rank_count == 1


def test_flush_draw_distinguishes_hero_from_board_only() -> None:
    hero_fd = derive_objective_semantics_v3(
        _item(((14, 0), (13, 0)), ((12, 0), (11, 0), (2, 1)))
    )
    assert not hero_fd.already_has_flush
    assert hero_fd.flush_completion_card_count == 9
    assert hero_fd.board_flush_completion_card_count == 0
    assert hero_fd.hero_adds_to_flush_draw == 1
    assert hero_fd.flush_draw_highest_hero_rank == 14
    assert hero_fd.flush_draw_higher_unseen_count == 0
    assert hero_fd.combo_draw == 1

    board_four_flush = derive_objective_semantics_v3(
        _item(((7, 1), (6, 2)), ((14, 0), (13, 0), (12, 0), (11, 0)))
    )
    assert board_four_flush.flush_completion_card_count == 9
    assert board_four_flush.board_flush_completion_card_count == 9
    assert board_four_flush.hero_adds_to_flush_draw == 0
    assert board_four_flush.flush_draw_highest_hero_rank == 0

    made_flush = derive_objective_semantics_v3(
        _item(((14, 0), (13, 0)), ((12, 0), (11, 0), (2, 0)))
    )
    assert made_flush.already_has_flush == 1
    assert made_flush.flush_completion_card_count == 0


def test_backdoors_are_two_card_only_not_direct_draw_aliases() -> None:
    backdoor = derive_objective_semantics_v3(
        _item(((14, 0), (13, 0)), ((12, 0), (7, 1), (2, 2)))
    )
    assert backdoor.flush_completion_card_count == 0
    assert backdoor.backdoor_flush == 1
    assert backdoor.backdoor_straight == 1

    direct = derive_objective_semantics_v3(
        _item(((8, 1), (7, 2)), ((6, 3), (5, 0), (2, 1)))
    )
    assert direct.straight_completion_card_count > 0
    assert direct.backdoor_straight == 0


def test_semantics_invariant_to_suit_names_hole_order_and_flop_order() -> None:
    hole = ((14, 0), (13, 1))
    flop = ((12, 0), (8, 1), (2, 2))
    baseline = derive_objective_semantics_v3(_item(hole, flop))

    for suit_perm in permutations(range(4)):
        transformed_hole = tuple((rank, suit_perm[suit]) for rank, suit in hole)
        transformed_flop = tuple((rank, suit_perm[suit]) for rank, suit in flop)
        assert derive_objective_semantics_v3(_item(transformed_hole, transformed_flop)) == baseline

    assert derive_objective_semantics_v3(_item((hole[1], hole[0]), flop)) == baseline
    for permuted_flop in permutations(flop):
        assert derive_objective_semantics_v3(_item(hole, tuple(permuted_flop))) == baseline


def test_pairwise_stack_geometry_keeps_both_three_way_opponents() -> None:
    sem = derive_objective_semantics_v3(
        _item(
            ((14, 0), (13, 1)),
            ((12, 2), (8, 3), (2, 0)),
            stacks=(12.0, 4.0, 20.0),
            totals=(3.0, 3.0, 3.0),
            pot=8.0,
        )
    )
    assert (sem.opponent1_present, sem.opponent2_present) == (1, 1)
    assert (sem.opponent1_effective_remaining_bb, sem.opponent2_effective_remaining_bb) == (4.0, 12.0)
    assert (sem.opponent1_pairwise_spr, sem.opponent2_pairwise_spr) == (0.5, 1.5)


def test_public_lineage_uses_commitments_not_allin_label() -> None:
    blinds = (
        _event(1, 0, 3, forced=1, paid=0.5, resulting=0.5),
        _event(2, 0, 3, forced=1, paid=1.0, resulting=1.0),
    )
    threebet = blinds + (
        _event(0, 0, 4, paid=2.5, resulting=2.5),
        _event(1, 0, 2, paid=2.0, resulting=2.5),
        _event(2, 0, 4, paid=7.0, resulting=8.0),
        _event(0, 0, 5, paid=5.5, resulting=8.0),  # AllIn type, but exact call.
        _event(1, 0, 2, paid=5.5, resulting=8.0),
    )
    called = derive_objective_semantics_v3(
        _item(((14, 0), (13, 1)), (), history=threebet)
    )
    assert called.preflop_aggression_count == 2
    assert called.preflop_pot_family == 3
    assert called.hero_called_last_preflop_aggression == 1
    assert called.preflop_first_aggressor_rel_plus1 == 1
    assert called.preflop_last_aggressor_rel_plus1 == 3

    # Same ExactActionType::AllIn becomes a third aggression when resulting
    # commitment exceeds the current bet.
    fourbet = blinds + (
        _event(0, 0, 4, paid=2.5, resulting=2.5),
        _event(1, 0, 2, paid=2.0, resulting=2.5),
        _event(2, 0, 4, paid=7.0, resulting=8.0),
        _event(0, 0, 5, paid=17.5, resulting=20.0),
    )
    raised = derive_objective_semantics_v3(
        _item(((14, 0), (13, 1)), (), history=fourbet)
    )
    assert raised.preflop_aggression_count == 3
    assert raised.preflop_pot_family == 4
    assert raised.hero_called_last_preflop_aggression == 0
    assert raised.preflop_last_aggressor_rel_plus1 == 1


def main() -> int:
    tests = [
        test_all_nine_made_hand_categories,
        test_best_five_hole_contribution_is_objective,
        test_wheel_and_open_ended_draw_geometry,
        test_flush_draw_distinguishes_hero_from_board_only,
        test_backdoors_are_two_card_only_not_direct_draw_aliases,
        test_semantics_invariant_to_suit_names_hole_order_and_flop_order,
        test_pairwise_stack_geometry_keeps_both_three_way_opponents,
        test_public_lineage_uses_commitments_not_allin_label,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("R7.5.3C objective semantics V3 adversarial tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
