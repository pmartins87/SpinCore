from __future__ import annotations

from spincore_nn.codec_v3 import DecodedInputV3
from spincore_nn.stack_geometry_v3 import derive_pairwise_stack_geometry_v3


def state(
    *,
    domain: int,
    pot: float,
    stacks: tuple[float, float, float],
    totals: tuple[float, float, float],
    statuses: tuple[int, int, int],
) -> DecodedInputV3:
    live = 2 if domain == 1 else 3
    dealer_rel = 0
    sb_rel = 0 if domain == 1 else 1
    bb_rel = 1 if domain == 1 else 2
    numeric = (
        pot, 0.0, 0.0,
        *stacks,
        0.0, 0.0, 0.0,
        *totals,
        0.5, 0.0, 2.0, 30.0,
    )
    return DecodedInputV3(
        categorical=(
            domain, 0, dealer_rel, sb_rel, bb_rel, live, 0,
            *statuses,
        ),
        rank_tokens=(14, 13, 0, 0, 0, 0, 0),
        same_suit=(0,) * 21,
        numeric=tuple(float(x) for x in numeric),
        primitive_legal=(1, 1, 1, 1, 1, 1),
        history=(),
    )


def test_three_way_keeps_both_effective_stacks() -> None:
    item = state(
        domain=0,
        pot=8.0,
        stacks=(12.0, 4.0, 20.0),
        totals=(3.0, 3.0, 3.0),
        statuses=(0, 0, 0),
    )
    g = derive_pairwise_stack_geometry_v3(item)
    assert g.opponent_present == (1, 1)
    assert g.opponent_contesting == (1, 1)
    assert g.opponent_actionable == (1, 1)
    assert g.effective_remaining_bb == (4.0, 12.0)
    assert g.pairwise_spr == (0.5, 1.5)
    assert g.effective_total_cap_bb == (7.0, 15.0)


def test_three_way_does_not_sort_away_position() -> None:
    a = state(
        domain=0,
        pot=8.0,
        stacks=(12.0, 4.0, 20.0),
        totals=(3.0, 3.0, 3.0),
        statuses=(0, 0, 0),
    )
    b = state(
        domain=0,
        pot=8.0,
        stacks=(12.0, 20.0, 4.0),
        totals=(3.0, 3.0, 3.0),
        statuses=(0, 0, 0),
    )
    ga = derive_pairwise_stack_geometry_v3(a)
    gb = derive_pairwise_stack_geometry_v3(b)
    assert ga.effective_remaining_bb == (4.0, 12.0)
    assert gb.effective_remaining_bb == (12.0, 4.0)
    assert ga != gb


def test_folded_and_allin_are_not_conflated() -> None:
    item = state(
        domain=0,
        pot=14.0,
        stacks=(10.0, 7.0, 0.0),
        totals=(5.0, 4.0, 9.0),
        statuses=(0, 1, 2),
    )
    g = derive_pairwise_stack_geometry_v3(item)
    assert g.opponent_present == (1, 1)
    assert g.opponent_contesting == (0, 1)
    assert g.opponent_actionable == (0, 0)
    assert g.effective_remaining_bb == (0.0, 0.0)
    assert g.pairwise_spr == (0.0, 0.0)
    assert g.effective_total_cap_bb == (0.0, 9.0)


def test_true_hu_dead_seat_is_explicitly_absent() -> None:
    # Canonical true-HU layout is [Hero, live opponent, absent].
    item = state(
        domain=1,
        pot=6.0,
        stacks=(15.0, 9.0, 0.0),
        totals=(2.0, 2.0, 0.0),
        statuses=(0, 0, 2),
    )
    g = derive_pairwise_stack_geometry_v3(item)
    assert g.opponent_present == (1, 0)
    assert g.opponent_contesting == (1, 0)
    assert g.opponent_actionable == (1, 0)
    assert g.effective_remaining_bb == (9.0, 0.0)
    assert g.pairwise_spr == (1.5, 0.0)
    assert g.effective_total_cap_bb == (11.0, 0.0)


def test_true_hu_real_allin_opponent_is_not_mistaken_for_absent() -> None:
    item = state(
        domain=1,
        pot=22.0,
        stacks=(8.0, 0.0, 0.0),
        totals=(7.0, 15.0, 0.0),
        statuses=(0, 2, 2),
    )
    g = derive_pairwise_stack_geometry_v3(item)
    assert g.opponent_present == (1, 0)
    assert g.opponent_contesting == (1, 0)
    assert g.opponent_actionable == (0, 0)
    assert g.effective_remaining_bb == (0.0, 0.0)
    assert g.effective_total_cap_bb == (15.0, 0.0)
    assert g.commitment_gap_bb == (8.0, 0.0)


def main() -> int:
    tests = [
        test_three_way_keeps_both_effective_stacks,
        test_three_way_does_not_sort_away_position,
        test_folded_and_allin_are_not_conflated,
        test_true_hu_dead_seat_is_explicitly_absent,
        test_true_hu_real_allin_opponent_is_not_mistaken_for_absent,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("R7.5.3C pairwise SPNNIV3 stack geometry tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
