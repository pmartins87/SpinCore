from __future__ import annotations

from dataclasses import replace

from spincore_nn.codec import DecodedInput
from spincore_nn.stack_geometry_v3 import derive_pairwise_stack_geometry_v3


def state(
    *,
    domain: int,
    pot: float,
    stacks: tuple[float, float, float],
    totals: tuple[float, float, float],
    statuses: tuple[int, int, int],
) -> DecodedInput:
    numeric = (
        pot, 0.0, 0.0,
        *stacks,
        0.0, 0.0, 0.0,
        *totals,
        0.5, 1.0, 0.0, 2.0 if domain == 1 else 3.0,
    )
    categorical = (domain, 1, 0, 2 if domain == 1 else 3, *statuses, 3)
    return DecodedInput(
        cards=(49, 46, 41, 30, 1, 0, 0),
        numeric=tuple(float(x) for x in numeric),
        categorical=tuple(int(x) for x in categorical),
        legal=(1, 1, 0, 1, 1, 1),
        history=(0,) * 32,
        history_len=0,
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


def main() -> int:
    tests = [
        test_three_way_keeps_both_effective_stacks,
        test_three_way_does_not_sort_away_position,
        test_folded_and_allin_are_not_conflated,
        test_true_hu_dead_seat_is_explicitly_absent,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("R7.5.3C pairwise stack geometry tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
