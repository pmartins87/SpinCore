from __future__ import annotations

import math
import numpy as np

import r7_5_arch_reset_v1plus_phase2c0_structural_reach_factorization as c0
import r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot as c2


def _assert_close(a: float, b: float, tol: float = 1e-12) -> None:
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f"{a!r} != {b!r}")


def test_constants_and_stage_geometry() -> None:
    assert c2.K == 64
    assert c2.STRATA_SIDE == 8
    assert c2.PILOT_ITERATIONS == 2
    assert c2.CHUNKS_PER_ITERATION == 2
    assert c2.ROOTS_PER_CHUNK == 32
    assert c2.ROOTS_PER_ITERATION_EFFECTIVE == 64
    assert c2.TOTAL_ROOTS == 128
    assert c2._stage_coords(1) == (1, 1)
    assert c2._stage_coords(2) == (1, 2)
    assert c2._stage_coords(3) == (2, 1)
    assert c2._stage_coords(4) == (2, 2)


def test_mean_targets() -> None:
    rows = [tuple(float(i + j) for j in range(10)) for i in range(c2.K)]
    mean = c2._mean_targets(rows)
    for j, value in enumerate(mean):
        _assert_close(value, 31.5 + j)


def test_cdf_draw() -> None:
    w = np.asarray([1.0, 2.0, 7.0], dtype=np.float64)
    assert c2._cdf_draw(w, 0.01) == 0
    assert c2._cdf_draw(w, 0.20) == 1
    assert c2._cdf_draw(w, 0.99) == 2


def test_structural_stratification_deterministic_and_legal() -> None:
    hands = c0._ordered_hands((0, 1))
    wa = np.linspace(0.1, 1.0, len(hands), dtype=np.float64)
    wb = np.linspace(1.0, 0.1, len(hands), dtype=np.float64)
    first, stats1 = c2._stratified_joint_indices(hands, wa, wb, seed=123456789)
    second, stats2 = c2._stratified_joint_indices(hands, wa, wb, seed=123456789)
    assert first == second
    assert stats1 == stats2
    assert len(first) == c2.K
    assert math.isfinite(float(stats1["joint_normalizer"]))
    assert float(stats1["joint_normalizer"]) > 0.0
    assert int(stats1["unique_seat_a_indices"]) > 0
    assert int(stats1["unique_joint_assignments"]) > 0
    for ia, ib in first:
        ha, hb = hands[ia], hands[ib]
        assert len({ha[0], ha[1], hb[0], hb[1]}) == 4


def test_board_generation() -> None:
    actor = (0, 1)
    hand_a = (2, 3)
    hand_b = (4, 5)
    board1 = c2._board_for_joint(actor, hand_a, hand_b, seed=999)
    board2 = c2._board_for_joint(actor, hand_a, hand_b, seed=999)
    assert board1 == board2
    assert len(board1) == 5
    assert len(set(board1)) == 5
    assert not (set(board1) & set(actor + hand_a + hand_b))


def test_multi_replacement_memory() -> None:
    class Sink:
        def __init__(self):
            self.items = []
        def add(self, sample):
            self.items.append(sample)

    sink = Sink()
    proxy = c2.MultiReplacingAdvantageMemory(
        sink,
        iteration=2,
        replacements=[
            {
                "label": "root",
                "observation": b"root",
                "target": [1.0] * 10,
                "legal_mask": [1] * 10,
            },
            {
                "label": "continuation",
                "observation": b"continuation",
                "target": [2.0] * 10,
                "legal_mask": [1] * 10,
            },
        ],
    )
    root = c2.ActionAdvantageSample(b"root", tuple([1] * 10), tuple([0.0] * 10), 2.0, 2)
    cont = c2.ActionAdvantageSample(b"continuation", tuple([1] * 10), tuple([0.0] * 10), 2.0, 2)
    other = c2.ActionAdvantageSample(b"other", tuple([1] * 10), tuple([3.0] * 10), 2.0, 2)
    proxy.add(root)
    proxy.add(other)
    proxy.add(cont)
    proxy.assert_complete()
    assert len(sink.items) == 3
    assert tuple(sink.items[0].target) == tuple([1.0] * 10)
    assert tuple(sink.items[1].target) == tuple([3.0] * 10)
    assert tuple(sink.items[2].target) == tuple([2.0] * 10)


def main() -> int:
    test_constants_and_stage_geometry()
    test_mean_targets()
    test_cdf_draw()
    test_structural_stratification_deterministic_and_legal()
    test_board_generation()
    test_multi_replacement_memory()
    print("R7.5 architecture-reset Phase2C2 structural range/reach synthetic tests PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
