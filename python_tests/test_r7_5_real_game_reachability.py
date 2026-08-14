from __future__ import annotations

from pathlib import Path

import pytest

from spincore.solver import Episode, SolverLibrary
from spincore_nn.codec import decode_spnniv1
from spincore_nn.codec_v2 import decode_spnniv2

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"
FULL_UNIVERSAL_MASK = 0x3FF


def _episodes() -> tuple[Episode, ...]:
    return (
        Episode(1500, True, 0, 10, 20, (0, 750, 750), 1, (0,)),
        Episode(1500, True, 0, 10, 20, (750, 0, 750), 2, (1,)),
        Episode(1500, False, 0, 10, 20, (500, 500, 500), 0, ()),
        Episode(1500, False, 0, 10, 20, (250, 500, 750), 1, ()),
        Episode(1500, False, 0, 10, 20, (700, 500, 300), 2, ()),
    )


def _assert_observation_matches_authoritative_legal_mask(state) -> None:
    legal = set(state.legal_actions())
    v1 = decode_spnniv1(state.neural_bytes())
    v2 = decode_spnniv2(state.neural_bytes_v2())
    expected = tuple(1 if action in legal else 0 for action in range(6))
    assert tuple(int(x) for x in v1.legal) == expected
    assert tuple(int(x) for x in v2.legal) == expected
    assert 0 <= int(v1.history_len) <= 32
    assert 0 <= int(v2.history_len) <= 32


def _assert_illegal_actions_fail_closed(state) -> None:
    legal = set(state.legal_actions())
    illegal = next((action for action in range(6) if action not in legal), None)
    if illegal is not None:
        clone = state.clone()
        try:
            with pytest.raises(RuntimeError):
                clone.apply(illegal)
        finally:
            clone.close()

    universal_legal = set(state.universal_legal_actions(FULL_UNIVERSAL_MASK))
    assert universal_legal
    illegal_universal = next(
        (action for action in range(10) if action not in universal_legal), None
    )
    if illegal_universal is not None:
        clone = state.clone()
        try:
            with pytest.raises(RuntimeError):
                clone.apply_universal(FULL_UNIVERSAL_MASK, illegal_universal)
        finally:
            clone.close()


def _walk_legacy(solver: SolverLibrary, episode: Episode, seed: int) -> None:
    state = solver.create(episode, seed)
    try:
        previous_v2_history_len = 0
        for depth in range(128):
            if state.terminal:
                assert state.legal_actions() == ()
                chip_delta = state.terminal_chip_delta()
                assert sum(chip_delta) == 0
                icm_delta = state.terminal_icm_delta((0.5, 0.3, 0.2))
                assert abs(sum(icm_delta)) <= 1e-10
                return

            assert state.actor in (0, 1, 2)
            legal = state.legal_actions()
            assert legal
            _assert_observation_matches_authoritative_legal_mask(state)
            _assert_illegal_actions_fail_closed(state)

            v2 = decode_spnniv2(state.neural_bytes_v2())
            # Public history can grow or saturate at its explicit capacity. It
            # must never move backwards along one authoritative trajectory.
            assert v2.history_len >= previous_v2_history_len
            previous_v2_history_len = v2.history_len

            action = legal[(seed + depth) % len(legal)]
            state.apply(action)
        raise AssertionError("reachable legacy trajectory did not terminate within 128 decisions")
    finally:
        state.close()


def _walk_universal(solver: SolverLibrary, episode: Episode, seed: int) -> None:
    state = solver.create(episode, seed)
    try:
        for depth in range(128):
            if state.terminal:
                assert state.legal_actions() == ()
                assert sum(state.terminal_chip_delta()) == 0
                return

            assert state.actor in (0, 1, 2)
            _assert_observation_matches_authoritative_legal_mask(state)
            universal = state.universal_legal_actions(FULL_UNIVERSAL_MASK)
            assert universal
            action = universal[(seed * 3 + depth) % len(universal)]
            state.apply_universal(FULL_UNIVERSAL_MASK, action)
        raise AssertionError("reachable universal trajectory did not terminate within 128 decisions")
    finally:
        state.close()


def test_real_game_reachability_across_hu_and_three_handed_trajectories() -> None:
    solver = SolverLibrary(LIB)
    seeds = (1, 7, 31, 127, 509, 20260814)
    for episode in _episodes():
        for seed in seeds:
            _walk_legacy(solver, episode, seed)
            _walk_universal(solver, episode, seed ^ 0x5A17)
