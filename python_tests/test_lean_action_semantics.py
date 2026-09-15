from __future__ import annotations

from pathlib import Path

from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import apply_lean, lean_legal_actions, resolve_lean_exact
from spincore.solver import Episode, SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"
MASK = FIRST_RELEASE_ACTION_SPEC.preflop_mask


def _root(seed: int = 7):
    solver = SolverLibrary(LIB)
    episode = Episode(1500, False, 0, 10, 20, (500, 500, 500), 0, ())
    return solver.create(episode, seed)


def test_lean_preflop_open_is_2bb_and_single_raise_response_is_5bb():
    state = _root()
    try:
        assert 3 in lean_legal_actions(state, MASK)  # legacy B33 label = normal open
        assert resolve_lean_exact(state, MASK, 3) == (4, 40)  # RaiseTo 2 BB
        apply_lean(state, MASK, 3)
        assert 5 in lean_legal_actions(state, MASK)  # legacy B50 label = 3-bet group
        assert resolve_lean_exact(state, MASK, 5) == (4, 100)  # 5 BB
    finally:
        state.close()


def test_lean_preflop_isolation_size_tracks_extra_limpers():
    state = _root(8)
    try:
        apply_lean(state, MASK, 1)  # BTN limp
        assert 7 in lean_legal_actions(state, MASK)
        assert resolve_lean_exact(state, MASK, 7) == (4, 50)  # 2.5 BB
        apply_lean(state, MASK, 1)  # SB completes: second limper
        assert 7 in lean_legal_actions(state, MASK)
        assert resolve_lean_exact(state, MASK, 7) == (4, 70)  # 3.5 BB
    finally:
        state.close()


def test_lean_preflop_3bet_size_tracks_caller_after_raise():
    state = _root(9)
    try:
        apply_lean(state, MASK, 3)  # BTN 2 BB open
        apply_lean(state, MASK, 1)  # SB calls
        assert 5 in lean_legal_actions(state, MASK)
        assert resolve_lean_exact(state, MASK, 5) == (4, 140)  # 5 BB + 2 BB/caller
    finally:
        state.close()


def test_lean_limper_facing_isolation_has_no_normal_reraise():
    state = _root(10)
    try:
        apply_lean(state, MASK, 1)  # BTN limp
        apply_lean(state, MASK, 7)  # SB isolation
        apply_lean(state, MASK, 0)  # BB folds
        legal = set(lean_legal_actions(state, MASK))
        assert legal <= {0, 1, 9}
        assert 3 not in legal and 5 not in legal and 7 not in legal and 8 not in legal
    finally:
        state.close()
