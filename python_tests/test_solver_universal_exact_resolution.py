from __future__ import annotations

from pathlib import Path

import pytest

from spincore.solver import Episode, ResolvedExactAction, SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"
FULL_MASK = 0x3FF


def _solver() -> SolverLibrary:
    return SolverLibrary(LIB)


def _three(stacks=(500, 500, 500)) -> Episode:
    return Episode(sum(stacks), False, 0, 10, 20, tuple(stacks), 0, ())


def test_every_effective_universal_slot_has_one_unique_exact_identity_and_query_is_read_only() -> None:
    state = _solver().create(_three(), 919191)
    try:
        before_actor = state.actor
        before_v1 = state.neural_bytes()
        before_v2 = state.neural_bytes_v2()
        legal = state.universal_legal_actions(FULL_MASK)
        rows = state.universal_resolved_actions(FULL_MASK)

        assert tuple(slot for slot, _ in rows) == legal
        exact = [action for _, action in rows]
        assert all(isinstance(action, ResolvedExactAction) for action in exact)
        assert len(exact) == len(set(exact)), "post-dedup effective slots must have unique exact identities"
        assert all(0 <= action.action_type <= 5 and action.amount_to >= 0 for action in exact)

        # Resolution is observational only: no actor, chips, history or cards move.
        assert state.actor == before_actor
        assert state.neural_bytes() == before_v1
        assert state.neural_bytes_v2() == before_v2
    finally:
        state.close()


def test_shallow_stack_allin_aliases_are_suppressed_and_cannot_be_resolved_as_effective_slots() -> None:
    # Mirrors the native all-in alias regression: shallow 30/30/30 stacks make
    # multiple nominal fractional sizings clamp to the same maximum exact action.
    state = _solver().create(_three((30, 30, 30)), 271828)
    try:
        legal = set(state.universal_legal_actions(FULL_MASK))
        rows = dict(state.universal_resolved_actions(FULL_MASK))
        allins = [(slot, action) for slot, action in rows.items() if action.action_type == 5]
        assert len(allins) == 1
        assert allins[0][0] == 9, "explicit ALL_IN must own the exact all-in identity"

        suppressed_fractional = [slot for slot in range(3, 9) if slot not in legal]
        assert suppressed_fractional, "shallow state must suppress at least one clamped fractional alias"
        for slot in suppressed_fractional:
            with pytest.raises(RuntimeError, match="alias|inactive|illegal"):
                state.resolve_universal_exact(FULL_MASK, slot)
    finally:
        state.close()


def test_min_raise_owns_minimum_alias_and_inactive_slots_fail_closed() -> None:
    state = _solver().create(_three(), 314159)
    try:
        active = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 9)
        legal = set(state.universal_legal_actions(active))
        assert 2 in legal
        minimum = state.resolve_universal_exact(active, 2)
        assert minimum.action_type in (3, 4, 5)
        for slot in legal:
            if slot == 2:
                continue
            assert state.resolve_universal_exact(active, slot) != minimum

        with pytest.raises(RuntimeError, match="inactive|illegal|alias"):
            state.resolve_universal_exact(active, 8)  # POT100 is inactive here.
    finally:
        state.close()


def test_bad_universal_resolution_arguments_fail_before_or_at_native_boundary() -> None:
    state = _solver().create(_three(), 1)
    try:
        with pytest.raises(ValueError):
            state.resolve_universal_exact(0x400, 1)
        with pytest.raises(ValueError):
            state.resolve_universal_exact(FULL_MASK, 10)
    finally:
        state.close()
