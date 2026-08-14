from __future__ import annotations

from pathlib import Path

import pytest

from spincore.solver import Episode, SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"

FOLD = 0
CHECK_CALL = 1
MIN_RAISE = 2
POT_33 = 3
POT_40 = 4
POT_50 = 5
POT_66 = 6
POT_75 = 7
POT_100 = 8
ALL_IN = 9

DENSE = sum(1 << slot for slot in range(10))
PRE_CONTROL = (1 << FOLD) | (1 << CHECK_CALL) | (1 << MIN_RAISE) | (1 << ALL_IN)
POST_CONTROL = (1 << FOLD) | (1 << CHECK_CALL) | (1 << POT_33) | (1 << POT_75) | (1 << ALL_IN)


def hu_episode(stacks=(0, 750, 750), dealer=1):
    return Episode(1500, True, 0, 10, 20, tuple(stacks), dealer, (0,))


def _same_state(a, b) -> bool:
    if a.terminal or b.terminal:
        return a.terminal == b.terminal and a.terminal_chip_delta() == b.terminal_chip_delta()
    return (
        a.actor == b.actor
        and a.legal_actions() == b.legal_actions()
        and a.neural_bytes() == b.neural_bytes()
        and a.neural_bytes_v2() == b.neural_bytes_v2()
    )


def _reach_flop(state):
    guard = 0
    while not state.terminal and state.neural_bytes_v2()[111 + 1] == 0 and guard < 16:
        guard += 1
        assert 1 in state.legal_actions()
        state.apply(1)
    assert not state.terminal
    assert state.neural_bytes_v2()[111 + 1] == 1
    return state


def test_universal_preflop_control_matches_frozen_six_slot_children() -> None:
    solver = SolverLibrary(LIB)
    root = solver.create(hu_episode(), 741852)
    try:
        old_to_new = {0: FOLD, 1: CHECK_CALL, 2: MIN_RAISE, 5: ALL_IN}
        legal_old = root.legal_actions()
        legal_new = root.universal_legal_actions(PRE_CONTROL)
        expected_new = tuple(old_to_new[action] for action in legal_old if action in old_to_new)
        assert legal_new == expected_new
        for old_action, new_action in old_to_new.items():
            if old_action not in legal_old:
                continue
            old_child = root.child(old_action)
            new_child = root.child_universal(PRE_CONTROL, new_action)
            try:
                assert _same_state(old_child, new_child)
            finally:
                old_child.close()
                new_child.close()
    finally:
        root.close()


def test_universal_postflop_33_75_control_matches_frozen_children() -> None:
    solver = SolverLibrary(LIB)
    root = solver.create(hu_episode(), 963258)
    try:
        _reach_flop(root)
        old_to_new = {0: FOLD, 1: CHECK_CALL, 3: POT_33, 4: POT_75, 5: ALL_IN}
        legal_old = root.legal_actions()
        legal_new = root.universal_legal_actions(POST_CONTROL)
        expected_new = tuple(old_to_new[action] for action in legal_old if action in old_to_new)
        assert legal_new == expected_new
        for old_action, new_action in old_to_new.items():
            if old_action not in legal_old:
                continue
            old_child = root.child(old_action)
            new_child = root.child_universal(POST_CONTROL, new_action)
            try:
                assert _same_state(old_child, new_child)
            finally:
                old_child.close()
                new_child.close()
    finally:
        root.close()


def test_universal_dense_mask_exposes_only_deduplicated_actions_and_alias_apply_fails() -> None:
    solver = SolverLibrary(LIB)
    # Very shallow effective stacks force several nominal sizes onto the same
    # exact maximum commitment, making state-local alias suppression observable.
    root = solver.create(Episode(90, True, 0, 10, 20, (0, 45, 45), 1, (0,)), 112358)
    try:
        combined = set(root.universal_legal_actions(DENSE))
        individually_legal = set()
        for slot in range(10):
            if root.universal_legal_actions(1 << slot):
                individually_legal.add(slot)
        suppressed = sorted(individually_legal - combined)
        assert suppressed, "fixture must exercise at least one state-local exact-action alias"
        assert ALL_IN in combined
        for slot in suppressed:
            clone = root.clone()
            try:
                with pytest.raises(RuntimeError, match="alias|inactive|illegal"):
                    clone.apply_universal(DENSE, slot)
            finally:
                clone.close()
        # Every effective action accepted by the mask must remain physically legal.
        for slot in sorted(combined):
            child = root.child_universal(DENSE, slot)
            child.close()
    finally:
        root.close()


def test_old_six_slot_api_remains_available_unchanged() -> None:
    solver = SolverLibrary(LIB)
    root = solver.create(hu_episode(), 424242)
    try:
        assert solver.lib.spincore_solver_c_abi_version() == 2
        old_legal = root.legal_actions()
        assert old_legal
        child = root.child(old_legal[0])
        child.close()
    finally:
        root.close()
