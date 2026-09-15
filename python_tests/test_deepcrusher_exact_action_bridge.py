from pathlib import Path

from spincore.deepcrusher_benchmark import ExternalExactAction, apply_external_exact
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import apply_lean, resolve_lean_exact
from spincore.solver import Episode, SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"
MASK = FIRST_RELEASE_ACTION_SPEC.preflop_mask


def _root(seed: int = 31):
    solver = SolverLibrary(LIB)
    episode = Episode(1500, False, 0, 10, 20, (500, 500, 500), 0, ())
    return solver.create(episode, seed)


def test_external_exact_raise_reproduces_resolved_lean_state_exactly():
    root = _root()
    lean = root.clone()
    external = root.clone()
    try:
        action_type, amount_to = resolve_lean_exact(root, MASK, 3)
        assert (action_type, amount_to) == (4, 40)
        apply_lean(lean, MASK, 3)
        apply_external_exact(external, ExternalExactAction(action_type, amount_to))
        assert lean.actor == external.actor
        assert lean.terminal == external.terminal
        assert lean.neural_bytes_v2() == external.neural_bytes_v2()
    finally:
        root.close()
        lean.close()
        external.close()


def test_external_exact_call_reproduces_lean_call_state():
    root = _root(32)
    try:
        apply_lean(root, MASK, 3)  # BTN raises to 2BB; SB now faces a raise.
        lean = root.clone()
        external = root.clone()
        try:
            action_type, amount_to = resolve_lean_exact(root, MASK, 1)
            assert action_type in (1, 2)  # CHECK/CALL carrier resolves from state.
            apply_lean(lean, MASK, 1)
            apply_external_exact(external, ExternalExactAction(action_type, amount_to))
            assert lean.actor == external.actor
            assert lean.neural_bytes_v2() == external.neural_bytes_v2()
        finally:
            lean.close()
            external.close()
    finally:
        root.close()
