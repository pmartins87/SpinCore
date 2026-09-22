from pathlib import Path

from spincore.deepcrusher_state import (
    DOMAIN_THREE_HANDED,
    STREET_PREFLOP,
    state_view,
)
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import apply_lean
from spincore.solver import Episode, SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"
MASK = FIRST_RELEASE_ACTION_SPEC.preflop_mask


def _root(seed: int = 41):
    solver = SolverLibrary(LIB)
    return solver.create(Episode(1500, False, 0, 10, 20, (700, 500, 300), 0, ()), seed)


def test_state_view_decodes_current_poker_geometry_and_history():
    state = _root()
    try:
        view = state_view(state)
        assert view.domain == DOMAIN_THREE_HANDED
        assert view.street == STREET_PREFLOP
        assert view.live_count == 3
        assert len(view.hero_hole_ranks) == 2
        assert all(2 <= rank <= 14 for rank in view.hero_hole_ranks)
        assert len(view.hero_hand_class) in (2, 3)
        assert view.hero_hand_class[0] in "AKQJT98765432"
        assert view.hero_hand_class[1] in "AKQJT98765432"
        if view.hero_hole_ranks[0] == view.hero_hole_ranks[1]:
            assert len(view.hero_hand_class) == 2
        else:
            assert view.hero_hand_class[-1] == ("s" if view.hole_suited else "o")
        assert view.board_ranks == ()
        assert len(view.same_suit) == 21
        assert len(view.primitive_legal) == 6
        # Blind posts are present but filtered from voluntary action history.
        assert len(view.history) >= 2
        assert view.voluntary_history() == ()

        apply_lean(state, MASK, 3)  # current actor opens to 2BB
        next_view = state_view(state)
        voluntary = next_view.voluntary_history(street=STREET_PREFLOP)
        assert len(voluntary) == 1
        assert voluntary[0].forced is False
        assert voluntary[0].action_type == 4  # RaiseTo
        assert voluntary[0].resulting_commitment_bb == 2.0
        assert next_view.to_call_bb >= 0.0
    finally:
        state.close()


def test_state_view_preserves_true_hu_topology():
    solver = SolverLibrary(LIB)
    state = solver.create(Episode(1500, True, 2, 20, 40, (900, 0, 600), 2, (1,)), 42)
    try:
        view = state_view(state)
        assert view.is_true_hu
        assert view.live_count == 2
        assert view.statuses[2] == 2  # canonical absent relative seat
    finally:
        state.close()
