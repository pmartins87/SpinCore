from pathlib import Path

from spincore.deepcrusher_benchmark import (
    DEEPC_RUSHER_POLICY_ID,
    SPINCORE_POLICY_ID,
    ExternalExactAction,
    OfflineHeadToHeadEngine,
    aggregate_policy_chip_delta,
)
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_functional_agent import _street_from_state
from spincore.lean_solver_actions import lean_legal_actions, resolve_lean_exact
from spincore.solver import Episode, SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"


class PassiveLeanPolicy:
    def __init__(self, policy_id: str):
        self.policy_id = policy_id

    def choose_exact(self, state, *, seat: int, rng):
        del seat, rng
        street = _street_from_state(state)
        mask = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal = lean_legal_actions(state, mask)
        slot = 1 if 1 in legal else 0
        action_type, amount_to = resolve_lean_exact(state, mask, slot)
        return ExternalExactAction(action_type, amount_to)


def _engine():
    return OfflineHeadToHeadEngine(
        SolverLibrary(LIB),
        spincore_policy=PassiveLeanPolicy(SPINCORE_POLICY_ID),
        deepcrusher_policy=PassiveLeanPolicy(DEEPC_RUSHER_POLICY_ID),
        master_seed=77,
    )


def test_balanced_hu_block_cancels_identical_policy_seat_effects():
    episode = Episode(1500, True, 1, 15, 30, (900, 600, 0), 0, (2,))
    rows = _engine().play_balanced_block(episode, deal_seed=12345, scenario_index=0)
    assert len(rows) == 2
    totals = aggregate_policy_chip_delta(rows)
    assert totals[SPINCORE_POLICY_ID] == 0
    assert totals[DEEPC_RUSHER_POLICY_ID] == 0


def test_balanced_three_handed_block_cancels_identical_policy_seat_effects():
    episode = Episode(1500, False, 0, 10, 20, (700, 500, 300), 1, ())
    rows = _engine().play_balanced_block(episode, deal_seed=54321, scenario_index=1)
    assert len(rows) == 6
    totals = aggregate_policy_chip_delta(rows)
    assert totals[SPINCORE_POLICY_ID] == 0
    assert totals[DEEPC_RUSHER_POLICY_ID] == 0
    assert all(row.decisions > 0 for row in rows)
