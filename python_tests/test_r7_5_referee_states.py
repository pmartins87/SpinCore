from __future__ import annotations

from pathlib import Path

import pytest

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_referee_states import (
    generate_heldout_referee_states,
    replay_heldout_referee_state,
)
from spincore.solver import SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"


def _uniform(_state, _observation: bytes, legal: tuple[int, ...]):
    out = [0.0] * 10
    for action in legal:
        out[action] = 1.0 / len(legal)
    return tuple(out)


def test_heldout_generation_and_replay_are_byte_identical_in_both_domains() -> None:
    solver = SolverLibrary(LIB)
    dense = postflop_candidate_specs(ROOT)["PF_DENSE_REFERENCE"]
    for domain in ("TRUE_HEADS_UP", "THREE_HANDED"):
        states = generate_heldout_referee_states(
            solver=solver,
            action_spec=dense,
            policy=_uniform,
            domain=domain,
            training_seed=1737995611,
            evaluation_seed=1817694185,
            count=48,
        )
        assert len(states) == 48
        assert [row.state_index for row in states] == list(range(48))
        for row in states:
            replayed = replay_heldout_referee_state(
                solver=solver,
                action_spec=dense,
                descriptor=row,
            )
            try:
                assert replayed.actor == row.actor
                assert replayed.neural_bytes() == row.observation_v1
                exact = tuple(
                    action
                    for _, action in replayed.universal_resolved_actions(row.dense_active_mask)
                )
                assert exact == row.dense_effective_actions
                assert len(exact) == len(set(exact))
            finally:
                replayed.close()


def test_heldout_generation_is_exactly_reproducible_and_eval_seed_sensitive() -> None:
    solver = SolverLibrary(LIB)
    dense = postflop_candidate_specs(ROOT)["PF_DENSE_REFERENCE"]
    kwargs = dict(
        solver=solver,
        action_spec=dense,
        policy=_uniform,
        domain="TRUE_HEADS_UP",
        training_seed=645939859,
        evaluation_seed=1617273629,
        count=32,
    )
    first = generate_heldout_referee_states(**kwargs)
    second = generate_heldout_referee_states(**kwargs)
    assert first == second
    changed = generate_heldout_referee_states(
        **{**kwargs, "evaluation_seed": 1817694185}
    )
    assert first != changed


def test_replay_fails_closed_on_descriptor_provenance_drift() -> None:
    solver = SolverLibrary(LIB)
    dense = postflop_candidate_specs(ROOT)["PF_DENSE_REFERENCE"]
    row = generate_heldout_referee_states(
        solver=solver,
        action_spec=dense,
        policy=_uniform,
        domain="THREE_HANDED",
        training_seed=1311335590,
        evaluation_seed=1817694185,
        count=1,
    )[0]
    drift = type(row)(**{**row.__dict__, "deck_seed": row.deck_seed ^ 1})
    with pytest.raises(ValueError, match="deck seed mismatch"):
        replay_heldout_referee_state(
            solver=solver,
            action_spec=dense,
            descriptor=drift,
        )
