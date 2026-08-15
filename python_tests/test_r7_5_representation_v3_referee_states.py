from __future__ import annotations

from pathlib import Path

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_representation_v3_referee_states import (
    generate_heldout_v3_states,
    replay_heldout_v3_state,
)
from spincore.solver import SolverLibrary


def test_phase2_heldout_states_are_deterministic_and_replay_byte_identical() -> None:
    library = Path("build/libspincore_solver_c.so")
    assert library.exists()
    solver = SolverLibrary(library)
    spec = postflop_candidate_specs(Path("."))["PF0_CONTROL_33_75_AI"]

    for domain in ("TRUE_HEADS_UP", "THREE_HANDED"):
        first = generate_heldout_v3_states(
            solver=solver,
            action_spec=spec,
            domain=domain,
            evaluation_seed=2029384436,
            count=64,
        )
        second = generate_heldout_v3_states(
            solver=solver,
            action_spec=spec,
            domain=domain,
            evaluation_seed=2029384436,
            count=64,
        )
        assert first == second
        assert len(first) == 64
        assert all(item.state_index == index for index, item in enumerate(first))
        assert all(item.observation_v3.startswith(b"SPNNIV3\0") for item in first)
        assert all(item.legal_slots for item in first)
        assert all(len(item.legal_slots) == len(item.exact_actions) for item in first)
        for descriptor in first[::7]:
            state = replay_heldout_v3_state(
                solver=solver,
                action_spec=spec,
                descriptor=descriptor,
            )
            state.close()


def test_phase2_heldout_seed_changes_chance_stream_but_not_contract() -> None:
    library = Path("build/libspincore_solver_c.so")
    assert library.exists()
    solver = SolverLibrary(library)
    spec = postflop_candidate_specs(Path("."))["PF0_CONTROL_33_75_AI"]
    a = generate_heldout_v3_states(
        solver=solver,
        action_spec=spec,
        domain="TRUE_HEADS_UP",
        evaluation_seed=2029384436,
        count=16,
    )
    b = generate_heldout_v3_states(
        solver=solver,
        action_spec=spec,
        domain="TRUE_HEADS_UP",
        evaluation_seed=1150634112,
        count=16,
    )
    assert a != b
    assert [item.state_index for item in a] == [item.state_index for item in b]
