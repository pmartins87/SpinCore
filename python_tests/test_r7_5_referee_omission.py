from __future__ import annotations

from pathlib import Path

import pytest

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_referee_omission import evaluate_heldout_omissions
from spincore.r7_5_referee_states import generate_heldout_referee_states
from spincore.solver import SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"


def _uniform(_state, _observation: bytes, legal: tuple[int, ...]):
    out = [0.0] * 10
    for action in legal:
        out[action] = 1.0 / len(legal)
    return tuple(out)


def _descriptors(solver, dense, domain: str, count: int = 6):
    return generate_heldout_referee_states(
        solver=solver,
        action_spec=dense,
        policy=_uniform,
        domain=domain,
        training_seed=1737995611,
        evaluation_seed=1817694185,
        count=count,
    )


def test_dense_referee_omission_against_itself_is_exactly_zero() -> None:
    solver = SolverLibrary(LIB)
    specs = postflop_candidate_specs(ROOT)
    dense = specs["PF_DENSE_REFERENCE"]
    for domain in ("TRUE_HEADS_UP", "THREE_HANDED"):
        results = evaluate_heldout_omissions(
            solver=solver,
            descriptors=_descriptors(solver, dense, domain, 5),
            dense_action_spec=dense,
            candidate_action_spec=dense,
            dense_policy=_uniform,
            exact_opponent_levels=1,
        )
        assert len(results) == 5
        assert all(result.omission == 0.0 for result in results)
        assert all(
            result.candidate_available_action_count == result.referee_action_count
            for result in results
        )


def test_compact_candidate_omission_is_nonnegative_and_deterministic() -> None:
    solver = SolverLibrary(LIB)
    specs = postflop_candidate_specs(ROOT)
    dense = specs["PF_DENSE_REFERENCE"]
    descriptors = _descriptors(solver, dense, "TRUE_HEADS_UP", 7)
    kwargs = dict(
        solver=solver,
        descriptors=descriptors,
        dense_action_spec=dense,
        candidate_action_spec=specs["PF0_CONTROL_33_75_AI"],
        dense_policy=_uniform,
        exact_opponent_levels=1,
    )
    first = evaluate_heldout_omissions(**kwargs)
    second = evaluate_heldout_omissions(**kwargs)
    assert first == second
    assert all(result.omission >= 0.0 for result in first)
    assert all(
        result.candidate_available_action_count <= result.referee_action_count
        for result in first
    )


def test_omission_credits_exact_action_alias_even_when_nominal_slot_differs() -> None:
    # A shallow 3H state creates fractional aliases. Exact-action intersection,
    # not slot-id intersection, is what determines candidate availability.
    solver = SolverLibrary(LIB)
    specs = postflop_candidate_specs(ROOT)
    dense = specs["PF_DENSE_REFERENCE"]
    descriptors = generate_heldout_referee_states(
        solver=solver,
        action_spec=dense,
        policy=_uniform,
        domain="THREE_HANDED",
        training_seed=645939859,
        evaluation_seed=1617273629,
        count=4,
    )
    results = evaluate_heldout_omissions(
        solver=solver,
        descriptors=descriptors,
        dense_action_spec=dense,
        candidate_action_spec=specs["PF4_CRUSHER_COMPACT_40_66_100_AI"],
        dense_policy=_uniform,
        exact_opponent_levels=0,
    )
    assert results
    assert all(result.candidate_available_action_count >= 1 for result in results)
    assert all(result.omission >= 0.0 for result in results)
