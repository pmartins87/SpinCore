from __future__ import annotations

from pathlib import Path

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_referee_crossplay import candidate_seats, paired_crossplay_scores
from spincore.solver import SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"


def _uniform(_state, _observation: bytes, legal: tuple[int, ...]):
    out = [0.0] * 10
    for action in legal:
        out[action] = 1.0 / len(legal)
    return tuple(out)


def test_dense_self_crossplay_is_exactly_zero_for_every_physical_candidate_seat() -> None:
    solver = SolverLibrary(LIB)
    dense = postflop_candidate_specs(ROOT)["PF_DENSE_REFERENCE"]
    for domain in ("TRUE_HEADS_UP", "THREE_HANDED"):
        for seat in candidate_seats(domain):
            scores = paired_crossplay_scores(
                solver=solver,
                dense_action_spec=dense,
                dense_policy=_uniform,
                candidate_action_spec=dense,
                candidate_policy=_uniform,
                domain=domain,
                training_seed=1737995611,
                evaluation_seed=1817694185,
                candidate_seat=seat,
                hand_count=24,
            )
            assert scores == (0.0,) * 24


def test_crossplay_is_reproducible_for_compact_candidate_and_seed_sensitive() -> None:
    solver = SolverLibrary(LIB)
    specs = postflop_candidate_specs(ROOT)
    kwargs = dict(
        solver=solver,
        dense_action_spec=specs["PF_DENSE_REFERENCE"],
        dense_policy=_uniform,
        candidate_action_spec=specs["PF0_CONTROL_33_75_AI"],
        candidate_policy=_uniform,
        domain="TRUE_HEADS_UP",
        training_seed=645939859,
        evaluation_seed=1617273629,
        candidate_seat=1,
        hand_count=32,
    )
    first = paired_crossplay_scores(**kwargs)
    second = paired_crossplay_scores(**kwargs)
    assert first == second
    changed = paired_crossplay_scores(**{**kwargs, "evaluation_seed": 1817694185})
    assert first != changed


def test_candidate_seat_contract_matches_exact_r7_4_topology() -> None:
    assert candidate_seats("TRUE_HEADS_UP") == (1, 2)
    assert candidate_seats("THREE_HANDED") == (0, 1, 2)
