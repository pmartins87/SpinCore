from __future__ import annotations

from pathlib import Path

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_representation_v3_crossplay import (
    common_reference_scores,
    live_candidate_seats,
    mirrored_h3_vs_h2_scores,
    uniform_pf0_policy,
)
from spincore.solver import SolverLibrary


def test_identical_policy_common_reference_is_exact_zero() -> None:
    library = Path("build/libspincore_solver_c.so")
    assert library.exists()
    solver = SolverLibrary(library)
    spec = postflop_candidate_specs(Path("."))["PF0_CONTROL_33_75_AI"]
    for domain in ("TRUE_HEADS_UP", "THREE_HANDED"):
        for seat in live_candidate_seats(domain):
            scores = common_reference_scores(
                solver=solver,
                action_spec=spec,
                candidate_policy=uniform_pf0_policy,
                domain=domain,
                evaluation_seed=2029384436,
                candidate_seat=seat,
                hand_count=48,
                rollout_batch_size=16,
            )
            assert len(scores) == 48
            assert all(score == 0.0 for score in scores)


def test_identical_h2_h3_pairwise_policy_is_exact_zero() -> None:
    library = Path("build/libspincore_solver_c.so")
    assert library.exists()
    solver = SolverLibrary(library)
    spec = postflop_candidate_specs(Path("."))["PF0_CONTROL_33_75_AI"]
    for domain in ("TRUE_HEADS_UP", "THREE_HANDED"):
        for seat in live_candidate_seats(domain):
            scores = mirrored_h3_vs_h2_scores(
                solver=solver,
                action_spec=spec,
                h2_policy=uniform_pf0_policy,
                h3_policy=uniform_pf0_policy,
                domain=domain,
                evaluation_seed=1150634112,
                candidate_seat=seat,
                hand_count=48,
                rollout_batch_size=16,
            )
            assert len(scores) == 48
            assert all(score == 0.0 for score in scores)
