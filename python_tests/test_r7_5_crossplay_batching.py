from __future__ import annotations

from pathlib import Path

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_referee_crossplay import (
    build_dense_crossplay_reference,
    score_candidate_from_crossplay_reference,
)
from spincore.solver import SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"


class BatchAwareUniformPolicy:
    def __init__(self):
        self.batch_calls = 0

    def __call__(self, _state, _observation: bytes, legal: tuple[int, ...]):
        out = [0.0] * 10
        for action in legal:
            out[action] = 1.0 / len(legal)
        return tuple(out)

    def batch_probabilities(self, observations, legal_sets):
        self.batch_calls += 1
        return tuple(self(None, observation, legal) for observation, legal in zip(observations, legal_sets))


def test_dense_reference_is_invariant_to_rollout_batch_partition() -> None:
    solver = SolverLibrary(LIB)
    dense = postflop_candidate_specs(ROOT)["PF_DENSE_REFERENCE"]
    one = BatchAwareUniformPolicy()
    many = BatchAwareUniformPolicy()
    scalar_partition = build_dense_crossplay_reference(
        solver=solver,
        dense_action_spec=dense,
        dense_policy=one,
        domain="TRUE_HEADS_UP",
        training_seed=1737995611,
        evaluation_seed=1817694185,
        hand_count=32,
        rollout_batch_size=1,
    )
    batched_partition = build_dense_crossplay_reference(
        solver=solver,
        dense_action_spec=dense,
        dense_policy=many,
        domain="TRUE_HEADS_UP",
        training_seed=1737995611,
        evaluation_seed=1817694185,
        hand_count=32,
        rollout_batch_size=8,
    )
    assert scalar_partition == batched_partition
    assert one.batch_calls > 0 and many.batch_calls > 0
    assert many.batch_calls < one.batch_calls


def test_candidate_scores_are_invariant_to_rollout_batch_partition() -> None:
    solver = SolverLibrary(LIB)
    specs = postflop_candidate_specs(ROOT)
    dense_policy = BatchAwareUniformPolicy()
    references = build_dense_crossplay_reference(
        solver=solver,
        dense_action_spec=specs["PF_DENSE_REFERENCE"],
        dense_policy=dense_policy,
        domain="THREE_HANDED",
        training_seed=645939859,
        evaluation_seed=1617273629,
        hand_count=24,
        rollout_batch_size=6,
    )
    candidate_one = BatchAwareUniformPolicy()
    candidate_many = BatchAwareUniformPolicy()
    common = dict(
        solver=solver,
        references=references,
        dense_action_spec=specs["PF_DENSE_REFERENCE"],
        dense_policy=dense_policy,
        candidate_action_spec=specs["PF1_33_50_75_AI"],
        domain="THREE_HANDED",
        training_seed=645939859,
        evaluation_seed=1617273629,
        candidate_seat=1,
    )
    first = score_candidate_from_crossplay_reference(
        **common, candidate_policy=candidate_one, rollout_batch_size=1
    )
    second = score_candidate_from_crossplay_reference(
        **common, candidate_policy=candidate_many, rollout_batch_size=8
    )
    assert first == second
    assert candidate_many.batch_calls > 0
