from __future__ import annotations

from pathlib import Path

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_action_cross_seed import (
    build_cross_seed_common_corpus,
    cross_seed_policy_stability,
)
from spincore.r7_5_action_stage_contract import POSTFLOP_TRAINING_SEEDS
from spincore.solver import SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"


def _uniform(_state, _observation: bytes, legal: tuple[int, ...]):
    out = [0.0] * 10
    for action in legal:
        out[action] = 1.0 / len(legal)
    return tuple(out)


def _first_legal(_state, _observation: bytes, legal: tuple[int, ...]):
    out = [0.0] * 10
    out[min(legal)] = 1.0
    return tuple(out)


def _last_legal(_state, _observation: bytes, legal: tuple[int, ...]):
    out = [0.0] * 10
    out[max(legal)] = 1.0
    return tuple(out)


def test_cross_seed_common_corpus_is_policy_independent_and_exact_size() -> None:
    solver = SolverLibrary(LIB)
    dense = postflop_candidate_specs(ROOT)["PF_DENSE_REFERENCE"]
    first = build_cross_seed_common_corpus(
        solver=solver,
        dense_action_spec=dense,
        domain="TRUE_HEADS_UP",
        per_training_seed=4,
    )
    second = build_cross_seed_common_corpus(
        solver=solver,
        dense_action_spec=dense,
        domain="TRUE_HEADS_UP",
        per_training_seed=4,
    )
    assert first == second
    assert len(first) == 12
    assert {row.training_seed for row in first} == set(POSTFLOP_TRAINING_SEEDS)


def test_identical_three_seed_policies_have_exact_zero_tv() -> None:
    solver = SolverLibrary(LIB)
    specs = postflop_candidate_specs(ROOT)
    corpus = build_cross_seed_common_corpus(
        solver=solver,
        dense_action_spec=specs["PF_DENSE_REFERENCE"],
        domain="TRUE_HEADS_UP",
        per_training_seed=4,
    )
    report = cross_seed_policy_stability(
        solver=solver,
        descriptors=corpus,
        dense_action_spec=specs["PF_DENSE_REFERENCE"],
        candidate_action_spec=specs["PF0_CONTROL_33_75_AI"],
        policies_by_seed={seed: _uniform for seed in POSTFLOP_TRAINING_SEEDS},
        candidate_id="PF0_CONTROL_33_75_AI",
        domain="TRUE_HEADS_UP",
    )
    assert report["common_state_count"] == 12
    assert report["pairwise_tv_count"] == 36
    assert report["mean_tv"] == 0.0
    assert report["p95_tv"] == 0.0
    assert report["gate_pass"] is True


def test_cross_seed_divergence_is_detected_on_same_common_states() -> None:
    solver = SolverLibrary(LIB)
    specs = postflop_candidate_specs(ROOT)
    corpus = build_cross_seed_common_corpus(
        solver=solver,
        dense_action_spec=specs["PF_DENSE_REFERENCE"],
        domain="THREE_HANDED",
        per_training_seed=4,
    )
    a, b, c = POSTFLOP_TRAINING_SEEDS
    report = cross_seed_policy_stability(
        solver=solver,
        descriptors=corpus,
        dense_action_spec=specs["PF_DENSE_REFERENCE"],
        candidate_action_spec=specs["PF1_33_50_75_AI"],
        policies_by_seed={a: _first_legal, b: _last_legal, c: _uniform},
        candidate_id="PF1_33_50_75_AI",
        domain="THREE_HANDED",
    )
    assert report["mean_tv"] > 0.0
    assert report["p95_tv"] > 0.0
    assert report["gate_pass"] is False
