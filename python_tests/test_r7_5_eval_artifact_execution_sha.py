from __future__ import annotations

import pytest

from spincore.r7_5_action_stage_contract import PAIRED_EVALUATION_SEEDS, POSTFLOP_TRAINING_SEEDS
from spincore.r7_5_eval_artifacts import (
    CANDIDATE_CELL_SCHEMA,
    DENSE_CACHE_SCHEMA,
    EXPECTED_EXECUTION_SHA,
    CandidateCellEvidence,
    DenseCellCache,
    validate_candidate_cell_evidence,
    validate_dense_cell_cache,
)

FUTURE_SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "89abcdef0123456789abcdef0123456789abcdef"


def _dense(sha: str) -> DenseCellCache:
    return DenseCellCache(
        schema=DENSE_CACHE_SCHEMA,
        execution_sha=sha,
        domain="TRUE_HEADS_UP",
        training_seed=POSTFLOP_TRAINING_SEEDS[0],
        evaluation_seed=PAIRED_EVALUATION_SEEDS[0],
        descriptors=(),
        q_references=(),
        diagnostics=(),
        crossplay_references=(),
    )


def _candidate(sha: str) -> CandidateCellEvidence:
    return CandidateCellEvidence(
        schema=CANDIDATE_CELL_SCHEMA,
        execution_sha=sha,
        candidate_id="PF0_CONTROL_33_75_AI",
        domain="TRUE_HEADS_UP",
        training_seed=POSTFLOP_TRAINING_SEEDS[0],
        evaluation_seed=PAIRED_EVALUATION_SEEDS[0],
        omission_samples=(),
        crossplay_samples=(),
        omission_summary={},
        crossplay_mean=0.0,
    )


def test_legacy_160_default_remains_strict() -> None:
    validate_dense_cell_cache(_dense(EXPECTED_EXECUTION_SHA), exact_counts=False)
    validate_candidate_cell_evidence(_candidate(EXPECTED_EXECUTION_SHA), exact_counts=False)
    with pytest.raises(ValueError, match="schema/execution mismatch"):
        validate_dense_cell_cache(_dense(FUTURE_SHA), exact_counts=False)
    with pytest.raises(ValueError, match="schema/execution mismatch"):
        validate_candidate_cell_evidence(_candidate(FUTURE_SHA), exact_counts=False)


def test_future_execution_sha_requires_exact_explicit_binding() -> None:
    validate_dense_cell_cache(
        _dense(FUTURE_SHA),
        exact_counts=False,
        expected_execution_sha=FUTURE_SHA,
    )
    validate_candidate_cell_evidence(
        _candidate(FUTURE_SHA),
        exact_counts=False,
        expected_execution_sha=FUTURE_SHA,
    )
    with pytest.raises(ValueError, match="schema/execution mismatch"):
        validate_dense_cell_cache(
            _dense(FUTURE_SHA),
            exact_counts=False,
            expected_execution_sha=OTHER_SHA,
        )
    with pytest.raises(ValueError, match="schema/execution mismatch"):
        validate_candidate_cell_evidence(
            _candidate(FUTURE_SHA),
            exact_counts=False,
            expected_execution_sha=OTHER_SHA,
        )


def test_explicit_execution_sha_must_be_lowercase_full_git_sha() -> None:
    for invalid in ("", "abc", "G" * 40, FUTURE_SHA.upper()):
        with pytest.raises(ValueError, match="lowercase 40-hex"):
            validate_dense_cell_cache(
                _dense(EXPECTED_EXECUTION_SHA),
                exact_counts=False,
                expected_execution_sha=invalid,
            )
