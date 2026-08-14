from __future__ import annotations

from pathlib import Path

from spincore.r7_5_eval_artifacts import (
    CANDIDATE_CELL_SCHEMA,
    DENSE_CACHE_SCHEMA,
    EXPECTED_EXECUTION_SHA,
    CandidateCellEvidence,
    DenseCellCache,
    MemoizedPolicy,
    StateDiagnostic,
    load_candidate_cell_evidence,
    load_dense_cell_cache,
    save_candidate_cell_evidence,
    save_dense_cell_cache,
    spr_bucket,
    summarize_omission,
)


def test_spr_buckets_and_objective_omission_groups_are_frozen() -> None:
    assert spr_bucket(0.0) == "SPR_0_1"
    assert spr_bucket(1.0) == "SPR_1_2"
    assert spr_bucket(2.0) == "SPR_2_4"
    assert spr_bucket(4.0) == "SPR_4_8"
    assert spr_bucket(8.0) == "SPR_8_PLUS"
    diagnostics = (
        StateDiagnostic(1, 0.5, "SPR_0_1", "ST1_M0_FD0_SD0_PB0_MS2"),
        StateDiagnostic(1, 1.5, "SPR_1_2", "ST1_M0_FD0_SD0_PB0_MS2"),
        StateDiagnostic(2, 5.0, "SPR_4_8", "ST2_M1_FD1_SD0_PB1_MS3"),
    )
    summary = summarize_omission((0.0, 0.1, 0.2), diagnostics)
    assert summary["overall"]["mean"] == 0.1
    assert summary["overall"]["p50"] == 0.1
    assert summary["overall"]["p95"] == 0.2
    assert summary["by_street"]["1"]["count"] == 2
    assert summary["by_spr"]["SPR_4_8"]["count"] == 1
    assert summary["by_semantic_sentinel"]["ST1_M0_FD0_SD0_PB0_MS2"]["count"] == 2


def test_evaluation_artifacts_roundtrip_without_production_authority(tmp_path: Path) -> None:
    dense = DenseCellCache(
        schema=DENSE_CACHE_SCHEMA,
        execution_sha=EXPECTED_EXECUTION_SHA,
        domain="TRUE_HEADS_UP",
        training_seed=1737995611,
        evaluation_seed=1817694185,
        descriptors=(),
        q_references=(),
        diagnostics=(),
        crossplay_references=(),
    )
    dense_path = tmp_path / "dense.pkl.gz"
    save_dense_cell_cache(dense_path, dense, exact_counts=False)
    assert load_dense_cell_cache(dense_path, exact_counts=False) == dense

    candidate = CandidateCellEvidence(
        schema=CANDIDATE_CELL_SCHEMA,
        execution_sha=EXPECTED_EXECUTION_SHA,
        candidate_id="PF0_CONTROL_33_75_AI",
        domain="TRUE_HEADS_UP",
        training_seed=1737995611,
        evaluation_seed=1817694185,
        omission_samples=(0.0, 0.01),
        crossplay_samples=(0.0, -0.01),
        omission_summary={"overall": {"mean": 0.005}},
        crossplay_mean=-0.005,
    )
    candidate_path = tmp_path / "candidate.pkl.gz"
    save_candidate_cell_evidence(candidate_path, candidate, exact_counts=False)
    assert load_candidate_cell_evidence(candidate_path, exact_counts=False) == candidate
    assert not candidate.production_training_authorized and not candidate.ready_for_tables


class _BatchBase:
    def __init__(self):
        self.batch_calls = 0

    def __call__(self, _state, _observation, legal):
        out = [0.0] * 10
        for action in legal:
            out[action] = 1.0 / len(legal)
        return tuple(out)

    def batch_probabilities(self, observations, legal_sets):
        self.batch_calls += 1
        return tuple(self(None, observation, legal) for observation, legal in zip(observations, legal_sets))


def test_memoized_policy_reuses_exact_observation_legal_identity_in_batch() -> None:
    base = _BatchBase()
    memo = MemoizedPolicy(base)
    observation = b"same-observation"
    legal = (0, 1, 3)
    first = memo.batch_probabilities([observation, observation], [legal, legal])
    assert first[0] == first[1]
    assert base.batch_calls == 1
    misses = memo.misses
    second = memo.batch_probabilities([observation], [legal])
    assert second[0] == first[0]
    assert memo.misses == misses
    assert memo.hits >= 1
