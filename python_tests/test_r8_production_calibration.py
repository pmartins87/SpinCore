from pathlib import Path

import pytest

from spincore.production_calibration import (
    CalibrationTrial,
    select_calibration,
    transaction_digest_map,
    transaction_semantic_digest,
    transaction_stream_key,
)
from spincore.production_transaction_checkpoint import (
    LoadedProductionTransaction,
    ProductionTransactionIdentity,
)


REF = {"hu-seed-a": "sha256:aaa", "3h-seed-b": "sha256:bbb"}


def _trial(concurrency, seconds, *, digests=None, error=None, cpu=None):
    return CalibrationTrial(
        concurrency=concurrency,
        elapsed_seconds=seconds,
        completed_work_units=100,
        stream_state_digests=REF if digests is None else digests,
        mean_cpu_percent=cpu,
        error=error,
    )


def _transaction(*, profile="profile-a", domain="TRUE_HEADS_UP", seed=11, suffix="a"):
    identity = ProductionTransactionIdentity(
        profile_id=profile,
        domain=domain,
        algorithm_seed=seed,
        completed_iteration=3,
        roots_per_iteration=64,
    )
    generation_id = "spingen-v1-" + suffix * 64
    return LoadedProductionTransaction(
        identity=identity,
        generation_id=generation_id,
        generation_dir=Path("unused"),
        component_paths={},
        manifest={
            "generation_id": generation_id,
            "semantic_consistency_validated": True,
            "ready_for_tables": False,
        },
    )


def test_fastest_exact_trial_wins():
    out = select_calibration(
        reference_stream_state_digests=REF,
        trials=[_trial(1, 100), _trial(2, 60), _trial(4, 50)],
    )
    assert out["calibration_pass"] is True
    assert out["selected_concurrency"] == 4
    assert out["cpu_utilization_is_acceptance_gate"] is False
    assert out["ready_for_official_training"] is False
    assert out["ready_for_tables"] is False


def test_faster_semantically_changed_trial_is_rejected():
    changed = dict(REF)
    changed["3h-seed-b"] = "sha256:CHANGED"
    out = select_calibration(
        reference_stream_state_digests=REF,
        trials=[_trial(1, 100), _trial(8, 10, digests=changed)],
    )
    assert out["calibration_pass"] is True
    assert out["selected_concurrency"] == 1
    by_c = {r["concurrency"]: r for r in out["trials"]}
    assert by_c[8]["semantic_exact"] is False
    assert by_c[8]["accepted"] is False


def test_error_trial_is_rejected_even_if_digest_matches():
    out = select_calibration(
        reference_stream_state_digests=REF,
        trials=[_trial(1, 100, error="worker crashed")],
    )
    assert out["calibration_pass"] is False
    assert out["selected_concurrency"] is None


def test_exact_throughput_tie_prefers_lower_concurrency():
    out = select_calibration(
        reference_stream_state_digests=REF,
        trials=[
            CalibrationTrial(2, 50, 100, REF),
            CalibrationTrial(4, 100, 200, REF),
        ],
    )
    assert out["selected_concurrency"] == 2


def test_cpu_utilization_never_overrides_semantic_throughput_rule():
    out = select_calibration(
        reference_stream_state_digests=REF,
        trials=[_trial(1, 80, cpu=30.0), _trial(2, 100, cpu=99.0)],
    )
    assert out["selected_concurrency"] == 1


def test_duplicate_concurrency_fails_closed():
    with pytest.raises(ValueError, match="duplicate concurrency"):
        select_calibration(
            reference_stream_state_digests=REF,
            trials=[_trial(2, 50), _trial(2, 40)],
        )


def test_invalid_trial_inputs_fail_closed():
    with pytest.raises(ValueError):
        CalibrationTrial(0, 1, 1, REF)
    with pytest.raises(ValueError):
        CalibrationTrial(1, 0, 1, REF)
    with pytest.raises(ValueError):
        CalibrationTrial(1, 1, 0, REF)


def test_integrated_transaction_generation_is_authoritative_digest():
    tx = _transaction()
    assert transaction_stream_key(tx) == "profile-a|TRUE_HEADS_UP|11"
    assert transaction_semantic_digest(tx) == "spingen-v1-" + "a" * 64
    assert transaction_digest_map([tx]) == {
        "profile-a|TRUE_HEADS_UP|11": "spingen-v1-" + "a" * 64
    }


def test_unvalidated_or_malformed_integrated_generation_fails_closed():
    tx = _transaction()
    tx.manifest["semantic_consistency_validated"] = False
    with pytest.raises(ValueError, match="semantic-consistency"):
        transaction_semantic_digest(tx)

    bad = _transaction(suffix="b")
    bad.manifest["generation_id"] = "not-the-loaded-generation"
    with pytest.raises(ValueError, match="identity mismatch"):
        transaction_semantic_digest(bad)


def test_duplicate_integrated_stream_identity_fails_closed():
    with pytest.raises(ValueError, match="duplicate calibration production stream"):
        transaction_digest_map([_transaction(suffix="a"), _transaction(suffix="b")])
