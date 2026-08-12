from __future__ import annotations

import copy
import hashlib

import pytest

from spincore.production_stream_scheduler import (
    DurableIterationReceipt,
    IndependentStreamScheduler,
    ProductionStreamKey,
    ProductionStreamPlan,
)


def _key(profile: str, domain: str, seed: int):
    return ProductionStreamKey(profile, domain, seed)


def _scheduler():
    return IndependentStreamScheduler(
        [
            ProductionStreamPlan(_key("p1", "TRUE_HEADS_UP", 11), 3),
            ProductionStreamPlan(_key("p1", "TRUE_HEADS_UP", 22), 3),
            ProductionStreamPlan(_key("p1", "THREE_HANDED", 11), 2),
            ProductionStreamPlan(_key("p2", "TRUE_HEADS_UP", 11), 2),
        ]
    )


def _receipt(lease, *, parent=None, tag=""):
    digest = hashlib.sha256(
        f"{lease.key.stream_id}|{lease.iteration}|{tag}".encode()
    ).hexdigest()
    return DurableIterationReceipt(
        key=lease.key,
        iteration=lease.iteration,
        lease_id=lease.lease_id,
        checkpoint_locator=f"artifact://{lease.key.stream_id}/{lease.iteration}/{tag or 'default'}",
        checkpoint_sha256=digest,
        checkpoint_size_bytes=1024 + lease.iteration,
        parent_checkpoint_sha256=parent,
    )


def _complete(scheduler, lease, *, tag=""):
    parent = scheduler.last_checkpoint_sha256(lease.key)
    receipt = _receipt(lease, parent=parent, tag=tag)
    scheduler.complete(lease, receipt)
    return receipt


def test_parallel_leases_are_always_distinct_streams():
    s = _scheduler()
    leases = s.lease(99)
    assert len(leases) == 4
    assert len({row.key for row in leases}) == len(leases)
    assert len(set(s.active_stream_ids)) == 4
    # No second lease can be issued while every independent stream is active.
    assert s.lease(99) == ()


def test_completion_requires_durable_receipt_and_never_skips_iteration():
    s = _scheduler()
    leases = s.lease(4)
    first = leases[0]
    with pytest.raises(ValueError, match="durable checkpoint receipt is required"):
        s.complete(first)
    assert first.key.stream_id in s.active_stream_ids

    receipt = _complete(s, first)
    assert s.last_checkpoint_sha256(first.key) == receipt.checkpoint_sha256
    next_rows = s.lease(4)
    assert len(next_rows) == 1
    assert next_rows[0].key == first.key
    assert next_rows[0].iteration == 2
    with pytest.raises(ValueError, match="stale or out of order"):
        s.complete(first, receipt)


def test_receipt_must_belong_to_active_lease():
    s = _scheduler()
    leases = s.lease(2)
    a, b = leases
    wrong = _receipt(b)
    with pytest.raises(ValueError, match="does not belong to the active lease"):
        s.complete(a, wrong)
    assert a.key.stream_id in s.active_stream_ids


def test_checkpoint_parent_must_match_exact_stream_lineage():
    s = _scheduler()
    first = s.lease(1)[0]
    first_receipt = _complete(s, first, tag="first")
    second = s.lease(1)[0]
    assert second.key == first.key and second.iteration == 2

    wrong_parent = hashlib.sha256(b"wrong-parent").hexdigest()
    bad = _receipt(second, parent=wrong_parent, tag="second")
    with pytest.raises(ValueError, match="parent does not match"):
        s.complete(second, bad)
    assert s.last_checkpoint_sha256(first.key) == first_receipt.checkpoint_sha256

    good = _receipt(second, parent=first_receipt.checkpoint_sha256, tag="second")
    s.complete(second, good)
    assert s.last_checkpoint_sha256(first.key) == good.checkpoint_sha256


def test_failure_retries_same_iteration_without_silent_progress():
    s = _scheduler()
    first = s.lease(1)[0]
    assert first.iteration == 1
    s.fail(first)
    retry = s.lease(1)[0]
    assert retry.key == first.key
    assert retry.iteration == 1
    assert retry.lease_id != first.lease_id
    assert s.last_checkpoint_sha256(first.key) is None
    row = next(x for x in s.state_dict()["streams"] if x["stream_id"] == first.key.stream_id)
    assert row["failed_attempts_for_next_iteration"] == 1


def test_crash_restore_clears_exclusive_lease_but_does_not_advance_iteration():
    s = _scheduler()
    lease = s.lease(1)[0]
    payload = copy.deepcopy(s.state_dict())
    resumed = IndependentStreamScheduler.from_state_dict(payload)
    retry = resumed.lease(1)[0]
    assert retry.key == lease.key
    assert retry.iteration == lease.iteration == 1
    assert retry.lease_id != lease.lease_id
    assert resumed.last_checkpoint_sha256(retry.key) is None


def test_checkpoint_round_trip_preserves_durable_lineage_deterministically():
    baseline = _scheduler()
    resumed = _scheduler()

    first = baseline.lease(4)
    first_receipts = {}
    for row in first:
        first_receipts[row.key] = _complete(baseline, row, tag="baseline-i1")
    second = baseline.lease(4)

    first2 = resumed.lease(4)
    for row in first2:
        _complete(resumed, row, tag="baseline-i1")
    resumed = IndependentStreamScheduler.from_state_dict(copy.deepcopy(resumed.state_dict()))
    second2 = resumed.lease(4)

    assert [(x.key, x.iteration) for x in second2] == [(x.key, x.iteration) for x in second]
    for key, receipt in first_receipts.items():
        assert resumed.last_checkpoint_sha256(key) == receipt.checkpoint_sha256


def test_restore_rejects_completed_iteration_without_durable_receipt_metadata():
    s = _scheduler()
    lease = s.lease(1)[0]
    _complete(s, lease)
    payload = copy.deepcopy(s.state_dict())
    row = next(x for x in payload["streams"] if x["stream_id"] == lease.key.stream_id)
    row["last_checkpoint_sha256"] = None
    with pytest.raises(ValueError, match="missing valid durable checkpoint SHA256"):
        IndependentStreamScheduler.from_state_dict(payload)


def test_receipt_from_file_hashes_existing_nonempty_checkpoint(tmp_path):
    s = _scheduler()
    lease = s.lease(1)[0]
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"durable-checkpoint-bytes")
    receipt = DurableIterationReceipt.from_file(
        lease,
        path,
        parent_checkpoint_sha256=None,
        locator="artifact://run/seed/checkpoint.bin",
    )
    assert receipt.checkpoint_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt.checkpoint_size_bytes == len(path.read_bytes())
    s.complete(lease, receipt)


def test_scheduler_rejects_duplicate_stream_identity():
    key = _key("p", "TRUE_HEADS_UP", 7)
    with pytest.raises(ValueError, match="duplicate"):
        IndependentStreamScheduler([ProductionStreamPlan(key, 2), ProductionStreamPlan(key, 2)])


def test_scheduler_checkpoint_never_authorizes_table_use():
    payload = _scheduler().state_dict()
    assert payload["ready_for_tables"] is False
    payload["ready_for_tables"] = True
    with pytest.raises(ValueError, match="cannot authorize table use"):
        IndependentStreamScheduler.from_state_dict(payload)
