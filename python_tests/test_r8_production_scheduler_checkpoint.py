from __future__ import annotations

import hashlib

import pytest

from spincore.production_scheduler_checkpoint import (
    load_scheduler_checkpoint,
    save_scheduler_checkpoint_atomic,
)
from spincore.production_stream_scheduler import (
    DurableIterationReceipt,
    IndependentStreamScheduler,
    ProductionStreamKey,
    ProductionStreamPlan,
)


def _scheduler() -> IndependentStreamScheduler:
    key = ProductionStreamKey("spinprofile-v3:test", "TRUE_HEADS_UP", 101)
    return IndependentStreamScheduler([ProductionStreamPlan(key, 3)])


def _complete_current(s: IndependentStreamScheduler, *, tag: str):
    lease = s.lease(1)[0]
    parent = s.last_checkpoint_sha256(lease.key)
    digest = hashlib.sha256(f"{lease.key.stream_id}|{lease.iteration}|{tag}".encode()).hexdigest()
    receipt = DurableIterationReceipt(
        key=lease.key,
        iteration=lease.iteration,
        lease_id=lease.lease_id,
        checkpoint_locator=f"artifact://stream/{lease.iteration}/{tag}",
        checkpoint_sha256=digest,
        checkpoint_size_bytes=4096 + lease.iteration,
        parent_checkpoint_sha256=parent,
    )
    s.complete(lease, receipt)
    return lease, receipt


def test_atomic_scheduler_checkpoint_round_trip_preserves_durable_lineage(tmp_path):
    s = _scheduler()
    lease, stream_receipt = _complete_current(s, tag="i1")
    path = tmp_path / "scheduler.json"
    durable = save_scheduler_checkpoint_atomic(path, s)

    assert path.exists()
    assert durable.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert durable.size_bytes == len(path.read_bytes())
    assert not list(tmp_path.glob("scheduler.json.tmp-*"))

    resumed = load_scheduler_checkpoint(path, expected_sha256=durable.sha256)
    assert resumed.last_checkpoint_sha256(lease.key) == stream_receipt.checkpoint_sha256
    next_lease = resumed.lease(1)[0]
    assert next_lease.iteration == 2


def test_crash_checkpoint_with_active_lease_retries_same_iteration(tmp_path):
    s = _scheduler()
    lease = s.lease(1)[0]
    durable = save_scheduler_checkpoint_atomic(tmp_path / "scheduler.json", s)

    resumed = load_scheduler_checkpoint(
        durable.path,
        expected_sha256=durable.sha256,
        clear_active_leases=True,
    )
    retry = resumed.lease(1)[0]
    assert retry.key == lease.key
    assert retry.iteration == lease.iteration == 1
    assert retry.lease_id != lease.lease_id
    assert resumed.last_checkpoint_sha256(retry.key) is None


def test_scheduler_checkpoint_hash_mismatch_fails_closed(tmp_path):
    path = tmp_path / "scheduler.json"
    durable = save_scheduler_checkpoint_atomic(path, _scheduler())
    wrong = hashlib.sha256(b"wrong").hexdigest()
    assert wrong != durable.sha256
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_scheduler_checkpoint(path, expected_sha256=wrong)


def test_truncated_scheduler_checkpoint_fails_closed(tmp_path):
    path = tmp_path / "scheduler.json"
    save_scheduler_checkpoint_atomic(path, _scheduler())
    path.write_bytes(b'{"schema":')
    with pytest.raises(ValueError, match="valid canonical JSON"):
        load_scheduler_checkpoint(path)


def test_durable_scheduler_checkpoint_never_authorizes_table_use(tmp_path):
    path = tmp_path / "scheduler.json"
    save_scheduler_checkpoint_atomic(path, _scheduler())
    text = path.read_text(encoding="utf-8")
    text = text.replace('"ready_for_tables":false', '"ready_for_tables":true', 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="cannot authorize table use"):
        load_scheduler_checkpoint(path)
