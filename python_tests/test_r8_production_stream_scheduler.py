from __future__ import annotations

import copy

import pytest

from spincore.production_stream_scheduler import (
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


def test_parallel_leases_are_always_distinct_streams():
    s = _scheduler()
    leases = s.lease(99)
    assert len(leases) == 4
    assert len({row.key for row in leases}) == len(leases)
    assert len(set(s.active_stream_ids)) == 4
    # No second lease can be issued while every independent stream is active.
    assert s.lease(99) == ()


def test_completion_advances_only_that_stream_and_never_skips_iteration():
    s = _scheduler()
    leases = s.lease(4)
    first = leases[0]
    s.complete(first)
    next_rows = s.lease(4)
    assert len(next_rows) == 1
    assert next_rows[0].key == first.key
    assert next_rows[0].iteration == 2
    with pytest.raises(ValueError, match="stale or out of order"):
        s.complete(first)


def test_failure_retries_same_iteration_without_silent_progress():
    s = _scheduler()
    first = s.lease(1)[0]
    assert first.iteration == 1
    s.fail(first)
    retry = s.lease(1)[0]
    assert retry.key == first.key
    assert retry.iteration == 1
    assert retry.lease_id != first.lease_id
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


def test_checkpoint_round_trip_after_completed_iterations_is_deterministic():
    baseline = _scheduler()
    resumed = _scheduler()

    first = baseline.lease(4)
    for row in first:
        baseline.complete(row)
    second = baseline.lease(4)

    first2 = resumed.lease(4)
    for row in first2:
        resumed.complete(row)
    resumed = IndependentStreamScheduler.from_state_dict(copy.deepcopy(resumed.state_dict()))
    second2 = resumed.lease(4)

    assert [(x.key, x.iteration) for x in second2] == [(x.key, x.iteration) for x in second]


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
