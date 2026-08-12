from __future__ import annotations

import copy

import pytest

from spincore.production_reservoir import CentralAlgorithmRReservoirs, RootSampleBatch
from spincore_nn.reservoir import AdvantageSample, StrategySample


def _adv(root: int, local: int) -> AdvantageSample:
    return AdvantageSample(
        observation=f"a:{root}:{local}".encode(),
        legal=(1, 1, 0, 0, 0, 0),
        target=(float(root), float(local), 0.0, 0.0, 0.0, 0.0),
        weight=1.0,
        iteration=1 + root // 5,
    )


def _pol(root: int, local: int) -> StrategySample:
    return StrategySample(
        observation=f"p:{root}:{local}".encode(),
        legal=(1, 1, 0, 0, 0, 0),
        target=(0.25 + 0.01 * local, 0.75 - 0.01 * local, 0.0, 0.0, 0.0, 0.0),
        weight=1.0,
        iteration=1 + root // 5,
    )


def _batch(root: int, *, profile: str = "gg:test", domain: str = "TRUE_HEADS_UP") -> RootSampleBatch:
    return RootSampleBatch(
        profile_id=profile,
        domain=domain,
        iteration=1 + root // 5,
        global_root=root,
        advantage=tuple(_adv(root, i) for i in range((root % 3) + 1)),
        strategy=tuple(_pol(root, i) for i in range((root % 2) + 1)),
    )


def _coordinator() -> CentralAlgorithmRReservoirs:
    return CentralAlgorithmRReservoirs(
        profile_id="gg:test",
        domain="TRUE_HEADS_UP",
        advantage_capacity=9,
        strategy_capacity=7,
        advantage_seed=112233,
        strategy_seed=445566,
    )


def _observable_state(obj: CentralAlgorithmRReservoirs) -> dict:
    state = copy.deepcopy(obj.state_dict())
    assert state["pending"] == []
    return state


def test_worker_completion_order_cannot_change_algorithm_r_state():
    ordered = _coordinator()
    scrambled = _coordinator()
    batches = [_batch(root) for root in range(30)]

    ordered.submit_many(batches)
    order = [7, 1, 12, 0, 3, 2, 6, 5, 4, 11, 10, 9, 8] + list(range(13, 30))
    scrambled.submit_many([batches[i] for i in order])

    ordered.assert_drained()
    scrambled.assert_drained()
    assert _observable_state(scrambled) == _observable_state(ordered)
    assert ordered.advantage.seen == sum(len(row.advantage) for row in batches)
    assert ordered.strategy.seen == sum(len(row.strategy) for row in batches)


def test_future_root_waits_for_gap_then_flushes_contiguously():
    c = _coordinator()
    assert c.submit(_batch(2)) == 0
    assert c.pending_roots == (2,)
    assert c.submit(_batch(0)) == 1
    assert c.next_global_root == 1
    assert c.submit(_batch(1)) == 2
    assert c.next_global_root == 3
    assert c.pending_roots == ()


def test_checkpoint_with_pending_roots_round_trips_exactly():
    baseline = _coordinator()
    resumed = _coordinator()

    baseline.submit_many([_batch(i) for i in range(20)])

    resumed.submit(_batch(3))
    resumed.submit(_batch(0))
    resumed.submit(_batch(2))
    state = copy.deepcopy(resumed.state_dict())
    resumed = CentralAlgorithmRReservoirs.from_state_dict(state)
    resumed.submit(_batch(1))
    resumed.submit_many([_batch(i) for i in range(4, 20)])

    baseline.assert_drained()
    resumed.assert_drained()
    assert _observable_state(resumed) == _observable_state(baseline)


def test_profile_domain_and_duplicate_roots_fail_closed():
    c = _coordinator()
    with pytest.raises(ValueError, match="profile"):
        c.submit(_batch(0, profile="wrong"))
    with pytest.raises(ValueError, match="domain"):
        c.submit(_batch(0, domain="THREE_HANDED"))

    c.submit(_batch(1))
    with pytest.raises(ValueError, match="duplicate pending"):
        c.submit(_batch(1))
    c.submit(_batch(0))
    with pytest.raises(ValueError, match="stale or duplicate"):
        c.submit(_batch(0))


def test_drained_guard_detects_missing_root():
    c = _coordinator()
    c.submit(_batch(4))
    with pytest.raises(RuntimeError, match="missing production roots"):
        c.assert_drained()
