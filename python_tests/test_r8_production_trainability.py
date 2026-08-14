from __future__ import annotations

import math

import pytest

from spincore.production_trainability import (
    HARD_CAP_DAYS,
    RESERVE_MULTIPLIER,
    MATURE_COST_BASIS,
    NON_ITERATION_SCOPE,
    MeasuredNonIterationTrainingWork,
    MeasuredTrainingStream,
    PlannedTrainingStream,
    project_production_trainability,
)


def _plan(stream_id: str, domain: str, seed: int, iterations: int = 10):
    return PlannedTrainingStream(stream_id, "profile-a", domain, seed, iterations)


def _measurement(stream_id: str, seconds: tuple[float, ...], concurrency: int = 2, **kwargs):
    return MeasuredTrainingStream(
        stream_id=stream_id,
        selected_concurrency=concurrency,
        iteration_seconds_samples=seconds,
        cost_basis=MATURE_COST_BASIS,
        **kwargs,
    )


def _overhead(seconds=(0.0, 0.0, 0.0), concurrency: int = 2, **kwargs):
    return MeasuredNonIterationTrainingWork(
        selected_concurrency=concurrency,
        seconds_samples=seconds,
        **kwargs,
    )


def _two_domain_fixture(seconds=(90.0, 100.0, 95.0), iterations=10):
    plans = [
        _plan("hu-a", "TRUE_HEADS_UP", 11, iterations),
        _plan("hu-b", "TRUE_HEADS_UP", 22, iterations),
        _plan("3h-a", "THREE_HANDED", 11, iterations),
        _plan("3h-b", "THREE_HANDED", 22, iterations),
    ]
    measurements = [_measurement(plan.stream_id, seconds) for plan in plans]
    return plans, measurements


def _project(plans, measurements, *, concurrency=2, overhead=None, **kwargs):
    return project_production_trainability(
        plans=plans,
        measurements=measurements,
        non_iteration_measurement=overhead or _overhead(concurrency=concurrency),
        selected_concurrency=concurrency,
        **kwargs,
    )


def test_small_complete_workload_passes_with_20_percent_reserve() -> None:
    plans, measurements = _two_domain_fixture()
    out = _project(plans, measurements)
    # Worst mature full iteration is 100 s. Four 1000-s streams, two workers:
    # LPT upper bound = 2000 s; no external work; with 20% reserve = 2400 s.
    assert out["stream_nominal_projected_upper_bound_seconds"] == 2000.0
    assert out["non_iteration_conservative_seconds"] == 0.0
    assert out["nominal_projected_upper_bound_seconds"] == 2000.0
    assert out["projected_with_reserve_seconds"] == 2400.0
    assert out["hard_cap_days"] == 90.0
    assert out["reserve_multiplier"] == 1.20
    assert out["implied_nominal_budget_days"] == 75.0
    assert out["trainability_pass"] is True
    assert out["workload_reduction_to_meet_budget_allowed"] is False
    assert out["intra_stream_parallelism_allowed"] is False
    assert out["ready_for_official_training"] is False
    assert out["ready_for_tables"] is False


def test_slowest_repeated_mature_full_iteration_is_used_not_mean() -> None:
    plans, measurements = _two_domain_fixture(seconds=(10.0, 30.0, 20.0), iterations=1)
    out = _project(plans, measurements)
    assert all(row["cost_basis"] == MATURE_COST_BASIS for row in out["streams"])
    assert all(row["conservative_seconds_per_iteration"] == 30.0 for row in out["streams"])
    assert out["nominal_projected_upper_bound_seconds"] == 60.0


def test_non_iteration_training_and_freeze_work_is_added_after_stream_makespan() -> None:
    plans, measurements = _two_domain_fixture(seconds=(10.0, 10.0, 10.0), iterations=1)
    out = _project(
        plans,
        measurements,
        overhead=_overhead((100.0, 200.0, 150.0)),
    )
    # Four 10-s streams on two workers => 20 s stream makespan; max external
    # training/freeze timing is 200 s => 220 s nominal total.
    assert out["stream_nominal_projected_upper_bound_seconds"] == 20.0
    assert out["non_iteration_conservative_seconds"] == 200.0
    assert out["nominal_projected_upper_bound_seconds"] == 220.0
    assert out["projected_with_reserve_seconds"] == 264.0


def test_exact_75_nominal_days_reaches_90_day_hard_cap_with_reserve() -> None:
    seconds = 75.0 * 86400.0
    plans = [
        _plan("hu", "TRUE_HEADS_UP", 11, 1),
        _plan("3h", "THREE_HANDED", 11, 1),
    ]
    measurements = [
        _measurement("hu", (seconds, seconds, seconds)),
        _measurement("3h", (seconds, seconds, seconds)),
    ]
    out = _project(plans, measurements)
    assert math.isclose(out["nominal_projected_upper_bound_days"], 75.0)
    assert math.isclose(out["projected_with_reserve_days"], HARD_CAP_DAYS)
    assert out["trainability_pass"] is True


def test_hidden_final_work_can_turn_nominal_75_day_stream_plan_into_failure() -> None:
    seconds = 75.0 * 86400.0
    plans = [
        _plan("hu", "TRUE_HEADS_UP", 11, 1),
        _plan("3h", "THREE_HANDED", 11, 1),
    ]
    measurements = [
        _measurement("hu", (seconds, seconds, seconds)),
        _measurement("3h", (seconds, seconds, seconds)),
    ]
    out = _project(
        plans,
        measurements,
        overhead=_overhead((3600.0, 3600.0, 3600.0)),
    )
    assert out["nominal_projected_upper_bound_days"] > 75.0
    assert out["projected_with_reserve_days"] > 90.0
    assert out["trainability_pass"] is False


def test_projection_over_90_days_fails_closed_without_reducing_workload() -> None:
    seconds = 76.0 * 86400.0
    plans = [
        _plan("hu", "TRUE_HEADS_UP", 11, 1),
        _plan("3h", "THREE_HANDED", 11, 1),
    ]
    measurements = [
        _measurement("hu", (seconds, seconds, seconds)),
        _measurement("3h", (seconds, seconds, seconds)),
    ]
    out = _project(plans, measurements)
    assert out["projected_with_reserve_days"] > 90.0
    assert out["trainability_pass"] is False
    assert out["workload_reduction_to_meet_budget_allowed"] is False


def test_semantic_mismatch_fails_even_when_runtime_is_fast() -> None:
    plans, measurements = _two_domain_fixture(seconds=(1.0, 1.0, 1.0))
    measurements[0] = _measurement(
        measurements[0].stream_id,
        (1.0, 1.0, 1.0),
        semantic_exact=False,
    )
    out = _project(plans, measurements)
    assert out["semantic_exact"] is False
    assert out["trainability_pass"] is False


def test_non_iteration_semantic_mismatch_also_fails() -> None:
    plans, measurements = _two_domain_fixture(seconds=(1.0, 1.0, 1.0))
    out = _project(
        plans,
        measurements,
        overhead=_overhead((1.0, 1.0, 1.0), semantic_exact=False),
    )
    assert out["semantic_exact"] is False
    assert out["trainability_pass"] is False


def test_measurement_must_cover_full_checkpoint_to_checkpoint_iteration() -> None:
    with pytest.raises(ValueError, match="full durable production iteration"):
        MeasuredTrainingStream(
            stream_id="hu",
            selected_concurrency=1,
            iteration_seconds_samples=(1.0, 1.0, 1.0),
            cost_basis=MATURE_COST_BASIS,
            measurement_scope="TRAVERSAL_ONLY",
        )


def test_measurement_must_be_mature_or_worst_case_certified() -> None:
    with pytest.raises(ValueError, match="mature or otherwise demonstrated worst-case"):
        MeasuredTrainingStream(
            stream_id="hu",
            selected_concurrency=1,
            iteration_seconds_samples=(1.0, 1.0, 1.0),
            cost_basis="EARLY_ITERATION_ONLY",
        )


def test_at_least_three_repeated_timings_are_required() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        _measurement("hu", (1.0, 1.0))
    with pytest.raises(ValueError, match="at least 3"):
        _overhead((1.0, 1.0))


def test_complete_stream_matrix_and_both_domains_are_required() -> None:
    plans, measurements = _two_domain_fixture()
    with pytest.raises(ValueError, match="stream matrix mismatch"):
        _project(plans, measurements[:-1])

    hu_only_plans = [_plan("hu-a", "TRUE_HEADS_UP", 11)]
    hu_only_measurements = [_measurement("hu-a", (1.0, 1.0, 1.0), concurrency=1)]
    with pytest.raises(ValueError, match="TRUE_HEADS_UP and THREE_HANDED"):
        _project(hu_only_plans, hu_only_measurements, concurrency=1)


def test_measurements_and_external_work_must_use_selected_concurrency() -> None:
    plans, measurements = _two_domain_fixture()
    with pytest.raises(ValueError, match="measurement concurrency differs from selected"):
        _project(plans, measurements, concurrency=4, overhead=_overhead(concurrency=4))
    with pytest.raises(ValueError, match="non-iteration measurement concurrency differs"):
        project_production_trainability(
            plans=plans,
            measurements=measurements,
            non_iteration_measurement=_overhead(concurrency=1),
            selected_concurrency=2,
        )


def test_non_iteration_scope_is_frozen() -> None:
    with pytest.raises(ValueError, match="scope is incomplete"):
        MeasuredNonIterationTrainingWork(
            selected_concurrency=1,
            seconds_samples=(0.0, 0.0, 0.0),
            measurement_scope="FINAL_FIT_ONLY",
        )
    assert NON_ITERATION_SCOPE == "ALL_FROZEN_NON_ITERATION_TRAINING_AND_FINAL_FREEZE_WORK_V1"


def test_invalid_budget_parameters_fail_closed() -> None:
    plans, measurements = _two_domain_fixture()
    with pytest.raises(ValueError):
        _project(plans, measurements, hard_cap_days=0.0)
    with pytest.raises(ValueError):
        _project(plans, measurements, reserve_multiplier=0.99)
    assert RESERVE_MULTIPLIER == 1.20
