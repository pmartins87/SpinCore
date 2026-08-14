from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .production_stream_scheduler import SUPPORTED_DOMAINS


SCHEMA = "SPINCORE_R8_PRODUCTION_TRAINABILITY_V1"
HARD_CAP_DAYS = 90.0
RESERVE_MULTIPLIER = 1.20
MIN_TIMING_SAMPLES_PER_STREAM = 3
FULL_ITERATION_SCOPE = "FULL_DURABLE_ITERATION_CHECKPOINT_TO_CHECKPOINT_V1"
SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class PlannedTrainingStream:
    stream_id: str
    profile_id: str
    domain: str
    algorithm_seed: int
    total_iterations: int

    def __post_init__(self) -> None:
        if not self.stream_id.strip():
            raise ValueError("trainability stream_id is required")
        if not self.profile_id.strip():
            raise ValueError("trainability profile_id is required")
        if self.domain not in SUPPORTED_DOMAINS:
            raise ValueError("unsupported trainability domain")
        if self.algorithm_seed <= 0:
            raise ValueError("trainability algorithm_seed must be positive")
        if self.total_iterations <= 0:
            raise ValueError("trainability total_iterations must be positive")


@dataclass(frozen=True)
class MeasuredTrainingStream:
    stream_id: str
    selected_concurrency: int
    iteration_seconds_samples: tuple[float, ...]
    semantic_exact: bool = True
    error: str | None = None
    measurement_scope: str = FULL_ITERATION_SCOPE

    def __post_init__(self) -> None:
        if not self.stream_id.strip():
            raise ValueError("measured trainability stream_id is required")
        if self.selected_concurrency <= 0:
            raise ValueError("selected_concurrency must be positive")
        if self.measurement_scope != FULL_ITERATION_SCOPE:
            raise ValueError("trainability timing must cover one full durable production iteration")
        if len(self.iteration_seconds_samples) < MIN_TIMING_SAMPLES_PER_STREAM:
            raise ValueError(
                f"at least {MIN_TIMING_SAMPLES_PER_STREAM} complete-iteration timing samples are required per stream"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in self.iteration_seconds_samples):
            raise ValueError("iteration timing samples must be finite and positive")

    @property
    def conservative_seconds_per_iteration(self) -> float:
        """Use the slowest valid complete-iteration observation, not the mean."""
        return float(max(self.iteration_seconds_samples))


def _lpt_stream_pinned_upper_bound_seconds(
    durations: Sequence[tuple[str, float]], *, concurrency: int
) -> tuple[float, tuple[float, ...]]:
    """Return a conservative makespan upper bound for independent serial streams.

    Each production stream is serial by the persistent-RNG contract.  Treating an
    entire stream as pinned to one worker is at least as restrictive as the actual
    scheduler, which may move a stream between workers at iteration boundaries.
    LPT therefore gives a simple deterministic *upper bound* without assuming
    unsafe intra-stream parallelism or optimistic perfect packing.
    """
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if not durations:
        raise ValueError("at least one projected stream duration is required")
    worker_count = min(int(concurrency), len(durations))
    loads = [0.0 for _ in range(worker_count)]
    for _, duration in sorted(durations, key=lambda row: (-row[1], row[0])):
        worker = min(range(worker_count), key=lambda index: (loads[index], index))
        loads[worker] += float(duration)
    return max(loads), tuple(loads)


def project_production_trainability(
    *,
    plans: Sequence[PlannedTrainingStream],
    measurements: Sequence[MeasuredTrainingStream],
    selected_concurrency: int,
    hard_cap_days: float = HARD_CAP_DAYS,
    reserve_multiplier: float = RESERVE_MULTIPLIER,
) -> dict:
    """Project complete R8 baseline-training wall time and enforce the hard cap.

    The caller must supply the *complete frozen* official workload: every selected
    production profile, both TRUE_HEADS_UP and THREE_HANDED domains, and every
    required algorithm-seed stream.  Timings must be measured on the intended
    production host at the already-selected semantically exact R8.2 concurrency.

    A timing sample is one full durable iteration from one accepted checkpoint to
    the next, so traversal, reservoir work, neural fitting and checkpoint overhead
    are included.  For every stream we use its slowest repeated observation.

    This function never reduces iterations, seeds, ensemble size, roots, optimizer
    steps or strategic coverage to make the result fit the deadline.
    """
    if selected_concurrency <= 0:
        raise ValueError("selected_concurrency must be positive")
    if not math.isfinite(hard_cap_days) or hard_cap_days <= 0.0:
        raise ValueError("hard_cap_days must be finite and positive")
    if not math.isfinite(reserve_multiplier) or reserve_multiplier < 1.0:
        raise ValueError("reserve_multiplier must be finite and >= 1")
    if not plans:
        raise ValueError("complete production training plan cannot be empty")
    if not measurements:
        raise ValueError("physical trainability measurements cannot be empty")

    plan_by_id: dict[str, PlannedTrainingStream] = {}
    for plan in plans:
        if plan.stream_id in plan_by_id:
            raise ValueError(f"duplicate planned trainability stream: {plan.stream_id}")
        plan_by_id[plan.stream_id] = plan

    measured_by_id: dict[str, MeasuredTrainingStream] = {}
    for row in measurements:
        if row.stream_id in measured_by_id:
            raise ValueError(f"duplicate measured trainability stream: {row.stream_id}")
        if row.selected_concurrency != int(selected_concurrency):
            raise ValueError("measurement concurrency differs from selected R8.2 concurrency")
        measured_by_id[row.stream_id] = row

    if set(plan_by_id) != set(measured_by_id):
        missing = sorted(set(plan_by_id) - set(measured_by_id))
        extra = sorted(set(measured_by_id) - set(plan_by_id))
        raise ValueError(f"trainability stream matrix mismatch: missing={missing} extra={extra}")

    domains = {plan.domain for plan in plans}
    if domains != set(SUPPORTED_DOMAINS):
        raise ValueError(
            "complete top-performance baseline projection must include TRUE_HEADS_UP and THREE_HANDED"
        )

    semantically_eligible = all(
        row.semantic_exact and row.error is None for row in measurements
    )

    rows: list[dict] = []
    durations: list[tuple[str, float]] = []
    for stream_id in sorted(plan_by_id):
        plan = plan_by_id[stream_id]
        measured = measured_by_id[stream_id]
        seconds_per_iteration = measured.conservative_seconds_per_iteration
        projected_serial_seconds = seconds_per_iteration * float(plan.total_iterations)
        durations.append((stream_id, projected_serial_seconds))
        rows.append(
            {
                "stream_id": stream_id,
                "profile_id": plan.profile_id,
                "domain": plan.domain,
                "algorithm_seed": int(plan.algorithm_seed),
                "total_iterations": int(plan.total_iterations),
                "timing_sample_count": len(measured.iteration_seconds_samples),
                "iteration_seconds_samples": [float(x) for x in measured.iteration_seconds_samples],
                "conservative_seconds_per_iteration": seconds_per_iteration,
                "projected_serial_seconds": projected_serial_seconds,
                "semantic_exact": bool(measured.semantic_exact),
                "error": measured.error,
            }
        )

    nominal_upper_seconds, worker_loads = _lpt_stream_pinned_upper_bound_seconds(
        durations, concurrency=int(selected_concurrency)
    )
    projected_with_reserve_seconds = nominal_upper_seconds * float(reserve_multiplier)
    hard_cap_seconds = float(hard_cap_days) * SECONDS_PER_DAY
    nominal_days = nominal_upper_seconds / SECONDS_PER_DAY
    projected_days = projected_with_reserve_seconds / SECONDS_PER_DAY
    trainability_pass = bool(
        semantically_eligible and projected_with_reserve_seconds <= hard_cap_seconds
    )

    return {
        "schema": SCHEMA,
        "measurement_scope": FULL_ITERATION_SCOPE,
        "selected_concurrency": int(selected_concurrency),
        "hard_cap_days": float(hard_cap_days),
        "hard_cap_seconds": hard_cap_seconds,
        "reserve_multiplier": float(reserve_multiplier),
        "reserve_fraction": float(reserve_multiplier - 1.0),
        "implied_nominal_budget_days": float(hard_cap_days) / float(reserve_multiplier),
        "stream_count": len(rows),
        "domains": sorted(domains),
        "timing_rule": "MAX_REPEATED_FULL_DURABLE_ITERATION_SECONDS_PER_STREAM",
        "scheduling_projection": "LPT_STREAM_PINNED_CONSERVATIVE_UPPER_BOUND",
        "worker_projected_load_seconds": [float(x) for x in worker_loads],
        "nominal_projected_upper_bound_seconds": nominal_upper_seconds,
        "nominal_projected_upper_bound_days": nominal_days,
        "projected_with_reserve_seconds": projected_with_reserve_seconds,
        "projected_with_reserve_days": projected_days,
        "semantic_exact": bool(semantically_eligible),
        "streams": rows,
        "trainability_pass": trainability_pass,
        "workload_reduction_to_meet_budget_allowed": False,
        "intra_stream_parallelism_allowed": False,
        "ready_for_official_training": False,
        "ready_for_tables": False,
    }
