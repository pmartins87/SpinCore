from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable


SUPPORTED_DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")


@dataclass(frozen=True, order=True)
class ProductionStreamKey:
    """Identity of one serial RNG/training stream.

    A stream owns one persistent live bundle RNG and one model/checkpoint lineage.
    Work inside this identity must never overlap in wall-clock time. Different
    keys are independent by contract and may execute concurrently.
    """

    profile_id: str
    domain: str
    algorithm_seed: int

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id is required")
        if self.domain not in SUPPORTED_DOMAINS:
            raise ValueError("unsupported strategy domain")
        if self.algorithm_seed <= 0:
            raise ValueError("algorithm_seed must be positive")

    @property
    def stream_id(self) -> str:
        raw = f"{self.profile_id}|{self.domain}|{self.algorithm_seed}".encode()
        return "spinstream-v1:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ProductionStreamPlan:
    key: ProductionStreamKey
    total_iterations: int

    def __post_init__(self) -> None:
        if self.total_iterations <= 0:
            raise ValueError("total_iterations must be positive")


@dataclass(frozen=True)
class IterationLease:
    """Exclusive authorization to run exactly one whole stream iteration."""

    key: ProductionStreamKey
    iteration: int
    lease_id: str


@dataclass
class _Progress:
    total_iterations: int
    next_iteration: int = 1
    active_lease_id: str | None = None
    failed_attempts_for_next_iteration: int = 0


class IndependentStreamScheduler:
    """Deterministic scheduler that parallelizes only independent streams.

    The unit of concurrency is a *whole training iteration* for a stream, not a
    root. This is intentional: the frozen R7/R8 semantic contract uses one
    persistent live RNG for traversal, reservoir replacement and minibatch
    sampling in execution order. Leasing no more than one iteration at a time
    for a key guarantees this scheduler cannot introduce intra-stream root/RNG
    parallelism.

    The scheduler owns no application RNG and never derives per-root seeds.
    """

    SCHEMA = "SPINCORE_R8_INDEPENDENT_STREAM_SCHEDULER_V1"

    def __init__(self, plans: Iterable[ProductionStreamPlan]) -> None:
        rows = sorted(plans, key=lambda row: row.key)
        if not rows:
            raise ValueError("at least one production stream plan is required")
        if len({row.key for row in rows}) != len(rows):
            raise ValueError("duplicate production stream key")
        self._progress: dict[ProductionStreamKey, _Progress] = {
            row.key: _Progress(total_iterations=int(row.total_iterations)) for row in rows
        }
        self._lease_counter = 0

    def _new_lease_id(self, key: ProductionStreamKey, iteration: int) -> str:
        self._lease_counter += 1
        raw = f"{key.stream_id}|{iteration}|{self._lease_counter}".encode()
        return "spinlease-v1:" + hashlib.sha256(raw).hexdigest()

    def lease(self, max_workers: int) -> tuple[IterationLease, ...]:
        """Lease work for distinct streams only, in deterministic key order."""
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        out: list[IterationLease] = []
        for key in sorted(self._progress):
            if len(out) >= max_workers:
                break
            state = self._progress[key]
            if state.active_lease_id is not None:
                continue
            if state.next_iteration > state.total_iterations:
                continue
            lease_id = self._new_lease_id(key, state.next_iteration)
            state.active_lease_id = lease_id
            out.append(IterationLease(key=key, iteration=state.next_iteration, lease_id=lease_id))
        return tuple(out)

    def complete(self, lease: IterationLease) -> None:
        state = self._validate_active(lease)
        state.active_lease_id = None
        state.next_iteration += 1
        state.failed_attempts_for_next_iteration = 0

    def fail(self, lease: IterationLease) -> None:
        """Release a failed lease without advancing its serial RNG lineage."""
        state = self._validate_active(lease)
        state.active_lease_id = None
        state.failed_attempts_for_next_iteration += 1

    def _validate_active(self, lease: IterationLease) -> _Progress:
        state = self._progress.get(lease.key)
        if state is None:
            raise ValueError("lease belongs to unknown stream")
        if lease.iteration != state.next_iteration:
            raise ValueError("lease iteration is stale or out of order")
        if state.active_lease_id != lease.lease_id:
            raise ValueError("lease is not the active exclusive lease")
        return state

    @property
    def complete_all(self) -> bool:
        return all(
            state.next_iteration > state.total_iterations and state.active_lease_id is None
            for state in self._progress.values()
        )

    @property
    def active_stream_ids(self) -> tuple[str, ...]:
        return tuple(
            key.stream_id
            for key in sorted(self._progress)
            if self._progress[key].active_lease_id is not None
        )

    def state_dict(self) -> dict:
        return {
            "schema": self.SCHEMA,
            "lease_counter": self._lease_counter,
            "streams": [
                {
                    "profile_id": key.profile_id,
                    "domain": key.domain,
                    "algorithm_seed": key.algorithm_seed,
                    "stream_id": key.stream_id,
                    "total_iterations": state.total_iterations,
                    "next_iteration": state.next_iteration,
                    "active_lease_id": state.active_lease_id,
                    "failed_attempts_for_next_iteration": state.failed_attempts_for_next_iteration,
                }
                for key, state in sorted(self._progress.items())
            ],
            "ready_for_tables": False,
        }

    @classmethod
    def from_state_dict(cls, payload: dict, *, clear_active_leases: bool = True) -> "IndependentStreamScheduler":
        if payload.get("schema") != cls.SCHEMA:
            raise ValueError("wrong production stream scheduler schema")
        if payload.get("ready_for_tables", False) is not False:
            raise ValueError("R8 scheduler checkpoint cannot authorize table use")
        rows = list(payload.get("streams") or [])
        plans: list[ProductionStreamPlan] = []
        keys: list[ProductionStreamKey] = []
        for row in rows:
            key = ProductionStreamKey(
                profile_id=str(row["profile_id"]),
                domain=str(row["domain"]),
                algorithm_seed=int(row["algorithm_seed"]),
            )
            if row.get("stream_id") != key.stream_id:
                raise ValueError("production stream identity hash mismatch")
            keys.append(key)
            plans.append(ProductionStreamPlan(key, int(row["total_iterations"])))
        obj = cls(plans)
        obj._lease_counter = int(payload.get("lease_counter", 0))
        for key, row in zip(keys, rows):
            state = obj._progress[key]
            state.next_iteration = int(row["next_iteration"])
            state.failed_attempts_for_next_iteration = int(row.get("failed_attempts_for_next_iteration", 0))
            active = row.get("active_lease_id")
            # A process crash makes an in-memory lease unverifiable. By default
            # it is cleared so the exact same iteration can be retried from its
            # durable stream checkpoint; it is never silently advanced.
            state.active_lease_id = None if clear_active_leases else active
            if state.next_iteration < 1 or state.next_iteration > state.total_iterations + 1:
                raise ValueError("invalid next_iteration in scheduler checkpoint")
        return obj
