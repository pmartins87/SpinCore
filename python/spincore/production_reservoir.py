from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from spincore_nn.reservoir import AdvantageSample, StrategySample, UniformReservoir


@dataclass(frozen=True)
class RootSampleBatch:
    """Samples emitted by exactly one deterministic production root.

    Workers may finish roots in any wall-clock order.  `global_root` is the
    authoritative logical stream position; sample tuple order is the order in
    which that root's collector produced samples for the corresponding memory.
    """

    profile_id: str
    domain: str
    iteration: int
    global_root: int
    advantage: tuple[AdvantageSample, ...]
    strategy: tuple[StrategySample, ...]

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id is required")
        if self.domain not in ("TRUE_HEADS_UP", "THREE_HANDED"):
            raise ValueError("unsupported production domain")
        if self.iteration <= 0:
            raise ValueError("iteration must be positive")
        if self.global_root < 0:
            raise ValueError("global_root must be non-negative")


class CentralAlgorithmRReservoirs:
    """Deterministic central Algorithm-R ingestion for parallel root workers.

    Parallel workers are allowed to complete out of order, but the reservoirs
    are mutated only in ascending `global_root` order.  Therefore worker count,
    scheduling and completion timing cannot change the logical Algorithm-R
    stream when workers return the same per-root samples.

    This class deliberately owns separate Advantage and AveragePolicy
    reservoirs.  It never merges worker-local reservoirs, which would alter the
    production sampling semantics and could accidentally weight workers rather
    than individual samples.
    """

    SCHEMA = "SPINCORE_R8_CENTRAL_ALGORITHM_R_V1"

    def __init__(
        self,
        *,
        profile_id: str,
        domain: str,
        advantage_capacity: int,
        strategy_capacity: int,
        advantage_seed: int,
        strategy_seed: int,
        next_global_root: int = 0,
    ) -> None:
        if not profile_id:
            raise ValueError("profile_id is required")
        if domain not in ("TRUE_HEADS_UP", "THREE_HANDED"):
            raise ValueError("unsupported production domain")
        if next_global_root < 0:
            raise ValueError("next_global_root must be non-negative")
        self.profile_id = str(profile_id)
        self.domain = str(domain)
        self.next_global_root = int(next_global_root)
        self.advantage = UniformReservoir[AdvantageSample](int(advantage_capacity), int(advantage_seed))
        self.strategy = UniformReservoir[StrategySample](int(strategy_capacity), int(strategy_seed))
        self._pending: dict[int, RootSampleBatch] = {}
        self.committed_roots = 0

    def _validate_batch(self, batch: RootSampleBatch) -> None:
        if batch.profile_id != self.profile_id:
            raise ValueError("production profile mismatch")
        if batch.domain != self.domain:
            raise ValueError("production domain mismatch")
        if batch.global_root < self.next_global_root:
            raise ValueError("stale or duplicate global_root")
        if batch.global_root in self._pending:
            raise ValueError("duplicate pending global_root")

    def submit(self, batch: RootSampleBatch) -> int:
        """Buffer one worker result and commit every newly contiguous root.

        Returns the number of roots committed by this call.  A future root can
        therefore return zero until the missing predecessor arrives.
        """

        self._validate_batch(batch)
        self._pending[int(batch.global_root)] = batch
        committed = 0
        while self.next_global_root in self._pending:
            row = self._pending.pop(self.next_global_root)
            for sample in row.advantage:
                self.advantage.add(sample)
            for sample in row.strategy:
                self.strategy.add(sample)
            self.next_global_root += 1
            self.committed_roots += 1
            committed += 1
        return committed

    def submit_many(self, batches: Iterable[RootSampleBatch]) -> int:
        committed = 0
        for batch in batches:
            committed += self.submit(batch)
        return committed

    @property
    def pending_roots(self) -> tuple[int, ...]:
        return tuple(sorted(self._pending))

    def assert_drained(self) -> None:
        if self._pending:
            raise RuntimeError(f"missing production roots before {self.pending_roots!r}")

    def state_dict(self) -> dict:
        return {
            "schema": self.SCHEMA,
            "profile_id": self.profile_id,
            "domain": self.domain,
            "next_global_root": self.next_global_root,
            "committed_roots": self.committed_roots,
            "advantage": self.advantage.state_dict(),
            "strategy": self.strategy.state_dict(),
            "pending": [self._pending[k] for k in sorted(self._pending)],
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "CentralAlgorithmRReservoirs":
        if state.get("schema") != cls.SCHEMA:
            raise ValueError("wrong central Algorithm-R state schema")
        advantage_state = state["advantage"]
        strategy_state = state["strategy"]
        obj = cls(
            profile_id=str(state["profile_id"]),
            domain=str(state["domain"]),
            advantage_capacity=int(advantage_state["capacity"]),
            strategy_capacity=int(strategy_state["capacity"]),
            advantage_seed=0,
            strategy_seed=0,
            next_global_root=int(state["next_global_root"]),
        )
        obj.advantage = UniformReservoir.from_state_dict(advantage_state)
        obj.strategy = UniformReservoir.from_state_dict(strategy_state)
        obj.committed_roots = int(state["committed_roots"])
        pending = list(state.get("pending") or [])
        obj._pending = {int(row.global_root): row for row in pending}
        if len(obj._pending) != len(pending):
            raise ValueError("duplicate pending roots in checkpoint")
        for row in obj._pending.values():
            if row.profile_id != obj.profile_id or row.domain != obj.domain:
                raise ValueError("pending checkpoint batch identity mismatch")
            if row.global_root < obj.next_global_root:
                raise ValueError("pending checkpoint contains stale root")
        return obj
