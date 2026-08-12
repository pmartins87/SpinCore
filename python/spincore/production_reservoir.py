from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from spincore_nn.reservoir import AdvantageSample, StrategySample, UniformReservoir


@dataclass(frozen=True)
class RootSampleBatch:
    """Samples emitted by exactly one deterministic production root.

    Workers may finish roots in any wall-clock order. `global_root` is the
    authoritative logical position inside one exact
    `(profile, domain, algorithm_seed)` stream; sample tuple order is the order
    in which that root's collector produced samples for the corresponding
    memory.
    """

    profile_id: str
    domain: str
    algorithm_seed: int
    iteration: int
    global_root: int
    advantage: tuple[AdvantageSample, ...]
    strategy: tuple[StrategySample, ...]

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id is required")
        if self.domain not in ("TRUE_HEADS_UP", "THREE_HANDED"):
            raise ValueError("unsupported production domain")
        if self.algorithm_seed <= 0:
            raise ValueError("algorithm_seed must be positive")
        if self.iteration <= 0:
            raise ValueError("iteration must be positive")
        if self.global_root < 0:
            raise ValueError("global_root must be non-negative")


class CentralAlgorithmRReservoirs:
    """Deterministic Algorithm-R ingestion for exactly one training stream.

    Parallel workers are allowed to complete roots out of wall-clock order, but
    the reservoirs are mutated only in ascending `global_root` order. Worker
    count, scheduling and completion timing therefore cannot change logical
    Algorithm-R insertion order when workers return the same per-root samples.

    Critically, V2 is keyed by algorithm seed as well as profile/domain. A
    reservoir is *not* a cross-seed mixer: independent seeds own independent RNG
    histories, model/checkpoint lineages and reservoir states unless a future
    algorithm explicitly precommits a different merge semantic.
    """

    SCHEMA = "SPINCORE_R8_CENTRAL_ALGORITHM_R_V2"

    def __init__(
        self,
        *,
        profile_id: str,
        domain: str,
        algorithm_seed: int,
        roots_per_iteration: int,
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
        if algorithm_seed <= 0:
            raise ValueError("algorithm_seed must be positive")
        if roots_per_iteration <= 0:
            raise ValueError("roots_per_iteration must be positive")
        if next_global_root < 0:
            raise ValueError("next_global_root must be non-negative")
        self.profile_id = str(profile_id)
        self.domain = str(domain)
        self.algorithm_seed = int(algorithm_seed)
        self.roots_per_iteration = int(roots_per_iteration)
        self.next_global_root = int(next_global_root)
        self.advantage = UniformReservoir[AdvantageSample](int(advantage_capacity), int(advantage_seed))
        self.strategy = UniformReservoir[StrategySample](int(strategy_capacity), int(strategy_seed))
        self._pending: dict[int, RootSampleBatch] = {}
        self.committed_roots = 0

    def _expected_iteration(self, global_root: int) -> int:
        return int(global_root) // self.roots_per_iteration + 1

    def _validate_batch_semantics(self, batch: RootSampleBatch) -> None:
        if batch.profile_id != self.profile_id:
            raise ValueError("production profile mismatch")
        if batch.domain != self.domain:
            raise ValueError("production domain mismatch")
        if batch.algorithm_seed != self.algorithm_seed:
            raise ValueError("production algorithm-seed mismatch")
        expected_iteration = self._expected_iteration(batch.global_root)
        if batch.iteration != expected_iteration:
            raise ValueError(
                f"production root iteration mismatch: global_root={batch.global_root} "
                f"requires iteration={expected_iteration}, got {batch.iteration}"
            )
        for sample in batch.advantage:
            if int(sample.iteration) != int(batch.iteration):
                raise ValueError("advantage sample iteration differs from root batch iteration")
        for sample in batch.strategy:
            if int(sample.iteration) != int(batch.iteration):
                raise ValueError("strategy sample iteration differs from root batch iteration")

    def _validate_batch(self, batch: RootSampleBatch) -> None:
        self._validate_batch_semantics(batch)
        if batch.global_root < self.next_global_root:
            raise ValueError("stale or duplicate global_root")
        if batch.global_root in self._pending:
            raise ValueError("duplicate pending global_root")

    def submit(self, batch: RootSampleBatch) -> int:
        """Buffer one worker result and commit every newly contiguous root.

        Returns the number of roots committed by this call. A future root can
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
            "algorithm_seed": self.algorithm_seed,
            "roots_per_iteration": self.roots_per_iteration,
            "next_global_root": self.next_global_root,
            "committed_roots": self.committed_roots,
            "advantage": self.advantage.state_dict(),
            "strategy": self.strategy.state_dict(),
            "pending": [self._pending[k] for k in sorted(self._pending)],
            "ready_for_tables": False,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "CentralAlgorithmRReservoirs":
        if state.get("schema") != cls.SCHEMA:
            raise ValueError("wrong central Algorithm-R state schema")
        if state.get("ready_for_tables", False) is not False:
            raise ValueError("R8 central reservoir checkpoint cannot authorize table use")
        advantage_state = state["advantage"]
        strategy_state = state["strategy"]
        obj = cls(
            profile_id=str(state["profile_id"]),
            domain=str(state["domain"]),
            algorithm_seed=int(state["algorithm_seed"]),
            roots_per_iteration=int(state["roots_per_iteration"]),
            advantage_capacity=int(advantage_state["capacity"]),
            strategy_capacity=int(strategy_state["capacity"]),
            advantage_seed=0,
            strategy_seed=0,
            next_global_root=int(state["next_global_root"]),
        )
        obj.advantage = UniformReservoir.from_state_dict(advantage_state)
        obj.strategy = UniformReservoir.from_state_dict(strategy_state)
        obj.committed_roots = int(state["committed_roots"])
        if obj.committed_roots < 0 or obj.committed_roots > obj.next_global_root:
            raise ValueError("invalid committed_roots in central reservoir checkpoint")
        pending = list(state.get("pending") or [])
        obj._pending = {int(row.global_root): row for row in pending}
        if len(obj._pending) != len(pending):
            raise ValueError("duplicate pending roots in checkpoint")
        for row in obj._pending.values():
            obj._validate_batch_semantics(row)
            if row.global_root < obj.next_global_root:
                raise ValueError("pending checkpoint contains stale root")
        return obj
