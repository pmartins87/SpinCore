from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable


SUPPORTED_DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


@dataclass(frozen=True)
class DurableIterationReceipt:
    """Proof that a leased iteration has a durable checkpoint before advance.

    The scheduler does not write model checkpoints itself.  It accepts progress
    only after the orchestrator supplies a receipt for already-persisted bytes.
    The parent digest makes the checkpoint lineage explicit and prevents a
    resumed scheduler from silently splicing iteration N+1 onto the wrong
    iteration-N model state.
    """

    key: ProductionStreamKey
    iteration: int
    lease_id: str
    checkpoint_locator: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    parent_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.iteration <= 0:
            raise ValueError("receipt iteration must be positive")
        if not self.lease_id:
            raise ValueError("receipt lease_id is required")
        if not self.checkpoint_locator.strip():
            raise ValueError("durable checkpoint locator is required")
        if not _SHA256_RE.fullmatch(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be 64 lowercase hex characters")
        if self.checkpoint_size_bytes <= 0:
            raise ValueError("durable checkpoint must contain bytes")
        if self.parent_checkpoint_sha256 is not None:
            if not _SHA256_RE.fullmatch(self.parent_checkpoint_sha256):
                raise ValueError("parent checkpoint SHA256 must be 64 lowercase hex characters")
            if self.parent_checkpoint_sha256 == self.checkpoint_sha256:
                raise ValueError("checkpoint cannot name itself as parent")

    @classmethod
    def from_file(
        cls,
        lease: IterationLease,
        path: str | Path,
        *,
        parent_checkpoint_sha256: str | None,
        locator: str | None = None,
    ) -> "DurableIterationReceipt":
        """Hash bytes only after a checkpoint file exists locally."""
        p = Path(path)
        raw = p.read_bytes()
        if not raw:
            raise ValueError("durable checkpoint file is empty")
        return cls(
            key=lease.key,
            iteration=lease.iteration,
            lease_id=lease.lease_id,
            checkpoint_locator=str(locator or p),
            checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
            checkpoint_size_bytes=len(raw),
            parent_checkpoint_sha256=parent_checkpoint_sha256,
        )

    @property
    def receipt_id(self) -> str:
        raw = (
            f"{self.key.stream_id}|{self.iteration}|{self.checkpoint_sha256}|"
            f"{self.parent_checkpoint_sha256 or 'GENESIS'}"
        ).encode()
        return "spinreceipt-v1:" + hashlib.sha256(raw).hexdigest()


@dataclass
class _Progress:
    total_iterations: int
    next_iteration: int = 1
    active_lease_id: str | None = None
    failed_attempts_for_next_iteration: int = 0
    last_checkpoint_sha256: str | None = None
    last_checkpoint_locator: str | None = None
    last_checkpoint_size_bytes: int | None = None
    last_receipt_id: str | None = None


class IndependentStreamScheduler:
    """Deterministic scheduler that parallelizes only independent streams.

    The unit of concurrency is a *whole training iteration* for a stream, not a
    root. This is intentional: the frozen R7/R8 semantic contract uses one
    persistent live RNG for traversal, reservoir replacement and minibatch
    sampling in execution order. Leasing no more than one iteration at a time
    for a key guarantees this scheduler cannot introduce intra-stream root/RNG
    parallelism.

    Progress is fail-closed as well: an iteration advances only after a durable
    checkpoint receipt whose parent digest matches this stream's last accepted
    checkpoint. The scheduler owns no application RNG and never derives per-root
    seeds.
    """

    SCHEMA = "SPINCORE_R8_INDEPENDENT_STREAM_SCHEDULER_V2"

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

    def complete(
        self,
        lease: IterationLease,
        receipt: DurableIterationReceipt | None = None,
    ) -> None:
        """Advance exactly once, but only after durable checkpoint proof."""
        state = self._validate_active(lease)
        if receipt is None:
            raise ValueError("durable checkpoint receipt is required before stream progress can advance")
        if receipt.key != lease.key or receipt.iteration != lease.iteration or receipt.lease_id != lease.lease_id:
            raise ValueError("durable checkpoint receipt does not belong to the active lease")
        if receipt.parent_checkpoint_sha256 != state.last_checkpoint_sha256:
            raise ValueError("durable checkpoint parent does not match accepted stream lineage")

        state.last_checkpoint_sha256 = receipt.checkpoint_sha256
        state.last_checkpoint_locator = receipt.checkpoint_locator
        state.last_checkpoint_size_bytes = receipt.checkpoint_size_bytes
        state.last_receipt_id = receipt.receipt_id
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

    def last_checkpoint_sha256(self, key: ProductionStreamKey) -> str | None:
        state = self._progress.get(key)
        if state is None:
            raise ValueError("unknown production stream")
        return state.last_checkpoint_sha256

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
                    "last_checkpoint_sha256": state.last_checkpoint_sha256,
                    "last_checkpoint_locator": state.last_checkpoint_locator,
                    "last_checkpoint_size_bytes": state.last_checkpoint_size_bytes,
                    "last_receipt_id": state.last_receipt_id,
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

            sha = row.get("last_checkpoint_sha256")
            locator = row.get("last_checkpoint_locator")
            size = row.get("last_checkpoint_size_bytes")
            receipt_id = row.get("last_receipt_id")
            completed = state.next_iteration - 1
            if completed == 0:
                if any(x is not None for x in (sha, locator, size, receipt_id)):
                    raise ValueError("scheduler checkpoint has durable receipt before any completed iteration")
            else:
                if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
                    raise ValueError("completed stream is missing valid durable checkpoint SHA256")
                if not isinstance(locator, str) or not locator.strip():
                    raise ValueError("completed stream is missing durable checkpoint locator")
                if not isinstance(size, int) or size <= 0:
                    raise ValueError("completed stream is missing durable checkpoint size")
                if not isinstance(receipt_id, str) or not receipt_id.startswith("spinreceipt-v1:"):
                    raise ValueError("completed stream is missing durable receipt identity")
            state.last_checkpoint_sha256 = sha
            state.last_checkpoint_locator = locator
            state.last_checkpoint_size_bytes = size
            state.last_receipt_id = receipt_id
        return obj
