from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import struct
from typing import Generic, Iterable, TypeVar

NUM_ACTIONS = 6


@dataclass(frozen=True)
class PairedSample:
    kind: str  # advantage | strategy
    domain: str
    corpus_seed: int
    observation_v1: bytes
    observation_v2: bytes
    legal: tuple[int, ...]
    target: tuple[float, ...]
    weight: float
    iteration: int

    def __post_init__(self) -> None:
        if self.kind not in {"advantage", "strategy"}:
            raise ValueError("paired sample kind must be advantage or strategy")
        if len(self.legal) != NUM_ACTIONS or len(self.target) != NUM_ACTIONS:
            raise ValueError("paired sample requires six-action legal/target vectors")
        if len(self.observation_v1) != 126 or not self.observation_v1.startswith(b"SPNNIV1\x00"):
            raise ValueError("paired sample has invalid SPNNIV1 payload")
        if len(self.observation_v2) != 830 or not self.observation_v2.startswith(b"SPNNIV2\x00"):
            raise ValueError("paired sample has invalid SPNNIV2 payload")


def immutable_sample_identity(sample: PairedSample) -> bytes:
    """Cross-process stable identity; does not use Python's salted hash()."""
    digest = hashlib.sha256()
    digest.update(b"SPINCORE_R7_5_PAIRED_SAMPLE_V1\x00")
    digest.update(sample.kind.encode("ascii"))
    digest.update(b"\x00")
    digest.update(sample.domain.encode("ascii"))
    digest.update(struct.pack("<q", int(sample.corpus_seed)))
    digest.update(sample.observation_v1)
    digest.update(sample.observation_v2)
    digest.update(bytes(int(value) & 0xFF for value in sample.legal))
    digest.update(struct.pack("<6d", *(float(value) for value in sample.target)))
    digest.update(struct.pack("<d", float(sample.weight)))
    digest.update(struct.pack("<q", int(sample.iteration)))
    return digest.digest()


def retention_key(sample: PairedSample) -> bytes:
    return hashlib.sha256(b"retain\x00" + immutable_sample_identity(sample)).digest()


def split_key(sample: PairedSample, split_seed: int) -> bytes:
    return hashlib.sha256(
        b"split\x00" + struct.pack("<q", int(split_seed)) + immutable_sample_identity(sample)
    ).digest()


def is_train_sample(sample: PairedSample, split_seed: int, train_fraction: float = 0.8) -> bool:
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    # Compare a 64-bit prefix against an integer threshold; no float RNG.
    value = int.from_bytes(split_key(sample, split_seed)[:8], "big")
    threshold = int(float(train_fraction) * (1 << 64))
    return value < threshold


T = TypeVar("T", bound=PairedSample)


class BottomHashCorpus(Generic[T]):
    """Streaming deterministic cap that never consumes traversal RNG.

    Keeps the lexicographically smallest SHA256 retention keys. Exact duplicate
    samples are intentionally not deduplicated: the frozen empirical sampling
    distribution is preserved. A monotonic insertion sequence is used only to
    prevent heapq from ever comparing PairedSample objects when two identities
    are byte-identical. Because such samples are themselves identical, this
    tie-break cannot change retained semantic content or any hash-based split.
    """

    def __init__(self, capacity: int):
        if int(capacity) <= 0:
            raise ValueError("positive paired-corpus capacity required")
        self.capacity = int(capacity)
        self.seen = 0
        self._sequence = 0
        # Max-heap via negated big-endian integer key. Sequence is a comparison
        # tie-breaker only; it never participates in retention/split hashes.
        self._heap: list[tuple[int, bytes, int, T]] = []

    def add(self, sample: T) -> None:
        key = retention_key(sample)
        key_int = int.from_bytes(key, "big")
        self.seen += 1
        self._sequence += 1
        item = (-key_int, key, self._sequence, sample)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, item)
            return
        largest_kept_int = -self._heap[0][0]
        if key_int < largest_kept_int:
            heapq.heapreplace(self._heap, item)

    @property
    def items(self) -> list[T]:
        return [sample for _, _, _, sample in sorted(self._heap, key=lambda row: (row[1], row[2]))]

    def state_summary(self) -> dict[str, object]:
        items = self.items
        digest = hashlib.sha256()
        for sample in items:
            digest.update(immutable_sample_identity(sample))
        return {
            "schema": "SPINCORE_R7_5_BOTTOM_HASH_CORPUS_V1",
            "capacity": self.capacity,
            "seen": self.seen,
            "kept": len(items),
            "ordered_identity_sha256": digest.hexdigest(),
        }


def split_items(
    samples: Iterable[T],
    *,
    split_seed: int,
    train_fraction: float = 0.8,
) -> tuple[list[T], list[T]]:
    train: list[T] = []
    heldout: list[T] = []
    for sample in samples:
        (train if is_train_sample(sample, split_seed, train_fraction) else heldout).append(sample)
    train.sort(key=immutable_sample_identity)
    heldout.sort(key=immutable_sample_identity)
    return train, heldout
