from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np

PREFIX = "SpinCore|R7.5.3C|PHASE2|REFV1"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_CONFIDENCE = 0.95


def _field(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite Phase2 referee RNG key field")
        return format(value, ".17g")
    text = str(value)
    if "|" in text:
        raise ValueError("Phase2 referee RNG fields may not contain '|'")
    return text


def canonical_key(*fields) -> str:
    return "|".join((PREFIX, *(_field(value) for value in fields)))


def stable_seed64(*fields) -> int:
    digest = hashlib.sha256(canonical_key(*fields).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def keyed_uniform01(*fields) -> float:
    raw = stable_seed64(*fields)
    return (float(raw) + 0.5) / float(1 << 64)


def sample_discrete_with_uniform(
    probabilities: Sequence[float],
    legal_actions: Sequence[int],
    uniform: float,
) -> int:
    if not 0.0 < float(uniform) < 1.0:
        raise ValueError("uniform must be strictly between zero and one")
    legal = tuple(int(action) for action in legal_actions)
    if not legal:
        raise ValueError("cannot sample an empty action set")
    if len(probabilities) <= max(legal):
        raise ValueError("probability vector does not cover legal actions")
    legal_set = set(legal)
    cleaned: list[tuple[int, float]] = []
    total = 0.0
    for index, raw in enumerate(probabilities):
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("invalid action probability")
        if index in legal_set:
            cleaned.append((index, value))
            total += value
        elif value != 0.0:
            raise ValueError("illegal action has probability mass")
    if total <= 0.0:
        cleaned = [(action, 1.0 / len(legal)) for action in legal]
        total = 1.0
    threshold = float(uniform) * total
    cumulative = 0.0
    for action, value in cleaned:
        cumulative += value
        if threshold < cumulative:
            return action
    return cleaned[-1][0]


def paired_bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed_fields: Sequence[object],
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    chunk_size: int = 64,
) -> dict[str, float | int]:
    data = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("paired bootstrap requires nonempty one-dimensional sample")
    if not np.isfinite(data).all():
        raise ValueError("paired bootstrap sample contains non-finite values")
    if int(replicates) <= 0 or int(chunk_size) <= 0:
        raise ValueError("positive bootstrap sizes required")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("bootstrap confidence must be between zero and one")
    rng = np.random.Generator(np.random.PCG64(stable_seed64(*seed_fields)))
    means = np.empty(int(replicates), dtype=np.float64)
    n = int(data.size)
    offset = 0
    while offset < int(replicates):
        count = min(int(chunk_size), int(replicates) - offset)
        indices = rng.integers(0, n, size=(count, n), endpoint=False)
        means[offset : offset + count] = data[indices].mean(axis=1)
        offset += count
    alpha = 1.0 - float(confidence)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "sample_count": n,
        "replicates": int(replicates),
        "confidence": float(confidence),
        "mean": float(data.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_seed": int(stable_seed64(*seed_fields)),
    }
