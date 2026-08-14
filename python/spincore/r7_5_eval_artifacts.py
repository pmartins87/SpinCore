from __future__ import annotations

from dataclasses import dataclass
import gzip
import math
import os
from pathlib import Path
import pickle
from typing import Sequence

from spincore.r7_5_action_stage_contract import PAIRED_EVALUATION_SEEDS, POSTFLOP_TRAINING_SEEDS
from spincore_nn.codec_v2 import decode_spnniv2

DENSE_CACHE_SCHEMA = "SPINCORE_R7_5_4A_DENSE_CELL_CACHE_V1"
CANDIDATE_CELL_SCHEMA = "SPINCORE_R7_5_4A_CANDIDATE_CELL_EVIDENCE_V1"
EXPECTED_EXECUTION_SHA = "457996944f76e9f1fa0475691df978f450259641"
OMISSION_COUNT_PER_CELL = 2048
CROSSPLAY_HANDS = {"TRUE_HEADS_UP": 20000, "THREE_HANDED": 10000}
CANDIDATE_SEAT_COUNT = {"TRUE_HEADS_UP": 2, "THREE_HANDED": 3}


@dataclass(frozen=True)
class StateDiagnostic:
    street: int
    spr: float
    spr_bucket: str
    semantic_sentinel: str


@dataclass(frozen=True)
class DenseCellCache:
    schema: str
    execution_sha: str
    domain: str
    training_seed: int
    evaluation_seed: int
    descriptors: tuple
    q_references: tuple
    diagnostics: tuple[StateDiagnostic, ...]
    crossplay_references: tuple
    production_training_authorized: bool = False
    ready_for_tables: bool = False


@dataclass(frozen=True)
class CandidateCellEvidence:
    schema: str
    execution_sha: str
    candidate_id: str
    domain: str
    training_seed: int
    evaluation_seed: int
    omission_samples: tuple[float, ...]
    crossplay_samples: tuple[float, ...]
    omission_summary: dict
    crossplay_mean: float
    production_training_authorized: bool = False
    ready_for_tables: bool = False


def spr_bucket(value: float) -> str:
    x = float(value)
    if not math.isfinite(x) or x < 0.0:
        raise ValueError("invalid SPR diagnostic")
    if x < 1.0:
        return "SPR_0_1"
    if x < 2.0:
        return "SPR_1_2"
    if x < 4.0:
        return "SPR_2_4"
    if x < 8.0:
        return "SPR_4_8"
    return "SPR_8_PLUS"


def state_diagnostic(state) -> StateDiagnostic:
    decoded = decode_spnniv2(state.neural_bytes_v2())
    street = int(decoded.categorical[1])
    spr = max(0.0, float(decoded.numeric[15]))
    made = int(decoded.categorical[27])
    flush_draw = int(bool(decoded.categorical[32]))
    straight_draw = int(bool(decoded.categorical[36]))
    paired_board = int(bool(decoded.categorical[52]))
    max_suit = min(int(decoded.categorical[44]), 3)
    sentinel = (
        f"ST{street}_M{made}_FD{flush_draw}_SD{straight_draw}_"
        f"PB{paired_board}_MS{max_suit}"
    )
    return StateDiagnostic(
        street=street,
        spr=spr,
        spr_bucket=spr_bucket(spr),
        semantic_sentinel=sentinel,
    )


def _nearest_rank(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires samples")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(float(q) * len(ordered)) - 1)]


def sample_summary(values: Sequence[float]) -> dict:
    rows = tuple(float(value) for value in values)
    if not rows or any(not math.isfinite(value) or value < 0.0 for value in rows):
        raise ValueError("omission summary requires finite nonnegative samples")
    return {
        "count": len(rows),
        "mean": float(sum(rows) / len(rows)),
        "p50": float(_nearest_rank(rows, 0.50)),
        "p95": float(_nearest_rank(rows, 0.95)),
        "max": float(max(rows)),
        "fraction_zero_loss": float(sum(value == 0.0 for value in rows) / len(rows)),
    }


def summarize_omission(
    omission_samples: Sequence[float],
    diagnostics: Sequence[StateDiagnostic],
) -> dict:
    values = tuple(float(value) for value in omission_samples)
    meta = tuple(diagnostics)
    if len(values) != len(meta):
        raise ValueError("omission/diagnostic count mismatch")

    def grouped(key_fn):
        groups: dict[str, list[float]] = {}
        for value, diagnostic in zip(values, meta):
            groups.setdefault(str(key_fn(diagnostic)), []).append(value)
        return {key: sample_summary(group) for key, group in sorted(groups.items())}

    return {
        "overall": sample_summary(values),
        "by_street": grouped(lambda row: row.street),
        "by_spr": grouped(lambda row: row.spr_bucket),
        "by_semantic_sentinel": grouped(lambda row: row.semantic_sentinel),
    }


def expected_candidate_crossplay_samples(domain: str) -> int:
    try:
        return CROSSPLAY_HANDS[str(domain)] * CANDIDATE_SEAT_COUNT[str(domain)]
    except KeyError as exc:
        raise ValueError(f"unsupported evaluation domain {domain!r}") from exc


def _required_execution_sha(expected_execution_sha: str | None) -> str:
    value = EXPECTED_EXECUTION_SHA if expected_execution_sha is None else str(expected_execution_sha)
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("expected evaluator execution SHA must be a lowercase 40-hex git SHA")
    return value


def validate_dense_cell_cache(
    cache: DenseCellCache,
    *,
    exact_counts: bool = True,
    expected_execution_sha: str | None = None,
) -> None:
    required_sha = _required_execution_sha(expected_execution_sha)
    if cache.schema != DENSE_CACHE_SCHEMA or cache.execution_sha != required_sha:
        raise ValueError("dense evaluation cache schema/execution mismatch")
    if cache.domain not in CROSSPLAY_HANDS:
        raise ValueError("dense evaluation cache domain mismatch")
    if int(cache.training_seed) not in POSTFLOP_TRAINING_SEEDS:
        raise ValueError("dense evaluation cache training seed mismatch")
    if int(cache.evaluation_seed) not in PAIRED_EVALUATION_SEEDS:
        raise ValueError("dense evaluation cache evaluation seed mismatch")
    if cache.production_training_authorized or cache.ready_for_tables:
        raise ValueError("dense evaluation cache illegally authorizes production/table use")
    if not (len(cache.descriptors) == len(cache.q_references) == len(cache.diagnostics)):
        raise ValueError("dense evaluation cache omission vectors differ in length")
    if exact_counts:
        if len(cache.descriptors) != OMISSION_COUNT_PER_CELL:
            raise ValueError("dense evaluation cache heldout count mismatch")
        if len(cache.crossplay_references) != CROSSPLAY_HANDS[cache.domain]:
            raise ValueError("dense evaluation cache crossplay count mismatch")


def validate_candidate_cell_evidence(
    evidence: CandidateCellEvidence,
    *,
    exact_counts: bool = True,
    expected_execution_sha: str | None = None,
) -> None:
    required_sha = _required_execution_sha(expected_execution_sha)
    if evidence.schema != CANDIDATE_CELL_SCHEMA or evidence.execution_sha != required_sha:
        raise ValueError("candidate evaluation evidence schema/execution mismatch")
    if evidence.candidate_id == "PF_DENSE_REFERENCE":
        raise ValueError("dense self-reference is not a candidate cell artifact")
    if evidence.domain not in CROSSPLAY_HANDS:
        raise ValueError("candidate evaluation domain mismatch")
    if int(evidence.training_seed) not in POSTFLOP_TRAINING_SEEDS:
        raise ValueError("candidate evaluation training seed mismatch")
    if int(evidence.evaluation_seed) not in PAIRED_EVALUATION_SEEDS:
        raise ValueError("candidate evaluation evaluation seed mismatch")
    if evidence.production_training_authorized or evidence.ready_for_tables:
        raise ValueError("candidate evaluation illegally authorizes production/table use")
    if any(not math.isfinite(value) or value < 0.0 for value in evidence.omission_samples):
        raise ValueError("candidate omission evidence contains invalid sample")
    if any(not math.isfinite(value) for value in evidence.crossplay_samples):
        raise ValueError("candidate crossplay evidence contains invalid sample")
    if exact_counts:
        if len(evidence.omission_samples) != OMISSION_COUNT_PER_CELL:
            raise ValueError("candidate omission sample count mismatch")
        if len(evidence.crossplay_samples) != expected_candidate_crossplay_samples(evidence.domain):
            raise ValueError("candidate crossplay sample count mismatch")


def _atomic_pickle(path: str | Path, value) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=5) as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, target)


def _load_pickle(path: str | Path):
    with gzip.open(Path(path), "rb") as handle:
        return pickle.load(handle)


def save_dense_cell_cache(
    path: str | Path,
    cache: DenseCellCache,
    *,
    exact_counts: bool = True,
    expected_execution_sha: str | None = None,
) -> None:
    validate_dense_cell_cache(
        cache,
        exact_counts=exact_counts,
        expected_execution_sha=expected_execution_sha,
    )
    _atomic_pickle(path, cache)


def load_dense_cell_cache(
    path: str | Path,
    *,
    exact_counts: bool = True,
    expected_execution_sha: str | None = None,
) -> DenseCellCache:
    value = _load_pickle(path)
    if not isinstance(value, DenseCellCache):
        raise ValueError("dense evaluation artifact has wrong type")
    validate_dense_cell_cache(
        value,
        exact_counts=exact_counts,
        expected_execution_sha=expected_execution_sha,
    )
    return value


def save_candidate_cell_evidence(
    path: str | Path,
    evidence: CandidateCellEvidence,
    *,
    exact_counts: bool = True,
    expected_execution_sha: str | None = None,
) -> None:
    validate_candidate_cell_evidence(
        evidence,
        exact_counts=exact_counts,
        expected_execution_sha=expected_execution_sha,
    )
    _atomic_pickle(path, evidence)


def load_candidate_cell_evidence(
    path: str | Path,
    *,
    exact_counts: bool = True,
    expected_execution_sha: str | None = None,
) -> CandidateCellEvidence:
    value = _load_pickle(path)
    if not isinstance(value, CandidateCellEvidence):
        raise ValueError("candidate evaluation artifact has wrong type")
    validate_candidate_cell_evidence(
        value,
        exact_counts=exact_counts,
        expected_execution_sha=expected_execution_sha,
    )
    return value


class MemoizedPolicy:
    """Evaluation-only exact memoization keyed by immutable observation+legal set."""

    def __init__(self, base):
        self.base = base
        self.cache: dict[tuple[bytes, tuple[int, ...]], tuple[float, ...]] = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]):
        key = (bytes(observation), tuple(int(x) for x in legal))
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        value = tuple(float(x) for x in self.base(state, observation, legal))
        self.cache[key] = value
        return value

    def batch_probabilities(self, observations, legal_sets):
        results: list[tuple[float, ...] | None] = [None] * len(observations)
        missing_keys: list[tuple[bytes, tuple[int, ...]]] = []
        missing_indices: list[int] = []
        seen_missing: dict[tuple[bytes, tuple[int, ...]], int] = {}
        for index, (observation, legal) in enumerate(zip(observations, legal_sets)):
            key = (bytes(observation), tuple(int(x) for x in legal))
            if key in self.cache:
                self.hits += 1
                results[index] = self.cache[key]
            elif key in seen_missing:
                missing_indices.append(index)
                missing_keys.append(key)
            else:
                seen_missing[key] = index
                missing_indices.append(index)
                missing_keys.append(key)
        unique_keys = list(seen_missing)
        if unique_keys:
            self.misses += len(unique_keys)
            batch_fn = getattr(self.base, "batch_probabilities", None)
            if callable(batch_fn):
                raw = batch_fn([key[0] for key in unique_keys], [key[1] for key in unique_keys])
            else:
                raw = tuple(self.base(None, key[0], key[1]) for key in unique_keys)
            if len(raw) != len(unique_keys):
                raise RuntimeError("memoized base batch output count mismatch")
            for key, value in zip(unique_keys, raw):
                self.cache[key] = tuple(float(x) for x in value)
        for index, key in zip(missing_indices, missing_keys):
            results[index] = self.cache[key]
        if any(value is None for value in results):
            raise RuntimeError("memoized policy failed to fill batch")
        return tuple(value for value in results if value is not None)
