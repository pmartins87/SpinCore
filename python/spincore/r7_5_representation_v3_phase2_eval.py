from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

import numpy as np

from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_stage_contract import (
    ADVANTAGE_NRMSE_MAX,
    CROSS_SEED_MEAN_TV_MAX,
    CROSS_SEED_P95_TV_MAX,
    ITERATIONS,
    POLICY_TV_MAX,
    TRAINING_SEEDS,
)

EVALUATION_FREEZE_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_EVALUATION_FREEZE_V1"
FINAL_REPORT_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_DOMAIN_FINAL_REPORT_V1"
BOOTSTRAP_REPLICATES = 2000
CONFIDENCE_LEVEL = 0.95
MATERIAL_FLOOR_ICM = 0.001
SENTINEL_SUM_TOLERANCE = 1e-7
SENTINEL_ILLEGAL_TOLERANCE = 1e-12
SENTINEL_DISTINCT_TOLERANCE = 1e-8

H2 = "H2"
H3 = "H3"
INCONCLUSIVE = "INCONCLUSIVE"
DOMAIN_CONFLICT = "DOMAIN_CONFLICT"
BLOCKED = "BLOCKED"


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _as_finite_array(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if array.ndim != 1 or array.size <= 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def linear_quantile(values: Sequence[float], q: float) -> float:
    """Torch-compatible linear empirical quantile for a finite 1-D sample."""
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    array = np.sort(_as_finite_array(values, name="quantile values"))
    if array.size == 1:
        return float(array[0])
    position = (array.size - 1) * float(q)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return float(array[lo])
    weight = position - lo
    return float(array[lo] * (1.0 - weight) + array[hi] * weight)


def summarize_distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = _as_finite_array(values, name="distribution")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": linear_quantile(array, 0.50),
        "p95": linear_quantile(array, 0.95),
        "max": float(array.max()),
    }


def total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    a = _as_finite_array(left, name="left policy")
    b = _as_finite_array(right, name="right policy")
    if a.shape != b.shape:
        raise ValueError("policy vectors have different shapes")
    if np.any(a < -1e-12) or np.any(b < -1e-12):
        raise ValueError("policy vector contains negative probability")
    if abs(float(a.sum()) - 1.0) > SENTINEL_SUM_TOLERANCE:
        raise ValueError("left policy does not sum to one")
    if abs(float(b.sum()) - 1.0) > SENTINEL_SUM_TOLERANCE:
        raise ValueError("right policy does not sum to one")
    return float(0.5 * np.abs(a - b).sum())


def cross_seed_policy_stability(
    seed_a: Sequence[Sequence[float]],
    seed_b: Sequence[Sequence[float]],
) -> dict:
    if len(seed_a) != len(seed_b) or not seed_a:
        raise ValueError("cross-seed policy rows must be aligned and non-empty")
    tv = [total_variation(a, b) for a, b in zip(seed_a, seed_b)]
    summary = summarize_distribution(tv)
    gate_pass = bool(
        float(summary["mean"]) <= CROSS_SEED_MEAN_TV_MAX
        and float(summary["p95"]) <= CROSS_SEED_P95_TV_MAX
    )
    return {
        **summary,
        "mean_tv_max": CROSS_SEED_MEAN_TV_MAX,
        "p95_tv_max": CROSS_SEED_P95_TV_MAX,
        "gate_pass": gate_pass,
    }


def aligned_series_mean(series: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not series:
        raise ValueError("at least one aligned series is required")
    arrays = [_as_finite_array(row, name="aligned series") for row in series]
    size = arrays[0].size
    if any(row.size != size for row in arrays[1:]):
        raise ValueError("aligned series length mismatch")
    return tuple(float(value) for value in np.stack(arrays, axis=0).mean(axis=0))


def paired_two_seed_representation_difference(
    *,
    h2_seed_values: Sequence[Sequence[float]],
    h3_seed_values: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """One H3-H2 value per paired state, averaging the two training seeds first.

    This preserves the frozen bootstrap unit as a heldout state and avoids treating
    the 2x2 training-seed combinations as independent observations.
    """
    if len(h2_seed_values) != len(TRAINING_SEEDS) or len(h3_seed_values) != len(TRAINING_SEEDS):
        raise ValueError("exactly two frozen training-seed series per representation are required")
    h2 = _as_finite_array(aligned_series_mean(h2_seed_values), name="H2 state means")
    h3 = _as_finite_array(aligned_series_mean(h3_seed_values), name="H3 state means")
    if h2.size != h3.size:
        raise ValueError("H2/H3 state count mismatch")
    return tuple(float(value) for value in (h3 - h2))


def stable_seed64(*parts: object) -> int:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed_parts: Sequence[object],
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence_level: float = CONFIDENCE_LEVEL,
    chunk_size: int = 16,
) -> dict:
    array = _as_finite_array(values, name="bootstrap values")
    reps = int(replicates)
    if reps <= 0:
        raise ValueError("bootstrap replicates must be positive")
    confidence = float(confidence_level)
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence level must be in (0, 1)")
    width = int(chunk_size)
    if width <= 0:
        raise ValueError("bootstrap chunk size must be positive")
    seed_key = tuple(seed_parts)
    seed = stable_seed64(*seed_key)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(reps, dtype=np.float64)
    cursor = 0
    while cursor < reps:
        count = min(width, reps - cursor)
        indices = rng.integers(0, array.size, size=(count, array.size), endpoint=False)
        draws[cursor : cursor + count] = array[indices].mean(axis=1)
        cursor += count
    alpha = (1.0 - confidence) / 2.0
    return {
        "unit_count": int(array.size),
        "estimate": float(array.mean()),
        "ci_low": linear_quantile(draws, alpha),
        "ci_high": linear_quantile(draws, 1.0 - alpha),
        "replicates": reps,
        "confidence_level": confidence,
        "seed": int(seed),
        "seed_key": [str(part) for part in seed_key],
    }


def equal_group_stratified_bootstrap_mean_ci(
    groups: Mapping[str, Sequence[float]],
    *,
    seed_parts: Sequence[object],
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence_level: float = CONFIDENCE_LEVEL,
    chunk_size: int = 8,
) -> dict:
    """Equal-weight group mean with within-group bootstrap resampling.

    Used to combine HU and 3H without letting their different frozen hand/state
    counts silently change the representation-selection weight.
    """
    if not groups:
        raise ValueError("at least one bootstrap group is required")
    ordered = [(str(name), _as_finite_array(values, name=f"group {name}")) for name, values in sorted(groups.items())]
    reps = int(replicates)
    if reps <= 0:
        raise ValueError("bootstrap replicates must be positive")
    confidence = float(confidence_level)
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence level must be in (0, 1)")
    width = int(chunk_size)
    if width <= 0:
        raise ValueError("bootstrap chunk size must be positive")
    seed_key = tuple(seed_parts)
    seed = stable_seed64(*seed_key)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(reps, dtype=np.float64)
    cursor = 0
    while cursor < reps:
        count = min(width, reps - cursor)
        group_draws = []
        for _name, array in ordered:
            indices = rng.integers(0, array.size, size=(count, array.size), endpoint=False)
            group_draws.append(array[indices].mean(axis=1))
        draws[cursor : cursor + count] = np.stack(group_draws, axis=0).mean(axis=0)
        cursor += count
    alpha = (1.0 - confidence) / 2.0
    group_means = {name: float(array.mean()) for name, array in ordered}
    return {
        "groups": {name: int(array.size) for name, array in ordered},
        "group_means": group_means,
        "estimate": float(sum(group_means.values()) / len(group_means)),
        "ci_low": linear_quantile(draws, alpha),
        "ci_high": linear_quantile(draws, 1.0 - alpha),
        "replicates": reps,
        "confidence_level": confidence,
        "seed": int(seed),
        "seed_key": [str(part) for part in seed_key],
        "group_weighting": "EQUAL_GROUP_WEIGHT_WITHIN_GROUP_RESAMPLE",
    }


def classify_local_deviation_ci(
    ci_low: float,
    ci_high: float,
    *,
    material_floor: float = MATERIAL_FLOOR_ICM,
) -> str:
    low, high, floor = float(ci_low), float(ci_high), float(material_floor)
    if not (_finite(low) and _finite(high)) or low > high or floor < 0.0:
        raise ValueError("invalid local-deviation confidence interval")
    if high <= -floor:
        return H3
    if low >= floor:
        return H2
    return INCONCLUSIVE


def classify_pairwise_crossplay_ci(
    ci_low: float,
    ci_high: float,
    *,
    material_floor: float = MATERIAL_FLOOR_ICM,
) -> str:
    low, high, floor = float(ci_low), float(ci_high), float(material_floor)
    if not (_finite(low) and _finite(high)) or low > high or floor < 0.0:
        raise ValueError("invalid pairwise confidence interval")
    if low >= floor:
        return H3
    if high <= -floor:
        return H2
    return INCONCLUSIVE


def combine_domain_directions(directions: Mapping[str, str]) -> str:
    if not directions:
        raise ValueError("domain directions are required")
    allowed = {H2, H3, INCONCLUSIVE}
    values = [str(value) for _domain, value in sorted(directions.items())]
    if any(value not in allowed for value in values):
        raise ValueError("unknown domain direction")
    if H2 in values and H3 in values:
        return DOMAIN_CONFLICT
    if H3 in values:
        return H3
    if H2 in values:
        return H2
    return INCONCLUSIVE


def validate_training_final_report(report: Mapping[str, object]) -> dict:
    failures: list[str] = []
    if report.get("schema") != FINAL_REPORT_SCHEMA:
        failures.append("SCHEMA")
    representation = str(report.get("representation"))
    if representation not in (H2_FINAL, H3_FINAL):
        failures.append("REPRESENTATION")
    if int(report.get("training_seed", -1)) not in TRAINING_SEEDS:
        failures.append("TRAINING_SEED")
    if int(report.get("iterations", -1)) != ITERATIONS:
        failures.append("ITERATION_COUNT")
    iteration_reports = list(report.get("iteration_reports") or [])
    if len(iteration_reports) != ITERATIONS:
        failures.append("ITERATION_REPORT_COUNT")
    else:
        for expected_iteration, row in enumerate(iteration_reports, start=1):
            if int(row.get("iteration", -1)) != expected_iteration:
                failures.append(f"ITERATION_{expected_iteration}_IDENTITY")
                continue
            value = float(row.get("ensemble_weighted_nrmse", math.nan))
            numeric_pass = _finite(value) and value <= ADVANTAGE_NRMSE_MAX
            if not numeric_pass:
                failures.append(f"ITERATION_{expected_iteration}_ADVANTAGE_NRMSE")
            if bool(row.get("ensemble_advantage_gate_pass")) != bool(numeric_pass):
                failures.append(f"ITERATION_{expected_iteration}_ADVANTAGE_GATE_BOOLEAN")
    policy_tv = float(report.get("final_policy_weighted_mean_tv", math.nan))
    policy_pass = _finite(policy_tv) and policy_tv <= POLICY_TV_MAX
    if not policy_pass:
        failures.append("FINAL_POLICY_TV")
    if bool(report.get("final_policy_gate_pass")) != bool(policy_pass):
        failures.append("FINAL_POLICY_GATE_BOOLEAN")
    return {
        "representation": representation,
        "training_seed": int(report.get("training_seed", -1)),
        "advantage_nrmse_max": ADVANTAGE_NRMSE_MAX,
        "final_policy_tv_max": POLICY_TV_MAX,
        "failures": failures,
        "gate_pass": not failures,
    }


def validate_sentinel_vectors(
    *,
    probabilities: Sequence[Sequence[float]],
    legal_sets: Sequence[Sequence[int]],
    logits: Sequence[Sequence[float]],
) -> dict:
    if not probabilities or len(probabilities) != len(legal_sets) or len(probabilities) != len(logits):
        raise ValueError("sentinel rows must be aligned and non-empty")
    failures: list[str] = []
    multi_action_vectors: list[np.ndarray] = []
    action_width: int | None = None
    for index, (raw_prob, raw_legal, raw_logits) in enumerate(zip(probabilities, legal_sets, logits)):
        prob = np.asarray(tuple(float(value) for value in raw_prob), dtype=np.float64)
        logit = np.asarray(tuple(float(value) for value in raw_logits), dtype=np.float64)
        if prob.ndim != 1 or logit.ndim != 1 or prob.size == 0 or prob.shape != logit.shape:
            failures.append(f"ROW_{index}_SHAPE")
            continue
        if action_width is None:
            action_width = int(prob.size)
        elif prob.size != action_width:
            failures.append(f"ROW_{index}_WIDTH")
            continue
        if not np.isfinite(logit).all():
            failures.append(f"ROW_{index}_NONFINITE_LOGITS")
        if not np.isfinite(prob).all() or np.any(prob < -SENTINEL_ILLEGAL_TOLERANCE):
            failures.append(f"ROW_{index}_INVALID_PROBABILITIES")
            continue
        legal = tuple(sorted(set(int(value) for value in raw_legal)))
        if not legal or legal[0] < 0 or legal[-1] >= prob.size:
            failures.append(f"ROW_{index}_LEGAL_SET")
            continue
        illegal = np.ones(prob.size, dtype=bool)
        illegal[list(legal)] = False
        if np.any(np.abs(prob[illegal]) > SENTINEL_ILLEGAL_TOLERANCE):
            failures.append(f"ROW_{index}_ILLEGAL_PROBABILITY")
        if abs(float(prob.sum()) - 1.0) > SENTINEL_SUM_TOLERANCE:
            failures.append(f"ROW_{index}_SUM")
        if len(legal) >= 2:
            multi_action_vectors.append(prob)
    distinct = False
    if len(multi_action_vectors) >= 2:
        first = multi_action_vectors[0]
        distinct = any(
            bool(np.max(np.abs(first - row)) > SENTINEL_DISTINCT_TOLERANCE)
            for row in multi_action_vectors[1:]
        )
    if not distinct:
        failures.append("POLICY_COLLAPSE_OR_NO_DISTINCT_MULTI_ACTION_VECTOR")
    return {
        "rows": len(probabilities),
        "multi_action_rows": len(multi_action_vectors),
        "distinct_multi_action_vectors": distinct,
        "failures": failures,
        "gate_pass": not failures,
    }


def resolve_frozen_winner(
    *,
    h2_hard_gate_pass: bool,
    h3_hard_gate_pass: bool,
    local_deviation_direction: str,
    pairwise_crossplay_direction: str,
) -> dict:
    allowed = {H2, H3, INCONCLUSIVE, DOMAIN_CONFLICT}
    local = str(local_deviation_direction)
    pairwise = str(pairwise_crossplay_direction)
    if local not in allowed or pairwise not in allowed:
        raise ValueError("unknown strategic direction")
    if local == DOMAIN_CONFLICT or pairwise == DOMAIN_CONFLICT:
        return {
            "status": BLOCKED,
            "winner": None,
            "reason": "MATERIAL_DOMAIN_CONFLICT",
        }

    h2_pass = bool(h2_hard_gate_pass)
    h3_pass = bool(h3_hard_gate_pass)
    if not h2_pass and not h3_pass:
        return {"status": BLOCKED, "winner": None, "reason": "BOTH_CANDIDATES_FAIL_HARD_GATES"}

    if h2_pass != h3_pass:
        passing = H2 if h2_pass else H3
        failed = H3 if h2_pass else H2
        if local == failed or pairwise == failed:
            return {
                "status": BLOCKED,
                "winner": None,
                "reason": "FAILED_CANDIDATE_MATERIALLY_FAVORED_BY_STRATEGIC_METRIC",
                "passing_candidate": passing,
                "failed_candidate": failed,
            }
        return {
            "status": "SELECTED",
            "winner": passing,
            "reason": "ONLY_CANDIDATE_PASSING_HARD_GATES_AND_NO_FAILED_CANDIDATE_STRATEGIC_GUARD",
        }

    if (local, pairwise) in ((H2, H3), (H3, H2)):
        return {"status": BLOCKED, "winner": None, "reason": "MATERIAL_STRATEGIC_METRIC_CONFLICT"}
    if local == H3 or pairwise == H3:
        return {"status": "SELECTED", "winner": H3, "reason": "CLEAR_H3_BY_FROZEN_RULE"}
    if local == H2 or pairwise == H2:
        return {"status": "SELECTED", "winner": H2, "reason": "CLEAR_H2_BY_FROZEN_RULE"}
    return {
        "status": "SELECTED",
        "winner": H2,
        "reason": "BOTH_STRATEGIC_COMPARISONS_INCONCLUSIVE_H2_SIZE_SPEED_TIEBREAK",
    }
