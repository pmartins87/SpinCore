from __future__ import annotations

from itertools import combinations
import math
from typing import Mapping, Sequence

from spincore.r7_5_action_cfr import uniform_policy, validate_policy
from spincore.r7_5_action_stage_contract import (
    CROSS_SEED_PER_SEED,
    PAIRED_EVALUATION_SEEDS,
    POSTFLOP_TRAINING_SEEDS,
)
from spincore.r7_5_referee_states import (
    HeldoutRefereeState,
    generate_heldout_referee_states,
    replay_heldout_referee_state,
    state_street,
)

MEAN_TV_MAX = 0.15
P95_TV_MAX = 0.35
INFERENCE_BATCH_SIZE = 256


def build_cross_seed_common_corpus(
    *,
    solver,
    dense_action_spec,
    domain: str,
    per_training_seed: int = CROSS_SEED_PER_SEED,
) -> tuple[HeldoutRefereeState, ...]:
    count = int(per_training_seed)
    if count <= 0 or count % len(PAIRED_EVALUATION_SEEDS) != 0:
        raise ValueError("cross-seed corpus count must be positive and divisible by evaluation-seed count")
    per_eval = count // len(PAIRED_EVALUATION_SEEDS)
    out: list[HeldoutRefereeState] = []
    for training_seed in POSTFLOP_TRAINING_SEEDS:
        for evaluation_seed in PAIRED_EVALUATION_SEEDS:
            out.extend(
                generate_heldout_referee_states(
                    solver=solver,
                    action_spec=dense_action_spec,
                    policy=uniform_policy,
                    domain=str(domain),
                    training_seed=int(training_seed),
                    evaluation_seed=int(evaluation_seed),
                    count=per_eval,
                )
            )
    expected = len(POSTFLOP_TRAINING_SEEDS) * count
    if len(out) != expected:
        raise RuntimeError("cross-seed common corpus size drift")
    return tuple(out)


def _tv(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != 10 or len(second) != 10:
        raise ValueError("cross-seed TV requires ten-action policies")
    return 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(first, second))


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 < float(quantile) <= 1.0:
        raise ValueError("quantile outside (0,1]")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(float(quantile) * len(ordered)) - 1)
    return float(ordered[index])


def _policy_rows(policy, states, observations, legal_sets):
    batch_fn = getattr(policy, "batch_probabilities", None)
    if callable(batch_fn):
        raw_rows = batch_fn(observations, legal_sets)
    else:
        raw_rows = tuple(
            policy(state, observation, legal)
            for state, observation, legal in zip(states, observations, legal_sets)
        )
    if len(raw_rows) != len(states):
        raise RuntimeError("cross-seed policy batch output count mismatch")
    return tuple(validate_policy(raw, legal) for raw, legal in zip(raw_rows, legal_sets))


def cross_seed_policy_stability(
    *,
    solver,
    descriptors: Sequence[HeldoutRefereeState],
    dense_action_spec,
    candidate_action_spec,
    policies_by_seed: Mapping[int, object],
    candidate_id: str,
    domain: str,
    inference_batch_size: int = INFERENCE_BATCH_SIZE,
) -> dict:
    if set(int(seed) for seed in policies_by_seed) != set(POSTFLOP_TRAINING_SEEDS):
        raise ValueError("cross-seed policy set differs from frozen training seeds")
    rows = tuple(descriptors)
    if not rows:
        raise ValueError("cross-seed stability requires common states")
    width = int(inference_batch_size)
    if width <= 0:
        raise ValueError("positive cross-seed inference batch size required")
    pair_ids = tuple(combinations(POSTFLOP_TRAINING_SEEDS, 2))
    tv_values: list[float] = []
    pair_values: dict[str, list[float]] = {f"{a}:{b}": [] for a, b in pair_ids}

    for start in range(0, len(rows), width):
        chunk = rows[start:start + width]
        states = []
        observations = []
        legal_sets = []
        try:
            for descriptor in chunk:
                if descriptor.domain != str(domain):
                    raise ValueError("cross-seed descriptor domain mismatch")
                state = replay_heldout_referee_state(
                    solver=solver,
                    action_spec=dense_action_spec,
                    descriptor=descriptor,
                )
                states.append(state)
                active_mask = int(candidate_action_spec.active_mask(state_street(state)))
                legal = state.universal_legal_actions(active_mask)
                if not legal:
                    raise RuntimeError("candidate has no effective legal action on common state")
                observations.append(state.neural_bytes())
                legal_sets.append(legal)
            probabilities = {
                int(seed): _policy_rows(
                    policies_by_seed[int(seed)], states, observations, legal_sets
                )
                for seed in POSTFLOP_TRAINING_SEEDS
            }
            for row_index in range(len(states)):
                for first_seed, second_seed in pair_ids:
                    value = _tv(
                        probabilities[first_seed][row_index],
                        probabilities[second_seed][row_index],
                    )
                    tv_values.append(value)
                    pair_values[f"{first_seed}:{second_seed}"].append(value)
        finally:
            for state in states:
                state.close()

    mean_tv = sum(tv_values) / len(tv_values)
    p95_tv = _nearest_rank(tv_values, 0.95)
    return {
        "schema": "SPINCORE_R7_5_4A_CROSS_SEED_POLICY_STABILITY_V1",
        "candidate_id": str(candidate_id),
        "domain": str(domain),
        "training_seeds": list(POSTFLOP_TRAINING_SEEDS),
        "common_state_count": len(rows),
        "pairwise_tv_count": len(tv_values),
        "mean_tv": float(mean_tv),
        "p95_tv": float(p95_tv),
        "pairwise": {
            key: {
                "count": len(values),
                "mean_tv": float(sum(values) / len(values)),
                "p95_tv": float(_nearest_rank(values, 0.95)),
            }
            for key, values in pair_values.items()
        },
        "mean_tv_max": MEAN_TV_MAX,
        "p95_tv_max": P95_TV_MAX,
        "gate_pass": bool(mean_tv <= MEAN_TV_MAX and p95_tv <= P95_TV_MAX),
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
