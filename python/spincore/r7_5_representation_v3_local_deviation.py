from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from spincore.r7_5_action_cfr import validate_policy
from spincore.r7_5_representation_v3_referee_rng import (
    keyed_uniform01,
    sample_discrete_with_uniform,
)
from spincore.r7_5_representation_v3_referee_states import (
    HeldoutV3State,
    effective_pf0,
    replay_heldout_v3_state,
)
from spincore.solver import ResolvedExactAction
from spincore.solver_v3 import neural_bytes_v3

PAYOUT = (0.5, 0.3, 0.2)
PolicyProvider = Callable[[object, bytes, tuple[int, ...]], Sequence[float]]


@dataclass(frozen=True)
class LocalDeviationStateResult:
    state_index: int
    actor: int
    policy_value: float
    best_local_value: float
    gain: float
    best_action: ResolvedExactAction
    q_values: tuple[tuple[ResolvedExactAction, float], ...]
    policy_probabilities: tuple[float, ...]


def candidate_continuation_value(
    state,
    *,
    target_player: int,
    action_spec,
    candidate_policy: PolicyProvider,
    exact_opponent_levels: int,
    continuation_key_prefix: Sequence[object],
    opponent_sample_ordinal: int = 0,
    depth: int = 0,
) -> float:
    if depth > 128:
        raise RuntimeError("Phase2 local-deviation continuation exceeded 128 decisions")
    if state.terminal:
        return float(state.terminal_icm_delta(PAYOUT)[int(target_player)])

    actor = int(state.actor)
    active_mask, legal, _exact = effective_pf0(state, action_spec)
    observation = neural_bytes_v3(state)
    sigma = validate_policy(candidate_policy(state, observation, legal), legal)

    enumerate_node = actor == int(target_player) or int(exact_opponent_levels) > 0
    if enumerate_node:
        value = 0.0
        next_exact = int(exact_opponent_levels)
        if actor != int(target_player):
            next_exact -= 1
        for slot in legal:
            probability = float(sigma[slot])
            if probability <= 0.0:
                continue
            child = state.child_universal(active_mask, slot)
            try:
                value += probability * candidate_continuation_value(
                    child,
                    target_player=int(target_player),
                    action_spec=action_spec,
                    candidate_policy=candidate_policy,
                    exact_opponent_levels=next_exact,
                    continuation_key_prefix=continuation_key_prefix,
                    opponent_sample_ordinal=int(opponent_sample_ordinal),
                    depth=depth + 1,
                )
            finally:
                child.close()
        return float(value)

    uniform = keyed_uniform01(
        *continuation_key_prefix,
        int(opponent_sample_ordinal),
    )
    action = sample_discrete_with_uniform(sigma, legal, uniform)
    child = state.child_universal(active_mask, action)
    try:
        return candidate_continuation_value(
            child,
            target_player=int(target_player),
            action_spec=action_spec,
            candidate_policy=candidate_policy,
            exact_opponent_levels=0,
            continuation_key_prefix=continuation_key_prefix,
            opponent_sample_ordinal=int(opponent_sample_ordinal) + 1,
            depth=depth + 1,
        )
    finally:
        child.close()


def evaluate_local_deviation_state(
    state,
    *,
    descriptor: HeldoutV3State,
    action_spec,
    candidate_policy: PolicyProvider,
    exact_opponent_levels: int = 2,
) -> LocalDeviationStateResult:
    if state.terminal:
        raise ValueError("local-deviation evaluation requires nonterminal state")
    actor = int(state.actor)
    if actor != int(descriptor.actor):
        raise ValueError("local-deviation replay actor mismatch")
    active_mask, legal, exact_actions = effective_pf0(state, action_spec)
    if active_mask != int(descriptor.active_mask) or legal != descriptor.legal_slots:
        raise ValueError("local-deviation replay action geometry mismatch")
    if exact_actions != descriptor.exact_actions:
        raise ValueError("local-deviation replay exact-action mismatch")

    observation = neural_bytes_v3(state)
    sigma = validate_policy(candidate_policy(state, observation, legal), legal)
    q_rows: list[tuple[ResolvedExactAction, float]] = []
    slot_q: dict[int, float] = {}
    for slot, exact in zip(legal, exact_actions):
        child = state.child_universal(active_mask, slot)
        try:
            q = candidate_continuation_value(
                child,
                target_player=actor,
                action_spec=action_spec,
                candidate_policy=candidate_policy,
                exact_opponent_levels=int(exact_opponent_levels),
                continuation_key_prefix=(
                    "localdev",
                    descriptor.domain,
                    int(descriptor.evaluation_seed),
                    int(descriptor.state_index),
                    int(exact.action_type),
                    int(exact.amount_to),
                ),
            )
        finally:
            child.close()
        slot_q[int(slot)] = float(q)
        q_rows.append((exact, float(q)))

    policy_value = sum(float(sigma[slot]) * slot_q[int(slot)] for slot in legal)
    best_slot = max(
        legal,
        key=lambda slot: (
            slot_q[int(slot)],
            -exact_actions[legal.index(slot)].action_type,
            -exact_actions[legal.index(slot)].amount_to,
        ),
    )
    best_index = legal.index(best_slot)
    best_action = exact_actions[best_index]
    best_value = float(slot_q[int(best_slot)])
    gain = best_value - float(policy_value)
    if gain < -1e-10:
        raise RuntimeError(f"local-deviation gain became negative: {gain}")
    gain = max(0.0, gain)
    q_rows.sort(key=lambda row: row[0])
    return LocalDeviationStateResult(
        state_index=int(descriptor.state_index),
        actor=actor,
        policy_value=float(policy_value),
        best_local_value=best_value,
        gain=float(gain),
        best_action=best_action,
        q_values=tuple(q_rows),
        policy_probabilities=tuple(float(x) for x in sigma),
    )


def evaluate_local_deviation_heldout(
    *,
    solver,
    descriptors: Sequence[HeldoutV3State],
    action_spec,
    candidate_policy: PolicyProvider,
    exact_opponent_levels: int = 2,
) -> tuple[LocalDeviationStateResult, ...]:
    out: list[LocalDeviationStateResult] = []
    for descriptor in descriptors:
        state = replay_heldout_v3_state(
            solver=solver,
            action_spec=action_spec,
            descriptor=descriptor,
        )
        try:
            out.append(
                evaluate_local_deviation_state(
                    state,
                    descriptor=descriptor,
                    action_spec=action_spec,
                    candidate_policy=candidate_policy,
                    exact_opponent_levels=exact_opponent_levels,
                )
            )
        finally:
            state.close()
    return tuple(out)
