from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from spincore.r7_5_action_cfr import validate_policy
from spincore.r7_5_referee_rng import keyed_uniform01, sample_discrete_with_uniform
from spincore.r7_5_referee_states import HeldoutRefereeState, replay_heldout_referee_state, state_street
from spincore.r7_5_action_stage_contract import PAYOUT
from spincore.solver import ResolvedExactAction

PolicyProvider = Callable[[object, bytes, tuple[int, ...]], Sequence[float]]


@dataclass(frozen=True)
class OmissionStateResult:
    state_index: int
    actor: int
    referee_best_q: float
    candidate_best_available_q: float
    omission: float
    referee_best_action: ResolvedExactAction
    candidate_best_action: ResolvedExactAction
    referee_action_count: int
    candidate_available_action_count: int


def _effective_rows(state, action_spec) -> tuple[int, tuple[tuple[int, ResolvedExactAction], ...]]:
    active_mask = int(action_spec.active_mask(state_street(state)))
    rows = state.universal_resolved_actions(active_mask)
    if not rows:
        raise RuntimeError("nonterminal referee state has no effective action")
    exact = [action for _, action in rows]
    if len(exact) != len(set(exact)):
        raise RuntimeError("effective exact-action identity is not unique")
    return active_mask, rows


def dense_partial_exact_value(
    state,
    *,
    target_player: int,
    dense_action_spec,
    dense_policy: PolicyProvider,
    exact_opponent_levels: int,
    continuation_key_prefix: Sequence[object],
    opponent_sample_ordinal: int = 0,
    depth: int = 0,
) -> float:
    if depth > 128:
        raise RuntimeError("dense referee continuation exceeded 128 decisions")
    if state.terminal:
        return float(state.terminal_icm_delta(PAYOUT)[int(target_player)])

    actor = int(state.actor)
    active_mask, rows = _effective_rows(state, dense_action_spec)
    legal = tuple(slot for slot, _ in rows)
    observation = state.neural_bytes()
    sigma = validate_policy(dense_policy(state, observation, legal), legal)

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
                value += probability * dense_partial_exact_value(
                    child,
                    target_player=int(target_player),
                    dense_action_spec=dense_action_spec,
                    dense_policy=dense_policy,
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
        return dense_partial_exact_value(
            child,
            target_player=int(target_player),
            dense_action_spec=dense_action_spec,
            dense_policy=dense_policy,
            exact_opponent_levels=0,
            continuation_key_prefix=continuation_key_prefix,
            opponent_sample_ordinal=int(opponent_sample_ordinal) + 1,
            depth=depth + 1,
        )
    finally:
        child.close()


def evaluate_omission_state(
    state,
    *,
    state_index: int,
    domain: str,
    training_seed: int,
    evaluation_seed: int,
    dense_action_spec,
    candidate_action_spec,
    dense_policy: PolicyProvider,
    exact_opponent_levels: int = 2,
) -> OmissionStateResult:
    if state.terminal:
        raise ValueError("omission evaluator requires a nonterminal decision state")
    target = int(state.actor)
    dense_mask, dense_rows = _effective_rows(state, dense_action_spec)
    _candidate_mask, candidate_rows = _effective_rows(state, candidate_action_spec)
    dense_by_exact = {exact: slot for slot, exact in dense_rows}
    candidate_exact = {exact for _, exact in candidate_rows}
    if not candidate_exact.issubset(set(dense_by_exact)):
        raise RuntimeError("candidate expresses an exact action absent from dense referee")
    available = set(dense_by_exact).intersection(candidate_exact)
    if not available:
        raise RuntimeError("candidate has no exact action in common with dense referee")

    q: dict[ResolvedExactAction, float] = {}
    for exact, slot in dense_by_exact.items():
        child = state.child_universal(dense_mask, slot)
        try:
            q[exact] = dense_partial_exact_value(
                child,
                target_player=target,
                dense_action_spec=dense_action_spec,
                dense_policy=dense_policy,
                exact_opponent_levels=int(exact_opponent_levels),
                continuation_key_prefix=(
                    "omission",
                    str(domain),
                    int(training_seed),
                    int(evaluation_seed),
                    int(state_index),
                    int(exact.action_type),
                    int(exact.amount_to),
                ),
            )
        finally:
            child.close()

    referee_best_action = max(q, key=lambda action: (q[action], -action.action_type, -action.amount_to))
    candidate_best_action = max(
        available,
        key=lambda action: (q[action], -action.action_type, -action.amount_to),
    )
    referee_best = float(q[referee_best_action])
    candidate_best = float(q[candidate_best_action])
    omission = referee_best - candidate_best
    if omission < -1e-12:
        raise RuntimeError("candidate omission became materially negative")
    if omission < 0.0:
        omission = 0.0
    return OmissionStateResult(
        state_index=int(state_index),
        actor=target,
        referee_best_q=referee_best,
        candidate_best_available_q=candidate_best,
        omission=float(omission),
        referee_best_action=referee_best_action,
        candidate_best_action=candidate_best_action,
        referee_action_count=len(dense_by_exact),
        candidate_available_action_count=len(available),
    )


def evaluate_heldout_omissions(
    *,
    solver,
    descriptors: Sequence[HeldoutRefereeState],
    dense_action_spec,
    candidate_action_spec,
    dense_policy: PolicyProvider,
    exact_opponent_levels: int = 2,
) -> tuple[OmissionStateResult, ...]:
    out: list[OmissionStateResult] = []
    for descriptor in descriptors:
        state = replay_heldout_referee_state(
            solver=solver,
            action_spec=dense_action_spec,
            descriptor=descriptor,
        )
        try:
            out.append(
                evaluate_omission_state(
                    state,
                    state_index=descriptor.state_index,
                    domain=descriptor.domain,
                    training_seed=descriptor.training_seed,
                    evaluation_seed=descriptor.evaluation_seed,
                    dense_action_spec=dense_action_spec,
                    candidate_action_spec=candidate_action_spec,
                    dense_policy=dense_policy,
                    exact_opponent_levels=int(exact_opponent_levels),
                )
            )
        finally:
            state.close()
    return tuple(out)
