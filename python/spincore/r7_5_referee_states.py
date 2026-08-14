from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_referee_rng import keyed_uniform01, sample_discrete_with_uniform, stable_seed64
from spincore.solver import ResolvedExactAction

PolicyProvider = Callable[[object, bytes, tuple[int, ...]], Sequence[float]]


@dataclass(frozen=True)
class HeldoutRefereeState:
    domain: str
    training_seed: int
    evaluation_seed: int
    state_index: int
    hand_index: int
    scenario_index: int
    deck_seed: int
    action_path: tuple[int, ...]
    actor: int
    observation_v1: bytes
    dense_active_mask: int
    dense_effective_actions: tuple[ResolvedExactAction, ...]


def state_street(state) -> int:
    payload = state.neural_bytes_v2()
    if len(payload) != 830 or not payload.startswith(b"SPNNIV2\x00"):
        raise RuntimeError("heldout referee state requires valid SPNNIV2 metadata")
    return int(payload[112])


def _effective(state, action_spec) -> tuple[int, tuple[int, ...], tuple[ResolvedExactAction, ...]]:
    active_mask = int(action_spec.active_mask(state_street(state)))
    legal = state.universal_legal_actions(active_mask)
    exact = tuple(action for _, action in state.universal_resolved_actions(active_mask))
    if len(legal) != len(exact) or len(exact) != len(set(exact)):
        raise RuntimeError("dense referee effective action identity drift")
    return active_mask, legal, exact


def generate_heldout_referee_states(
    *,
    solver,
    action_spec,
    policy: PolicyProvider,
    domain: str,
    training_seed: int,
    evaluation_seed: int,
    count: int,
) -> tuple[HeldoutRefereeState, ...]:
    if int(count) <= 0:
        raise ValueError("positive heldout state count required")
    scenarios = action_scenario_cycle(str(domain))
    out: list[HeldoutRefereeState] = []
    hand_index = 0
    while len(out) < int(count):
        scenario_index = hand_index % len(scenarios)
        deck = stable_seed64(
            "heldout", domain, int(training_seed), int(evaluation_seed), hand_index, "deck"
        )
        state = solver.create(scenarios[scenario_index], deck)
        path: list[int] = []
        seat_ordinals = [0, 0, 0]
        guard = 0
        try:
            while not state.terminal and len(out) < int(count):
                guard += 1
                if guard > 128:
                    raise RuntimeError("heldout referee hand exceeded 128 decisions")
                actor = int(state.actor)
                if actor not in (0, 1, 2):
                    raise RuntimeError("heldout referee state has invalid actor")
                active_mask, legal, exact = _effective(state, action_spec)
                observation = state.neural_bytes()
                out.append(
                    HeldoutRefereeState(
                        domain=str(domain),
                        training_seed=int(training_seed),
                        evaluation_seed=int(evaluation_seed),
                        state_index=len(out),
                        hand_index=int(hand_index),
                        scenario_index=int(scenario_index),
                        deck_seed=int(deck),
                        action_path=tuple(path),
                        actor=actor,
                        observation_v1=observation,
                        dense_active_mask=active_mask,
                        dense_effective_actions=exact,
                    )
                )
                if len(out) >= int(count):
                    break
                probabilities = tuple(float(x) for x in policy(state, observation, legal))
                ordinal = seat_ordinals[actor]
                uniform = keyed_uniform01(
                    "heldout",
                    domain,
                    int(training_seed),
                    int(evaluation_seed),
                    hand_index,
                    actor,
                    ordinal,
                )
                action = sample_discrete_with_uniform(probabilities, legal, uniform)
                seat_ordinals[actor] += 1
                state.apply_universal(active_mask, action)
                path.append(int(action))
        finally:
            state.close()
        hand_index += 1
    return tuple(out)


def replay_heldout_referee_state(*, solver, action_spec, descriptor: HeldoutRefereeState):
    scenarios = action_scenario_cycle(descriptor.domain)
    expected_index = descriptor.hand_index % len(scenarios)
    if int(descriptor.scenario_index) != expected_index:
        raise ValueError("heldout replay scenario index mismatch")
    expected_deck = stable_seed64(
        "heldout",
        descriptor.domain,
        descriptor.training_seed,
        descriptor.evaluation_seed,
        descriptor.hand_index,
        "deck",
    )
    if int(descriptor.deck_seed) != expected_deck:
        raise ValueError("heldout replay deck seed mismatch")
    state = solver.create(scenarios[descriptor.scenario_index], descriptor.deck_seed)
    try:
        for action in descriptor.action_path:
            if state.terminal:
                raise ValueError("heldout replay path continues after terminal state")
            active_mask = int(action_spec.active_mask(state_street(state)))
            legal = state.universal_legal_actions(active_mask)
            if int(action) not in legal:
                raise ValueError("heldout replay path contains ineffective action")
            state.apply_universal(active_mask, int(action))
        if state.terminal:
            raise ValueError("heldout descriptor unexpectedly replays to terminal")
        active_mask, _legal, exact = _effective(state, action_spec)
        if state.actor != descriptor.actor:
            raise ValueError("heldout replay actor mismatch")
        if state.neural_bytes() != descriptor.observation_v1:
            raise ValueError("heldout replay SPNNIV1 bytes mismatch")
        if active_mask != descriptor.dense_active_mask:
            raise ValueError("heldout replay dense active-mask mismatch")
        if exact != descriptor.dense_effective_actions:
            raise ValueError("heldout replay exact-action set mismatch")
        return state
    except Exception:
        state.close()
        raise
