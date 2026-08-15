from __future__ import annotations

from dataclasses import dataclass

from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3_referee_rng import (
    keyed_uniform01,
    sample_discrete_with_uniform,
    stable_seed64,
)
from spincore.solver import ResolvedExactAction
from spincore.solver_v3 import neural_bytes_v3
from spincore_nn.codec_v3 import decode_spnniv3


@dataclass(frozen=True)
class HeldoutV3State:
    domain: str
    evaluation_seed: int
    state_index: int
    hand_index: int
    scenario_index: int
    deck_seed: int
    action_path: tuple[int, ...]
    actor: int
    observation_v3: bytes
    active_mask: int
    legal_slots: tuple[int, ...]
    exact_actions: tuple[ResolvedExactAction, ...]


def state_street_v3(state) -> int:
    payload = neural_bytes_v3(state)
    decoded = decode_spnniv3(payload)
    return int(decoded.categorical[1])


def effective_pf0(state, action_spec):
    active_mask = int(action_spec.active_mask(state_street_v3(state)))
    legal = tuple(int(x) for x in state.universal_legal_actions(active_mask))
    resolved = tuple(state.universal_resolved_actions(active_mask))
    exact = tuple(action for slot, action in resolved if int(slot) in set(legal))
    resolved_slots = tuple(int(slot) for slot, _ in resolved if int(slot) in set(legal))
    if resolved_slots != legal:
        raise RuntimeError(
            f"PF0 resolved/legal slot ordering drift: {resolved_slots!r} != {legal!r}"
        )
    if len(exact) != len(legal) or len(exact) != len(set(exact)):
        raise RuntimeError("PF0 exact-action identity is not state-local deduplicated")
    return active_mask, legal, exact


def uniform_policy(legal: tuple[int, ...]) -> tuple[float, ...]:
    if not legal:
        raise ValueError("empty PF0 legal set")
    out = [0.0] * 10
    probability = 1.0 / len(legal)
    for action in legal:
        out[action] = probability
    return tuple(out)


def generate_heldout_v3_states(
    *,
    solver,
    action_spec,
    domain: str,
    evaluation_seed: int,
    count: int,
) -> tuple[HeldoutV3State, ...]:
    if int(count) <= 0:
        raise ValueError("positive heldout state count required")
    scenarios = action_scenario_cycle(str(domain))
    out: list[HeldoutV3State] = []
    hand_index = 0
    while len(out) < int(count):
        scenario_index = hand_index % len(scenarios)
        deck_seed = stable_seed64(
            "heldout", domain, int(evaluation_seed), hand_index, "deck"
        )
        state = solver.create(scenarios[scenario_index], deck_seed)
        path: list[int] = []
        seat_ordinals = [0, 0, 0]
        guard = 0
        try:
            while not state.terminal and len(out) < int(count):
                guard += 1
                if guard > 128:
                    raise RuntimeError("Phase2 heldout hand exceeded 128 decisions")
                actor = int(state.actor)
                if actor not in (0, 1, 2):
                    raise RuntimeError("Phase2 heldout state has invalid actor")
                active_mask, legal, exact = effective_pf0(state, action_spec)
                observation = neural_bytes_v3(state)
                out.append(
                    HeldoutV3State(
                        domain=str(domain),
                        evaluation_seed=int(evaluation_seed),
                        state_index=len(out),
                        hand_index=int(hand_index),
                        scenario_index=int(scenario_index),
                        deck_seed=int(deck_seed),
                        action_path=tuple(path),
                        actor=actor,
                        observation_v3=observation,
                        active_mask=active_mask,
                        legal_slots=legal,
                        exact_actions=exact,
                    )
                )
                if len(out) >= int(count):
                    break
                probabilities = uniform_policy(legal)
                ordinal = seat_ordinals[actor]
                uniform = keyed_uniform01(
                    "heldout",
                    domain,
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


def replay_heldout_v3_state(*, solver, action_spec, descriptor: HeldoutV3State):
    scenarios = action_scenario_cycle(descriptor.domain)
    expected_index = descriptor.hand_index % len(scenarios)
    if int(descriptor.scenario_index) != expected_index:
        raise ValueError("Phase2 heldout replay scenario index mismatch")
    expected_deck = stable_seed64(
        "heldout",
        descriptor.domain,
        descriptor.evaluation_seed,
        descriptor.hand_index,
        "deck",
    )
    if int(descriptor.deck_seed) != expected_deck:
        raise ValueError("Phase2 heldout replay deck seed mismatch")
    state = solver.create(scenarios[descriptor.scenario_index], descriptor.deck_seed)
    try:
        for action in descriptor.action_path:
            if state.terminal:
                raise ValueError("Phase2 heldout replay path continues after terminal")
            active_mask, legal, _exact = effective_pf0(state, action_spec)
            if int(action) not in legal:
                raise ValueError("Phase2 heldout replay contains ineffective action")
            state.apply_universal(active_mask, int(action))
        if state.terminal:
            raise ValueError("Phase2 heldout descriptor unexpectedly terminal")
        active_mask, legal, exact = effective_pf0(state, action_spec)
        if int(state.actor) != int(descriptor.actor):
            raise ValueError("Phase2 heldout replay actor mismatch")
        if neural_bytes_v3(state) != descriptor.observation_v3:
            raise ValueError("Phase2 heldout replay SPNNIV3 bytes mismatch")
        if active_mask != descriptor.active_mask:
            raise ValueError("Phase2 heldout replay active-mask mismatch")
        if legal != descriptor.legal_slots:
            raise ValueError("Phase2 heldout replay legal-slot mismatch")
        if exact != descriptor.exact_actions:
            raise ValueError("Phase2 heldout replay exact-action mismatch")
        return state
    except Exception:
        state.close()
        raise
