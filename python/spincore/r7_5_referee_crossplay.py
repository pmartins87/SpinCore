from __future__ import annotations

from typing import Callable, Sequence

from spincore.r7_5_action_cfr import validate_policy
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_action_stage_contract import PAYOUT
from spincore.r7_5_referee_rng import keyed_uniform01, sample_discrete_with_uniform, stable_seed64
from spincore.r7_5_referee_states import state_street

PolicyProvider = Callable[[object, bytes, tuple[int, ...]], Sequence[float]]


def candidate_seats(domain: str) -> tuple[int, ...]:
    if domain == "TRUE_HEADS_UP":
        return (1, 2)
    if domain == "THREE_HANDED":
        return (0, 1, 2)
    raise ValueError(f"unsupported crossplay domain {domain!r}")


def _run_hand(
    *,
    solver,
    episode,
    deck_seed: int,
    domain: str,
    training_seed: int,
    evaluation_seed: int,
    hand_index: int,
    seat_specs: Sequence,
    seat_policies: Sequence[PolicyProvider],
) -> tuple[float, float, float]:
    if len(seat_specs) != 3 or len(seat_policies) != 3:
        raise ValueError("crossplay requires one action spec and policy per physical seat")
    state = solver.create(episode, int(deck_seed))
    ordinals = [0, 0, 0]
    guard = 0
    try:
        while not state.terminal:
            guard += 1
            if guard > 128:
                raise RuntimeError("crossplay hand exceeded 128 decisions")
            actor = int(state.actor)
            if actor not in (0, 1, 2):
                raise RuntimeError("crossplay state has invalid actor")
            spec = seat_specs[actor]
            active_mask = int(spec.active_mask(state_street(state)))
            legal = state.universal_legal_actions(active_mask)
            if not legal:
                raise RuntimeError("crossplay nonterminal state has no effective legal action")
            observation = state.neural_bytes()
            sigma = validate_policy(seat_policies[actor](state, observation, legal), legal)
            ordinal = ordinals[actor]
            uniform = keyed_uniform01(
                "crossplay",
                str(domain),
                int(training_seed),
                int(evaluation_seed),
                int(hand_index),
                actor,
                ordinal,
            )
            action = sample_discrete_with_uniform(sigma, legal, uniform)
            ordinals[actor] += 1
            state.apply_universal(active_mask, action)
        delta = state.terminal_icm_delta(PAYOUT)
        return tuple(float(value) for value in delta)
    finally:
        state.close()


def paired_crossplay_scores(
    *,
    solver,
    dense_action_spec,
    dense_policy: PolicyProvider,
    candidate_action_spec,
    candidate_policy: PolicyProvider,
    domain: str,
    training_seed: int,
    evaluation_seed: int,
    candidate_seat: int,
    hand_count: int,
) -> tuple[float, ...]:
    if int(hand_count) <= 0:
        raise ValueError("positive crossplay hand count required")
    allowed = candidate_seats(str(domain))
    if int(candidate_seat) not in allowed:
        raise ValueError("candidate seat is not live/eligible for this domain")
    scenarios = action_scenario_cycle(str(domain))
    out: list[float] = []
    for hand_index in range(int(hand_count)):
        scenario = scenarios[hand_index % len(scenarios)]
        deck = stable_seed64(
            "crossplay",
            str(domain),
            int(training_seed),
            int(evaluation_seed),
            hand_index,
            "deck",
        )
        reference_specs = (dense_action_spec, dense_action_spec, dense_action_spec)
        reference_policies = (dense_policy, dense_policy, dense_policy)
        test_specs = list(reference_specs)
        test_policies = list(reference_policies)
        test_specs[int(candidate_seat)] = candidate_action_spec
        test_policies[int(candidate_seat)] = candidate_policy

        reference = _run_hand(
            solver=solver,
            episode=scenario,
            deck_seed=deck,
            domain=str(domain),
            training_seed=int(training_seed),
            evaluation_seed=int(evaluation_seed),
            hand_index=hand_index,
            seat_specs=reference_specs,
            seat_policies=reference_policies,
        )
        tested = _run_hand(
            solver=solver,
            episode=scenario,
            deck_seed=deck,
            domain=str(domain),
            training_seed=int(training_seed),
            evaluation_seed=int(evaluation_seed),
            hand_index=hand_index,
            seat_specs=tuple(test_specs),
            seat_policies=tuple(test_policies),
        )
        out.append(float(tested[int(candidate_seat)] - reference[int(candidate_seat)]))
    return tuple(out)
