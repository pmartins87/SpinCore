from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from spincore.r7_5_action_cfr import validate_policy
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_action_stage_contract import PAYOUT
from spincore.r7_5_referee_rng import keyed_uniform01, sample_discrete_with_uniform, stable_seed64
from spincore.r7_5_referee_states import state_street

PolicyProvider = Callable[[object, bytes, tuple[int, ...]], Sequence[float]]


@dataclass(frozen=True)
class DenseCrossplayReferenceHand:
    hand_index: int
    scenario_index: int
    deck_seed: int
    terminal_icm_delta: tuple[float, float, float]


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


def build_dense_crossplay_reference(
    *,
    solver,
    dense_action_spec,
    dense_policy: PolicyProvider,
    domain: str,
    training_seed: int,
    evaluation_seed: int,
    hand_count: int,
) -> tuple[DenseCrossplayReferenceHand, ...]:
    if int(hand_count) <= 0:
        raise ValueError("positive crossplay hand count required")
    scenarios = action_scenario_cycle(str(domain))
    specs = (dense_action_spec, dense_action_spec, dense_action_spec)
    policies = (dense_policy, dense_policy, dense_policy)
    out: list[DenseCrossplayReferenceHand] = []
    for hand_index in range(int(hand_count)):
        scenario_index = hand_index % len(scenarios)
        deck = stable_seed64(
            "crossplay",
            str(domain),
            int(training_seed),
            int(evaluation_seed),
            hand_index,
            "deck",
        )
        terminal = _run_hand(
            solver=solver,
            episode=scenarios[scenario_index],
            deck_seed=deck,
            domain=str(domain),
            training_seed=int(training_seed),
            evaluation_seed=int(evaluation_seed),
            hand_index=hand_index,
            seat_specs=specs,
            seat_policies=policies,
        )
        out.append(
            DenseCrossplayReferenceHand(
                hand_index=int(hand_index),
                scenario_index=int(scenario_index),
                deck_seed=int(deck),
                terminal_icm_delta=terminal,
            )
        )
    return tuple(out)


def score_candidate_from_crossplay_reference(
    *,
    solver,
    references: Sequence[DenseCrossplayReferenceHand],
    dense_action_spec,
    dense_policy: PolicyProvider,
    candidate_action_spec,
    candidate_policy: PolicyProvider,
    domain: str,
    training_seed: int,
    evaluation_seed: int,
    candidate_seat: int,
) -> tuple[float, ...]:
    allowed = candidate_seats(str(domain))
    if int(candidate_seat) not in allowed:
        raise ValueError("candidate seat is not live/eligible for this domain")
    if not references:
        raise ValueError("crossplay reference cache cannot be empty")
    scenarios = action_scenario_cycle(str(domain))
    reference_specs = (dense_action_spec, dense_action_spec, dense_action_spec)
    reference_policies = (dense_policy, dense_policy, dense_policy)
    test_specs = list(reference_specs)
    test_policies = list(reference_policies)
    test_specs[int(candidate_seat)] = candidate_action_spec
    test_policies[int(candidate_seat)] = candidate_policy

    out: list[float] = []
    for expected_index, reference in enumerate(references):
        if int(reference.hand_index) != expected_index:
            raise ValueError("crossplay reference hand indices are not canonical/contiguous")
        expected_scenario = expected_index % len(scenarios)
        if int(reference.scenario_index) != expected_scenario:
            raise ValueError("crossplay reference scenario index mismatch")
        expected_deck = stable_seed64(
            "crossplay",
            str(domain),
            int(training_seed),
            int(evaluation_seed),
            expected_index,
            "deck",
        )
        if int(reference.deck_seed) != expected_deck:
            raise ValueError("crossplay reference deck seed mismatch")
        tested = _run_hand(
            solver=solver,
            episode=scenarios[reference.scenario_index],
            deck_seed=reference.deck_seed,
            domain=str(domain),
            training_seed=int(training_seed),
            evaluation_seed=int(evaluation_seed),
            hand_index=expected_index,
            seat_specs=tuple(test_specs),
            seat_policies=tuple(test_policies),
        )
        out.append(
            float(
                tested[int(candidate_seat)]
                - reference.terminal_icm_delta[int(candidate_seat)]
            )
        )
    return tuple(out)


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
    references = build_dense_crossplay_reference(
        solver=solver,
        dense_action_spec=dense_action_spec,
        dense_policy=dense_policy,
        domain=domain,
        training_seed=training_seed,
        evaluation_seed=evaluation_seed,
        hand_count=hand_count,
    )
    return score_candidate_from_crossplay_reference(
        solver=solver,
        references=references,
        dense_action_spec=dense_action_spec,
        dense_policy=dense_policy,
        candidate_action_spec=candidate_action_spec,
        candidate_policy=candidate_policy,
        domain=domain,
        training_seed=training_seed,
        evaluation_seed=evaluation_seed,
        candidate_seat=candidate_seat,
    )
