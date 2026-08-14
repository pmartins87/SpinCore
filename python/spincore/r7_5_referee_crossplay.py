from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from spincore.r7_5_action_cfr import validate_policy
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_action_stage_contract import PAYOUT
from spincore.r7_5_referee_rng import keyed_uniform01, sample_discrete_with_uniform, stable_seed64
from spincore.r7_5_referee_states import state_street

PolicyProvider = Callable[[object, bytes, tuple[int, ...]], Sequence[float]]
DEFAULT_ROLLOUT_BATCH_SIZE = 256


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


def _policy_batch(policy, states, observations, legal_sets) -> tuple[tuple[float, ...], ...]:
    if not (len(states) == len(observations) == len(legal_sets)):
        raise ValueError("crossplay policy batch shape mismatch")
    batch_fn = getattr(policy, "batch_probabilities", None)
    if callable(batch_fn):
        raw_rows = batch_fn(observations, legal_sets)
    else:
        raw_rows = tuple(
            policy(state, observation, legal)
            for state, observation, legal in zip(states, observations, legal_sets)
        )
    if len(raw_rows) != len(states):
        raise RuntimeError("crossplay policy batch output count mismatch")
    return tuple(
        validate_policy(raw, legal)
        for raw, legal in zip(raw_rows, legal_sets)
    )


def _run_hands_batched(
    *,
    solver,
    episodes: Sequence,
    deck_seeds: Sequence[int],
    hand_indices: Sequence[int],
    domain: str,
    training_seed: int,
    evaluation_seed: int,
    seat_specs: Sequence,
    seat_policies: Sequence[PolicyProvider],
) -> tuple[tuple[float, float, float], ...]:
    if not (len(episodes) == len(deck_seeds) == len(hand_indices)):
        raise ValueError("batched crossplay hand vectors differ in length")
    if len(seat_specs) != 3 or len(seat_policies) != 3:
        raise ValueError("crossplay requires one action spec and policy per physical seat")
    if not episodes:
        return ()
    states = [solver.create(episode, int(deck)) for episode, deck in zip(episodes, deck_seeds)]
    ordinals = [[0, 0, 0] for _ in states]
    decision_counts = [0 for _ in states]
    terminal: list[tuple[float, float, float] | None] = [None for _ in states]
    try:
        while any(value is None for value in terminal):
            groups: dict[int, dict] = {}
            for index, state in enumerate(states):
                if terminal[index] is not None:
                    continue
                if state.terminal:
                    delta = state.terminal_icm_delta(PAYOUT)
                    terminal[index] = tuple(float(value) for value in delta)
                    continue
                decision_counts[index] += 1
                if decision_counts[index] > 128:
                    raise RuntimeError("crossplay hand exceeded 128 decisions")
                actor = int(state.actor)
                if actor not in (0, 1, 2):
                    raise RuntimeError("crossplay state has invalid actor")
                spec = seat_specs[actor]
                policy = seat_policies[actor]
                active_mask = int(spec.active_mask(state_street(state)))
                legal = state.universal_legal_actions(active_mask)
                if not legal:
                    raise RuntimeError("crossplay nonterminal state has no effective legal action")
                record = (
                    index,
                    state,
                    actor,
                    active_mask,
                    legal,
                    state.neural_bytes(),
                    ordinals[index][actor],
                )
                bucket = groups.setdefault(id(policy), {"policy": policy, "records": []})
                if bucket["policy"] is not policy:
                    raise RuntimeError("crossplay policy identity collision")
                bucket["records"].append(record)

            for bucket in groups.values():
                policy = bucket["policy"]
                records = bucket["records"]
                probabilities = _policy_batch(
                    policy,
                    [row[1] for row in records],
                    [row[5] for row in records],
                    [row[4] for row in records],
                )
                for record, sigma in zip(records, probabilities):
                    index, state, actor, active_mask, legal, _observation, ordinal = record
                    uniform = keyed_uniform01(
                        "crossplay",
                        str(domain),
                        int(training_seed),
                        int(evaluation_seed),
                        int(hand_indices[index]),
                        int(actor),
                        int(ordinal),
                    )
                    action = sample_discrete_with_uniform(sigma, legal, uniform)
                    ordinals[index][actor] += 1
                    state.apply_universal(active_mask, action)
        return tuple(value for value in terminal if value is not None)
    finally:
        for state in states:
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
    rollout_batch_size: int = DEFAULT_ROLLOUT_BATCH_SIZE,
) -> tuple[DenseCrossplayReferenceHand, ...]:
    count = int(hand_count)
    width = int(rollout_batch_size)
    if count <= 0:
        raise ValueError("positive crossplay hand count required")
    if width <= 0:
        raise ValueError("positive crossplay rollout batch size required")
    scenarios = action_scenario_cycle(str(domain))
    specs = (dense_action_spec, dense_action_spec, dense_action_spec)
    policies = (dense_policy, dense_policy, dense_policy)
    out: list[DenseCrossplayReferenceHand] = []
    for start in range(0, count, width):
        indices = list(range(start, min(count, start + width)))
        scenario_indices = [index % len(scenarios) for index in indices]
        decks = [
            stable_seed64(
                "crossplay", str(domain), int(training_seed), int(evaluation_seed), index, "deck"
            )
            for index in indices
        ]
        terminals = _run_hands_batched(
            solver=solver,
            episodes=[scenarios[index] for index in scenario_indices],
            deck_seeds=decks,
            hand_indices=indices,
            domain=str(domain),
            training_seed=int(training_seed),
            evaluation_seed=int(evaluation_seed),
            seat_specs=specs,
            seat_policies=policies,
        )
        for hand_index, scenario_index, deck, terminal in zip(
            indices, scenario_indices, decks, terminals
        ):
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
    rollout_batch_size: int = DEFAULT_ROLLOUT_BATCH_SIZE,
) -> tuple[float, ...]:
    allowed = candidate_seats(str(domain))
    if int(candidate_seat) not in allowed:
        raise ValueError("candidate seat is not live/eligible for this domain")
    if not references:
        raise ValueError("crossplay reference cache cannot be empty")
    width = int(rollout_batch_size)
    if width <= 0:
        raise ValueError("positive crossplay rollout batch size required")
    scenarios = action_scenario_cycle(str(domain))
    reference_specs = (dense_action_spec, dense_action_spec, dense_action_spec)
    reference_policies = (dense_policy, dense_policy, dense_policy)
    test_specs = list(reference_specs)
    test_policies = list(reference_policies)
    test_specs[int(candidate_seat)] = candidate_action_spec
    test_policies[int(candidate_seat)] = candidate_policy

    out: list[float] = []
    for start in range(0, len(references), width):
        chunk = tuple(references[start:start + width])
        indices: list[int] = []
        episodes: list = []
        decks: list[int] = []
        for offset, reference in enumerate(chunk):
            expected_index = start + offset
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
            indices.append(expected_index)
            episodes.append(scenarios[expected_scenario])
            decks.append(reference.deck_seed)
        tested = _run_hands_batched(
            solver=solver,
            episodes=episodes,
            deck_seeds=decks,
            hand_indices=indices,
            domain=str(domain),
            training_seed=int(training_seed),
            evaluation_seed=int(evaluation_seed),
            seat_specs=tuple(test_specs),
            seat_policies=tuple(test_policies),
        )
        for reference, terminal in zip(chunk, tested):
            out.append(
                float(
                    terminal[int(candidate_seat)]
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
    rollout_batch_size: int = DEFAULT_ROLLOUT_BATCH_SIZE,
) -> tuple[float, ...]:
    references = build_dense_crossplay_reference(
        solver=solver,
        dense_action_spec=dense_action_spec,
        dense_policy=dense_policy,
        domain=domain,
        training_seed=training_seed,
        evaluation_seed=evaluation_seed,
        hand_count=hand_count,
        rollout_batch_size=rollout_batch_size,
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
        rollout_batch_size=rollout_batch_size,
    )
