from __future__ import annotations

from typing import Callable, Sequence

from spincore.r7_5_action_cfr import validate_policy
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3_referee_rng import (
    keyed_uniform01,
    sample_discrete_with_uniform,
    stable_seed64,
)
from spincore.r7_5_representation_v3_referee_states import effective_pf0, uniform_policy
from spincore.solver_v3 import neural_bytes_v3

PAYOUT = (0.5, 0.3, 0.2)
PolicyProvider = Callable[[object, bytes, tuple[int, ...]], Sequence[float]]
DEFAULT_BATCH_SIZE = 256


def live_candidate_seats(domain: str) -> tuple[int, ...]:
    if domain == "TRUE_HEADS_UP":
        return (1, 2)
    if domain == "THREE_HANDED":
        return (0, 1, 2)
    raise ValueError(f"unsupported Phase2 crossplay domain {domain!r}")


def uniform_pf0_policy(_state, _observation: bytes, legal: tuple[int, ...]) -> tuple[float, ...]:
    return uniform_policy(tuple(legal))


def _policy_batch(policy, records):
    observations = [record[4] for record in records]
    legal_sets = [record[3] for record in records]
    batch_fn = getattr(policy, "batch_probabilities", None)
    if callable(batch_fn):
        raw_rows = batch_fn(observations, legal_sets)
    else:
        raw_rows = tuple(
            policy(record[1], record[4], record[3])
            for record in records
        )
    if len(raw_rows) != len(records):
        raise RuntimeError("Phase2 crossplay policy batch output count mismatch")
    return tuple(
        validate_policy(raw, legal)
        for raw, legal in zip(raw_rows, legal_sets)
    )


def run_v3_crossplay_hands(
    *,
    solver,
    action_spec,
    domain: str,
    evaluation_seed: int,
    hand_count: int,
    seat_policies: Sequence[PolicyProvider],
    rng_scope: str,
    rollout_batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[tuple[float, float, float], ...]:
    count = int(hand_count)
    width = int(rollout_batch_size)
    if count <= 0 or width <= 0:
        raise ValueError("positive Phase2 crossplay hand/batch count required")
    if len(seat_policies) != 3:
        raise ValueError("Phase2 crossplay requires one policy per physical seat")
    if not str(rng_scope) in ("commonref", "pairwise"):
        raise ValueError("Phase2 crossplay rng_scope must be commonref or pairwise")
    scenarios = action_scenario_cycle(str(domain))
    out: list[tuple[float, float, float]] = []

    for start in range(0, count, width):
        indices = list(range(start, min(count, start + width)))
        states = []
        ordinals = []
        decision_counts = []
        terminal: list[tuple[float, float, float] | None] = [None] * len(indices)
        try:
            for hand_index in indices:
                scenario_index = hand_index % len(scenarios)
                deck_seed = stable_seed64(
                    "crossplay",
                    str(rng_scope),
                    str(domain),
                    int(evaluation_seed),
                    int(hand_index),
                    "deck",
                )
                states.append(solver.create(scenarios[scenario_index], deck_seed))
                ordinals.append([0, 0, 0])
                decision_counts.append(0)

            while any(value is None for value in terminal):
                groups: dict[int, dict] = {}
                for local_index, state in enumerate(states):
                    if terminal[local_index] is not None:
                        continue
                    if state.terminal:
                        terminal[local_index] = tuple(
                            float(x) for x in state.terminal_icm_delta(PAYOUT)
                        )
                        continue
                    decision_counts[local_index] += 1
                    if decision_counts[local_index] > 128:
                        raise RuntimeError("Phase2 crossplay hand exceeded 128 decisions")
                    actor = int(state.actor)
                    if actor not in (0, 1, 2):
                        raise RuntimeError("Phase2 crossplay state has invalid actor")
                    active_mask, legal, _exact = effective_pf0(state, action_spec)
                    observation = neural_bytes_v3(state)
                    policy = seat_policies[actor]
                    record = (
                        local_index,
                        state,
                        active_mask,
                        legal,
                        observation,
                        actor,
                        ordinals[local_index][actor],
                        indices[local_index],
                    )
                    bucket = groups.setdefault(id(policy), {"policy": policy, "records": []})
                    if bucket["policy"] is not policy:
                        raise RuntimeError("Phase2 crossplay policy identity collision")
                    bucket["records"].append(record)

                for bucket in groups.values():
                    policy = bucket["policy"]
                    records = bucket["records"]
                    probabilities = _policy_batch(policy, records)
                    for record, sigma in zip(records, probabilities):
                        local_index, state, active_mask, legal, _obs, actor, ordinal, hand_index = record
                        uniform = keyed_uniform01(
                            "crossplay",
                            str(rng_scope),
                            str(domain),
                            int(evaluation_seed),
                            int(hand_index),
                            int(actor),
                            int(ordinal),
                        )
                        action = sample_discrete_with_uniform(sigma, legal, uniform)
                        ordinals[local_index][actor] += 1
                        state.apply_universal(active_mask, action)
            out.extend(value for value in terminal if value is not None)
        finally:
            for state in states:
                state.close()
    if len(out) != count:
        raise RuntimeError("Phase2 crossplay terminal count drift")
    return tuple(out)


def common_reference_scores(
    *,
    solver,
    action_spec,
    candidate_policy: PolicyProvider,
    domain: str,
    evaluation_seed: int,
    candidate_seat: int,
    hand_count: int,
    rollout_batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[float, ...]:
    if int(candidate_seat) not in live_candidate_seats(domain):
        raise ValueError("Phase2 common-reference candidate seat is not live")
    uniform = uniform_pf0_policy
    reference = run_v3_crossplay_hands(
        solver=solver,
        action_spec=action_spec,
        domain=domain,
        evaluation_seed=evaluation_seed,
        hand_count=hand_count,
        seat_policies=(uniform, uniform, uniform),
        rng_scope="commonref",
        rollout_batch_size=rollout_batch_size,
    )
    policies = [uniform, uniform, uniform]
    policies[int(candidate_seat)] = candidate_policy
    tested = run_v3_crossplay_hands(
        solver=solver,
        action_spec=action_spec,
        domain=domain,
        evaluation_seed=evaluation_seed,
        hand_count=hand_count,
        seat_policies=tuple(policies),
        rng_scope="commonref",
        rollout_batch_size=rollout_batch_size,
    )
    return tuple(
        float(test[int(candidate_seat)] - ref[int(candidate_seat)])
        for ref, test in zip(reference, tested)
    )


def mirrored_h3_vs_h2_scores(
    *,
    solver,
    action_spec,
    h2_policy: PolicyProvider,
    h3_policy: PolicyProvider,
    domain: str,
    evaluation_seed: int,
    candidate_seat: int,
    hand_count: int,
    rollout_batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[float, ...]:
    if int(candidate_seat) not in live_candidate_seats(domain):
        raise ValueError("Phase2 pairwise candidate seat is not live")
    h3_seat = [h2_policy, h2_policy, h2_policy]
    h3_seat[int(candidate_seat)] = h3_policy
    h2_seat = [h3_policy, h3_policy, h3_policy]
    h2_seat[int(candidate_seat)] = h2_policy
    h3_test = run_v3_crossplay_hands(
        solver=solver,
        action_spec=action_spec,
        domain=domain,
        evaluation_seed=evaluation_seed,
        hand_count=hand_count,
        seat_policies=tuple(h3_seat),
        rng_scope="pairwise",
        rollout_batch_size=rollout_batch_size,
    )
    h2_test = run_v3_crossplay_hands(
        solver=solver,
        action_spec=action_spec,
        domain=domain,
        evaluation_seed=evaluation_seed,
        hand_count=hand_count,
        seat_policies=tuple(h2_seat),
        rng_scope="pairwise",
        rollout_batch_size=rollout_batch_size,
    )
    return tuple(
        0.5 * float(a[int(candidate_seat)] - b[int(candidate_seat)])
        for a, b in zip(h3_test, h2_test)
    )
