from __future__ import annotations

import random
from pathlib import Path

from spincore.deep_cfr import icm_delta_utility
from spincore.r7_5_action_cfr import UniversalPartialExactCollector, uniform_policy
from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.solver import Episode, SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"
PAYOUT = (0.5, 0.3, 0.2)


class ListMemory:
    def __init__(self) -> None:
        self.items = []

    def add(self, sample) -> None:
        self.items.append(sample)


def episode() -> Episode:
    return Episode(1500, True, 0, 10, 20, (0, 750, 750), 1, (0,))


def _street(state) -> int:
    payload = state.neural_bytes_v2()
    assert len(payload) == 830 and payload.startswith(b"SPNNIV2\x00")
    return int(payload[112])


def _active_legal(state, spec):
    active = spec.active_mask(_street(state))
    legal = state.universal_legal_actions(active)
    assert legal
    return active, legal


def _uniform(legal):
    value = 1.0 / len(legal)
    return tuple(value if action in legal else 0.0 for action in range(10))


def _sample_uniform(legal, rng):
    x = rng.random()
    step = 1.0 / len(legal)
    cumulative = 0.0
    for action in legal:
        cumulative += step
        if x < cumulative:
            return action
    return legal[-1]


def _reference_advantage(
    state,
    *,
    spec,
    terminal,
    rng,
    traverser: int,
    iteration: int,
    exact_levels: int,
    explicit_mass: float,
    samples: list,
):
    if state.terminal:
        return float(terminal(state)[traverser]), 1, 0
    actor = state.actor
    active, legal = _active_legal(state, spec)
    sigma = _uniform(legal)
    observation = state.neural_bytes()

    if actor == traverser:
        values = [0.0] * 10
        nodes = 1
        added = 0
        for action in legal:
            child = state.child_universal(active, action)
            try:
                value, child_nodes, child_added = _reference_advantage(
                    child,
                    spec=spec,
                    terminal=terminal,
                    rng=rng,
                    traverser=traverser,
                    iteration=iteration,
                    exact_levels=exact_levels,
                    explicit_mass=explicit_mass,
                    samples=samples,
                )
            finally:
                child.close()
            values[action] = value
            nodes += child_nodes
            added += child_added
        node_value = sum(sigma[action] * values[action] for action in legal)
        target = tuple(
            values[action] - node_value if action in legal else 0.0
            for action in range(10)
        )
        if explicit_mass > 0.0:
            samples.append(
                (
                    observation,
                    tuple(1 if action in legal else 0 for action in range(10)),
                    target,
                    float(iteration) * float(explicit_mass),
                    iteration,
                )
            )
            added += 1
        return node_value, nodes, added

    if exact_levels > 0:
        value = 0.0
        nodes = 1
        added = 0
        for action in legal:
            probability = sigma[action]
            child = state.child_universal(active, action)
            try:
                child_value, child_nodes, child_added = _reference_advantage(
                    child,
                    spec=spec,
                    terminal=terminal,
                    rng=rng,
                    traverser=traverser,
                    iteration=iteration,
                    exact_levels=exact_levels - 1,
                    explicit_mass=explicit_mass * probability,
                    samples=samples,
                )
            finally:
                child.close()
            value += probability * child_value
            nodes += child_nodes
            added += child_added
        return value, nodes, added

    action = _sample_uniform(legal, rng)
    child = state.child_universal(active, action)
    try:
        value, nodes, added = _reference_advantage(
            child,
            spec=spec,
            terminal=terminal,
            rng=rng,
            traverser=traverser,
            iteration=iteration,
            exact_levels=0,
            explicit_mass=explicit_mass,
            samples=samples,
        )
    finally:
        child.close()
    return value, nodes + 1, added


def _reference_strategy(
    state,
    *,
    spec,
    rng,
    target_player: int,
    iteration: int,
    samples: list,
):
    if state.terminal:
        return 0
    actor = state.actor
    active, legal = _active_legal(state, spec)
    sigma = _uniform(legal)
    observation = state.neural_bytes()
    if actor == target_player:
        samples.append(
            (
                observation,
                tuple(1 if action in legal else 0 for action in range(10)),
                sigma,
                float(iteration),
                iteration,
            )
        )
        action = _sample_uniform(legal, rng)
        child = state.child_universal(active, action)
        try:
            return 1 + _reference_strategy(
                child,
                spec=spec,
                rng=rng,
                target_player=target_player,
                iteration=iteration,
                samples=samples,
            )
        finally:
            child.close()
    total = 0
    for action in legal:
        child = state.child_universal(active, action)
        try:
            total += _reference_strategy(
                child,
                spec=spec,
                rng=rng,
                target_player=target_player,
                iteration=iteration,
                samples=samples,
            )
        finally:
            child.close()
    return total


def _as_tuple(sample):
    return (
        sample.observation,
        tuple(sample.legal),
        tuple(float(value) for value in sample.target),
        float(sample.weight),
        int(sample.iteration),
    )


def test_pf0_partial_exact_matches_independent_deduplicated_reference() -> None:
    solver = SolverLibrary(LIB)
    terminal = icm_delta_utility(PAYOUT)
    reference_rng = random.Random(90311)
    collector_rng = random.Random(90311)
    reference_samples = []
    collector_adv = ListMemory()
    collector_strategy = ListMemory()
    pf0 = postflop_candidate_specs(ROOT)["PF0_CONTROL_33_75_AI"]
    collector = UniversalPartialExactCollector(
        action_spec=pf0,
        selected_representation="C0_V1_FROZEN_CONTROL",
        policy=uniform_policy,
        terminal_utility=terminal,
        rng=collector_rng,
        advantage_memory=collector_adv,
        strategy_memory=collector_strategy,
    )

    root_reference = solver.create(episode(), 551122)
    try:
        ref_utility, ref_nodes, ref_added = _reference_advantage(
            root_reference,
            spec=pf0,
            terminal=terminal,
            rng=reference_rng,
            traverser=1,
            iteration=2,
            exact_levels=2,
            explicit_mass=1.0,
            samples=reference_samples,
        )
    finally:
        root_reference.close()
    root_collector = solver.create(episode(), 551122)
    try:
        result = collector.collect_advantage_partial_exact(
            root_collector,
            traverser=1,
            iteration=2,
            exact_opponent_levels=2,
        )
    finally:
        root_collector.close()

    assert result.utility == ref_utility
    assert result.nodes == ref_nodes
    assert result.samples_added == ref_added
    assert collector_rng.getstate() == reference_rng.getstate()
    assert [_as_tuple(sample) for sample in collector_adv.items] == reference_samples


def test_pf0_strategy_matches_independent_deduplicated_reference() -> None:
    solver = SolverLibrary(LIB)
    terminal = icm_delta_utility(PAYOUT)
    reference_rng = random.Random(177013)
    collector_rng = random.Random(177013)
    reference_samples = []
    collector_adv = ListMemory()
    collector_strategy = ListMemory()
    pf0 = postflop_candidate_specs(ROOT)["PF0_CONTROL_33_75_AI"]
    collector = UniversalPartialExactCollector(
        action_spec=pf0,
        selected_representation="C0_V1_FROZEN_CONTROL",
        policy=uniform_policy,
        terminal_utility=terminal,
        rng=collector_rng,
        advantage_memory=collector_adv,
        strategy_memory=collector_strategy,
    )

    root_reference = solver.create(episode(), 771199)
    try:
        reference_count = _reference_strategy(
            root_reference,
            spec=pf0,
            rng=reference_rng,
            target_player=1,
            iteration=2,
            samples=reference_samples,
        )
    finally:
        root_reference.close()
    root_collector = solver.create(episode(), 771199)
    try:
        collector_count = collector.collect_strategy_own_reach(
            root_collector,
            target_player=1,
            iteration=2,
        )
    finally:
        root_collector.close()

    assert collector_count == reference_count
    assert collector_rng.getstate() == reference_rng.getstate()
    assert [_as_tuple(sample) for sample in collector_strategy.items] == reference_samples
