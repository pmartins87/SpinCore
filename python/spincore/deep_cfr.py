from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Protocol, Sequence

from spincore_nn.reservoir import AdvantageSample, StrategySample, UniformReservoir

from .solver import SolverState

NUM_ACTIONS = 6
Policy = tuple[float, ...]
TerminalUtility = Callable[[SolverState], tuple[float, float, float]]


class PolicyProvider(Protocol):
    def __call__(self, state: SolverState, observation: bytes, legal: tuple[int, ...]) -> Policy: ...


def _validate_policy(policy: Sequence[float], legal: tuple[int, ...]) -> Policy:
    if len(policy) != NUM_ACTIONS:
        raise ValueError(f"policy must have {NUM_ACTIONS} actions")
    legal_set = set(legal)
    if not legal_set:
        raise ValueError("nonterminal state has no legal actions")
    out = [0.0] * NUM_ACTIONS
    total = 0.0
    for a, raw in enumerate(policy):
        p = float(raw)
        if not math.isfinite(p) or p < 0.0:
            raise ValueError("invalid policy probability")
        if a in legal_set:
            out[a] = p
            total += p
        elif p != 0.0:
            raise ValueError("policy assigns mass to illegal action")
    if total <= 0.0:
        u = 1.0 / len(legal)
        for a in legal:
            out[a] = u
    else:
        inv = 1.0 / total
        for a in legal:
            out[a] *= inv
    return tuple(out)


def uniform_policy(_state: SolverState, _observation: bytes, legal: tuple[int, ...]) -> Policy:
    if not legal:
        raise ValueError("empty legal set")
    out = [0.0] * NUM_ACTIONS
    p = 1.0 / len(legal)
    for a in legal:
        out[a] = p
    return tuple(out)


def regret_matching_policy(advantages: Sequence[float], legal: tuple[int, ...]) -> Policy:
    if len(advantages) != NUM_ACTIONS:
        raise ValueError(f"advantages must have {NUM_ACTIONS} actions")
    out = [0.0] * NUM_ACTIONS
    positives = [max(0.0, float(advantages[a])) for a in legal]
    z = sum(positives)
    if z <= 0.0:
        p = 1.0 / len(legal)
        for a in legal:
            out[a] = p
    else:
        for a, x in zip(legal, positives):
            out[a] = x / z
    return tuple(out)


def sample_action(policy: Sequence[float], legal: tuple[int, ...], rng: random.Random) -> int:
    p = _validate_policy(policy, legal)
    x = rng.random()
    acc = 0.0
    last = legal[-1]
    for a in legal:
        acc += p[a]
        if x < acc:
            return a
    return last


@dataclass(frozen=True)
class TraversalResult:
    utility: float
    nodes: int
    samples_added: int


class ExternalSamplingCollector:
    """R6 correctness-first Deep CFR collector over the authoritative C++ state.

    Advantage traversal follows external-sampling MCCFR: enumerate every action
    at the traverser's nodes and sample one action at other players' nodes.

    Average-policy collection follows the project's own-reach rule: at the
    target player's nodes record sigma(I) and sample the target's action;
    enumerate non-target players. The later R7 native frontier is a performance
    optimization of this same semantic contract, not a different algorithm.
    """

    def __init__(
        self,
        *,
        policy: PolicyProvider,
        terminal_utility: TerminalUtility,
        rng: random.Random,
        advantage_memory: UniformReservoir[AdvantageSample],
        strategy_memory: UniformReservoir[StrategySample],
    ) -> None:
        self.policy = policy
        self.terminal_utility = terminal_utility
        self.rng = rng
        self.advantage_memory = advantage_memory
        self.strategy_memory = strategy_memory

    def _policy(self, state: SolverState, obs: bytes, legal: tuple[int, ...]) -> Policy:
        return _validate_policy(self.policy(state, obs, legal), legal)

    def collect_advantage(self, root: SolverState, *, traverser: int, iteration: int) -> TraversalResult:
        if iteration <= 0:
            raise ValueError("LCFR iteration must be positive")
        utility, nodes, added = self._adv(root, traverser=traverser, iteration=iteration)
        return TraversalResult(utility, nodes, added)

    def _adv(self, state: SolverState, *, traverser: int, iteration: int) -> tuple[float, int, int]:
        if state.terminal:
            utility = self.terminal_utility(state)
            return float(utility[traverser]), 1, 0

        actor = state.actor
        legal = state.legal_actions()
        obs = state.neural_bytes()
        sigma = self._policy(state, obs, legal)

        if actor == traverser:
            values = [0.0] * NUM_ACTIONS
            nodes = 1
            added = 0
            for a in legal:
                child = state.child(a)
                try:
                    v, n, s = self._adv(child, traverser=traverser, iteration=iteration)
                finally:
                    child.close()
                values[a] = v
                nodes += n
                added += s
            node_value = sum(sigma[a] * values[a] for a in legal)
            target = [0.0] * NUM_ACTIONS
            for a in legal:
                target[a] = values[a] - node_value
            self.advantage_memory.add(
                AdvantageSample(
                    observation=obs,
                    legal=tuple(1 if a in legal else 0 for a in range(NUM_ACTIONS)),
                    target=tuple(target),
                    weight=float(iteration),
                    iteration=int(iteration),
                )
            )
            return node_value, nodes, added + 1

        action = sample_action(sigma, legal, self.rng)
        child = state.child(action)
        try:
            v, n, s = self._adv(child, traverser=traverser, iteration=iteration)
        finally:
            child.close()
        return v, n + 1, s

    def collect_strategy_own_reach(self, root: SolverState, *, target_player: int, iteration: int) -> int:
        if iteration <= 0:
            raise ValueError("LCFR iteration must be positive")
        return self._strategy(root, target_player=target_player, iteration=iteration)

    def _strategy(self, state: SolverState, *, target_player: int, iteration: int) -> int:
        if state.terminal:
            return 0
        actor = state.actor
        legal = state.legal_actions()
        obs = state.neural_bytes()
        sigma = self._policy(state, obs, legal)

        if actor == target_player:
            self.strategy_memory.add(
                StrategySample(
                    observation=obs,
                    legal=tuple(1 if a in legal else 0 for a in range(NUM_ACTIONS)),
                    target=tuple(sigma),
                    weight=float(iteration),
                    iteration=int(iteration),
                )
            )
            action = sample_action(sigma, legal, self.rng)
            child = state.child(action)
            try:
                return 1 + self._strategy(child, target_player=target_player, iteration=iteration)
            finally:
                child.close()

        added = 0
        for action in legal:
            child = state.child(action)
            try:
                added += self._strategy(child, target_player=target_player, iteration=iteration)
            finally:
                child.close()
        return added


def chip_delta_utility(state: SolverState) -> tuple[float, float, float]:
    """Structural test utility only; production SpinCore may use continuation value."""
    return tuple(float(x) for x in state.terminal_chip_delta())  # type: ignore[return-value]
