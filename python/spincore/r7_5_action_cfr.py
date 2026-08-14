from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

from spincore.r7_5_action_contract import ActionCandidateSpec, UNIVERSAL_ACTION_COUNT
from spincore_nn.action_models import collate_action_observations, representation_wire

NUM_ACTIONS = UNIVERSAL_ACTION_COUNT
Policy = tuple[float, ...]


@dataclass(frozen=True)
class ActionAdvantageSample:
    observation: bytes
    legal: tuple[int, ...]
    target: tuple[float, ...]
    weight: float
    iteration: int

    def __post_init__(self) -> None:
        if len(self.legal) != NUM_ACTIONS or len(self.target) != NUM_ACTIONS:
            raise ValueError("R7.5.4 advantage sample requires ten actions")


@dataclass(frozen=True)
class ActionStrategySample:
    observation: bytes
    legal: tuple[int, ...]
    target: tuple[float, ...]
    weight: float
    iteration: int

    def __post_init__(self) -> None:
        if len(self.legal) != NUM_ACTIONS or len(self.target) != NUM_ACTIONS:
            raise ValueError("R7.5.4 strategy sample requires ten actions")


@dataclass(frozen=True)
class ActionTraversalResult:
    utility: float
    nodes: int
    samples_added: int


class PolicyProvider(Protocol):
    def __call__(self, state, observation: bytes, legal: tuple[int, ...]) -> Policy: ...


def legal_mask(legal: Sequence[int]) -> tuple[int, ...]:
    legal_set = {int(action) for action in legal}
    if not legal_set or any(action < 0 or action >= NUM_ACTIONS for action in legal_set):
        raise ValueError("invalid universal legal set")
    return tuple(1 if action in legal_set else 0 for action in range(NUM_ACTIONS))


def validate_policy(policy: Sequence[float], legal: tuple[int, ...]) -> Policy:
    if len(policy) != NUM_ACTIONS:
        raise ValueError("universal policy must have ten actions")
    if not legal:
        raise ValueError("empty universal legal set")
    legal_set = set(legal)
    out = [0.0] * NUM_ACTIONS
    total = 0.0
    for action, raw in enumerate(policy):
        probability = float(raw)
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("invalid universal action probability")
        if action in legal_set:
            out[action] = probability
            total += probability
        elif probability != 0.0:
            raise ValueError("illegal universal action has probability mass")
    if total <= 0.0:
        for action in legal:
            out[action] = 1.0 / len(legal)
    else:
        for action in legal:
            out[action] /= total
    return tuple(out)


def uniform_policy(_state, _observation: bytes, legal: tuple[int, ...]) -> Policy:
    out = [0.0] * NUM_ACTIONS
    for action in legal:
        out[action] = 1.0 / len(legal)
    return tuple(out)


def regret_matching_policy(advantages: Sequence[float], legal: tuple[int, ...]) -> Policy:
    if len(advantages) != NUM_ACTIONS:
        raise ValueError("ten universal advantages required")
    out = [0.0] * NUM_ACTIONS
    total = sum(max(0.0, float(advantages[action])) for action in legal)
    if total <= 0.0:
        for action in legal:
            out[action] = 1.0 / len(legal)
    else:
        for action in legal:
            out[action] = max(0.0, float(advantages[action])) / total
    return tuple(out)


def sample_action(policy: Sequence[float], legal: tuple[int, ...], rng) -> int:
    probabilities = validate_policy(policy, legal)
    x = rng.random()
    cumulative = 0.0
    for action in legal:
        cumulative += probabilities[action]
        if x < cumulative:
            return action
    return legal[-1]


class NeuralActionAdvantagePolicy:
    def __init__(
        self,
        model,
        *,
        selected_representation: str,
        device: str = "cpu",
        ready: bool = True,
    ):
        self.model = model
        self.selected_representation = str(selected_representation)
        self.device = device
        self.ready = bool(ready)

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]) -> Policy:
        if not self.ready:
            return uniform_policy(state, observation, legal)
        import torch

        batch = collate_action_observations(
            self.selected_representation,
            [observation],
            [legal_mask(legal)],
            device=self.device,
        )
        self.model.eval()
        with torch.no_grad():
            raw = self.model(batch)[0].detach().cpu().tolist()
        return regret_matching_policy(raw, legal)


class UniversalPartialExactCollector:
    """R7.4 partial-exact semantics generalized from 6 to 10 action slots.

    No old collector is modified. Exact child creation is delegated to the
    solver's state-local universal-action resolver, so aliases never enter CFR
    as duplicate actions.

    The collector also records action-geometry telemetry on *advantage traversal
    decision visits only*.  These counters are observational: they never affect
    policy, RNG, recursion or child creation.  Because `legal` is returned after
    state-local exact-action deduplication, the aggressive-branch counter measures
    realized unique neural branches rather than nominal sizing labels.
    """

    def __init__(
        self,
        *,
        action_spec: ActionCandidateSpec,
        selected_representation: str,
        policy: PolicyProvider,
        terminal_utility,
        rng,
        advantage_memory,
        strategy_memory,
    ):
        self.action_spec = action_spec
        self.selected_representation = str(selected_representation)
        self.policy = policy
        self.terminal_utility = terminal_utility
        self.rng = rng
        self.advantage_memory = advantage_memory
        self.strategy_memory = strategy_memory
        self._wire = representation_wire(self.selected_representation)
        self.telemetry = {
            "advantage_decision_visits": 0,
            "nominal_aggressive_branches": 0,
            "effective_unique_aggressive_branches": 0,
        }

    def reset_telemetry(self) -> None:
        for key in self.telemetry:
            self.telemetry[key] = 0

    def telemetry_snapshot(self) -> dict[str, float | int]:
        visits = int(self.telemetry["advantage_decision_visits"])
        nominal = int(self.telemetry["nominal_aggressive_branches"])
        effective = int(self.telemetry["effective_unique_aggressive_branches"])
        return {
            **{key: int(value) for key, value in self.telemetry.items()},
            "nominal_aggressive_branches_per_decision": (
                float(nominal) / visits if visits else 0.0
            ),
            "effective_unique_aggressive_branches_per_decision": (
                float(effective) / visits if visits else 0.0
            ),
        }

    def _observation(self, state) -> bytes:
        return state.neural_bytes() if self._wire == "SPNNIV1" else state.neural_bytes_v2()

    @staticmethod
    def _street(state) -> int:
        payload = state.neural_bytes_v2()
        if len(payload) != 830 or not payload.startswith(b"SPNNIV2\x00"):
            raise RuntimeError("universal action traversal requires valid SPNNIV2 state metadata")
        return int(payload[111 + 1])

    def _active_and_legal(self, state) -> tuple[int, tuple[int, ...]]:
        active_mask = self.action_spec.active_mask(self._street(state))
        legal = state.universal_legal_actions(active_mask)
        if not legal:
            raise RuntimeError("nonterminal universal-action state has no effective legal action")
        return active_mask, legal

    def _record_action_geometry(self, active_mask: int, legal: tuple[int, ...]) -> None:
        # Slots 0/1 are FOLD/CHECK_CALL; every other universal slot is an
        # aggressive primitive. The active count is nominal. The legal count is
        # already state-local deduplicated by the authoritative C++ resolver.
        nominal = sum(1 for action in range(2, NUM_ACTIONS) if int(active_mask) & (1 << action))
        effective = sum(1 for action in legal if int(action) >= 2)
        self.telemetry["advantage_decision_visits"] += 1
        self.telemetry["nominal_aggressive_branches"] += nominal
        self.telemetry["effective_unique_aggressive_branches"] += effective

    def _p(self, state, observation: bytes, legal: tuple[int, ...]) -> Policy:
        return validate_policy(self.policy(state, observation, legal), legal)

    def collect_advantage_partial_exact(
        self,
        root,
        *,
        traverser: int,
        iteration: int,
        exact_opponent_levels: int,
    ) -> ActionTraversalResult:
        if iteration <= 0:
            raise ValueError("positive iteration required")
        if exact_opponent_levels < 0:
            raise ValueError("nonnegative exact_opponent_levels required")
        utility, nodes, added = self._adv_partial(
            root,
            int(traverser),
            int(iteration),
            int(exact_opponent_levels),
            1.0,
        )
        return ActionTraversalResult(float(utility), int(nodes), int(added))

    def _adv_partial(
        self,
        state,
        traverser: int,
        iteration: int,
        exact_levels: int,
        explicit_opponent_mass: float,
    ):
        if state.terminal:
            return float(self.terminal_utility(state)[traverser]), 1, 0

        actor = state.actor
        active_mask, legal = self._active_and_legal(state)
        self._record_action_geometry(active_mask, legal)
        observation = self._observation(state)
        sigma = self._p(state, observation, legal)

        if actor == traverser:
            values = [0.0] * NUM_ACTIONS
            nodes = 1
            added = 0
            for action in legal:
                child = state.child_universal(active_mask, action)
                try:
                    value, child_nodes, child_added = self._adv_partial(
                        child,
                        traverser,
                        iteration,
                        exact_levels,
                        explicit_opponent_mass,
                    )
                finally:
                    child.close()
                values[action] = float(value)
                nodes += int(child_nodes)
                added += int(child_added)

            node_value = sum(float(sigma[action]) * float(values[action]) for action in legal)
            target = [0.0] * NUM_ACTIONS
            for action in legal:
                target[action] = float(values[action]) - float(node_value)
            if explicit_opponent_mass > 0.0:
                self.advantage_memory.add(
                    ActionAdvantageSample(
                        observation=observation,
                        legal=legal_mask(legal),
                        target=tuple(target),
                        weight=float(iteration) * float(explicit_opponent_mass),
                        iteration=int(iteration),
                    )
                )
                added += 1
            return float(node_value), int(nodes), int(added)

        if exact_levels > 0:
            value = 0.0
            nodes = 1
            added = 0
            for action in legal:
                probability = float(sigma[action])
                if probability <= 0.0:
                    continue
                child = state.child_universal(active_mask, action)
                try:
                    child_value, child_nodes, child_added = self._adv_partial(
                        child,
                        traverser,
                        iteration,
                        exact_levels - 1,
                        float(explicit_opponent_mass) * probability,
                    )
                finally:
                    child.close()
                value += probability * float(child_value)
                nodes += int(child_nodes)
                added += int(child_added)
            return float(value), int(nodes), int(added)

        action = sample_action(sigma, legal, self.rng)
        child = state.child_universal(active_mask, action)
        try:
            value, nodes, added = self._adv_partial(
                child,
                traverser,
                iteration,
                0,
                explicit_opponent_mass,
            )
        finally:
            child.close()
        return float(value), int(nodes) + 1, int(added)

    def collect_strategy_own_reach(self, root, *, target_player: int, iteration: int) -> int:
        if iteration <= 0:
            raise ValueError("positive iteration required")
        return self._strategy(root, int(target_player), int(iteration))

    def _strategy(self, state, target_player: int, iteration: int) -> int:
        if state.terminal:
            return 0
        actor = state.actor
        active_mask, legal = self._active_and_legal(state)
        observation = self._observation(state)
        sigma = self._p(state, observation, legal)
        if actor == target_player:
            self.strategy_memory.add(
                ActionStrategySample(
                    observation=observation,
                    legal=legal_mask(legal),
                    target=tuple(sigma),
                    weight=float(iteration),
                    iteration=int(iteration),
                )
            )
            action = sample_action(sigma, legal, self.rng)
            child = state.child_universal(active_mask, action)
            try:
                return 1 + self._strategy(child, target_player, iteration)
            finally:
                child.close()

        total = 0
        for action in legal:
            child = state.child_universal(active_mask, action)
            try:
                total += self._strategy(child, target_player, iteration)
            finally:
                child.close()
        return total
