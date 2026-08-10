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
    if not legal:
        raise ValueError("empty legal set")
    legal_set = set(legal)
    out = [0.0] * NUM_ACTIONS
    total = 0.0
    for action, raw in enumerate(policy):
        p = float(raw)
        if not math.isfinite(p) or p < 0.0:
            raise ValueError("invalid probability")
        if action in legal_set:
            out[action] = p
            total += p
        elif p != 0.0:
            raise ValueError("illegal action has mass")
    if total <= 0.0:
        for action in legal:
            out[action] = 1.0 / len(legal)
    else:
        for action in legal:
            out[action] /= total
    return tuple(out)


def uniform_policy(_state, _observation, legal):
    out = [0.0] * NUM_ACTIONS
    for action in legal:
        out[action] = 1.0 / len(legal)
    return tuple(out)


def regret_matching_policy(advantages: Sequence[float], legal: tuple[int, ...]) -> Policy:
    if len(advantages) != NUM_ACTIONS:
        raise ValueError("six advantages required")
    out = [0.0] * NUM_ACTIONS
    total = sum(max(0.0, float(advantages[action])) for action in legal)
    if total <= 0.0:
        for action in legal:
            out[action] = 1.0 / len(legal)
    else:
        for action in legal:
            out[action] = max(0.0, float(advantages[action])) / total
    return tuple(out)


def sample_action(policy, legal, rng):
    probs = _validate_policy(policy, legal)
    x = rng.random()
    acc = 0.0
    for action in legal:
        acc += probs[action]
        if x < acc:
            return action
    return legal[-1]


@dataclass(frozen=True)
class TraversalResult:
    utility: float
    nodes: int
    samples_added: int


class ExternalSamplingCollector:
    def __init__(self, *, policy, terminal_utility, rng, advantage_memory, strategy_memory):
        self.policy = policy
        self.terminal_utility = terminal_utility
        self.rng = rng
        self.advantage_memory = advantage_memory
        self.strategy_memory = strategy_memory

    def _p(self, state, observation, legal):
        return _validate_policy(self.policy(state, observation, legal), legal)

    def collect_advantage(self, root, *, traverser, iteration):
        if iteration <= 0:
            raise ValueError("positive iteration required")
        utility, nodes, added = self._adv(root, traverser, iteration)
        return TraversalResult(utility, nodes, added)

    def _adv(self, state, traverser, iteration):
        if state.terminal:
            return float(self.terminal_utility(state)[traverser]), 1, 0
        actor = state.actor
        legal = state.legal_actions()
        observation = state.neural_bytes()
        sigma = self._p(state, observation, legal)
        if actor == traverser:
            values = [0.0] * NUM_ACTIONS
            nodes = 1
            added = 0
            for action in legal:
                child = state.child(action)
                try:
                    value, child_nodes, child_added = self._adv(child, traverser, iteration)
                finally:
                    child.close()
                values[action] = value
                nodes += child_nodes
                added += child_added
            node_value = sum(sigma[action] * values[action] for action in legal)
            target = [0.0] * NUM_ACTIONS
            for action in legal:
                target[action] = values[action] - node_value
            self.advantage_memory.add(
                AdvantageSample(
                    observation,
                    tuple(1 if action in legal else 0 for action in range(NUM_ACTIONS)),
                    tuple(target),
                    float(iteration),
                    iteration,
                )
            )
            return node_value, nodes, added + 1
        action = sample_action(sigma, legal, self.rng)
        child = state.child(action)
        try:
            value, nodes, added = self._adv(child, traverser, iteration)
        finally:
            child.close()
        return value, nodes + 1, added

    def collect_strategy_own_reach(self, root, *, target_player, iteration):
        if iteration <= 0:
            raise ValueError("positive iteration required")
        return self._strategy(root, target_player, iteration)

    def _strategy(self, state, target_player, iteration):
        if state.terminal:
            return 0
        actor = state.actor
        legal = state.legal_actions()
        observation = state.neural_bytes()
        sigma = self._p(state, observation, legal)
        if actor == target_player:
            self.strategy_memory.add(
                StrategySample(
                    observation,
                    tuple(1 if action in legal else 0 for action in range(NUM_ACTIONS)),
                    tuple(sigma),
                    float(iteration),
                    iteration,
                )
            )
            action = sample_action(sigma, legal, self.rng)
            child = state.child(action)
            try:
                return 1 + self._strategy(child, target_player, iteration)
            finally:
                child.close()
        total = 0
        for action in legal:
            child = state.child(action)
            try:
                total += self._strategy(child, target_player, iteration)
            finally:
                child.close()
        return total


def chip_delta_utility(state):
    return tuple(float(x) for x in state.terminal_chip_delta())


def icm_delta_utility(payout_by_place: Sequence[float]) -> TerminalUtility:
    payouts = tuple(float(x) for x in payout_by_place)
    if len(payouts) != 3:
        raise ValueError("three payouts required")
    return lambda state: state.terminal_icm_delta(payouts)


class NeuralAdvantagePolicy:
    """Regret-matching behavior with an explicit zero-regret bootstrap state.

    An untrained random neural network is not a valid representation of the
    initial CFR regret table: initial cumulative regret is exactly zero, so the
    first behavior policy must be uniform over legal actions.  `ready=False`
    enforces that invariant until the first fitted AdvantageNet exists.
    """

    def __init__(self, model, *, device="cpu", ready: bool = True):
        self.model = model
        self.device = device
        self.ready = bool(ready)

    def __call__(self, _state, observation, legal):
        if not self.ready:
            return uniform_policy(_state, observation, legal)

        import torch
        from spincore_nn.codec import collate_inputs, decode_spnniv1

        batch = collate_inputs([decode_spnniv1(observation)], device=self.device)
        self.model.eval()
        with torch.no_grad():
            raw = self.model(batch)[0].cpu().tolist()
        return regret_matching_policy(raw, legal)


def _batch(samples, device):
    import torch
    from spincore_nn.codec import collate_inputs, decode_spnniv1

    batch = collate_inputs([decode_spnniv1(x.observation) for x in samples], device=device)
    target = torch.tensor([x.target for x in samples], dtype=torch.float32, device=device)
    weights = torch.tensor([x.weight for x in samples], dtype=torch.float32, device=device)
    return batch, target, weights


class DeepCFRDomainSession:
    def __init__(self, *, solver_library, bundle, terminal_utility, device="cpu"):
        self.solver_library = solver_library
        self.bundle = bundle
        self.terminal_utility = terminal_utility
        self.device = device

        for key in (
            "iteration",
            "roots",
            "advantage_samples",
            "strategy_samples",
            "nodes",
            "adv_optimizer_steps",
            "policy_optimizer_steps",
            "advantage_resets",
        ):
            bundle.counters.setdefault(key, 0)
        bundle.counters.setdefault(
            "advantage_ready", 1 if int(bundle.counters.get("adv_optimizer_steps", 0)) > 0 else 0
        )

        self.behavior = NeuralAdvantagePolicy(
            bundle.advantage,
            device=device,
            ready=bool(bundle.counters["advantage_ready"]),
        )
        self.collector = ExternalSamplingCollector(
            policy=self.behavior,
            terminal_utility=terminal_utility,
            rng=bundle.batch_rng,
            advantage_memory=bundle.adv_mem,
            strategy_memory=bundle.pol_mem,
        )

    def collect_root(self, episode, *, iteration, deck_seed=None):
        if deck_seed is None:
            deck_seed = self.bundle.batch_rng.getrandbits(64)
        live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
        nodes = advantage_added = strategy_added = 0
        for player in live:
            root = self.solver_library.create(episode, deck_seed)
            try:
                result = self.collector.collect_advantage(root, traverser=player, iteration=iteration)
            finally:
                root.close()
            nodes += result.nodes
            advantage_added += result.samples_added
        for player in live:
            root = self.solver_library.create(episode, deck_seed)
            try:
                strategy_added += self.collector.collect_strategy_own_reach(
                    root, target_player=player, iteration=iteration
                )
            finally:
                root.close()
        counters = self.bundle.counters
        counters["iteration"] = max(counters["iteration"], iteration)
        counters["roots"] += 1
        counters["nodes"] += nodes
        counters["advantage_samples"] += advantage_added
        counters["strategy_samples"] += strategy_added
        return {
            "nodes": nodes,
            "advantage_samples": advantage_added,
            "strategy_samples": strategy_added,
        }

    def _train(self, memory, model, optimizer, kind, steps, batch_size):
        from spincore_nn.training import train_step

        if steps and not memory.items:
            raise ValueError("empty memory")
        losses = []
        for _ in range(steps):
            samples = memory.sample(min(batch_size, len(memory.items)), self.bundle.batch_rng)
            batch, target, weights = _batch(samples, self.device)
            losses.append(train_step(model, optimizer, batch, target, weights, kind))
        return losses

    def train_advantage(self, *, steps, batch_size):
        losses = self._train(
            self.bundle.adv_mem,
            self.bundle.advantage,
            self.bundle.adv_opt,
            "advantage",
            steps,
            batch_size,
        )
        self.bundle.counters["adv_optimizer_steps"] += len(losses)
        if losses:
            self.behavior.ready = True
            self.bundle.counters["advantage_ready"] = 1
        return losses

    def train_average_policy(self, *, steps, batch_size):
        losses = self._train(
            self.bundle.pol_mem,
            self.bundle.policy,
            self.bundle.pol_opt,
            "strategy",
            steps,
            batch_size,
        )
        self.bundle.counters["policy_optimizer_steps"] += len(losses)
        return losses

    def reset_advantage_network(self, *, init_seed: int, lr: float | None = None):
        """Reinitialize AdvantageNet and optimizer without clearing memories.

        Deep CFR fits a fresh advantage approximator after each CFR iteration.
        Between reset and fit the neural model is explicitly *not* a valid regret
        approximator, so behavior reverts to zero-regret uniform until training
        marks the new model ready.
        """
        import torch
        from spincore_nn import AdvantageNet

        optimizer_defaults = dict(self.bundle.adv_opt.defaults)
        if lr is not None:
            optimizer_defaults["lr"] = float(lr)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            model = AdvantageNet(self.bundle.config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_defaults)
        self.bundle.advantage = model
        self.bundle.adv_opt = optimizer
        self.behavior.model = model
        self.behavior.ready = False
        self.bundle.counters["advantage_ready"] = 0
        self.bundle.counters["advantage_resets"] += 1
        return model
