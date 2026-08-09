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


class NeuralAdvantagePolicy:
    """Current Deep CFR behavior policy obtained by regret matching AdvantageNet output."""
    def __init__(self, model, *, device: str = 'cpu') -> None:
        self.model = model
        self.device = device

    def __call__(self, _state: SolverState, observation: bytes, legal: tuple[int, ...]) -> Policy:
        import torch
        from spincore_nn.codec import decode_spnniv1, collate_inputs
        item = decode_spnniv1(observation)
        batch = collate_inputs([item], device=self.device)
        self.model.eval()
        with torch.no_grad():
            raw = self.model(batch)[0].detach().cpu().tolist()
        return regret_matching_policy(raw, legal)


def _samples_to_training_batch(samples, *, device: str):
    import torch
    from spincore_nn.codec import decode_spnniv1, collate_inputs
    observations = [decode_spnniv1(x.observation) for x in samples]
    batch = collate_inputs(observations, device=device)
    target = torch.tensor([x.target for x in samples], dtype=torch.float32, device=device)
    weights = torch.tensor([x.weight for x in samples], dtype=torch.float32, device=device)
    return batch, target, weights


class DeepCFRDomainSession:
    """Minimal resumable R6 domain session using the R4 DomainBundle contract.

    It intentionally does not encode R7 scheduling/calibration policy. R7 layers
    checkpoint cadence, mid-iteration progress, distributed audits and native
    own-reach frontier performance on top of this correctness core.
    """
    def __init__(self, *, solver_library, bundle, terminal_utility: TerminalUtility, device: str = 'cpu') -> None:
        self.solver_library = solver_library
        self.bundle = bundle
        self.terminal_utility = terminal_utility
        self.device = device
        self.behavior = NeuralAdvantagePolicy(bundle.advantage, device=device)
        self.collector = ExternalSamplingCollector(
            policy=self.behavior,
            terminal_utility=terminal_utility,
            rng=bundle.batch_rng,
            advantage_memory=bundle.adv_mem,
            strategy_memory=bundle.pol_mem,
        )
        self.bundle.counters.setdefault('iteration', 0)
        self.bundle.counters.setdefault('roots', 0)
        self.bundle.counters.setdefault('advantage_samples', 0)
        self.bundle.counters.setdefault('strategy_samples', 0)
        self.bundle.counters.setdefault('nodes', 0)
        self.bundle.counters.setdefault('adv_optimizer_steps', 0)
        self.bundle.counters.setdefault('policy_optimizer_steps', 0)

    def collect_root(self, episode, *, iteration: int, deck_seed: int | None = None) -> dict[str, int]:
        if iteration <= 0:
            raise ValueError('iteration must be positive')
        if deck_seed is None:
            deck_seed = self.bundle.batch_rng.getrandbits(64)
        live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
        if len(live) not in (2, 3):
            raise ValueError('SpinCore root must have two or three live players')

        nodes = adv_added = strat_added = 0
        for player in live:
            root = self.solver_library.create(episode, deck_seed)
            try:
                r = self.collector.collect_advantage(root, traverser=player, iteration=iteration)
            finally:
                root.close()
            nodes += r.nodes
            adv_added += r.samples_added

        for player in live:
            root = self.solver_library.create(episode, deck_seed)
            try:
                strat_added += self.collector.collect_strategy_own_reach(root, target_player=player, iteration=iteration)
            finally:
                root.close()

        c = self.bundle.counters
        c['iteration'] = max(int(c.get('iteration', 0)), int(iteration))
        c['roots'] = int(c.get('roots', 0)) + 1
        c['nodes'] = int(c.get('nodes', 0)) + nodes
        c['advantage_samples'] = int(c.get('advantage_samples', 0)) + adv_added
        c['strategy_samples'] = int(c.get('strategy_samples', 0)) + strat_added
        return {'nodes': nodes, 'advantage_samples': adv_added, 'strategy_samples': strat_added}

    def _train(self, *, memory, model, optimizer, kind: str, steps: int, batch_size: int) -> list[float]:
        from spincore_nn.training import train_step
        if steps < 0 or batch_size <= 0:
            raise ValueError('bad training schedule')
        if steps and not memory.items:
            raise ValueError(f'{kind} memory is empty')
        losses: list[float] = []
        for _ in range(steps):
            samples = memory.sample(min(batch_size, len(memory.items)), self.bundle.batch_rng)
            batch, target, weights = _samples_to_training_batch(samples, device=self.device)
            losses.append(train_step(model, optimizer, batch, target, weights, kind))
        return losses

    def train_advantage(self, *, steps: int, batch_size: int) -> list[float]:
        losses = self._train(memory=self.bundle.adv_mem, model=self.bundle.advantage, optimizer=self.bundle.adv_opt,
                             kind='advantage', steps=steps, batch_size=batch_size)
        self.bundle.counters['adv_optimizer_steps'] += len(losses)
        return losses

    def train_average_policy(self, *, steps: int, batch_size: int) -> list[float]:
        losses = self._train(memory=self.bundle.pol_mem, model=self.bundle.policy, optimizer=self.bundle.pol_opt,
                             kind='strategy', steps=steps, batch_size=batch_size)
        self.bundle.counters['policy_optimizer_steps'] += len(losses)
        return losses
