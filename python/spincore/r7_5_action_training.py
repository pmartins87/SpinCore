from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from spincore.r7_5_action_cfr import (
    NeuralActionAdvantagePolicy,
    UniversalPartialExactCollector,
)
from spincore.r7_5_action_contract import ActionCandidateSpec
from spincore_nn.action_models import (
    collate_action_observations,
    make_action_models,
    make_advantage_action_model,
)
from spincore_nn.reservoir import UniformReservoir
from spincore_nn.training import train_step


@dataclass
class ActionDomainBundle:
    domain: str
    seed: int
    selected_representation: str
    action_candidate: str
    config: object
    advantage: object
    policy: object
    adv_opt: object
    pol_opt: object
    adv_mem: UniformReservoir
    pol_mem: UniformReservoir
    batch_rng: random.Random
    counters: dict


def make_action_bundle(
    seed: int,
    *,
    domain: str,
    selected_representation: str,
    action_spec: ActionCandidateSpec,
    device: str = "cpu",
    reservoir_capacity: int = 100000,
    lr: float = 0.001,
) -> ActionDomainBundle:
    seed = int(seed)
    config, advantage, policy = make_action_models(
        selected_representation,
        device=device,
        advantage_seed=seed & 0x7FFFFFFF,
        policy_seed=(seed ^ 0x5DEECE66D) & 0x7FFFFFFF,
    )
    adv_opt = torch.optim.Adam(advantage.parameters(), lr=float(lr))
    pol_opt = torch.optim.Adam(policy.parameters(), lr=float(lr))
    return ActionDomainBundle(
        domain=str(domain),
        seed=seed,
        selected_representation=str(selected_representation),
        action_candidate=action_spec.candidate_id,
        config=config,
        advantage=advantage,
        policy=policy,
        adv_opt=adv_opt,
        pol_opt=pol_opt,
        adv_mem=UniformReservoir(int(reservoir_capacity), seed ^ 0xA5A5A5A5),
        pol_mem=UniformReservoir(int(reservoir_capacity), seed ^ 0x5A5A5A5A),
        batch_rng=random.Random(seed ^ 0xC0FFEE),
        counters={
            "iteration": 0,
            "roots": 0,
            "nodes": 0,
            "advantage_samples": 0,
            "strategy_samples": 0,
            "adv_optimizer_steps": 0,
            "policy_optimizer_steps": 0,
            "advantage_resets": 0,
            "advantage_ready": 0,
        },
    )


class ActionDeepCFRSession:
    def __init__(
        self,
        *,
        solver_library,
        bundle: ActionDomainBundle,
        action_spec: ActionCandidateSpec,
        terminal_utility,
        device: str = "cpu",
    ):
        if bundle.action_candidate != action_spec.candidate_id:
            raise ValueError("action bundle/spec mismatch")
        self.solver_library = solver_library
        self.bundle = bundle
        self.action_spec = action_spec
        self.terminal_utility = terminal_utility
        self.device = device
        self.behavior = NeuralActionAdvantagePolicy(
            bundle.advantage,
            selected_representation=bundle.selected_representation,
            device=device,
            ready=bool(bundle.counters.get("advantage_ready", 0)),
        )
        self.collector = UniversalPartialExactCollector(
            action_spec=action_spec,
            selected_representation=bundle.selected_representation,
            policy=self.behavior,
            terminal_utility=terminal_utility,
            rng=bundle.batch_rng,
            advantage_memory=bundle.adv_mem,
            strategy_memory=bundle.pol_mem,
        )

    def collect_root(
        self,
        episode,
        *,
        iteration: int,
        exact_opponent_levels: int,
        deck_seed: int | None = None,
    ) -> dict[str, int]:
        if iteration <= 0:
            raise ValueError("positive iteration required")
        if exact_opponent_levels < 0:
            raise ValueError("nonnegative exact_opponent_levels required")
        if deck_seed is None:
            deck_seed = self.bundle.batch_rng.getrandbits(64)
        live = [index for index, stack in enumerate(episode.stacks) if stack > 0]
        nodes = advantage_added = strategy_added = 0
        for player in live:
            root = self.solver_library.create(episode, int(deck_seed))
            try:
                result = self.collector.collect_advantage_partial_exact(
                    root,
                    traverser=int(player),
                    iteration=int(iteration),
                    exact_opponent_levels=int(exact_opponent_levels),
                )
            finally:
                root.close()
            nodes += int(result.nodes)
            advantage_added += int(result.samples_added)
        for player in live:
            root = self.solver_library.create(episode, int(deck_seed))
            try:
                strategy_added += int(
                    self.collector.collect_strategy_own_reach(
                        root,
                        target_player=int(player),
                        iteration=int(iteration),
                    )
                )
            finally:
                root.close()

        counters = self.bundle.counters
        counters["iteration"] = max(int(counters["iteration"]), int(iteration))
        counters["roots"] += 1
        counters["nodes"] += nodes
        counters["advantage_samples"] += advantage_added
        counters["strategy_samples"] += strategy_added
        return {
            "nodes": nodes,
            "advantage_samples": advantage_added,
            "strategy_samples": strategy_added,
        }

    def _batch(self, samples):
        batch = collate_action_observations(
            self.bundle.selected_representation,
            [sample.observation for sample in samples],
            [sample.legal for sample in samples],
            device=self.device,
        )
        target = torch.tensor(
            [sample.target for sample in samples], dtype=torch.float32, device=self.device
        )
        weights = torch.tensor(
            [sample.weight for sample in samples], dtype=torch.float32, device=self.device
        )
        return batch, target, weights

    def _train(self, memory, model, optimizer, kind: str, steps: int, batch_size: int):
        if steps and not memory.items:
            raise ValueError("empty action-training memory")
        losses = []
        for _ in range(int(steps)):
            samples = memory.sample(min(int(batch_size), len(memory.items)), self.bundle.batch_rng)
            batch, target, weights = self._batch(samples)
            losses.append(train_step(model, optimizer, batch, target, weights, kind))
        return losses

    def train_advantage(self, *, steps: int, batch_size: int):
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

    def train_average_policy(self, *, steps: int, batch_size: int):
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
        optimizer_defaults = dict(self.bundle.adv_opt.defaults)
        if lr is not None:
            optimizer_defaults["lr"] = float(lr)
        config, model = make_advantage_action_model(
            self.bundle.selected_representation,
            device=self.device,
            seed=int(init_seed),
        )
        if config.to_dict() != self.bundle.config.to_dict():
            raise RuntimeError("R7.5.4 action model config drift during reset")
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_defaults)
        self.bundle.advantage = model
        self.bundle.adv_opt = optimizer
        self.behavior.model = model
        self.behavior.ready = False
        self.bundle.counters["advantage_ready"] = 0
        self.bundle.counters["advantage_resets"] += 1
        return model
