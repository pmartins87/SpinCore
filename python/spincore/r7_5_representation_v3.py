from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from spincore.r7_5_action_cfr import (
    UniversalPartialExactCollector,
    legal_mask,
    regret_matching_policy,
    uniform_policy,
)
from spincore.r7_5_action_contract import ActionCandidateSpec
from spincore.solver_v3 import neural_bytes_v3
from spincore_nn.models_v3_final import (
    collate_v3_observations,
    make_h2_final_v3,
    make_h3_final_v3,
)
from spincore_nn.reservoir import UniformReservoir
from spincore_nn.training import train_step

H2_FINAL = "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL"
H3_FINAL = "H3_HYBRID_EXACT_SEMANTIC_FINAL"
V3_REPRESENTATIONS = (H2_FINAL, H3_FINAL)


def _uses_semantics(representation: str) -> bool:
    if representation == H2_FINAL:
        return False
    if representation == H3_FINAL:
        return True
    raise ValueError(f"unsupported Phase-2 V3 representation {representation!r}")


def _make_model(representation: str, *, device: str, seed: int):
    if representation == H2_FINAL:
        return make_h2_final_v3(device=device, seed=seed)
    if representation == H3_FINAL:
        return make_h3_final_v3(device=device, seed=seed)
    raise ValueError(f"unsupported Phase-2 V3 representation {representation!r}")


class V3NeuralAdvantagePolicy:
    def __init__(
        self,
        model,
        *,
        representation: str,
        device: str = "cpu",
        ready: bool = True,
    ):
        self.model = model
        self.representation = str(representation)
        self.device = device
        self.ready = bool(ready)
        self.with_semantics = _uses_semantics(self.representation)

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]):
        if not self.ready:
            return uniform_policy(state, observation, legal)
        batch = collate_v3_observations(
            [observation],
            [legal_mask(legal)],
            with_semantics=self.with_semantics,
            device=self.device,
        )
        self.model.eval()
        with torch.no_grad():
            raw = self.model(batch)[0].detach().cpu().tolist()
        return regret_matching_policy(raw, legal)


class UniversalPartialExactCollectorV3(UniversalPartialExactCollector):
    """Same audited universal CFR recursion, authoritative SPNNIV3 observation."""

    def __init__(
        self,
        *,
        action_spec: ActionCandidateSpec,
        policy,
        terminal_utility,
        rng,
        advantage_memory,
        strategy_memory,
    ):
        # The parent needs a historically supported representation only to set
        # its old observation-wire tag. Recursion/action semantics are otherwise
        # representation-agnostic. Override observation immediately below.
        super().__init__(
            action_spec=action_spec,
            selected_representation="C0_V1_FROZEN_CONTROL",
            policy=policy,
            terminal_utility=terminal_utility,
            rng=rng,
            advantage_memory=advantage_memory,
            strategy_memory=strategy_memory,
        )
        self._wire = "SPNNIV3"

    def _observation(self, state) -> bytes:
        return neural_bytes_v3(state)


@dataclass
class RepresentationV3Bundle:
    representation: str
    seed: int
    config: object
    advantage: object
    policy: object
    adv_opt: object
    pol_opt: object
    adv_mem: UniformReservoir
    pol_mem: UniformReservoir
    batch_rng: random.Random
    counters: dict


def make_representation_v3_bundle(
    representation: str,
    seed: int,
    *,
    device: str = "cpu",
    reservoir_capacity: int = 100000,
    lr: float = 0.001,
) -> RepresentationV3Bundle:
    representation = str(representation)
    seed = int(seed)
    cfg_a, advantage = _make_model(
        representation, device=device, seed=seed & 0x7FFFFFFF
    )
    cfg_p, policy = _make_model(
        representation,
        device=device,
        seed=(seed ^ 0x5DEECE66D) & 0x7FFFFFFF,
    )
    if cfg_a.to_dict() != cfg_p.to_dict():
        raise RuntimeError("Phase-2 V3 Advantage/Policy config drift")
    return RepresentationV3Bundle(
        representation=representation,
        seed=seed,
        config=cfg_a,
        advantage=advantage,
        policy=policy,
        adv_opt=torch.optim.Adam(advantage.parameters(), lr=float(lr)),
        pol_opt=torch.optim.Adam(policy.parameters(), lr=float(lr)),
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


class RepresentationV3DeepCFRSession:
    def __init__(
        self,
        *,
        solver_library,
        bundle: RepresentationV3Bundle,
        action_spec: ActionCandidateSpec,
        terminal_utility,
        device: str = "cpu",
    ):
        self.solver_library = solver_library
        self.bundle = bundle
        self.action_spec = action_spec
        self.terminal_utility = terminal_utility
        self.device = device
        self.behavior = V3NeuralAdvantagePolicy(
            bundle.advantage,
            representation=bundle.representation,
            device=device,
            ready=bool(bundle.counters.get("advantage_ready", 0)),
        )
        self.collector = UniversalPartialExactCollectorV3(
            action_spec=action_spec,
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
            raise ValueError("nonnegative exact-opponent level required")
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

        c = self.bundle.counters
        c["iteration"] = max(int(c["iteration"]), int(iteration))
        c["roots"] += 1
        c["nodes"] += nodes
        c["advantage_samples"] += advantage_added
        c["strategy_samples"] += strategy_added
        return {
            "nodes": nodes,
            "advantage_samples": advantage_added,
            "strategy_samples": strategy_added,
        }

    def _batch(self, samples):
        batch = collate_v3_observations(
            [sample.observation for sample in samples],
            [sample.legal for sample in samples],
            with_semantics=_uses_semantics(self.bundle.representation),
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
            raise ValueError("empty Phase-2 V3 training memory")
        losses = []
        for _ in range(int(steps)):
            samples = memory.sample(
                min(int(batch_size), len(memory.items)), self.bundle.batch_rng
            )
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
        config, model = _make_model(
            self.bundle.representation,
            device=self.device,
            seed=int(init_seed),
        )
        if config.to_dict() != self.bundle.config.to_dict():
            raise RuntimeError("Phase-2 V3 config drift during Advantage reset")
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_defaults)
        self.bundle.advantage = model
        self.bundle.adv_opt = optimizer
        self.behavior.model = model
        self.behavior.ready = False
        self.bundle.counters["advantage_ready"] = 0
        self.bundle.counters["advantage_resets"] += 1
        return model
