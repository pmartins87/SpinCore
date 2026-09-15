from __future__ import annotations

"""Lean functional Deep-CFR path for SpinCore.

This is the first training path that combines the pieces we actually want to
keep for the functional agent:

- the legacy empirical SpinGo scenario distribution (3H + HU, all blind levels);
- compact SPNNIV1 exact-state observation;
- the mature legacy seven-action vocabulary represented inside the current
  ten-slot universal resolver;
- external-sampling Deep CFR (exact_opponent_levels=0 by default);
- repaired all-nonpositive advantage fallback;
- WTA chip EV with one global /1500 numeric scale;
- separate 3H and HU brains, trained from their own realistic conditional
  state distributions;
- resumable checkpoints after every iteration.

It intentionally omits the old R7.5 certification ensemble, referee matrix,
bootstrap gates and fixed 10/20 scenario cycle.  Those mechanisms do not make
the first functional agent play better.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from spincore.lean_action_policy import LeanNeuralActionAdvantagePolicy
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_training_scope import LeanTrainingScope
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import UniversalPartialExactCollector
from spincore.r7_5_action_training import ActionDeepCFRSession, make_action_bundle
from spincore.solver import SolverLibrary
from spincore_nn.reservoir import UniformReservoir

SCHEMA = "SPINCORE_LEAN_FUNCTIONAL_TRAINING_V1"
REPRESENTATION = "C0_V1_FROZEN_CONTROL"
DOMAINS = ("THREE_HANDED", "TRUE_HEADS_UP")


@dataclass(frozen=True)
class LeanFunctionalConfig:
    iterations: int = 1
    roots_per_iteration: int = 2
    exact_opponent_levels: int = 0
    reservoir_capacity: int = 100_000
    advantage_steps: int = 2
    policy_steps: int = 2
    batch_size: int = 64
    learning_rate: float = 1e-3
    heads_up_prob: float = 0.4548

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.roots_per_iteration < 2:
            raise ValueError("need positive iterations and at least two roots per iteration")
        if self.exact_opponent_levels < 0:
            raise ValueError("exact_opponent_levels must be nonnegative")
        if self.reservoir_capacity <= 0 or self.batch_size <= 0:
            raise ValueError("reservoir_capacity and batch_size must be positive")
        if self.advantage_steps < 0 or self.policy_steps < 0:
            raise ValueError("optimizer steps must be nonnegative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.heads_up_prob < 1.0:
            raise ValueError("heads_up_prob must be strictly between 0 and 1")

    def roots_by_domain(self) -> dict[str, int]:
        hu = int(round(self.roots_per_iteration * self.heads_up_prob))
        hu = max(1, min(self.roots_per_iteration - 1, hu))
        return {
            "TRUE_HEADS_UP": hu,
            "THREE_HANDED": self.roots_per_iteration - hu,
        }


@dataclass
class DomainRuntime:
    bundle: Any
    session: ActionDeepCFRSession


def _domain_seed(seed: int, domain: str) -> int:
    return int(seed) ^ (0x13579BDF if domain == "THREE_HANDED" else 0x2468ACE0)


def _advantage_reset_seed(seed: int, domain: str, iteration: int) -> int:
    return (_domain_seed(seed, domain) ^ (int(iteration) * 0x45D9F3B)) & 0x7FFFFFFF


def _make_runtime(
    solver: SolverLibrary,
    *,
    seed: int,
    domain: str,
    config: LeanFunctionalConfig,
) -> DomainRuntime:
    bundle = make_action_bundle(
        _domain_seed(seed, domain),
        domain=domain,
        selected_representation=REPRESENTATION,
        action_spec=FIRST_RELEASE_ACTION_SPEC,
        device="cpu",
        reservoir_capacity=config.reservoir_capacity,
        lr=config.learning_rate,
    )
    scope = LeanTrainingScope()
    session = ActionDeepCFRSession(
        solver_library=solver,
        bundle=bundle,
        action_spec=FIRST_RELEASE_ACTION_SPEC,
        terminal_utility=scope.terminal_utility,
        device="cpu",
    )
    # Replace only the behavior-policy adapter.  The audited universal-action
    # recursion remains unchanged; this restores the repaired legacy fallback.
    behavior = LeanNeuralActionAdvantagePolicy(
        bundle.advantage,
        selected_representation=REPRESENTATION,
        device="cpu",
        ready=bool(bundle.counters.get("advantage_ready", 0)),
    )
    session.behavior = behavior
    session.collector.policy = behavior
    return DomainRuntime(bundle=bundle, session=session)


def _bundle_state(bundle) -> dict[str, Any]:
    return {
        "domain": bundle.domain,
        "seed": int(bundle.seed),
        "config": bundle.config.to_dict(),
        "advantage": bundle.advantage.state_dict(),
        "policy": bundle.policy.state_dict(),
        "adv_opt": bundle.adv_opt.state_dict(),
        "pol_opt": bundle.pol_opt.state_dict(),
        "adv_mem": bundle.adv_mem.state_dict(),
        "pol_mem": bundle.pol_mem.state_dict(),
        "batch_rng": bundle.batch_rng.getstate(),
        "counters": dict(bundle.counters),
    }


def _restore_bundle(runtime: DomainRuntime, state: dict[str, Any]) -> None:
    bundle = runtime.bundle
    if state["domain"] != bundle.domain or int(state["seed"]) != int(bundle.seed):
        raise ValueError("lean checkpoint domain/seed mismatch")
    if dict(state["config"]) != bundle.config.to_dict():
        raise ValueError("lean checkpoint network config mismatch")
    bundle.advantage.load_state_dict(state["advantage"])
    bundle.policy.load_state_dict(state["policy"])
    bundle.adv_opt.load_state_dict(state["adv_opt"])
    bundle.pol_opt.load_state_dict(state["pol_opt"])
    bundle.adv_mem = UniformReservoir.from_state_dict(state["adv_mem"])
    bundle.pol_mem = UniformReservoir.from_state_dict(state["pol_mem"])
    bundle.batch_rng.setstate(state["batch_rng"])
    bundle.counters = dict(state["counters"])
    runtime.session.bundle = bundle
    runtime.session.collector.advantage_memory = bundle.adv_mem
    runtime.session.collector.strategy_memory = bundle.pol_mem
    runtime.session.collector.rng = bundle.batch_rng
    runtime.session.behavior.model = bundle.advantage
    runtime.session.behavior.ready = bool(bundle.counters.get("advantage_ready", 0))


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    path: str | Path,
    *,
    seed: int,
    config: LeanFunctionalConfig,
    completed_iteration: int,
    sampler: LegacyScenarioSampler,
    runtimes: dict[str, DomainRuntime],
    history: list[dict[str, Any]],
    finalized: bool,
) -> None:
    payload = {
        "schema": SCHEMA,
        "seed": int(seed),
        "config": asdict(config),
        "completed_iteration": int(completed_iteration),
        "sampler_rng_state": sampler.rng.bit_generator.state,
        "domains": {domain: _bundle_state(runtime.bundle) for domain, runtime in runtimes.items()},
        "history": list(history),
        "finalized": bool(finalized),
        "representation": REPRESENTATION,
        "action_candidate": FIRST_RELEASE_ACTION_SPEC.candidate_id,
        "utility": asdict(LeanTrainingScope()),
    }
    _atomic_torch_save(payload, Path(path))


def load_checkpoint(
    path: str | Path,
    *,
    solver: SolverLibrary,
) -> tuple[int, LeanFunctionalConfig, int, LegacyScenarioSampler, dict[str, DomainRuntime], list[dict[str, Any]], bool]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise ValueError("wrong lean functional checkpoint schema")
    if payload.get("representation") != REPRESENTATION:
        raise ValueError("lean checkpoint representation drift")
    if payload.get("action_candidate") != FIRST_RELEASE_ACTION_SPEC.candidate_id:
        raise ValueError("lean checkpoint action scope drift")
    seed = int(payload["seed"])
    config = LeanFunctionalConfig(**dict(payload["config"]))
    sampler = LegacyScenarioSampler(
        seed=seed ^ 0xA0F5A0F5,
        config=LegacyScenarioConfig(heads_up_prob=config.heads_up_prob),
    )
    sampler.rng.bit_generator.state = payload["sampler_rng_state"]
    runtimes = {
        domain: _make_runtime(solver, seed=seed, domain=domain, config=config)
        for domain in DOMAINS
    }
    for domain in DOMAINS:
        _restore_bundle(runtimes[domain], payload["domains"][domain])
    return (
        seed,
        config,
        int(payload["completed_iteration"]),
        sampler,
        runtimes,
        list(payload.get("history") or []),
        bool(payload.get("finalized")),
    )


def new_run(
    solver: SolverLibrary,
    *,
    seed: int,
    config: LeanFunctionalConfig,
) -> tuple[LegacyScenarioSampler, dict[str, DomainRuntime]]:
    sampler = LegacyScenarioSampler(
        seed=seed ^ 0xA0F5A0F5,
        config=LegacyScenarioConfig(heads_up_prob=config.heads_up_prob),
    )
    runtimes = {
        domain: _make_runtime(solver, seed=seed, domain=domain, config=config)
        for domain in DOMAINS
    }
    return sampler, runtimes


def run_iteration(
    *,
    seed: int,
    iteration: int,
    config: LeanFunctionalConfig,
    sampler: LegacyScenarioSampler,
    runtimes: dict[str, DomainRuntime],
) -> dict[str, Any]:
    counts = config.roots_by_domain()
    report: dict[str, Any] = {"iteration": int(iteration), "domains": {}}

    for domain in DOMAINS:
        runtime = runtimes[domain]
        roots = int(counts[domain])
        nodes_before = int(runtime.bundle.counters["nodes"])
        adv_before = int(runtime.bundle.adv_mem.seen)
        pol_before = int(runtime.bundle.pol_mem.seen)
        blind_counts: dict[str, int] = {}
        started = time.perf_counter()
        for local_root in range(roots):
            episode = sampler.sample_episode(force_domain=domain)
            key = f"{episode.small_blind}/{episode.big_blind}"
            blind_counts[key] = blind_counts.get(key, 0) + 1
            deck_seed = (
                int(seed)
                ^ (int(iteration) * 0x9E3779B1)
                ^ (_domain_seed(seed, domain) << 1)
                ^ int(local_root)
            ) & ((1 << 63) - 1)
            runtime.session.collect_root(
                episode,
                iteration=int(iteration),
                exact_opponent_levels=int(config.exact_opponent_levels),
                deck_seed=deck_seed,
            )
        tree_seconds = time.perf_counter() - started

        fit_started = time.perf_counter()
        runtime.session.reset_advantage_network(
            init_seed=_advantage_reset_seed(seed, domain, iteration),
            lr=config.learning_rate,
        )
        adv_losses = runtime.session.train_advantage(
            steps=config.advantage_steps,
            batch_size=config.batch_size,
        )
        fit_seconds = time.perf_counter() - fit_started

        report["domains"][domain] = {
            "roots": roots,
            "nodes": int(runtime.bundle.counters["nodes"]) - nodes_before,
            "advantage_samples": int(runtime.bundle.adv_mem.seen) - adv_before,
            "strategy_samples": int(runtime.bundle.pol_mem.seen) - pol_before,
            "tree_seconds": float(tree_seconds),
            "seconds_per_root": float(tree_seconds / roots),
            "advantage_fit_seconds": float(fit_seconds),
            "advantage_loss_last": float(adv_losses[-1]) if adv_losses else None,
            "blind_counts": blind_counts,
        }
    return report


def finalize(
    *,
    config: LeanFunctionalConfig,
    runtimes: dict[str, DomainRuntime],
) -> dict[str, Any]:
    out: dict[str, Any] = {"domains": {}}
    for domain in DOMAINS:
        runtime = runtimes[domain]
        started = time.perf_counter()
        losses = runtime.session.train_average_policy(
            steps=config.policy_steps,
            batch_size=config.batch_size,
        )
        out["domains"][domain] = {
            "policy_fit_seconds": float(time.perf_counter() - started),
            "policy_loss_last": float(losses[-1]) if losses else None,
            "roots": int(runtime.bundle.counters["roots"]),
            "nodes": int(runtime.bundle.counters["nodes"]),
            "advantage_seen": int(runtime.bundle.adv_mem.seen),
            "strategy_seen": int(runtime.bundle.pol_mem.seen),
        }
    return out


def compact_report(
    *,
    seed: int,
    config: LeanFunctionalConfig,
    history: list[dict[str, Any]],
    final: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "SPINCORE_LEAN_FUNCTIONAL_REPORT_V1",
        "seed": int(seed),
        "config": asdict(config),
        "representation": REPRESENTATION,
        "action_scope": FIRST_RELEASE_ACTION_SPEC.candidate_id,
        "utility_id": LeanTrainingScope().utility_id,
        "utility_scale_id": LeanTrainingScope().utility_scale_id,
        "history": history,
        "final": final,
    }


def write_json_report(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
