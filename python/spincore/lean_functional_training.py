from __future__ import annotations

"""Lean functional Deep-CFR path for SpinCore.

This is the first training path that combines the pieces we actually want to
keep for the functional agent:

- the legacy empirical SpinGo scenario distribution (3H + HU, all blind levels);
- compact SPNNIV1 exact-state observation;
- the mature legacy seven-action vocabulary with its context-sensitive action
  semantics;
- external-sampling Deep CFR (exact_opponent_levels=0 by default);
- legacy-style sampled average-policy collection rather than exact opponent
  expansion;
- repaired all-nonpositive advantage fallback;
- WTA chip EV with one global /1500 numeric scale;
- separate 3H and HU brains, trained from their own realistic conditional
  state distributions;
- resumable checkpoints;
- optional root-level multiprocessing that changes execution throughput only,
  not poker/state/action/utility semantics.

It intentionally omits the old R7.5 certification ensemble, referee matrix,
bootstrap gates and fixed 10/20 scenario cycle. Those mechanisms do not make
the first functional agent play better.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch

from spincore.lean_action_policy import LeanNeuralActionAdvantagePolicy
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_parallel import RootJob
from spincore.lean_solver_actions import LeanSolverState, apply_lean
from spincore.lean_training_scope import LeanTrainingScope
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import ActionStrategySample, legal_mask, sample_action
from spincore.r7_5_action_training import ActionDeepCFRSession, make_action_bundle
from spincore.solver import SolverLibrary
from spincore_nn.reservoir import UniformReservoir

SCHEMA = "SPINCORE_LEAN_FUNCTIONAL_TRAINING_V1"
REPRESENTATION = "C0_V1_FROZEN_CONTROL"
DOMAINS = ("THREE_HANDED", "TRUE_HEADS_UP")

# Historical DeepSpin defaults were 512 advantage traversals per player and 256
# sampled policy episodes per iteration: 3*512 / 256 = 6 advantage traversals
# per policy episode. Current roots traverse every live player, so preserving
# that ratio is a natural scale-independent default.
LEGACY_ADV_TRAVERSALS_PER_POLICY_EPISODE = 6


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

    def policy_episodes_by_domain(self) -> dict[str, int]:
        roots = self.roots_by_domain()
        return {
            "THREE_HANDED": max(
                1,
                int(
                    math.ceil(
                        roots["THREE_HANDED"]
                        * 3
                        / LEGACY_ADV_TRAVERSALS_PER_POLICY_EPISODE
                    )
                ),
            ),
            "TRUE_HEADS_UP": max(
                1,
                int(
                    math.ceil(
                        roots["TRUE_HEADS_UP"]
                        * 2
                        / LEGACY_ADV_TRAVERSALS_PER_POLICY_EPISODE
                    )
                ),
            ),
        }


@dataclass
class DomainRuntime:
    bundle: Any
    session: ActionDeepCFRSession


def _domain_seed(seed: int, domain: str) -> int:
    return int(seed) ^ (0x13579BDF if domain == "THREE_HANDED" else 0x2468ACE0)


def _advantage_reset_seed(seed: int, domain: str, iteration: int) -> int:
    return (_domain_seed(seed, domain) ^ (int(iteration) * 0x45D9F3B)) & 0x7FFFFFFF


def _root_deck_seed(seed: int, domain: str, iteration: int, local_root: int) -> int:
    return (
        int(seed)
        ^ (int(iteration) * 0x9E3779B1)
        ^ (_domain_seed(seed, domain) << 1)
        ^ int(local_root)
    ) & ((1 << 63) - 1)


def _root_policy_seed(seed: int, domain: str, iteration: int, local_root: int) -> int:
    # Parallel roots need independent sampling streams so scheduling/worker
    # count cannot change the stochastic distribution inside a root.
    return (
        int(seed)
        ^ 0xD1B54A32D192ED03
        ^ (int(iteration) * 0x94D049BB133111EB)
        ^ (_domain_seed(seed, domain) << 7)
        ^ (int(local_root) * 0x9E3779B97F4A7C15)
    ) & ((1 << 63) - 1)


def _policy_deck_seed(seed: int, domain: str, iteration: int, episode_index: int) -> int:
    return (
        int(seed)
        ^ 0x6A09E667
        ^ (int(iteration) * 0xBB67AE85)
        ^ (_domain_seed(seed, domain) << 1)
        ^ int(episode_index)
    ) & ((1 << 63) - 1)


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


def _collect_sampled_policy(
    *,
    seed: int,
    iteration: int,
    domain: str,
    episodes: int,
    sampler: LegacyScenarioSampler,
    runtime: DomainRuntime,
) -> dict[str, Any]:
    """Collect AveragePolicy targets along ordinary sampled game trajectories."""
    before = int(runtime.bundle.pol_mem.seen)
    decisions = 0
    action_counts = [0] * 10
    blind_counts: dict[str, int] = {}
    started = time.perf_counter()

    for episode_index in range(int(episodes)):
        episode = sampler.sample_episode(force_domain=domain)
        blind_key = f"{episode.small_blind}/{episode.big_blind}"
        blind_counts[blind_key] = blind_counts.get(blind_key, 0) + 1
        raw = runtime.session.solver_library.create(
            episode,
            _policy_deck_seed(seed, domain, iteration, episode_index),
        )
        state = LeanSolverState(raw)
        try:
            while not state.terminal:
                active_mask, legal = runtime.session.collector._active_and_legal(state)
                observation = runtime.session.collector._observation(state)
                sigma = runtime.session.behavior(state, observation, legal)
                runtime.bundle.pol_mem.add(
                    ActionStrategySample(
                        observation=observation,
                        legal=legal_mask(legal),
                        target=tuple(float(x) for x in sigma),
                        weight=float(iteration),
                        iteration=int(iteration),
                    )
                )
                decisions += 1
                action = sample_action(sigma, legal, runtime.bundle.batch_rng)
                action_counts[int(action)] += 1
                apply_lean(state.inner, active_mask, action)
        finally:
            state.close()

    added = int(runtime.bundle.pol_mem.seen) - before
    if added != decisions:
        raise RuntimeError("sampled policy accounting drift")
    runtime.bundle.counters["strategy_samples"] += added
    return {
        "episodes": int(episodes),
        "samples": int(added),
        "seconds": float(time.perf_counter() - started),
        "action_counts": action_counts,
        "blind_counts": blind_counts,
    }


def run_iteration(
    *,
    seed: int,
    iteration: int,
    config: LeanFunctionalConfig,
    sampler: LegacyScenarioSampler,
    runtimes: dict[str, DomainRuntime],
    parallel_executor=None,
) -> dict[str, Any]:
    counts = config.roots_by_domain()
    policy_counts = config.policy_episodes_by_domain()
    report: dict[str, Any] = {"iteration": int(iteration), "domains": {}}

    for domain in DOMAINS:
        runtime = runtimes[domain]
        roots = int(counts[domain])
        nodes_before = int(runtime.bundle.counters["nodes"])
        adv_before = int(runtime.bundle.adv_mem.seen)
        pol_before = int(runtime.bundle.pol_mem.seen)
        blind_counts: dict[str, int] = {}

        # Sample scenarios in the parent so the empirical scenario stream remains
        # authoritative and independent of worker scheduling.
        jobs: list[RootJob] = []
        for local_root in range(roots):
            episode = sampler.sample_episode(force_domain=domain)
            key = f"{episode.small_blind}/{episode.big_blind}"
            blind_counts[key] = blind_counts.get(key, 0) + 1
            jobs.append(
                RootJob(
                    root_index=int(local_root),
                    episode=episode,
                    deck_seed=_root_deck_seed(seed, domain, iteration, local_root),
                    policy_seed=_root_policy_seed(seed, domain, iteration, local_root),
                )
            )

        if parallel_executor is not None:
            parallel_stats = parallel_executor.collect(
                domain=domain,
                bundle=runtime.bundle,
                iteration=int(iteration),
                exact_opponent_levels=int(config.exact_opponent_levels),
                jobs=jobs,
            )
            tree_seconds = float(parallel_stats["seconds"])
            execution_mode = f"parallel_{parallel_executor.workers}x1"
        else:
            started = time.perf_counter()
            # Historical single-process path retained for portability/CI.
            for job in jobs:
                runtime.session.collect_root(
                    job.episode,
                    iteration=int(iteration),
                    exact_opponent_levels=int(config.exact_opponent_levels),
                    deck_seed=int(job.deck_seed),
                )
            tree_seconds = time.perf_counter() - started
            execution_mode = "serial"

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

        policy_report = _collect_sampled_policy(
            seed=seed,
            iteration=iteration,
            domain=domain,
            episodes=int(policy_counts[domain]),
            sampler=sampler,
            runtime=runtime,
        )

        report["domains"][domain] = {
            "roots": roots,
            "nodes": int(runtime.bundle.counters["nodes"]) - nodes_before,
            "advantage_samples": int(runtime.bundle.adv_mem.seen) - adv_before,
            "strategy_samples": int(runtime.bundle.pol_mem.seen) - pol_before,
            "tree_seconds": float(tree_seconds),
            "seconds_per_root": float(tree_seconds / roots),
            "execution_mode": execution_mode,
            "advantage_fit_seconds": float(fit_seconds),
            "advantage_fit_profile": dict(runtime.session.last_fit_profile),
            "advantage_loss_last": float(adv_losses[-1]) if adv_losses else None,
            "blind_counts": blind_counts,
            "sampled_policy": policy_report,
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
        "roots_by_domain": config.roots_by_domain(),
        "policy_episodes_by_domain": config.policy_episodes_by_domain(),
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
