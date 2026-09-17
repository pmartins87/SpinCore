from __future__ import annotations

"""Semantics-preserving concurrent-domain iteration for the lean SpinCore trainer.

The canonical sequential iteration interleaves 3H root collection, Advantage fit,
3H sampled-policy collection, then the same HU phases.  The two Advantage models
and their optimizer/RNG states are domain-local, but the scenario sampler is
shared.  To overlap only the independent Advantage optimizer loops without
changing the sampler stream, this module first pre-samples root and policy
episodes in the exact canonical call order, collects both root batches with the
pre-fit domain models, resets both Advantage networks sequentially, fits 3H/HU
concurrently, then replays sampled-policy episodes in canonical domain order.

This path was admitted only after a read-only iteration-3001 gate demonstrated
exact equality of model/optimizer hashes, domain RNG/counters, reservoir RNGs,
added-sample streams, sampler RNG state and non-timing iteration report versus
the canonical sequential implementation.
"""

from concurrent.futures import ThreadPoolExecutor
import math
import time
from typing import Any

from spincore.lean_functional_training import (
    DOMAINS,
    _advantage_reset_seed,
    _policy_deck_seed,
    _root_deck_seed,
    _root_policy_seed,
)
from spincore.lean_parallel import RootJob
from spincore.lean_solver_actions import LeanSolverState, apply_lean
from spincore.r7_5_action_cfr import ActionStrategySample, legal_mask, sample_action


def _collect_policy_from_episodes(
    *,
    seed: int,
    iteration: int,
    domain: str,
    episodes,
    runtime,
) -> dict[str, Any]:
    before = int(runtime.bundle.pol_mem.seen)
    decisions = 0
    action_counts = [0] * 10
    blind_counts: dict[str, int] = {}
    started = time.perf_counter()

    for episode_index, episode in enumerate(episodes):
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
        "episodes": int(len(episodes)),
        "samples": int(added),
        "seconds": float(time.perf_counter() - started),
        "action_counts": action_counts,
        "blind_counts": blind_counts,
    }


def _presample_plan(*, seed: int, iteration: int, config, sampler):
    counts = config.roots_by_domain()
    policy_counts = config.policy_episodes_by_domain()
    plans = {}

    # Reproduce the canonical sampler call order exactly: domain roots, then
    # that domain's policy episodes, then the next domain.
    for domain in DOMAINS:
        roots = int(counts[domain])
        blind_counts: dict[str, int] = {}
        jobs = []
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
        policy_episodes = [
            sampler.sample_episode(force_domain=domain)
            for _ in range(int(policy_counts[domain]))
        ]
        plans[domain] = {
            "roots": roots,
            "jobs": jobs,
            "blind_counts": blind_counts,
            "policy_episodes": policy_episodes,
        }
    return plans


def run_iteration_concurrent_fit(
    *,
    seed: int,
    iteration: int,
    config,
    sampler,
    runtimes,
    parallel_executor=None,
) -> dict[str, Any]:
    """Run one canonical-equivalent iteration with only 3H/HU fits overlapped."""

    plans = _presample_plan(
        seed=seed,
        iteration=iteration,
        config=config,
        sampler=sampler,
    )
    report: dict[str, Any] = {"iteration": int(iteration), "domains": {}}
    base = {}

    # Both root batches must use the pre-fit model for their own domain, exactly
    # as in the canonical iteration.  Root collection remains sequential by
    # domain because it already uses the persistent 31-process executor.
    for domain in DOMAINS:
        runtime = runtimes[domain]
        plan = plans[domain]
        base[domain] = {
            "nodes_before": int(runtime.bundle.counters["nodes"]),
            "adv_before": int(runtime.bundle.adv_mem.seen),
            "pol_before": int(runtime.bundle.pol_mem.seen),
        }
        if parallel_executor is not None:
            stats = parallel_executor.collect(
                domain=domain,
                bundle=runtime.bundle,
                iteration=int(iteration),
                exact_opponent_levels=int(config.exact_opponent_levels),
                jobs=plan["jobs"],
            )
            tree_seconds = float(stats["seconds"])
            execution_mode = f"parallel_{parallel_executor.workers}x1"
        else:
            started = time.perf_counter()
            for job in plan["jobs"]:
                runtime.session.collect_root(
                    job.episode,
                    iteration=int(iteration),
                    exact_opponent_levels=int(config.exact_opponent_levels),
                    deck_seed=int(job.deck_seed),
                )
            tree_seconds = time.perf_counter() - started
            execution_mode = "serial"
        base[domain]["tree_seconds"] = float(tree_seconds)
        base[domain]["execution_mode"] = execution_mode

    # Model construction/reset uses a forked Torch RNG scope.  Keep resets
    # sequential; only the independent optimizer loops may overlap.
    for domain in DOMAINS:
        runtimes[domain].session.reset_advantage_network(
            init_seed=_advantage_reset_seed(seed, domain, iteration),
            lr=config.learning_rate,
        )

    def fit_domain(domain: str):
        runtime = runtimes[domain]
        started = time.perf_counter()
        losses = runtime.session.train_advantage(
            steps=config.advantage_steps,
            batch_size=config.batch_size,
        )
        elapsed = time.perf_counter() - started
        if not losses or not all(math.isfinite(float(x)) for x in losses):
            raise RuntimeError(f"nonfinite/empty fit: {domain}")
        return list(losses), float(elapsed), dict(runtime.session.last_fit_profile)

    fit_wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="spincore-fit") as pool:
        futures = {domain: pool.submit(fit_domain, domain) for domain in DOMAINS}
        fit_results = {domain: futures[domain].result() for domain in DOMAINS}
    shared_fit_wall = float(time.perf_counter() - fit_wall_started)

    # Replay policy trajectories in canonical domain order.  Scenario sampling
    # already occurred in canonical order, and each domain consumes only its
    # own batch RNG here.
    for domain in DOMAINS:
        runtime = runtimes[domain]
        plan = plans[domain]
        losses, fit_seconds, fit_profile = fit_results[domain]
        policy_report = _collect_policy_from_episodes(
            seed=seed,
            iteration=iteration,
            domain=domain,
            episodes=plan["policy_episodes"],
            runtime=runtime,
        )
        before = base[domain]
        report["domains"][domain] = {
            "roots": int(plan["roots"]),
            "nodes": int(runtime.bundle.counters["nodes"]) - int(before["nodes_before"]),
            "advantage_samples": int(runtime.bundle.adv_mem.seen) - int(before["adv_before"]),
            "strategy_samples": int(runtime.bundle.pol_mem.seen) - int(before["pol_before"]),
            "tree_seconds": float(before["tree_seconds"]),
            "seconds_per_root": float(before["tree_seconds"] / plan["roots"]),
            "execution_mode": before["execution_mode"],
            "advantage_fit_seconds": float(fit_seconds),
            "advantage_fit_profile": fit_profile,
            "advantage_loss_last": float(losses[-1]),
            "blind_counts": dict(plan["blind_counts"]),
            "sampled_policy": policy_report,
        }
    report["concurrent_fit_wall_seconds"] = shared_fit_wall
    return report
