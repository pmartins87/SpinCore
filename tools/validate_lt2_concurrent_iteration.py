#!/usr/bin/env python3
from __future__ import annotations

"""Validate one full LT2 iteration with concurrent 3H/HU Advantage fitting.

This is a finite, read-only integration gate.  The production trainer is not
modified here.  We run iteration 3001 twice from the same finalized LT2-A
checkpoint:

1. canonical sequential `run_iteration`;
2. a staged candidate that preserves the exact parent scenario stream by
   pre-sampling root + policy episodes in canonical order, collects both domain
   root batches, resets both Advantage networks sequentially, overlaps only the
   two independent optimizer loops, then replays the pre-sampled policy
   episodes in canonical domain order.

The gate compares semantic state rather than timing fields.  It requires exact
parity for model/optimizer hashes, per-domain RNG states/counters, reservoir RNG
states, every sample added during the iteration (streaming SHA-256), sampler RNG
state, and the non-timing iteration report.  The source checkpoint is never
written.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import torch

from spincore.lean_functional_training import (
    DOMAINS,
    _advantage_reset_seed,
    _policy_deck_seed,
    _root_deck_seed,
    _root_policy_seed,
    load_checkpoint,
    run_iteration,
)
from spincore.lean_parallel import ParallelRootExecutor, RootJob
from spincore.lean_solver_actions import LeanSolverState, apply_lean
from spincore.r7_5_action_cfr import ActionStrategySample, legal_mask, sample_action
from spincore.solver import SolverLibrary


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _update_obj(h: "hashlib._Hash", value: Any) -> None:
    if value is None:
        h.update(b"N")
    elif isinstance(value, bool):
        h.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        h.update(b"I" + str(value).encode() + b";")
    elif isinstance(value, float):
        h.update(b"F" + struct.pack(">d", float(value)))
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        h.update(b"S" + len(raw).to_bytes(8, "big") + raw)
    elif isinstance(value, bytes):
        h.update(b"Y" + len(value).to_bytes(8, "big") + value)
    elif torch.is_tensor(value):
        t = value.detach().cpu().contiguous()
        h.update(b"T" + str(t.dtype).encode() + b"|")
        _update_obj(h, tuple(int(x) for x in t.shape))
        h.update(t.numpy().tobytes())
    elif isinstance(value, dict):
        h.update(b"D")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_obj(h, key)
            _update_obj(h, value[key])
        h.update(b"d")
    elif isinstance(value, (list, tuple)):
        h.update(b"L" if isinstance(value, list) else b"Q")
        h.update(len(value).to_bytes(8, "big"))
        for item in value:
            _update_obj(h, item)
    else:
        # The objects hashed by this gate are ordinary checkpoint state values.
        # Fail closed if a new unsupported type appears instead of silently
        # hashing an unstable object repr.
        raise TypeError(f"unsupported digest type: {type(value)!r}")


def object_digest(value: Any) -> str:
    h = hashlib.sha256()
    _update_obj(h, value)
    return h.hexdigest()


def sample_update(h: "hashlib._Hash", sample: Any) -> None:
    h.update(type(sample).__name__.encode("ascii") + b"\0")
    _update_obj(h, sample.observation)
    _update_obj(h, tuple(sample.legal))
    _update_obj(h, tuple(float(x) for x in sample.target))
    _update_obj(h, float(sample.weight))
    _update_obj(h, int(sample.iteration))


def install_add_traces(runtimes) -> dict[str, dict[str, dict[str, Any]]]:
    traces: dict[str, dict[str, dict[str, Any]]] = {}
    for domain in DOMAINS:
        traces[domain] = {}
        bundle = runtimes[domain].bundle
        for label, mem in (("adv", bundle.adv_mem), ("policy", bundle.pol_mem)):
            state = {"count": 0, "hash": hashlib.sha256()}
            original_add = mem.add

            def traced_add(item, *, _state=state, _original=original_add):
                sample_update(_state["hash"], item)
                _state["count"] += 1
                return _original(item)

            mem.add = traced_add
            traces[domain][label] = state
    return traces


def traces_signature(traces) -> dict[str, Any]:
    return {
        domain: {
            label: {
                "count": int(state["count"]),
                "sha256": state["hash"].hexdigest(),
            }
            for label, state in traces[domain].items()
        }
        for domain in DOMAINS
    }


def model_digest(model) -> str:
    return object_digest(model.state_dict())


def semantic_signature(sampler, runtimes, traces, report) -> dict[str, Any]:
    domains = {}
    for domain in DOMAINS:
        b = runtimes[domain].bundle
        domains[domain] = {
            "advantage_model_sha256": model_digest(b.advantage),
            "policy_model_sha256": model_digest(b.policy),
            "adv_optimizer_sha256": object_digest(b.adv_opt.state_dict()),
            "policy_optimizer_sha256": object_digest(b.pol_opt.state_dict()),
            "batch_rng_sha256": object_digest(b.batch_rng.getstate()),
            "counters": dict(b.counters),
            "adv_mem": {
                "seen": int(b.adv_mem.seen),
                "length": int(len(b.adv_mem.items)),
                "rng_sha256": object_digest(b.adv_mem.rng.getstate()),
            },
            "policy_mem": {
                "seen": int(b.pol_mem.seen),
                "length": int(len(b.pol_mem.items)),
                "rng_sha256": object_digest(b.pol_mem.rng.getstate()),
            },
        }
    return {
        "sampler_rng_sha256": object_digest(sampler.rng.bit_generator.state),
        "domains": domains,
        "added_sample_streams": traces_signature(traces),
        "report_semantics": strip_timing(report),
    }


def strip_timing(value: Any) -> Any:
    timing_keys = {
        "seconds",
        "tree_seconds",
        "seconds_per_root",
        "advantage_fit_seconds",
        "advantage_fit_profile",
    }
    if isinstance(value, dict):
        return {
            key: strip_timing(item)
            for key, item in value.items()
            if key not in timing_keys
        }
    if isinstance(value, list):
        return [strip_timing(item) for item in value]
    return value


def collect_policy_from_episodes(*, seed, iteration, domain, episodes, runtime) -> dict[str, Any]:
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


def presample_plan(*, seed, iteration, config, sampler):
    counts = config.roots_by_domain()
    policy_counts = config.policy_episodes_by_domain()
    plans = {}
    # Important: this loop reproduces the canonical sampler call order exactly:
    # domain roots, then that domain's policy episodes, then next domain.
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


def run_candidate(*, seed, iteration, config, sampler, runtimes, parallel_executor):
    plans = presample_plan(
        seed=seed,
        iteration=iteration,
        config=config,
        sampler=sampler,
    )
    report: dict[str, Any] = {"iteration": int(iteration), "domains": {}}
    base = {}

    # Collect both independent root batches with the pre-fit models, exactly as
    # the canonical iteration does for each domain.
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

    # Reset sequentially: model construction uses a forked Torch RNG scope and
    # must not race.  This is the same reset contract used by the microbenchmark.
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

    # Replay policy trajectories in canonical domain order.  Their scenarios
    # were pre-sampled in canonical order, and each domain consumes its own
    # batch RNG only after its Advantage fit, as before.
    for domain in DOMAINS:
        runtime = runtimes[domain]
        plan = plans[domain]
        losses, fit_seconds, fit_profile = fit_results[domain]
        policy_report = collect_policy_from_episodes(
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


def run_once(*, source, solver_path, workers, threads, candidate):
    solver = SolverLibrary(solver_path)
    seed, config, completed, sampler, runtimes, _history, finalized = load_checkpoint(
        source,
        solver=solver,
    )
    if not finalized or completed != config.iterations:
        raise ValueError("expected finalized LT2-A checkpoint")
    iteration = int(completed) + 1
    for runtime in runtimes.values():
        runtime.session.batch_mode = "vectorized"
    torch.set_num_threads(int(threads))
    traces = install_add_traces(runtimes)
    executor = ParallelRootExecutor(solver_path, workers) if workers > 1 else None
    started = time.perf_counter()
    try:
        if candidate:
            report = run_candidate(
                seed=seed,
                iteration=iteration,
                config=config,
                sampler=sampler,
                runtimes=runtimes,
                parallel_executor=executor,
            )
        else:
            report = run_iteration(
                seed=seed,
                iteration=iteration,
                config=config,
                sampler=sampler,
                runtimes=runtimes,
                parallel_executor=executor,
            )
        wall = float(time.perf_counter() - started)
        sig = semantic_signature(sampler, runtimes, traces, report)
        return {
            "wall_seconds": wall,
            "report": report,
            "signature": sig,
        }
    finally:
        if executor is not None:
            executor.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=31)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    source = args.checkpoint.resolve(strict=True)
    solver_path = args.solver.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.workers < 1 or args.threads < 1:
        raise ValueError("workers/threads must be positive")

    source_hash = digest_file(source)
    print(f"SOURCE read_only={source}", flush=True)
    print("RUN reference iteration", flush=True)
    reference = run_once(
        source=source,
        solver_path=solver_path,
        workers=args.workers,
        threads=args.threads,
        candidate=False,
    )
    gc.collect()
    print("RUN concurrent-fit candidate iteration", flush=True)
    candidate = run_once(
        source=source,
        solver_path=solver_path,
        workers=args.workers,
        threads=args.threads,
        candidate=True,
    )

    source_unchanged = digest_file(source) == source_hash
    semantic_parity = reference["signature"] == candidate["signature"]
    result = {
        "schema": "LT2_CONCURRENT_ITERATION_PARITY_V1",
        "source": str(source),
        "source_sha256": source_hash,
        "source_iteration": 3000,
        "tested_iteration": 3001,
        "workers": int(args.workers),
        "threads": int(args.threads),
        "batch_mode": "vectorized",
        "reference_wall_seconds": float(reference["wall_seconds"]),
        "candidate_wall_seconds": float(candidate["wall_seconds"]),
        "candidate_speedup": float(reference["wall_seconds"] / candidate["wall_seconds"]),
        "semantic_parity": bool(semantic_parity),
        "source_unchanged": bool(source_unchanged),
        "reference_signature": reference["signature"],
        "candidate_signature": candidate["signature"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    result["status"] = (
        "LT2_CONCURRENT_ITERATION_PARITY_PASS"
        if semantic_parity and source_unchanged
        else "LT2_CONCURRENT_ITERATION_PARITY_FAIL"
    )
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not source_unchanged:
        raise RuntimeError("source checkpoint changed")
    if not semantic_parity:
        print("LT2_CONCURRENT_ITERATION_PARITY_FAIL", flush=True)
        print(f"report={report_path}", flush=True)
        return 2

    print("LT2_CONCURRENT_ITERATION_PARITY_PASS", flush=True)
    print(f"reference_wall_seconds={reference['wall_seconds']:.3f}", flush=True)
    print(f"candidate_wall_seconds={candidate['wall_seconds']:.3f}", flush=True)
    print(f"candidate_speedup={result['candidate_speedup']:.3f}", flush=True)
    print("source_unchanged=true", flush=True)
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
