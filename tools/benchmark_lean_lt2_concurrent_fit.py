#!/usr/bin/env python3
from __future__ import annotations

"""Bounded read-only benchmark for concurrent 3H/HU Advantage fitting.

The source checkpoint is never modified.  Both domain networks are reset
sequentially with the exact production reset seeds, then only the independent
optimizer loops are allowed to overlap.  Same-thread sequential/concurrent
model hashes, losses and per-domain batch RNG states must match exactly.

This benchmark does NOT authorize concurrent training by itself.  It only asks
whether there is enough measured fit-wall gain to justify implementing and
validating a semantics-preserving full-iteration candidate.
"""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import torch

from spincore.lean_functional_training import DOMAINS, _advantage_reset_seed, load_checkpoint
from spincore.solver import SolverLibrary


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_digest(model) -> str:
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", default="4,8")
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    source = args.checkpoint.resolve(strict=True)
    args.solver = args.solver.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    threads = list(dict.fromkeys(int(v) for v in args.threads.split(",")))
    cpus = os.cpu_count() or 1
    if any(v < 1 or v > cpus for v in threads):
        raise ValueError("thread candidate outside CPU budget")

    source_hash = digest_file(source)
    print(f"LOADING_CHECKPOINT read_only={source}", flush=True)
    solver = SolverLibrary(args.solver)
    seed, config, completed, _sampler, runtimes, _history, finalized = load_checkpoint(source, solver=solver)
    if not finalized or completed != config.iterations:
        raise ValueError("expected completed finalized checkpoint")

    snapshots = {}
    for domain, runtime in runtimes.items():
        b = runtime.bundle
        runtime.session.batch_mode = "vectorized"
        snapshots[domain] = copy.deepcopy({
            "batch_rng": b.batch_rng.getstate(),
            "counters": dict(b.counters),
        })

    def prepare_domain(domain: str) -> None:
        runtime = runtimes[domain]
        b = runtime.bundle
        snap = snapshots[domain]
        b.batch_rng.setstate(snap["batch_rng"])
        b.counters = dict(snap["counters"])
        runtime.session.batch_mode = "vectorized"
        runtime.session.reset_advantage_network(
            init_seed=_advantage_reset_seed(seed, domain, completed + 1),
            lr=config.learning_rate,
        )

    def fit_domain(domain: str) -> dict:
        runtime = runtimes[domain]
        started = time.perf_counter()
        losses = runtime.session.train_advantage(
            steps=config.advantage_steps,
            batch_size=config.batch_size,
        )
        seconds = time.perf_counter() - started
        if not losses or not all(math.isfinite(float(x)) for x in losses):
            raise RuntimeError(f"nonfinite/empty fit: {domain}")
        return {
            "seconds": float(seconds),
            "loss_last": float(losses[-1]),
            "model_sha256": model_digest(runtime.bundle.advantage),
            "batch_rng_state": runtime.bundle.batch_rng.getstate(),
            "profile": dict(runtime.session.last_fit_profile),
        }

    # Warm both domains once, outside all timed trials.
    torch.set_num_threads(max(threads))
    for domain in DOMAINS:
        prepare_domain(domain)
        runtime = runtimes[domain]
        runtime.session.train_advantage(steps=2, batch_size=config.batch_size)

    rows = []
    signatures = {}
    for count in threads:
        torch.set_num_threads(count)
        for mode in ("sequential", "concurrent"):
            trial_seconds = []
            trial_domain_seconds = {domain: [] for domain in DOMAINS}
            signature = None
            for trial in range(args.repeats):
                # Resets are intentionally sequential: model construction uses a
                # forked global Torch RNG scope and must never race across threads.
                for domain in DOMAINS:
                    prepare_domain(domain)
                started = time.perf_counter()
                if mode == "sequential":
                    results = {domain: fit_domain(domain) for domain in DOMAINS}
                else:
                    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="spincore-fit") as pool:
                        futures = {domain: pool.submit(fit_domain, domain) for domain in DOMAINS}
                        results = {domain: futures[domain].result() for domain in DOMAINS}
                elapsed = time.perf_counter() - started
                current_sig = {
                    domain: {
                        "loss_last": results[domain]["loss_last"],
                        "model_sha256": results[domain]["model_sha256"],
                        "batch_rng_state": repr(results[domain]["batch_rng_state"]),
                    }
                    for domain in DOMAINS
                }
                if signature is None:
                    signature = current_sig
                elif current_sig != signature:
                    raise RuntimeError(f"within-case nondeterminism at threads={count} mode={mode}")
                trial_seconds.append(float(elapsed))
                for domain in DOMAINS:
                    trial_domain_seconds[domain].append(float(results[domain]["seconds"]))

            row = {
                "threads": int(count),
                "mode": mode,
                "repeats": int(args.repeats),
                "median_fit_wall_seconds": float(statistics.median(trial_seconds)),
                "trial_fit_wall_seconds": trial_seconds,
                "median_domain_seconds": {
                    domain: float(statistics.median(trial_domain_seconds[domain]))
                    for domain in DOMAINS
                },
                "signature": signature,
            }
            rows.append(row)
            signatures[(count, mode)] = signature
            print("FIT_CONCURRENCY_CASE " + json.dumps({
                k: v for k, v in row.items() if k != "signature"
            }, sort_keys=True), flush=True)

        if signatures[(count, "sequential")] != signatures[(count, "concurrent")]:
            raise RuntimeError(f"sequential/concurrent parity failed at {count} threads")

    baseline = next((r for r in rows if r["threads"] == 8 and r["mode"] == "sequential"), None)
    if baseline is None:
        raise RuntimeError("8-thread sequential baseline is required")
    concurrent_rows = [r for r in rows if r["mode"] == "concurrent"]
    fastest = min(concurrent_rows, key=lambda r: r["median_fit_wall_seconds"])
    speedup = baseline["median_fit_wall_seconds"] / fastest["median_fit_wall_seconds"]
    worth_integrating = fastest["median_fit_wall_seconds"] <= baseline["median_fit_wall_seconds"] * 0.95

    if digest_file(source) != source_hash:
        raise RuntimeError("source checkpoint changed during benchmark")

    report = {
        "schema": "LT2_CONCURRENT_FIT_BENCHMARK_V1",
        "source": str(source),
        "source_sha256": source_hash,
        "source_iteration": int(completed),
        "config": config.__dict__,
        "torch_version": torch.__version__,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "rows": rows,
        "baseline": {"threads": 8, "mode": "sequential"},
        "fastest_concurrent": {
            "threads": fastest["threads"],
            "mode": fastest["mode"],
            "median_fit_wall_seconds": fastest["median_fit_wall_seconds"],
        },
        "fit_speedup_vs_sequential8": float(speedup),
        "worth_full_iteration_integration": bool(worth_integrating),
        "source_unchanged": True,
        "status": "LT2_CONCURRENT_FIT_BENCHMARK_PASS",
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("LT2_CONCURRENT_FIT_BENCHMARK_PASS", flush=True)
    print(f"speedup_vs_sequential8={speedup:.3f}", flush=True)
    print(f"worth_full_iteration_integration={str(worth_integrating).lower()}", flush=True)
    print(f"report={report_path}", flush=True)
    print("Source checkpoint unchanged. No training continuation started.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
