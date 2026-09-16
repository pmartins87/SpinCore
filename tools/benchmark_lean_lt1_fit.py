#!/usr/bin/env python3
"""Finite fit benchmark on real LT1 reservoirs; source checkpoint is read-only.

Ten cases: reference/vectorized x 1/2/4/8/16 threads. Same reset seed,
random.sample sequence, full configured steps/batch and unchanged loss/Adam.
No policy selection and no long-training auto-launch. One resumed iteration
checks integration; original LT1 is never saved over.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import torch
from spincore.lean_functional_training import (
    DOMAINS, _advantage_reset_seed, load_checkpoint, run_iteration,
)
from spincore.lean_parallel import ParallelRootExecutor
from spincore.solver import SolverLibrary


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_digest(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=31)
    p.add_argument("--threads", default="8,1,2,4,16")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.checkpoint.resolve(strict=True)
    if args.out.resolve() == source.parent:
        raise ValueError("benchmark output must be separate from LT1")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    threads = list(dict.fromkeys(int(t) for t in args.threads.split(",")))
    cpus = os.cpu_count() or 1
    if any(t < 1 or t > cpus for t in threads):
        raise ValueError("thread candidate outside CPU budget")
    source_hash = digest_file(source)
    print("LOADING_CHECKPOINT read_only=" + str(source), flush=True)
    solver = SolverLibrary(args.solver)
    seed, config, completed, sampler, runtimes, history, finalized = load_checkpoint(source, solver=solver)
    if not finalized or completed != config.iterations:
        raise ValueError("expected a completed, finalized LT1 checkpoint")
    # Only small model/optimizer/RNG state is copied, not the multi-GB reservoirs.
    snapshots = {}
    for domain, runtime in runtimes.items():
        b = runtime.bundle
        snapshots[domain] = copy.deepcopy(dict(
            advantage=b.advantage.state_dict(), adv_opt=b.adv_opt.state_dict(),
            batch_rng=b.batch_rng.getstate(), counters=b.counters,
        ))
        samples = b.adv_mem.sample(min(config.batch_size, len(b.adv_mem.items)), random.Random(19))
        runtime.session.batch_mode = "reference"
        a = runtime.session._batch(samples)
        runtime.session.batch_mode = "vectorized"
        z = runtime.session._batch(samples)
        for key in a[0]:
            if not torch.equal(a[0][key], z[0][key]):
                raise RuntimeError(f"tensor parity failed: {domain}/{key}")
        if not torch.equal(a[1], z[1]) or not torch.equal(a[2], z[2]):
            raise RuntimeError("target/weight parity failed")
    rows = []
    started = time.perf_counter()
    for index, count in enumerate(threads):
        torch.set_num_threads(count)
        modes = ("reference", "vectorized") if index % 2 == 0 else ("vectorized", "reference")
        pair = {}
        for mode in modes:
            row = dict(threads=count, batch_mode=mode, domains={})
            for domain in DOMAINS:
                runtime = runtimes[domain]
                session, b = runtime.session, runtime.bundle
                snap = snapshots[domain]
                session.batch_mode = mode
                # Warm kernels on disposable model state, then restore reset/RNG.
                for warm in (True, False):
                    session.reset_advantage_network(init_seed=_advantage_reset_seed(seed, domain, completed + 1), lr=config.learning_rate)
                    b.batch_rng.setstate(snap["batch_rng"])
                    b.counters = dict(snap["counters"])
                    fit_started = time.perf_counter()
                    losses = session.train_advantage(steps=2 if warm else config.advantage_steps, batch_size=config.batch_size)
                    seconds = time.perf_counter() - fit_started
                    if not all(math.isfinite(x) for x in losses):
                        raise RuntimeError("nonfinite fit loss")
                row["domains"][domain] = dict(seconds=seconds, profile=dict(session.last_fit_profile), loss_last=losses[-1], model_sha256=model_digest(b.advantage))
                if b.batch_rng.getstate() != pair.get(("rng", domain), b.batch_rng.getstate()):
                    raise RuntimeError("batch RNG drift")
                pair[("rng", domain)] = b.batch_rng.getstate()
            row["fit_seconds"] = sum(d["seconds"] for d in row["domains"].values())
            pair[mode] = row
            rows.append(row)
            print("FIT_CASE " + json.dumps(row, sort_keys=True), flush=True)
        for domain in DOMAINS:
            if pair["reference"]["domains"][domain]["model_sha256"] != pair["vectorized"]["domains"][domain]["model_sha256"]:
                raise RuntimeError(f"optimizer/model parity failed at {count} threads: {domain}")
    baseline = next((r for r in rows if r["threads"] == 8 and r["batch_mode"] == "reference"), rows[0])
    fastest = min(rows, key=lambda r: r["fit_seconds"])
    # <5% gain is inconclusive: keep baseline, stop rather than repeat tuning.
    selected = fastest if fastest["fit_seconds"] <= baseline["fit_seconds"] * 0.95 else baseline
    torch.set_num_threads(selected["threads"])
    for domain, runtime in runtimes.items():
        b, snap = runtime.bundle, snapshots[domain]
        b.advantage.load_state_dict(snap["advantage"])
        b.adv_opt.load_state_dict(snap["adv_opt"])
        b.batch_rng.setstate(snap["batch_rng"])
        b.counters = dict(snap["counters"])
        runtime.session.behavior.ready = bool(b.counters["advantage_ready"])
        runtime.session.batch_mode = selected["batch_mode"]
    print("RESUME_CHECK iteration=" + str(completed + 1), flush=True)
    executor = ParallelRootExecutor(args.solver, args.workers) if args.workers > 1 else None
    try:
        resumed = run_iteration(seed=seed, iteration=completed + 1, config=config, sampler=sampler, runtimes=runtimes, parallel_executor=executor)
    finally:
        if executor is not None:
            executor.close()
    if sum(d["roots"] for d in resumed["domains"].values()) != config.roots_per_iteration:
        raise RuntimeError("resumed root count mismatch")
    if digest_file(source) != source_hash:
        raise RuntimeError("source checkpoint changed during benchmark")
    result = dict(schema="LT1_FIT_BENCHMARK_V1", source=str(source), source_sha256=source_hash,
                  source_iteration=completed, config=config.__dict__, torch_version=torch.__version__,
                  solver_sha256=digest_file(args.solver), baseline=dict(threads=baseline["threads"], batch_mode=baseline["batch_mode"]),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  rows=rows, selected=dict(threads=selected["threads"], batch_mode=selected["batch_mode"]),
                  fit_speedup=baseline["fit_seconds"] / selected["fit_seconds"],
                  resume_iteration=resumed, wall_seconds=time.perf_counter()-started,
                  source_unchanged=True, status="LT1_FIT_BENCHMARK_PASS")
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print("LT1_FIT_BENCHMARK_PASS", flush=True)
    print("selected=" + json.dumps(result["selected"]) + f" fit_speedup={result['fit_speedup']:.3f}", flush=True)
    print("report=" + str(args.out / "report.json"), flush=True)
    print("Original LT1 unchanged. No long training started; send report.json for LT2 decision.", flush=True)


if __name__ == "__main__":
    main()
