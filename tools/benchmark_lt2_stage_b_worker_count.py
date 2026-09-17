#!/usr/bin/env python3
from __future__ import annotations

"""Read-only production-shaped worker-count benchmark for LT2 Stage B.

Each worker-count case runs in its own Python process so the 2M reservoirs and
PyTorch/process-pool state are fully released between cases.  Every case starts
from the exact preserved LT2-A iteration-3000 checkpoint, performs one warmup
iteration plus five timed production concurrent-fit iterations, and saves no
checkpoint.

Worker count is an execution-only knob: root results are merged in deterministic
root order.  The benchmark therefore also requires an exact post-run semantic
signature across all tested worker counts before recommending any change.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch

from spincore.lean_concurrent_iteration import run_iteration_concurrent_fit
from spincore.lean_functional_training import DOMAINS, load_checkpoint
from spincore.lean_parallel import ParallelRootExecutor
from spincore.solver import SolverLibrary
from validate_lt2_concurrent_iteration import (
    install_add_traces,
    model_digest,
    object_digest,
    traces_signature,
)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_signature(sampler, runtimes, traces):
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
    }


def run_case(args) -> int:
    source = args.checkpoint.resolve(strict=True)
    solver_path = args.solver.resolve(strict=True)
    workers = int(args.case_workers)
    solver = SolverLibrary(solver_path)
    seed, config, completed, sampler, runtimes, _history, finalized = load_checkpoint(
        source, solver=solver
    )
    if not finalized or completed != 3000:
        raise ValueError("expected finalized LT2-A checkpoint at iteration 3000")
    for runtime in runtimes.values():
        runtime.session.batch_mode = "vectorized"
    torch.set_num_threads(int(args.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    traces = install_add_traces(runtimes)
    executor = ParallelRootExecutor(solver_path, workers)
    rows = []
    try:
        for offset in range(1, int(args.iterations) + 1):
            iteration = completed + offset
            started = time.perf_counter()
            report = run_iteration_concurrent_fit(
                seed=seed,
                iteration=iteration,
                config=config,
                sampler=sampler,
                runtimes=runtimes,
                parallel_executor=executor,
            )
            wall = float(time.perf_counter() - started)
            rows.append({
                "iteration": int(iteration),
                "warmup": bool(offset == 1),
                "wall_seconds": wall,
                "tree_seconds": float(sum(
                    report["domains"][d]["tree_seconds"] for d in DOMAINS
                )),
                "fit_wall_seconds": float(report["concurrent_fit_wall_seconds"]),
                "policy_seconds": float(sum(
                    report["domains"][d]["sampled_policy"]["seconds"] for d in DOMAINS
                )),
            })
    finally:
        executor.close()

    timed = rows[1:]
    result = {
        "workers": workers,
        "threads": int(args.threads),
        "iterations": int(args.iterations),
        "warmup_iterations": 1,
        "timed_iterations": len(timed),
        "median_wall_seconds": float(statistics.median(r["wall_seconds"] for r in timed)),
        "median_tree_seconds": float(statistics.median(r["tree_seconds"] for r in timed)),
        "median_fit_wall_seconds": float(statistics.median(r["fit_wall_seconds"] for r in timed)),
        "median_policy_seconds": float(statistics.median(r["policy_seconds"] for r in timed)),
        "rows": rows,
        "signature": state_signature(sampler, runtimes, traces),
    }
    args.case_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("WORKER_CASE " + json.dumps({k: v for k, v in result.items() if k not in {"rows", "signature"}}, sort_keys=True), flush=True)
    del runtimes, sampler, solver
    gc.collect()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", default="31,16,20,24,28")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--case-workers", type=int)
    p.add_argument("--case-out", type=Path)
    args = p.parse_args()

    if args.case_workers is not None:
        if args.case_out is None:
            raise ValueError("--case-out is required in case mode")
        return run_case(args)

    if args.out is None:
        raise ValueError("--out is required")
    if args.iterations < 3:
        raise ValueError("need at least three iterations per case")
    source = args.checkpoint.resolve(strict=True)
    solver = args.solver.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    source_hash = digest_file(source)
    candidates = list(dict.fromkeys(int(x) for x in args.workers.split(",")))
    if 31 not in candidates:
        raise ValueError("31-worker production baseline must be included")

    cases = []
    for workers in candidates:
        case_path = args.out / f"workers_{workers}.json"
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--checkpoint", str(source),
            "--solver", str(solver),
            "--threads", str(args.threads),
            "--iterations", str(args.iterations),
            "--case-workers", str(workers),
            "--case-out", str(case_path),
        ]
        print(f"RUN workers={workers}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        cases.append(json.loads(case_path.read_text(encoding="utf-8")))

    baseline = next(c for c in cases if c["workers"] == 31)
    base_sig = baseline["signature"]
    parity = all(c["signature"] == base_sig for c in cases)
    fastest = min(cases, key=lambda c: c["median_wall_seconds"])
    speedup = baseline["median_wall_seconds"] / fastest["median_wall_seconds"]
    # Avoid switching worker count for noise-level wins.  Require >=3% whole-
    # iteration improvement plus exact learning-state parity.
    recommend = 31
    if parity and fastest["median_wall_seconds"] <= baseline["median_wall_seconds"] * 0.97:
        recommend = int(fastest["workers"])

    source_unchanged = digest_file(source) == source_hash
    report = {
        "schema": "LT2_STAGE_B_WORKER_COUNT_BENCHMARK_V1",
        "source": str(source),
        "source_sha256": source_hash,
        "source_iteration": 3000,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "threads": int(args.threads),
        "batch_mode": "vectorized",
        "iteration_mode": "concurrent_fit",
        "cases": cases,
        "baseline_workers": 31,
        "fastest_workers": int(fastest["workers"]),
        "speedup_vs_31": float(speedup),
        "semantic_parity_across_workers": bool(parity),
        "source_unchanged": bool(source_unchanged),
        "recommended_workers": int(recommend),
        "selection_threshold": "exact parity and >=3% median whole-iteration gain vs workers=31",
    }
    report["status"] = (
        "LT2_STAGE_B_WORKER_COUNT_BENCHMARK_PASS"
        if parity and source_unchanged
        else "LT2_STAGE_B_WORKER_COUNT_BENCHMARK_FAIL"
    )
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["status"], flush=True)
    print(f"fastest_workers={fastest['workers']} speedup_vs_31={speedup:.3f}", flush=True)
    print(f"recommended_workers={recommend}", flush=True)
    print(f"semantic_parity_across_workers={str(parity).lower()}", flush=True)
    print(f"source_unchanged={str(source_unchanged).lower()}", flush=True)
    print(f"report={report_path}", flush=True)
    return 0 if report["status"].endswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
