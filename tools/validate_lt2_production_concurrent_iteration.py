#!/usr/bin/env python3
from __future__ import annotations

"""Final admission gate for the production concurrent-fit iteration function.

Runs iteration 3001 twice from the same finalized LT2-A checkpoint: once through
the canonical sequential function and once through the production
`run_iteration_concurrent_fit`. The source checkpoint is read-only. Exact
semantic parity is required for models, optimizers, RNG/counters, reservoir RNG
states, added sample streams, sampler RNG state and non-timing report fields.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch

from spincore.lean_concurrent_iteration import run_iteration_concurrent_fit
from spincore.lean_functional_training import load_checkpoint, run_iteration
from spincore.lean_parallel import ParallelRootExecutor
from spincore.solver import SolverLibrary
from validate_lt2_concurrent_iteration import install_add_traces, semantic_signature


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _signature_report(report: dict[str, Any]) -> dict[str, Any]:
    """Remove candidate-only wall telemetry before semantic comparison.

    `concurrent_fit_wall_seconds` is intentionally emitted only by the concurrent
    implementation. It is timing telemetry, not poker/training state. The shared
    semantic helper already strips per-domain timing fields; this gate must also
    strip this top-level candidate-only timing field or it produces a false FAIL.
    """
    out = dict(report)
    out.pop("concurrent_fit_wall_seconds", None)
    return out


def _first_diff(a: Any, b: Any, path: str = "$") -> str | None:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        ak = set(a)
        bk = set(b)
        if ak != bk:
            return f"{path}: keys only_reference={sorted(ak-bk)} only_candidate={sorted(bk-ak)}"
        for key in sorted(ak):
            diff = _first_diff(a[key], b[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (av, bv) in enumerate(zip(a, b)):
            diff = _first_diff(av, bv, f"{path}[{i}]")
            if diff:
                return diff
        return None
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def run_once(*, source: Path, solver_path: Path, workers: int, threads: int, candidate: bool):
    solver = SolverLibrary(solver_path)
    seed, config, completed, sampler, runtimes, _history, finalized = load_checkpoint(
        source,
        solver=solver,
    )
    if not finalized or completed != config.iterations or completed != 3000:
        raise ValueError("expected finalized LT2-A checkpoint at iteration 3000")
    iteration = completed + 1
    for runtime in runtimes.values():
        runtime.session.batch_mode = "vectorized"
    torch.set_num_threads(int(threads))
    traces = install_add_traces(runtimes)
    executor = ParallelRootExecutor(solver_path, workers) if workers > 1 else None
    started = time.perf_counter()
    try:
        if candidate:
            report = run_iteration_concurrent_fit(
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
        signature = semantic_signature(
            sampler,
            runtimes,
            traces,
            _signature_report(report),
        )
        return {"wall_seconds": wall, "signature": signature}
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
    source_hash = digest_file(source)

    print(f"SOURCE read_only={source}", flush=True)
    print("RUN canonical sequential iteration", flush=True)
    reference = run_once(
        source=source,
        solver_path=solver_path,
        workers=args.workers,
        threads=args.threads,
        candidate=False,
    )
    gc.collect()
    print("RUN production concurrent-fit iteration", flush=True)
    candidate = run_once(
        source=source,
        solver_path=solver_path,
        workers=args.workers,
        threads=args.threads,
        candidate=True,
    )

    source_unchanged = digest_file(source) == source_hash
    first_difference = _first_diff(reference["signature"], candidate["signature"])
    semantic_parity = first_difference is None
    result = {
        "schema": "LT2_PRODUCTION_CONCURRENT_ITERATION_PARITY_V2",
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
        "first_difference": first_difference,
        "reference_signature": reference["signature"],
        "candidate_signature": candidate["signature"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    result["status"] = (
        "LT2_PRODUCTION_CONCURRENT_ITERATION_PARITY_PASS"
        if semantic_parity and source_unchanged
        else "LT2_PRODUCTION_CONCURRENT_ITERATION_PARITY_FAIL"
    )
    report = args.out / "report.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not source_unchanged:
        raise RuntimeError("source checkpoint changed")
    if not semantic_parity:
        print("LT2_PRODUCTION_CONCURRENT_ITERATION_PARITY_FAIL", flush=True)
        print(f"first_difference={first_difference}", flush=True)
        print(f"report={report}", flush=True)
        return 2

    print("LT2_PRODUCTION_CONCURRENT_ITERATION_PARITY_PASS", flush=True)
    print(f"reference_wall_seconds={reference['wall_seconds']:.3f}", flush=True)
    print(f"candidate_wall_seconds={candidate['wall_seconds']:.3f}", flush=True)
    print(f"candidate_speedup={result['candidate_speedup']:.3f}", flush=True)
    print("source_unchanged=true", flush=True)
    print(f"report={report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
