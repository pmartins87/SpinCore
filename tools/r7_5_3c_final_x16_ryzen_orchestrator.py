from __future__ import annotations

"""Resumable Ryzen orchestrator for the frozen final R7.5.3C x16 contingency.

Cells are strategically independent. Parallelism only overlaps independent
subprocesses; every individual cell preserves its exact 3 x 16 chunk sequence,
global-root order, checkpoint chain, and frozen two-thread Phase-2 contract.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from spincore.r7_5_representation_v3_stage_contract import DOMAINS, REPRESENTATIONS, TRAINING_SEEDS

WORKER_SCHEMA = "SPINCORE_R7_5_3C_FINAL_CHANCE_COVERAGE_X16_STAGED_WORKER_V1"
ORCHESTRATOR_SCHEMA = "SPINCORE_R7_5_3C_FINAL_CHANCE_COVERAGE_X16_RYZEN_ORCHESTRATOR_V1"
ITERATIONS = 3
CHUNKS = 16


def _git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _discover_solver(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser()
        if not path.is_absolute():
            path = repo / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"solver not found: {path}")
        return path
    candidates = [
        repo / "build" / "Release" / "spincore_solver_c.dll",
        repo / "build" / "spincore_solver_c.dll",
        repo / "build" / "libspincore_solver_c.so",
        repo / "build" / "libspincore_solver_c.dylib",
    ]
    found = [path.resolve() for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            "could not uniquely discover solver; build spincore_solver_c or pass --solver. "
            f"candidates={candidates} found={found}"
        )
    return found[0]


def _stage_dir(root: Path, rep: str, domain: str, seed: int, iteration: int, chunk: int) -> Path:
    return root / rep / domain / str(seed) / f"i{iteration}c{chunk}"


def _validate_existing(report_path: Path, checkpoint_path: Path, *, sha: str, rep: str, domain: str, seed: int, iteration: int, chunk: int) -> None:
    if report_path.exists() != checkpoint_path.exists():
        raise RuntimeError(f"partial stage collision: {report_path.parent}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "schema": WORKER_SCHEMA,
        "execution_sha": sha,
        "representation": rep,
        "domain": domain,
        "training_seed": int(seed),
        "target_iteration": int(iteration),
        "root_chunk": int(chunk),
        "chance_coverage_multiplier": 16,
        "effective_roots_per_iteration": 1024,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"existing x16 stage identity drift {report_path}: {key}={payload.get(key)!r} expected={value!r}")
    if int(payload.get("chunk_report", {}).get("roots", -1)) != 64:
        raise RuntimeError(f"existing x16 stage root-count drift: {report_path}")
    should_finalize = iteration == ITERATIONS and chunk == CHUNKS
    if bool(payload.get("finalized")) != should_finalize:
        raise RuntimeError(f"existing x16 finalization identity drift: {report_path}")


def _run_cell(*, repo: Path, solver: Path, output_root: Path, sha: str, rep: str, domain: str, seed: int) -> dict:
    worker = repo / "tools" / "r7_5_3c_final_x16_domain_worker.py"
    previous: Path | None = None
    started = time.perf_counter()
    executed = 0
    skipped = 0
    child_env = dict(os.environ)
    paths = [str(repo / "python"), str(repo / "tools")]
    if child_env.get("PYTHONPATH"):
        paths.append(child_env["PYTHONPATH"])
    child_env["PYTHONPATH"] = os.pathsep.join(paths)
    child_env["SPINCORE_TORCH_THREADS"] = "2"
    child_env["OMP_NUM_THREADS"] = "2"
    child_env["MKL_NUM_THREADS"] = "2"

    for iteration in range(1, ITERATIONS + 1):
        for chunk in range(1, CHUNKS + 1):
            stage_dir = _stage_dir(output_root, rep, domain, int(seed), iteration, chunk)
            stage_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = stage_dir / "checkpoint.pt"
            report = stage_dir / "report.json"
            if checkpoint.exists() or report.exists():
                if not (checkpoint.exists() and report.exists()):
                    raise RuntimeError(f"incomplete existing x16 stage: {stage_dir}")
                _validate_existing(
                    report,
                    checkpoint,
                    sha=sha,
                    rep=rep,
                    domain=domain,
                    seed=int(seed),
                    iteration=iteration,
                    chunk=chunk,
                )
                previous = checkpoint
                skipped += 1
                print(f"[x16 resume] {rep} {domain} seed={seed} i{iteration}c{chunk} already valid", flush=True)
                continue

            command = [
                sys.executable,
                str(worker),
                "--repo-root", str(repo),
                "--solver", str(solver),
                "--representation", rep,
                "--domain", domain,
                "--training-seed", str(seed),
                "--target-iteration", str(iteration),
                "--root-chunk", str(chunk),
                "--checkpoint-out", str(checkpoint),
                "--report-out", str(report),
                "--execution-sha", sha,
            ]
            if previous is not None:
                command.extend(["--resume", str(previous)])
            if iteration == ITERATIONS and chunk == CHUNKS:
                command.append("--finalize")
            print(f"[x16 run] {rep} {domain} seed={seed} i{iteration}c{chunk}", flush=True)
            proc = subprocess.run(command, cwd=str(repo), env=child_env)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"x16 worker failed rc={proc.returncode}: {rep}/{domain}/{seed}/i{iteration}c{chunk}"
                )
            _validate_existing(
                report,
                checkpoint,
                sha=sha,
                rep=rep,
                domain=domain,
                seed=int(seed),
                iteration=iteration,
                chunk=chunk,
            )
            previous = checkpoint
            executed += 1

    if previous is None:
        raise RuntimeError("x16 cell produced no checkpoint")
    final_report = json.loads((previous.parent / "report.json").read_text(encoding="utf-8"))
    final_payload = dict(final_report.get("final_report") or {})
    if int(final_payload.get("roots", -1)) != 3072:
        raise RuntimeError(f"x16 final cell root total drift: {rep}/{domain}/{seed}")
    return {
        "representation": rep,
        "domain": domain,
        "training_seed": int(seed),
        "final_checkpoint": str(previous),
        "final_checkpoint_sha256": _sha256(previous),
        "final_report": str(previous.parent / "report.json"),
        "final_report_sha256": _sha256(previous.parent / "report.json"),
        "executed_stages": executed,
        "resumed_stages": skipped,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all eight frozen x16 R7.5.3C cells on Ryzen")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("ryzen_x16_final"))
    parser.add_argument("--execution-sha", default="", help="defaults to exact git HEAD")
    parser.add_argument("--cell-workers", type=int, default=4, choices=tuple(range(1, 9)))
    args = parser.parse_args()

    repo = args.repo_root.expanduser().resolve()
    sha = str(args.execution_sha).strip() or _git_head(repo)
    if _git_head(repo) != sha:
        raise SystemExit(f"execution SHA must equal HEAD: HEAD={_git_head(repo)} requested={sha}")
    solver = _discover_solver(repo, args.solver)
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = repo / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cells = [
        (rep, domain, int(seed))
        for rep in REPRESENTATIONS
        for domain in DOMAINS
        for seed in TRAINING_SEEDS
    ]
    print(f"[x16] HEAD={sha} solver={solver} cells={len(cells)} parallel={args.cell_workers}", flush=True)
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=int(args.cell_workers)) as executor:
        futures = {
            executor.submit(
                _run_cell,
                repo=repo,
                solver=solver,
                output_root=output_root,
                sha=sha,
                rep=rep,
                domain=domain,
                seed=seed,
            ): (rep, domain, seed)
            for rep, domain, seed in cells
        }
        try:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"[x16 cell complete] {result['representation']} {result['domain']} seed={result['training_seed']}", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    results.sort(key=lambda row: (row["representation"], row["domain"], row["training_seed"]))
    if len(results) != 8:
        raise RuntimeError(f"expected 8 completed x16 cells, observed {len(results)}")
    summary = {
        "schema": ORCHESTRATOR_SCHEMA,
        "status": "TRAINING_COMPLETE_PENDING_INDEPENDENT_STABILITY_CERTIFICATION",
        "execution_sha": sha,
        "solver": str(solver),
        "solver_sha256": _sha256(solver),
        "cell_workers": int(args.cell_workers),
        "cells": results,
        "cell_count": len(results),
        "roots_per_iteration": 1024,
        "iterations": 3,
        "roots_per_seed": 3072,
        "coverage_multiplier": 16,
        "elapsed_seconds": float(time.perf_counter() - started),
        "representation_winner": None,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    summary_path = output_root / "orchestrator_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "cell_count": summary["cell_count"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "summary": str(summary_path),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
