#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.lean_concurrent_iteration import run_iteration_concurrent_fit  # noqa: E402
from spincore.lean_functional_training import (  # noqa: E402
    LeanFunctionalConfig,
    compact_report,
    finalize,
    load_checkpoint,
    new_run,
    run_iteration,
    save_checkpoint,
    write_json_report,
)
from spincore.lean_parallel import ParallelRootExecutor, recommended_ryzen_workers  # noqa: E402
from spincore.solver import SolverLibrary  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lean functional SpinCore Deep-CFR trainer")
    p.add_argument("--solver", type=Path, default=ROOT / "build" / "libspincore_solver_c.so")
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--roots-per-iteration", type=int, default=2)
    p.add_argument("--exact-opponent-levels", type=int, default=0)
    p.add_argument("--reservoir-capacity", type=int, default=100000)
    p.add_argument("--advantage-steps", type=int, default=2)
    p.add_argument(
        "--hu-advantage-steps",
        type=int,
        default=None,
        help=(
            "optional TRUE_HEADS_UP-only Advantage fit budget; when omitted, "
            "HU uses --advantage-steps. On resume, this may be changed only "
            "together with --additional-iterations."
        ),
    )
    p.add_argument("--policy-steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--heads-up-prob", type=float, default=0.4548)
    p.add_argument(
        "--hu-preflop-board-average-k",
        type=int,
        default=None,
        help=(
            "opt-in TRUE_HEADS_UP preflop Advantage target averaging over K future boards; "
            "1 preserves canonical training; values >1 require parallel workers and exact-opponent-levels=0"
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="root-collection worker processes; 0=auto (logical CPUs minus one), 1=serial",
    )
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "lean_functional" / "checkpoint.pt")
    p.add_argument("--report", type=Path, default=ROOT / "runs" / "lean_functional" / "report.json")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-mode", choices=("reference", "vectorized"), default="reference")
    p.add_argument(
        "--iteration-mode",
        choices=("sequential", "concurrent_fit"),
        default="sequential",
        help=(
            "iteration execution path; concurrent_fit overlaps only the independently validated "
            "3H/HU Advantage optimizer loops and preserves canonical sampler semantics"
        ),
    )
    p.add_argument(
        "--additional-iterations",
        type=int,
        default=0,
        help="when resuming, extend the loaded run by this many iterations; works even from a finalized checkpoint",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="save an in-progress checkpoint every N completed iterations (final checkpoint is always saved)",
    )
    return p.parse_args()


def _save_with_metrics(path: Path, save_fn) -> dict[str, float | int | str]:
    started = time.perf_counter()
    save_fn()
    seconds = float(time.perf_counter() - started)
    size_bytes = int(path.stat().st_size) if path.exists() else -1
    metrics = {
        "path": str(path),
        "seconds": seconds,
        "size_bytes": size_bytes,
        "size_gib": float(size_bytes / (1024 ** 3)) if size_bytes >= 0 else -1.0,
    }
    print("CHECKPOINT_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
    return metrics


def main() -> int:
    args = parse_args()
    if not args.solver.exists():
        raise SystemExit(f"solver not found: {args.solver}")
    if int(args.checkpoint_every) <= 0:
        raise SystemExit("--checkpoint-every must be positive")
    if int(args.additional_iterations) < 0:
        raise SystemExit("--additional-iterations must be nonnegative")
    if int(args.workers) < 0:
        raise SystemExit("--workers must be >= 0")
    if args.hu_preflop_board_average_k is not None and int(args.hu_preflop_board_average_k) <= 0:
        raise SystemExit("--hu-preflop-board-average-k must be positive")
    if args.hu_advantage_steps is not None and int(args.hu_advantage_steps) < 0:
        raise SystemExit("--hu-advantage-steps must be nonnegative")
    if args.resume and args.hu_advantage_steps is not None and int(args.additional_iterations) <= 0:
        raise SystemExit("--hu-advantage-steps on resume requires --additional-iterations > 0")

    workers = int(args.workers)
    if workers == 0:
        workers = recommended_ryzen_workers(os.cpu_count())
    solver = SolverLibrary(args.solver)

    if args.resume:
        if not args.checkpoint.exists():
            raise SystemExit(f"checkpoint not found for --resume: {args.checkpoint}")
        seed, config, completed, sampler, runtimes, history, finalized = load_checkpoint(
            args.checkpoint, solver=solver
        )
        if int(args.additional_iterations) > 0:
            replace_kwargs = {
                "iterations": int(completed) + int(args.additional_iterations),
            }
            if args.hu_preflop_board_average_k is not None:
                replace_kwargs["hu_preflop_board_average_k"] = int(args.hu_preflop_board_average_k)
            if args.hu_advantage_steps is not None:
                replace_kwargs["hu_advantage_steps"] = int(args.hu_advantage_steps)
            config = replace(config, **replace_kwargs)
            finalized = False
        elif finalized:
            print(
                "checkpoint already finalized; use --resume --additional-iterations N to extend it",
                flush=True,
            )
            return 0
        print(
            f"RESUME seed={seed} completed_iteration={completed}/{config.iterations} "
            f"workers={workers} iteration_mode={args.iteration_mode}",
            flush=True,
        )
    else:
        if int(args.additional_iterations) != 0:
            raise SystemExit("--additional-iterations is valid only with --resume")
        seed = int(args.seed)
        config = LeanFunctionalConfig(
            iterations=int(args.iterations),
            roots_per_iteration=int(args.roots_per_iteration),
            exact_opponent_levels=int(args.exact_opponent_levels),
            reservoir_capacity=int(args.reservoir_capacity),
            advantage_steps=int(args.advantage_steps),
            hu_advantage_steps=(
                None
                if args.hu_advantage_steps is None
                else int(args.hu_advantage_steps)
            ),
            policy_steps=int(args.policy_steps),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            heads_up_prob=float(args.heads_up_prob),
            hu_preflop_board_average_k=(
                1
                if args.hu_preflop_board_average_k is None
                else int(args.hu_preflop_board_average_k)
            ),
        )
        completed = 0
        history = []
        sampler, runtimes = new_run(solver, seed=seed, config=config)
        print(
            "START "
            + json.dumps(
                {
                    "seed": seed,
                    "config": config.__dict__,
                    "roots_by_domain": config.roots_by_domain(),
                    "solver": str(args.solver),
                    "checkpoint_every": int(args.checkpoint_every),
                    "workers": workers,
                    "iteration_mode": args.iteration_mode,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if int(config.hu_preflop_board_average_k) > 1 and workers <= 1:
        raise SystemExit("HU preflop board averaging requires --workers > 1")
    for runtime in runtimes.values():
        runtime.session.batch_mode = args.batch_mode
    if "SPINCORE_TORCH_THREADS" in os.environ:
        torch.set_num_threads(int(os.environ["SPINCORE_TORCH_THREADS"]))
    print(
        f"FIT_RUNTIME threads={torch.get_num_threads()} batch_mode={args.batch_mode} "
        f"iteration_mode={args.iteration_mode} "
        f"advantage_steps_3h={config.advantage_steps_for_domain('THREE_HANDED')} "
        f"advantage_steps_hu={config.advantage_steps_for_domain('TRUE_HEADS_UP')} "
        f"hu_preflop_board_average_k={config.hu_preflop_board_average_k}",
        flush=True,
    )
    iteration_fn = (
        run_iteration_concurrent_fit
        if args.iteration_mode == "concurrent_fit"
        else run_iteration
    )
    executor = ParallelRootExecutor(args.solver, workers) if workers > 1 else None
    overall_started = time.perf_counter()
    checkpoint_metrics: list[dict[str, float | int | str]] = []
    try:
        for iteration in range(completed + 1, config.iterations + 1):
            print(f"ITERATION {iteration}/{config.iterations} collect+fit", flush=True)
            report = iteration_fn(
                seed=seed,
                iteration=iteration,
                config=config,
                sampler=sampler,
                runtimes=runtimes,
                parallel_executor=executor,
            )
            history.append(report)
            completed = iteration
            should_checkpoint = (
                completed % int(args.checkpoint_every) == 0
                or completed == int(config.iterations)
            )
            if should_checkpoint:
                metrics = _save_with_metrics(
                    args.checkpoint,
                    lambda: save_checkpoint(
                        args.checkpoint,
                        seed=seed,
                        config=config,
                        completed_iteration=completed,
                        sampler=sampler,
                        runtimes=runtimes,
                        history=history,
                        finalized=False,
                    ),
                )
                metrics["iteration"] = int(completed)
                checkpoint_metrics.append(metrics)
                print(f"CHECKPOINT {args.checkpoint}", flush=True)
            print("ITERATION_REPORT " + json.dumps(report, sort_keys=True), flush=True)
    finally:
        if executor is not None:
            executor.close()

    print("FINALIZE average-policy", flush=True)
    final = finalize(config=config, runtimes=runtimes)
    report = compact_report(seed=seed, config=config, history=history, final=final)
    report["wall_seconds"] = float(time.perf_counter() - overall_started)
    report["execution_workers"] = int(workers)
    report["checkpoint_metrics"] = checkpoint_metrics
    final_ckpt = _save_with_metrics(
        args.checkpoint,
        lambda: save_checkpoint(
            args.checkpoint,
            seed=seed,
            config=config,
            completed_iteration=completed,
            sampler=sampler,
            runtimes=runtimes,
            history=history,
            finalized=True,
        ),
    )
    final_ckpt["iteration"] = int(completed)
    final_ckpt["finalized"] = True
    checkpoint_metrics.append(final_ckpt)
    report["wall_seconds"] = float(time.perf_counter() - overall_started)
    report["wall_scope"] = "iterations_through_final_checkpoint_excludes_initial_load"
    report["batch_mode"] = args.batch_mode
    report["iteration_mode"] = args.iteration_mode
    report["torch_threads"] = torch.get_num_threads()
    write_json_report(args.report, report)
    print("FINAL_SUMMARY " + json.dumps(report["final"], sort_keys=True), flush=True)
    print(f"PASS report={args.report} checkpoint={args.checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
