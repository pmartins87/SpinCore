#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

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
    p.add_argument("--policy-steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--heads-up-prob", type=float, default=0.4548)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "lean_functional" / "checkpoint.pt")
    p.add_argument("--report", type=Path, default=ROOT / "runs" / "lean_functional" / "report.json")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.solver.exists():
        raise SystemExit(f"solver not found: {args.solver}")
    solver = SolverLibrary(args.solver)

    if args.resume:
        if not args.checkpoint.exists():
            raise SystemExit(f"checkpoint not found for --resume: {args.checkpoint}")
        seed, config, completed, sampler, runtimes, history, finalized = load_checkpoint(
            args.checkpoint, solver=solver
        )
        if finalized:
            print("checkpoint already finalized; nothing to resume", flush=True)
            return 0
        print(
            f"RESUME seed={seed} completed_iteration={completed}/{config.iterations}",
            flush=True,
        )
    else:
        seed = int(args.seed)
        config = LeanFunctionalConfig(
            iterations=int(args.iterations),
            roots_per_iteration=int(args.roots_per_iteration),
            exact_opponent_levels=int(args.exact_opponent_levels),
            reservoir_capacity=int(args.reservoir_capacity),
            advantage_steps=int(args.advantage_steps),
            policy_steps=int(args.policy_steps),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            heads_up_prob=float(args.heads_up_prob),
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
                },
                sort_keys=True,
            ),
            flush=True,
        )

    overall_started = time.perf_counter()
    for iteration in range(completed + 1, config.iterations + 1):
        print(f"ITERATION {iteration}/{config.iterations} collect+fit", flush=True)
        report = run_iteration(
            seed=seed,
            iteration=iteration,
            config=config,
            sampler=sampler,
            runtimes=runtimes,
        )
        history.append(report)
        completed = iteration
        save_checkpoint(
            args.checkpoint,
            seed=seed,
            config=config,
            completed_iteration=completed,
            sampler=sampler,
            runtimes=runtimes,
            history=history,
            finalized=False,
        )
        print("ITERATION_REPORT " + json.dumps(report, sort_keys=True), flush=True)
        print(f"CHECKPOINT {args.checkpoint}", flush=True)

    print("FINALIZE average-policy", flush=True)
    final = finalize(config=config, runtimes=runtimes)
    report = compact_report(seed=seed, config=config, history=history, final=final)
    report["wall_seconds"] = float(time.perf_counter() - overall_started)
    write_json_report(args.report, report)
    save_checkpoint(
        args.checkpoint,
        seed=seed,
        config=config,
        completed_iteration=completed,
        sampler=sampler,
        runtimes=runtimes,
        history=history,
        finalized=True,
    )
    print("FINAL_REPORT " + json.dumps(report, sort_keys=True), flush=True)
    print(f"PASS report={args.report} checkpoint={args.checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
