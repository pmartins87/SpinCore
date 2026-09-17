#!/usr/bin/env python3
from __future__ import annotations

"""Controlled held-out optimizer-budget sweep on the preserved LT2 Stage-B checkpoint.

This is a causal diagnostic, not long training:
- no roots are collected;
- source checkpoint is read-only and hash-checked by the launcher;
- deterministic held-out reservoir items are excluded from optimization sampling;
- Advantage starts from a fresh deterministic reset and includes the canonical
  100-step budget on a geometric curve;
- AveragePolicy starts from the stored Stage-B model+optimizer state and measures
  additional fit budget, including the canonical +4000 finalization increment.

The report intentionally has no arbitrary PASS threshold. It exposes held-out
curves so the next decision can distinguish optimizer-budget limitation from a
capacity/representation/target problem.
"""

import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch

from audit_lt2_checkpoint_fit import _accumulate_advantage, _accumulate_policy
from spincore.lean_functional_training import DOMAINS, _advantage_reset_seed, load_checkpoint
from spincore.solver import SolverLibrary
from spincore_nn.training import train_step

ADV_BUDGETS = (0, 25, 50, 100, 200, 400, 800, 1600)
POLICY_EXTRA_BUDGETS = (0, 1000, 2000, 4000, 8000)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--heldout", type=int, default=25000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=2048)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _holdout(memory, count: int, seed: int):
    n = len(memory.items)
    if count <= 0 or count >= n:
        raise ValueError(f"heldout must be in [1,{n-1}]")
    rng = random.Random(int(seed))
    idx = rng.sample(range(n), int(count))
    hold_set = set(idx)
    samples = [memory.items[i] for i in idx]
    return hold_set, samples


def _sample_excluding(memory, heldout: set[int], n: int, rng: random.Random):
    population = len(memory.items)
    eligible = population - len(heldout)
    k = min(int(n), eligible)
    if k <= 0:
        raise RuntimeError("no training items outside holdout")
    selected: list[int] = []
    seen: set[int] = set()
    while len(selected) < k:
        i = rng.randrange(population)
        if i in heldout or i in seen:
            continue
        seen.add(i)
        selected.append(i)
    return [memory.items[i] for i in selected]


def _fit_steps(runtime, *, memory, model, optimizer, kind: str, heldout: set[int], rng: random.Random, steps: int, batch_size: int) -> float:
    runtime.session.batch_mode = "vectorized"
    started = time.perf_counter()
    for _ in range(int(steps)):
        samples = _sample_excluding(memory, heldout, batch_size, rng)
        batch, target, weights = runtime.session._batch(samples)
        train_step(model, optimizer, batch, target, weights, kind)
    return float(time.perf_counter() - started)


def _curve_row(step: int, metrics: dict[str, Any], wall_seconds: float) -> dict[str, Any]:
    return {"steps": int(step), "heldout": metrics, "cumulative_fit_seconds": float(wall_seconds)}


def _sweep_advantage(runtime, *, domain: str, seed: int, completed: int, heldout_count: int, batch_size: int, eval_batch: int) -> dict[str, Any]:
    memory = runtime.bundle.adv_mem
    hold_set, hold_samples = _holdout(memory, heldout_count, 0xA51D0000 ^ int(seed) ^ (0 if domain == "THREE_HANDED" else 0x12345))

    # Fresh reset is deliberate: canonical training also resets Advantage each iteration.
    reset_seed = _advantage_reset_seed(int(seed), domain, int(completed) + 1)
    runtime.session.reset_advantage_network(init_seed=reset_seed, lr=runtime.bundle.adv_opt.defaults.get("lr", 1e-3))
    model = runtime.bundle.advantage
    optimizer = runtime.bundle.adv_opt
    train_rng = random.Random(0xAD710000 ^ int(seed) ^ (17 if domain == "THREE_HANDED" else 29))

    curve: list[dict[str, Any]] = []
    done = 0
    cumulative_seconds = 0.0
    for budget in ADV_BUDGETS:
        add = int(budget) - done
        if add < 0:
            raise RuntimeError("non-monotone advantage budgets")
        if add:
            cumulative_seconds += _fit_steps(
                runtime,
                memory=memory,
                model=model,
                optimizer=optimizer,
                kind="advantage",
                heldout=hold_set,
                rng=train_rng,
                steps=add,
                batch_size=batch_size,
            )
        metrics = _accumulate_advantage(runtime, hold_samples, eval_batch)
        curve.append(_curve_row(budget, metrics, cumulative_seconds))
        w = metrics["weighted"]
        print(
            f"ADV {domain} steps={budget}: mse={w['mse']:.6f} "
            f"fit_vs_zero={w['fit_fraction_vs_zero']:+.4f} tv={w['tv']:.4f} argmax={w['argmax']:.4f}",
            flush=True,
        )
        done = int(budget)

    del hold_samples, hold_set
    gc.collect()
    return {
        "reset_seed": int(reset_seed),
        "memory_items": len(memory.items),
        "memory_seen": int(memory.seen),
        "heldout_count": int(heldout_count),
        "heldout_excluded_from_optimization": True,
        "batch_size": int(batch_size),
        "budgets": list(ADV_BUDGETS),
        "canonical_budget_steps": 100,
        "curve": curve,
    }


def _sweep_policy(runtime, *, domain: str, seed: int, heldout_count: int, batch_size: int, eval_batch: int) -> dict[str, Any]:
    memory = runtime.bundle.pol_mem
    hold_set, hold_samples = _holdout(memory, heldout_count, 0xB61C0000 ^ int(seed) ^ (0 if domain == "THREE_HANDED" else 0x54321))

    # Start from the actual stored Stage-B policy and optimizer state.
    model = runtime.bundle.policy
    optimizer = runtime.bundle.pol_opt
    train_rng = random.Random(0xC0710000 ^ int(seed) ^ (41 if domain == "THREE_HANDED" else 53))

    curve: list[dict[str, Any]] = []
    done = 0
    cumulative_seconds = 0.0
    for budget in POLICY_EXTRA_BUDGETS:
        add = int(budget) - done
        if add < 0:
            raise RuntimeError("non-monotone policy budgets")
        if add:
            cumulative_seconds += _fit_steps(
                runtime,
                memory=memory,
                model=model,
                optimizer=optimizer,
                kind="strategy",
                heldout=hold_set,
                rng=train_rng,
                steps=add,
                batch_size=batch_size,
            )
        metrics = _accumulate_policy(runtime, hold_samples, eval_batch)
        curve.append(_curve_row(budget, metrics, cumulative_seconds))
        w = metrics["weighted"]
        print(
            f"POL {domain} extra_steps={budget}: ce={w['ce']:.6f} kl={w['kl']:.6f} "
            f"fit_gap={w['fit_fraction_uniform_to_target']:+.4f} tv={w['tv']:.4f} argmax={w['argmax']:.4f}",
            flush=True,
        )
        done = int(budget)

    del hold_samples, hold_set
    gc.collect()
    return {
        "start_state": "stored Stage-B AveragePolicy model and optimizer",
        "memory_items": len(memory.items),
        "memory_seen": int(memory.seen),
        "heldout_count": int(heldout_count),
        "heldout_excluded_from_optimization": True,
        "batch_size": int(batch_size),
        "additional_budgets": list(POLICY_EXTRA_BUDGETS),
        "canonical_finalization_increment_steps": 4000,
        "curve": curve,
    }


def main() -> int:
    args = parse_args()
    if args.heldout <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0 or args.threads <= 0:
        raise SystemExit("heldout/batch/eval-batch/threads must be positive")
    checkpoint = args.checkpoint.resolve(strict=True)
    solver_path = args.solver.resolve(strict=True)

    torch.set_num_threads(int(args.threads))
    solver = SolverLibrary(solver_path)
    seed, config, completed, sampler, runtimes, history, finalized = load_checkpoint(checkpoint, solver=solver)
    if not finalized or int(completed) != int(config.iterations):
        raise SystemExit("expected finalized milestone checkpoint")
    if int(config.batch_size) != int(args.batch_size):
        raise SystemExit(f"sweep batch size must equal canonical checkpoint batch size {config.batch_size}")

    report: dict[str, Any] = {
        "schema": "SPINCORE_LT2_STAGE_B_FIT_BUDGET_SWEEP_V1",
        "checkpoint": str(checkpoint),
        "completed_iteration": int(completed),
        "seed": int(seed),
        "config": dict(config.__dict__),
        "method": {
            "no_roots": True,
            "source_checkpoint_read_only": True,
            "heldout_count_per_memory": int(args.heldout),
            "heldout_excluded_from_optimization": True,
            "optimizer_batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "threads": int(args.threads),
            "advantage_design": "fresh deterministic reset; cumulative geometric budgets around canonical 100 steps",
            "policy_design": "continue stored Stage-B policy+optimizer; cumulative additional budgets around canonical +4000 finalization increment",
            "interpretation": "diagnostic curves only; no arbitrary PASS threshold",
        },
        "domains": {},
    }

    for domain in DOMAINS:
        print(f"=== domain={domain} ===", flush=True)
        runtime = runtimes[domain]
        adv = _sweep_advantage(
            runtime,
            domain=domain,
            seed=seed,
            completed=completed,
            heldout_count=args.heldout,
            batch_size=args.batch_size,
            eval_batch=args.eval_batch_size,
        )
        pol = _sweep_policy(
            runtime,
            domain=domain,
            seed=seed,
            heldout_count=args.heldout,
            batch_size=args.batch_size,
            eval_batch=args.eval_batch_size,
        )
        report["domains"][domain] = {"advantage": adv, "average_policy": pol}

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("LT2_STAGE_B_FIT_BUDGET_SWEEP_COMPLETE", flush=True)
    print(f"report={args.report.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
