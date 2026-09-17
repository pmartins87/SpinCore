#!/usr/bin/env python3
from __future__ import annotations

"""Stage-B Advantage-only held-out budget sweep with production policy semantics.

Why V2 exists:
The first fit audit/sweep computed Advantage-derived TV/argmax with the historical
uniform all-nonpositive fallback. Production SpinCore uses the repaired Lean
masked-softmax fallback. MSE results from the first sweep remain valid, but its
Advantage policy-space metrics are not canonical.

This bounded diagnostic:
- collects no roots;
- never mutates the source checkpoint on disk;
- excludes a deterministic 25k holdout from optimization;
- repeats the same cumulative budget curve over multiple deterministic reset
  seeds to avoid deciding from one initialization;
- evaluates Advantage-derived policy metrics with the exact production Lean
  regret-matching + softmax-fallback semantics from audit_lt2_checkpoint_fit.py.
"""

import argparse
import gc
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch

from audit_lt2_checkpoint_fit import _accumulate_advantage, _selftest_lean_rm_policy_tensor
from spincore.lean_functional_training import DOMAINS, _advantage_reset_seed, load_checkpoint
from spincore.solver import SolverLibrary
from spincore_nn.training import train_step

BUDGETS = (0, 25, 50, 100, 200, 400, 800, 1600)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--heldout", type=int, default=25000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=2048)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _holdout(memory, count: int, seed: int):
    n = len(memory.items)
    if count <= 0 or count >= n:
        raise ValueError(f"heldout must be in [1,{n-1}]")
    rng = random.Random(int(seed))
    idx = rng.sample(range(n), int(count))
    held = set(idx)
    samples = [memory.items[i] for i in idx]
    return held, samples


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


def _fit_steps(runtime, heldout: set[int], rng: random.Random, steps: int, batch_size: int) -> float:
    runtime.session.batch_mode = "vectorized"
    memory = runtime.bundle.adv_mem
    model = runtime.bundle.advantage
    optimizer = runtime.bundle.adv_opt
    started = time.perf_counter()
    for _ in range(int(steps)):
        samples = _sample_excluding(memory, heldout, batch_size, rng)
        batch, target, weights = runtime.session._batch(samples)
        train_step(model, optimizer, batch, target, weights, "advantage")
    return float(time.perf_counter() - started)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("mse", "fit_fraction_vs_zero", "tv", "argmax", "pred_all_nonpos")
    out: dict[str, Any] = {}
    for name in fields:
        values = [float(r["heldout"]["weighted"][name]) for r in rows]
        out[name] = {
            "mean": float(statistics.fmean(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        }
    return out


def main() -> int:
    args = parse_args()
    if args.heldout <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise SystemExit("heldout/batch sizes must be positive")
    if args.threads <= 0 or args.replicates <= 0:
        raise SystemExit("threads/replicates must be positive")

    checkpoint = args.checkpoint.resolve(strict=True)
    solver_path = args.solver.resolve(strict=True)
    torch.set_num_threads(int(args.threads))
    _selftest_lean_rm_policy_tensor()

    solver = SolverLibrary(solver_path)
    seed, config, completed, sampler, runtimes, history, finalized = load_checkpoint(
        checkpoint, solver=solver
    )
    if not finalized or int(completed) != int(config.iterations):
        raise SystemExit("expected finalized Stage-B milestone checkpoint")
    if int(config.batch_size) != int(args.batch_size):
        raise SystemExit(
            f"optimizer batch size must equal canonical checkpoint batch size {config.batch_size}"
        )

    report: dict[str, Any] = {
        "schema": "SPINCORE_LT2_STAGE_B_ADVANTAGE_BUDGET_SWEEP_V2",
        "checkpoint": str(checkpoint),
        "completed_iteration": int(completed),
        "seed": int(seed),
        "config": dict(config.__dict__),
        "method": {
            "no_roots": True,
            "source_checkpoint_read_only": True,
            "heldout_count_per_domain": int(args.heldout),
            "heldout_excluded_from_optimization": True,
            "optimizer_batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "threads": int(args.threads),
            "replicates": int(args.replicates),
            "budgets": list(BUDGETS),
            "canonical_budget_steps": 100,
            "policy_semantics": "production Lean: positive regret matching; all-nonpositive -> stable masked softmax",
            "replicate_design": "fixed heldout per domain; fresh canonical future-iteration reset seed and independent training sampler per replicate",
            "warning": "diagnostic only; no arbitrary PASS threshold",
        },
        "domains": {},
    }

    for domain_index, domain in enumerate(DOMAINS):
        runtime = runtimes[domain]
        memory = runtime.bundle.adv_mem
        held_set, held_samples = _holdout(
            memory,
            int(args.heldout),
            0xA51D2000 ^ int(seed) ^ (domain_index * 0x13579),
        )
        by_budget: dict[int, list[dict[str, Any]]] = {int(b): [] for b in BUDGETS}
        replicates: list[dict[str, Any]] = []

        print(f"=== domain={domain} ===", flush=True)
        for rep in range(int(args.replicates)):
            future_iteration = int(completed) + 1 + rep
            reset_seed = _advantage_reset_seed(int(seed), domain, future_iteration)
            runtime.session.reset_advantage_network(
                init_seed=reset_seed,
                lr=float(config.learning_rate),
            )
            rng = random.Random(
                0xAD720000 ^ int(seed) ^ (domain_index * 0x2468B) ^ (rep * 0x9E3779B1)
            )
            done = 0
            cumulative_seconds = 0.0
            curve: list[dict[str, Any]] = []
            for budget in BUDGETS:
                add = int(budget) - int(done)
                if add < 0:
                    raise RuntimeError("non-monotone budget sequence")
                if add:
                    cumulative_seconds += _fit_steps(
                        runtime,
                        heldout=held_set,
                        rng=rng,
                        steps=add,
                        batch_size=int(args.batch_size),
                    )
                metrics = _accumulate_advantage(
                    runtime, held_samples, int(args.eval_batch_size)
                )
                row = {
                    "replicate": int(rep),
                    "future_iteration_seed_slot": int(future_iteration),
                    "reset_seed": int(reset_seed),
                    "steps": int(budget),
                    "cumulative_fit_seconds": float(cumulative_seconds),
                    "heldout": metrics,
                }
                curve.append(row)
                by_budget[int(budget)].append(row)
                w = metrics["weighted"]
                print(
                    f"ADV2 {domain} rep={rep} steps={budget}: "
                    f"mse={w['mse']:.6f} fit_vs_zero={w['fit_fraction_vs_zero']:+.4f} "
                    f"lean_tv={w['tv']:.4f} lean_argmax={w['argmax']:.4f} "
                    f"pred_all_nonpos={w['pred_all_nonpos']:.4f}",
                    flush=True,
                )
                done = int(budget)
            replicates.append(
                {
                    "replicate": int(rep),
                    "future_iteration_seed_slot": int(future_iteration),
                    "reset_seed": int(reset_seed),
                    "curve": curve,
                }
            )

        aggregate = []
        for budget in BUDGETS:
            s = _summary(by_budget[int(budget)])
            aggregate.append({"steps": int(budget), "replicate_summary": s})
            print(
                f"ADV2_MEAN {domain} steps={budget}: "
                f"mse={s['mse']['mean']:.6f} lean_tv={s['tv']['mean']:.4f} "
                f"lean_argmax={s['argmax']['mean']:.4f}",
                flush=True,
            )

        report["domains"][domain] = {
            "memory_items": len(memory.items),
            "memory_seen": int(memory.seen),
            "heldout_count": int(args.heldout),
            "heldout_excluded_from_optimization": True,
            "target_all_nonpos_weighted": float(
                _accumulate_advantage(runtime, held_samples, int(args.eval_batch_size))["weighted"][
                    "target_all_nonpos"
                ]
            ),
            "replicates": replicates,
            "aggregate": aggregate,
        }
        del held_samples, held_set
        gc.collect()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("LT2_STAGE_B_ADVANTAGE_BUDGET_SWEEP_V2_COMPLETE", flush=True)
    print(f"report={args.report.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
