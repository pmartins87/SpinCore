#!/usr/bin/env python3
from __future__ import annotations

"""Read-only fit audit for finalized LT2 training checkpoints.

This does not train or mutate the source checkpoints. It samples each checkpoint's
Advantage and AveragePolicy reservoirs deterministically and measures how well the
stored networks fit the stored targets. Advantage-derived policy metrics use the
same positive-regret matching + masked-softmax all-nonpositive fallback as the
production lean behavior policy.
"""

import argparse
import gc
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import torch

from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_functional_training import DOMAINS, load_checkpoint
from spincore.solver import SolverLibrary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--stage-b", type=Path, required=True)
    p.add_argument("--samples-per-memory", type=int, default=25000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _quantiles(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0}
    s = sorted(values)

    def q(frac: float) -> float:
        pos = frac * (len(s) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(s[lo])
        a = pos - lo
        return float(s[lo] * (1.0 - a) + s[hi] * a)

    return {
        "min": int(s[0]),
        "p25": q(0.25),
        "median": q(0.5),
        "p75": q(0.75),
        "max": int(s[-1]),
    }


def _sample(memory, n: int, seed: int):
    count = min(int(n), len(memory.items))
    if count <= 0:
        raise RuntimeError("cannot audit empty reservoir")
    rng = random.Random(int(seed))
    indices = rng.sample(range(len(memory.items)), count)
    return [memory.items[i] for i in indices]


def _lean_rm_policy_tensor(values: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Vectorized production Lean regret matching semantics.

    Positive legal regrets are normalized. If no legal regret is positive, use
    the same stable masked softmax over raw legal advantages as
    lean_regret_matching_policy().
    """
    if values.ndim != 2 or legal.shape != values.shape:
        raise ValueError("values/legal shape mismatch")
    legal_f = legal.float()
    positive = torch.clamp(values, min=0.0) * legal_f
    positive_total = positive.sum(dim=1, keepdim=True)
    rm = positive / positive_total.clamp_min(1e-12)

    masked = values.masked_fill(~legal, float("-inf"))
    maximum = masked.max(dim=1, keepdim=True).values
    # The scalar production adapter clamps exp arguments to [-60, 60]. Since
    # maximum is a legal value, legal differences are <= 0; keeping the full
    # clamp reproduces the scalar contract and protects extreme inputs.
    shifted = (values - maximum).clamp(min=-60.0, max=60.0)
    soft_weights = torch.exp(shifted) * legal_f
    softmax_fallback = soft_weights / soft_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.where(positive_total > 0.0, rm, softmax_fallback)


def _selftest_lean_rm_policy_tensor() -> None:
    values = torch.tensor(
        [
            [-1.0, -2.0, 0.0, 0.0],
            [1.0, -1.0, 3.0, 7.0],
            [-100.0, -1.0, -2.0, 9.0],
        ],
        dtype=torch.float32,
    )
    legal = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    # Production carrier has ten actions; pad the synthetic parity cases.
    pad = 10 - values.shape[1]
    values10 = torch.nn.functional.pad(values, (0, pad), value=0.0)
    legal10 = torch.nn.functional.pad(legal, (0, pad), value=False)
    tensor = _lean_rm_policy_tensor(values10, legal10).cpu()
    for row in range(values10.shape[0]):
        legal_tuple = tuple(i for i, x in enumerate(legal10[row].tolist()) if x)
        scalar = torch.tensor(
            lean_regret_matching_policy(values10[row].tolist(), legal_tuple),
            dtype=torch.float32,
        )
        if not torch.allclose(tensor[row], scalar, atol=1e-7, rtol=1e-6):
            raise RuntimeError("lean regret-matching tensor/scalar parity failed")


def _accumulate_policy(runtime, samples, batch_size: int) -> dict[str, Any]:
    runtime.session.batch_mode = "vectorized"
    model = runtime.bundle.policy
    model.eval()
    sums = {k: 0.0 for k in ("w", "ce", "entropy", "kl", "uniform_ce", "tv", "argmax")}
    raw = {k: 0.0 for k in ("n", "ce", "entropy", "kl", "uniform_ce", "tv", "argmax")}
    iterations: list[int] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            batch, target, weights = runtime.session._batch(chunk)
            legal = batch["legal"]
            logits = model(batch).masked_fill(~legal, -1e9)
            logp = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            ce = -(target * logp).sum(dim=1)
            entropy = -(target * torch.log(target.clamp_min(1e-12))).sum(dim=1)
            kl = ce - entropy
            uniform_ce = torch.log(legal.float().sum(dim=1).clamp_min(1.0))
            tv = 0.5 * torch.abs(target - probs).sum(dim=1)
            targ_arg = target.masked_fill(~legal, -1.0).argmax(dim=1)
            pred_arg = probs.masked_fill(~legal, -1.0).argmax(dim=1)
            arg = (targ_arg == pred_arg).float()
            w = weights.float()
            sums["w"] += float(w.sum().item())
            for name, value in (
                ("ce", ce),
                ("entropy", entropy),
                ("kl", kl),
                ("uniform_ce", uniform_ce),
                ("tv", tv),
                ("argmax", arg),
            ):
                sums[name] += float((value * w).sum().item())
                raw[name] += float(value.sum().item())
            raw["n"] += float(len(chunk))
            iterations.extend(int(x.iteration) for x in chunk)
    w = max(sums["w"], 1e-12)
    n = max(raw["n"], 1.0)
    weighted = {
        name: sums[name] / w
        for name in ("ce", "entropy", "kl", "uniform_ce", "tv", "argmax")
    }
    unweighted = {
        name: raw[name] / n
        for name in ("ce", "entropy", "kl", "uniform_ce", "tv", "argmax")
    }
    denom = weighted["uniform_ce"] - weighted["entropy"]
    weighted["fit_fraction_uniform_to_target"] = (
        (weighted["uniform_ce"] - weighted["ce"]) / denom
        if denom > 1e-12
        else float("nan")
    )
    return {
        "sample_count": int(raw["n"]),
        "weighted": weighted,
        "unweighted": unweighted,
        "sample_iteration_quantiles": _quantiles(iterations),
    }


def _accumulate_advantage(runtime, samples, batch_size: int) -> dict[str, Any]:
    runtime.session.batch_mode = "vectorized"
    model = runtime.bundle.advantage
    model.eval()
    sums = {
        k: 0.0
        for k in (
            "w",
            "mse",
            "zero_mse",
            "tv",
            "argmax",
            "target_all_nonpos",
            "pred_all_nonpos",
        )
    }
    raw = {
        k: 0.0
        for k in (
            "n",
            "mse",
            "zero_mse",
            "tv",
            "argmax",
            "target_all_nonpos",
            "pred_all_nonpos",
        )
    }
    iterations: list[int] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            batch, target, weights = runtime.session._batch(chunk)
            legal = batch["legal"]
            out = model(batch)
            mask = legal.float()
            count = mask.sum(dim=1).clamp_min(1.0)
            mse = (((out - target) ** 2) * mask).sum(dim=1) / count
            zero_mse = ((target**2) * mask).sum(dim=1) / count
            tp = _lean_rm_policy_tensor(target, legal)
            pp = _lean_rm_policy_tensor(out, legal)
            tv = 0.5 * torch.abs(tp - pp).sum(dim=1)
            targ_arg = tp.masked_fill(~legal, -1.0).argmax(dim=1)
            pred_arg = pp.masked_fill(~legal, -1.0).argmax(dim=1)
            arg = (targ_arg == pred_arg).float()
            targ_nonpos = (
                target.masked_fill(~legal, float("-inf")).max(dim=1).values <= 0.0
            ).float()
            pred_nonpos = (
                out.masked_fill(~legal, float("-inf")).max(dim=1).values <= 0.0
            ).float()
            w = weights.float()
            sums["w"] += float(w.sum().item())
            for name, value in (
                ("mse", mse),
                ("zero_mse", zero_mse),
                ("tv", tv),
                ("argmax", arg),
                ("target_all_nonpos", targ_nonpos),
                ("pred_all_nonpos", pred_nonpos),
            ):
                sums[name] += float((value * w).sum().item())
                raw[name] += float(value.sum().item())
            raw["n"] += float(len(chunk))
            iterations.extend(int(x.iteration) for x in chunk)
    w = max(sums["w"], 1e-12)
    n = max(raw["n"], 1.0)
    weighted = {
        name: sums[name] / w
        for name in (
            "mse",
            "zero_mse",
            "tv",
            "argmax",
            "target_all_nonpos",
            "pred_all_nonpos",
        )
    }
    unweighted = {
        name: raw[name] / n
        for name in (
            "mse",
            "zero_mse",
            "tv",
            "argmax",
            "target_all_nonpos",
            "pred_all_nonpos",
        )
    }
    weighted["fit_fraction_vs_zero"] = 1.0 - weighted["mse"] / max(
        weighted["zero_mse"], 1e-12
    )
    return {
        "sample_count": int(raw["n"]),
        "weighted": weighted,
        "unweighted": unweighted,
        "sample_iteration_quantiles": _quantiles(iterations),
    }


def _audit_one(
    label: str,
    path: Path,
    solver: SolverLibrary,
    sample_n: int,
    batch_size: int,
) -> dict[str, Any]:
    source = path.resolve(strict=True)
    print(f"LOADING {label} read_only={source}", flush=True)
    seed, config, completed, sampler, runtimes, history, finalized = load_checkpoint(
        source, solver=solver
    )
    if not finalized or completed != config.iterations:
        raise RuntimeError(f"{label} is not a finalized milestone checkpoint")
    result: dict[str, Any] = {
        "label": label,
        "checkpoint": str(source),
        "seed": int(seed),
        "completed_iteration": int(completed),
        "config": dict(config.__dict__),
        "domains": {},
    }
    for i, domain in enumerate(DOMAINS):
        runtime = runtimes[domain]
        b = runtime.bundle
        adv_samples = _sample(b.adv_mem, sample_n, 0xA100 + completed * 7 + i)
        pol_samples = _sample(b.pol_mem, sample_n, 0xB200 + completed * 11 + i)
        block = {
            "counters": dict(b.counters),
            "advantage_memory": {
                "capacity": int(b.adv_mem.capacity),
                "items": len(b.adv_mem.items),
                "seen": int(b.adv_mem.seen),
            },
            "policy_memory": {
                "capacity": int(b.pol_mem.capacity),
                "items": len(b.pol_mem.items),
                "seen": int(b.pol_mem.seen),
            },
            "advantage_fit": _accumulate_advantage(runtime, adv_samples, batch_size),
            "policy_fit": _accumulate_policy(runtime, pol_samples, batch_size),
        }
        result["domains"][domain] = block
        a = block["advantage_fit"]["weighted"]
        p = block["policy_fit"]["weighted"]
        print(
            f"{label} {domain}: adv_fit_vs_zero={a['fit_fraction_vs_zero']:+.4f} "
            f"adv_policy_tv={a['tv']:.4f} policy_KL={p['kl']:.4f} "
            f"policy_TV={p['tv']:.4f} policy_argmax={p['argmax']:.4f}",
            flush=True,
        )
    del runtimes, sampler, history
    gc.collect()
    return result


def main() -> int:
    args = parse_args()
    if args.samples_per_memory <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise SystemExit("samples/batch/threads must be positive")
    if not args.solver.is_file():
        raise SystemExit(f"solver not found: {args.solver}")
    _selftest_lean_rm_policy_tensor()
    torch.set_num_threads(int(args.threads))
    solver = SolverLibrary(args.solver)
    a = _audit_one(
        "LT2A_1P8M", args.stage_a, solver, args.samples_per_memory, args.batch_size
    )
    b = _audit_one(
        "LT2B_4P5M", args.stage_b, solver, args.samples_per_memory, args.batch_size
    )
    report = {
        "schema": "SPINCORE_LT2_TRAINING_FIT_AUDIT_V2",
        "method": {
            "read_only": True,
            "sampling": "deterministic uniform sample from each stored reservoir",
            "samples_per_memory": int(args.samples_per_memory),
            "batch_size": int(args.batch_size),
            "threads": int(args.threads),
            "policy_metrics": "cross-entropy, target entropy, KL, TV and argmax agreement; iteration weights reported through weighted metrics",
            "advantage_metrics": "legal-action MSE vs zero predictor plus production-Lean regret-matching/softmax-fallback policy TV and argmax agreement",
            "advantage_policy_semantics": "positive legal advantages normalized; if none positive, stable masked softmax over raw legal advantages",
            "warning": "diagnostic fit audit only; no arbitrary PASS threshold and no source checkpoint mutation",
        },
        "stage_a": a,
        "stage_b": b,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("LT2_TRAINING_FIT_AUDIT_COMPLETE", flush=True)
    print(f"report={args.report.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
