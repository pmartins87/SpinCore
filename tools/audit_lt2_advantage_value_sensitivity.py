#!/usr/bin/env python3
from __future__ import annotations

"""Read-only Stage-B Advantage target/value-sensitivity audit.

The corrected budget sweep showed that more optimizer steps reduce held-out MSE
but do not reduce production-policy TV.  This audit asks the more decision-
relevant question: how much target value is actually lost by the model-induced
policy, and where does that loss occur?

No roots are collected.  No optimizer step is taken.  The source checkpoint is
read only.  All value metrics are reported in the canonical utility scale and in
chip-equivalent units using the fixed `/1500` training scale.
"""

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch

from audit_lt2_checkpoint_fit import _lean_rm_policy_tensor, _selftest_lean_rm_policy_tensor
from spincore.lean_functional_training import DOMAINS, load_checkpoint
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore.solver import SolverLibrary

CHIP_SCALE = 1500.0
STREET_NAMES = {0: "PREFLOP", 1: "FLOP", 2: "TURN", 3: "RIVER"}
SPAN_BINS = (
    ("SPAN_0_TO_1_CHIP", 0.0, 1.0),
    ("SPAN_1_TO_5_CHIPS", 1.0, 5.0),
    ("SPAN_5_TO_20_CHIPS", 5.0, 20.0),
    ("SPAN_20_TO_100_CHIPS", 20.0, 100.0),
    ("SPAN_100_PLUS_CHIPS", 100.0, float("inf")),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--samples-per-domain", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _sample(memory, n: int, seed: int):
    count = min(int(n), len(memory.items))
    if count <= 0:
        raise RuntimeError("cannot audit empty Advantage reservoir")
    rng = random.Random(int(seed))
    indices = rng.sample(range(len(memory.items)), count)
    return [memory.items[i] for i in indices]


def _new_acc() -> dict[str, Any]:
    return {
        "n": 0,
        "weight": 0.0,
        "weighted": {
            "tv": 0.0,
            "argmax": 0.0,
            "target_all_nonpos": 0.0,
            "pred_all_nonpos": 0.0,
            "branch_mismatch": 0.0,
            "false_positive_rm_branch": 0.0,
            "false_negative_rm_branch": 0.0,
            "target_span_chips": 0.0,
            "max_positive_target_chips": 0.0,
            "target_policy_regret_to_best_chips": 0.0,
            "model_policy_regret_to_best_chips": 0.0,
            "signed_model_minus_target_policy_value_gap_chips": 0.0,
            "positive_model_value_loss_vs_target_policy_chips": 0.0,
            "absolute_model_target_policy_value_gap_chips": 0.0,
        },
        "target_action_mass": [0.0] * 10,
        "model_action_mass": [0.0] * 10,
    }


def _add(acc: dict[str, Any], *, weights: torch.Tensor, metrics: dict[str, torch.Tensor], tp: torch.Tensor, pp: torch.Tensor) -> None:
    w = weights.float()
    wsum = float(w.sum().item())
    acc["n"] += int(w.numel())
    acc["weight"] += wsum
    for name, value in metrics.items():
        acc["weighted"][name] += float((value.float() * w).sum().item())
    for slot in range(10):
        acc["target_action_mass"][slot] += float((tp[:, slot] * w).sum().item())
        acc["model_action_mass"][slot] += float((pp[:, slot] * w).sum().item())


def _finalize(acc: dict[str, Any], total_weight: float) -> dict[str, Any]:
    w = max(float(acc["weight"]), 1e-12)
    out = {
        "n": int(acc["n"]),
        "weight_fraction": float(acc["weight"] / max(total_weight, 1e-12)),
        "weighted": {k: float(v / w) for k, v in acc["weighted"].items()},
        "target_action_mass": {
            NAME_BY_SLOT[i]: float(acc["target_action_mass"][i] / w) for i in range(10)
        },
        "model_action_mass": {
            NAME_BY_SLOT[i]: float(acc["model_action_mass"][i] / w) for i in range(10)
        },
    }
    return out


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": float("nan"), "p75": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan")}
    s = sorted(float(x) for x in values)
    def q(frac: float) -> float:
        pos = frac * (len(s) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return float(s[lo])
        a = pos - lo
        return float(s[lo] * (1.0 - a) + s[hi] * a)
    return {"p50": q(0.50), "p75": q(0.75), "p90": q(0.90), "p95": q(0.95), "p99": q(0.99)}


def _span_bin_masks(span_chips: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    out = []
    for name, low, high in SPAN_BINS:
        if math.isinf(high):
            mask = span_chips >= low
        elif low == 0.0:
            mask = (span_chips >= low) & (span_chips < high)
        else:
            mask = (span_chips >= low) & (span_chips < high)
        out.append((name, mask))
    return out


def _audit_domain(runtime, samples, batch_size: int) -> dict[str, Any]:
    runtime.session.batch_mode = "vectorized"
    model = runtime.bundle.advantage
    model.eval()

    groups: dict[str, dict[str, Any]] = {"OVERALL": _new_acc()}
    for name in ("TARGET_ALL_NONPOS", "TARGET_HAS_POSITIVE"):
        groups[name] = _new_acc()
    for street_name in STREET_NAMES.values():
        groups[f"STREET_{street_name}"] = _new_acc()
    for name, _, _ in SPAN_BINS:
        groups[name] = _new_acc()

    total_weight = 0.0
    geometry_lists = {
        "target_span_chips": [],
        "max_positive_target_chips": [],
        "model_policy_regret_to_best_chips": [],
        "absolute_model_target_policy_value_gap_chips": [],
        "tv": [],
    }

    with torch.no_grad():
        for start in range(0, len(samples), int(batch_size)):
            chunk = samples[start : start + int(batch_size)]
            batch, target, weights = runtime.session._batch(chunk)
            legal = batch["legal"]
            out = model(batch)
            tp = _lean_rm_policy_tensor(target, legal)
            pp = _lean_rm_policy_tensor(out, legal)

            legal_target_max = target.masked_fill(~legal, float("-inf")).max(dim=1).values
            legal_target_min = target.masked_fill(~legal, float("inf")).min(dim=1).values
            target_span_chips = (legal_target_max - legal_target_min).clamp_min(0.0) * CHIP_SCALE
            max_positive_target_chips = legal_target_max.clamp_min(0.0) * CHIP_SCALE

            target_positive_total = (torch.clamp(target, min=0.0) * legal.float()).sum(dim=1)
            pred_positive_total = (torch.clamp(out, min=0.0) * legal.float()).sum(dim=1)
            target_has_positive = target_positive_total > 0.0
            pred_has_positive = pred_positive_total > 0.0
            target_all_nonpos = ~target_has_positive
            pred_all_nonpos = ~pred_has_positive

            target_value = (tp * target).sum(dim=1)
            model_value = (pp * target).sum(dim=1)
            target_regret = (legal_target_max - target_value).clamp_min(0.0) * CHIP_SCALE
            model_regret = (legal_target_max - model_value).clamp_min(0.0) * CHIP_SCALE
            signed_gap = (target_value - model_value) * CHIP_SCALE
            positive_gap = signed_gap.clamp_min(0.0)
            absolute_gap = signed_gap.abs()

            tv = 0.5 * torch.abs(tp - pp).sum(dim=1)
            target_arg = tp.masked_fill(~legal, -1.0).argmax(dim=1)
            model_arg = pp.masked_fill(~legal, -1.0).argmax(dim=1)
            argmax = (target_arg == model_arg).float()
            branch_mismatch = (target_has_positive != pred_has_positive).float()
            false_positive = (target_all_nonpos & pred_has_positive).float()
            false_negative = (target_has_positive & pred_all_nonpos).float()

            metrics = {
                "tv": tv,
                "argmax": argmax,
                "target_all_nonpos": target_all_nonpos.float(),
                "pred_all_nonpos": pred_all_nonpos.float(),
                "branch_mismatch": branch_mismatch,
                "false_positive_rm_branch": false_positive,
                "false_negative_rm_branch": false_negative,
                "target_span_chips": target_span_chips,
                "max_positive_target_chips": max_positive_target_chips,
                "target_policy_regret_to_best_chips": target_regret,
                "model_policy_regret_to_best_chips": model_regret,
                "signed_model_minus_target_policy_value_gap_chips": signed_gap,
                "positive_model_value_loss_vs_target_policy_chips": positive_gap,
                "absolute_model_target_policy_value_gap_chips": absolute_gap,
            }

            total_weight += float(weights.float().sum().item())
            _add(groups["OVERALL"], weights=weights, metrics=metrics, tp=tp, pp=pp)

            masks: list[tuple[str, torch.Tensor]] = [
                ("TARGET_ALL_NONPOS", target_all_nonpos),
                ("TARGET_HAS_POSITIVE", target_has_positive),
            ]
            streets = batch["categorical"][:, 1].long()
            for code, street_name in STREET_NAMES.items():
                masks.append((f"STREET_{street_name}", streets == int(code)))
            masks.extend(_span_bin_masks(target_span_chips))

            for group_name, mask in masks:
                if not bool(mask.any().item()):
                    continue
                submetrics = {k: v[mask] for k, v in metrics.items()}
                _add(groups[group_name], weights=weights[mask], metrics=submetrics, tp=tp[mask], pp=pp[mask])

            geometry_lists["target_span_chips"].extend(target_span_chips.detach().cpu().tolist())
            geometry_lists["max_positive_target_chips"].extend(max_positive_target_chips.detach().cpu().tolist())
            geometry_lists["model_policy_regret_to_best_chips"].extend(model_regret.detach().cpu().tolist())
            geometry_lists["absolute_model_target_policy_value_gap_chips"].extend(absolute_gap.detach().cpu().tolist())
            geometry_lists["tv"].extend(tv.detach().cpu().tolist())

    return {
        "sample_count": len(samples),
        "weighted_groups": {
            name: _finalize(acc, total_weight) for name, acc in groups.items() if int(acc["n"]) > 0
        },
        "unweighted_quantiles": {name: _quantiles(values) for name, values in geometry_lists.items()},
        "span_bins_chips": [
            {"name": name, "low_inclusive": low, "high_exclusive": None if math.isinf(high) else high}
            for name, low, high in SPAN_BINS
        ],
        "street_codes": {str(k): v for k, v in STREET_NAMES.items()},
    }


def main() -> int:
    args = parse_args()
    if args.samples_per_domain <= 0 or args.batch_size <= 0 or args.threads <= 0:
        raise SystemExit("samples/batch/threads must be positive")

    solver_path = args.solver.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    torch.set_num_threads(int(args.threads))
    _selftest_lean_rm_policy_tensor()

    solver = SolverLibrary(solver_path)
    seed, config, completed, sampler, runtimes, history, finalized = load_checkpoint(checkpoint, solver=solver)
    if not finalized or int(completed) != int(config.iterations):
        raise SystemExit("expected finalized milestone checkpoint")

    report: dict[str, Any] = {
        "schema": "SPINCORE_LT2_ADVANTAGE_VALUE_SENSITIVITY_V1",
        "checkpoint": str(checkpoint),
        "completed_iteration": int(completed),
        "seed": int(seed),
        "config": dict(config.__dict__),
        "method": {
            "read_only": True,
            "no_roots": True,
            "no_optimizer_steps": True,
            "samples_per_domain": int(args.samples_per_domain),
            "sampling": "deterministic uniform sample from stored Stage-B Advantage reservoir",
            "batch_size": int(args.batch_size),
            "threads": int(args.threads),
            "policy_semantics": "production Lean positive-regret matching with masked-softmax all-nonpositive fallback",
            "value_scale": "target utilities are terminal chip delta / 1500; chip-equivalent diagnostics multiply target-value gaps by 1500",
            "interpretation": "diagnostic only; span bins are descriptive, not PASS thresholds",
        },
        "domains": {},
    }

    for i, domain in enumerate(DOMAINS):
        runtime = runtimes[domain]
        samples = _sample(
            runtime.bundle.adv_mem,
            int(args.samples_per_domain),
            0xA51D5000 ^ int(seed) ^ (i * 0x13579),
        )
        block = _audit_domain(runtime, samples, int(args.batch_size))
        report["domains"][domain] = block
        overall = block["weighted_groups"]["OVERALL"]["weighted"]
        print(
            f"VALUE_SENSITIVITY {domain}: tv={overall['tv']:.4f} "
            f"branch_mismatch={overall['branch_mismatch']:.4f} "
            f"model_regret={overall['model_policy_regret_to_best_chips']:.3f} chips/decision "
            f"positive_loss_vs_target={overall['positive_model_value_loss_vs_target_policy_chips']:.3f}",
            flush=True,
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("LT2_ADVANTAGE_VALUE_SENSITIVITY_COMPLETE", flush=True)
    print(f"report={args.report.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
