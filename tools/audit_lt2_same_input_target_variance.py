#!/usr/bin/env python3
from __future__ import annotations

"""Read-only same-neural-input Advantage target variance audit for LT2 Stage B.

This audit groups stored Advantage samples that have exactly the same SPNNIV1
observation bytes and the same 10-action legal mask. Any deterministic network
fed that representation must emit the same prediction for every item in a group.
For duplicate groups we therefore decompose the training-weighted target MSE into
within-input target variance plus current-model error to the group conditional
mean target.

The within-input term is an empirical variance floor under the stored reservoir
distribution. It can contain hidden-card/future-chance sampling, opponent-action
external-sampling noise, and historical-policy drift; it is not claimed to be an
intrinsic game-theoretic irreducible variance.
"""

import argparse
import gc
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch

from audit_lt2_checkpoint_fit import _lean_rm_policy_tensor, _selftest_lean_rm_policy_tensor
from spincore.lean_functional_training import DOMAINS, REPRESENTATION, load_checkpoint
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore.solver import SolverLibrary
from spincore_nn.action_models import collate_action_observations
from spincore_nn.codec import decode_spnniv1

CHIP_SCALE = 1500.0
STREET_NAMES = {0: "PREFLOP", 1: "FLOP", 2: "TURN", 3: "RIVER"}
SPNNIV1_STREET_OFFSET = 80
MIN_GROUP_SIZES = (2, 4, 8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "p90": float("nan"), "max": float("nan")}
    s = sorted(float(x) for x in values)
    def q(frac: float) -> float:
        pos = frac * (len(s) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return float(s[lo])
        a = pos - lo
        return float(s[lo] * (1.0 - a) + s[hi] * a)
    return {"min": s[0], "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "p90": q(0.90), "max": s[-1]}


def _street_from_observation(observation: bytes) -> int:
    if len(observation) <= SPNNIV1_STREET_OFFSET or not observation.startswith(b"SPNNIV1\x00"):
        raise ValueError("same-input audit requires SPNNIV1 observation")
    code = int(observation[SPNNIV1_STREET_OFFSET])
    if code not in STREET_NAMES:
        raise ValueError(f"invalid SPNNIV1 street code {code}")
    return code


def _new_threshold_acc() -> dict[str, Any]:
    return {
        "groups": 0,
        "items": 0,
        "weight": 0.0,
        "within_mse_weighted_sum": 0.0,
        "model_mse_weighted_sum": 0.0,
        "tv_weighted_sum": 0.0,
        "argmax_weighted_sum": 0.0,
        "branch_mismatch_weighted_sum": 0.0,
        "signed_gap_chips_weighted_sum": 0.0,
        "model_regret_chips_weighted_sum": 0.0,
        "target_action_mass_weighted_sum": [0.0] * 10,
        "model_action_mass_weighted_sum": [0.0] * 10,
        "group_sizes": [],
        "iteration_spans": [],
    }


def _finalize_acc(acc: dict[str, Any], *, total_items: int, total_weight: float) -> dict[str, Any]:
    w = max(float(acc["weight"]), 1e-12)
    within = float(acc["within_mse_weighted_sum"] / w)
    model = float(acc["model_mse_weighted_sum"] / w)
    total = within + model
    return {
        "group_count": int(acc["groups"]),
        "covered_item_count": int(acc["items"]),
        "covered_item_fraction": float(acc["items"] / max(int(total_items), 1)),
        "covered_training_weight_fraction": float(acc["weight"] / max(float(total_weight), 1e-12)),
        "group_size_quantiles": _quantiles(acc["group_sizes"]),
        "group_iteration_span_quantiles": _quantiles(acc["iteration_spans"]),
        "training_weighted_proxy": {
            "within_same_input_target_mse": within,
            "model_mse_to_same_input_mean": model,
            "sample_target_mse": total,
            "conditional_variance_fraction": float(within / max(total, 1e-12)),
            "conditional_mean_model_error_fraction": float(model / max(total, 1e-12)),
            "model_policy_tv_to_same_input_mean_policy": float(acc["tv_weighted_sum"] / w),
            "model_policy_argmax_agreement": float(acc["argmax_weighted_sum"] / w),
            "branch_mismatch_rate": float(acc["branch_mismatch_weighted_sum"] / w),
            "signed_same_input_mean_policy_minus_model_policy_value_gap_chips": float(acc["signed_gap_chips_weighted_sum"] / w),
            "model_policy_regret_to_same_input_mean_best_action_chips": float(acc["model_regret_chips_weighted_sum"] / w),
            "target_action_mass": {NAME_BY_SLOT[i]: float(acc["target_action_mass_weighted_sum"][i] / w) for i in range(10)},
            "model_action_mass": {NAME_BY_SLOT[i]: float(acc["model_action_mass_weighted_sum"][i] / w) for i in range(10)},
        },
    }


def _audit_street(runtime, street_code: int, batch_size: int) -> dict[str, Any]:
    memory = runtime.bundle.adv_mem
    model = runtime.bundle.advantage
    model.eval()
    counts: dict[tuple[bytes, tuple[int, ...]], int] = {}
    total_items = 0
    total_weight = 0.0
    for sample in memory.items:
        if _street_from_observation(sample.observation) != int(street_code):
            continue
        total_items += 1
        total_weight += float(sample.weight)
        key = (sample.observation, sample.legal)
        counts[key] = counts.get(key, 0) + 1

    duplicate_counts = {key: count for key, count in counts.items() if count >= 2}
    unique_input_count = len(counts)
    del counts
    gc.collect()

    if not duplicate_counts:
        return {
            "street": STREET_NAMES[int(street_code)],
            "street_item_count": int(total_items),
            "street_training_weight": float(total_weight),
            "unique_input_count": int(unique_input_count),
            "duplicate_input_group_count": 0,
            "thresholds": {},
        }

    moments: dict[tuple[bytes, tuple[int, ...]], list[Any]] = {}
    for sample in memory.items:
        if _street_from_observation(sample.observation) != int(street_code):
            continue
        key = (sample.observation, sample.legal)
        if key not in duplicate_counts:
            continue
        rec = moments.get(key)
        if rec is None:
            rec = [0, 0.0, [0.0] * 10, [0.0] * 10, int(sample.iteration), int(sample.iteration)]
            moments[key] = rec
        rec[0] += 1
        w = float(sample.weight)
        rec[1] += w
        rec[4] = min(int(rec[4]), int(sample.iteration))
        rec[5] = max(int(rec[5]), int(sample.iteration))
        for action, value in enumerate(sample.target):
            v = float(value)
            rec[2][action] += w * v
            rec[3][action] += w * v * v

    del duplicate_counts
    gc.collect()

    threshold_acc = {size: _new_threshold_acc() for size in MIN_GROUP_SIZES}
    keys = list(moments.keys())

    with torch.no_grad():
        for start in range(0, len(keys), int(batch_size)):
            batch_keys = keys[start : start + int(batch_size)]
            observations = [key[0] for key in batch_keys]
            legal_masks = [key[1] for key in batch_keys]
            batch = collate_action_observations(REPRESENTATION, observations, legal_masks, device="cpu")
            pred = model(batch)
            legal = batch["legal"]

            mean_targets = []
            within_values = []
            group_weights = []
            group_counts = []
            iteration_spans = []
            for key in batch_keys:
                rec = moments[key]
                sum_w = max(float(rec[1]), 1e-12)
                mean = [float(x) / sum_w for x in rec[2]]
                second = [float(x) / sum_w for x in rec[3]]
                legal_actions = [i for i, flag in enumerate(key[1]) if int(flag)]
                if not legal_actions:
                    raise RuntimeError("duplicate Advantage sample has empty legal mask")
                within = sum(max(0.0, second[i] - mean[i] * mean[i]) for i in legal_actions) / len(legal_actions)
                mean_targets.append(mean)
                within_values.append(within)
                group_weights.append(sum_w)
                group_counts.append(int(rec[0]))
                iteration_spans.append(int(rec[5]) - int(rec[4]))

            target = torch.tensor(mean_targets, dtype=torch.float32)
            within_t = torch.tensor(within_values, dtype=torch.float32)
            group_w = torch.tensor(group_weights, dtype=torch.float32)
            legal_f = legal.float()
            action_count = legal_f.sum(dim=1).clamp_min(1.0)
            model_mse = (((pred - target) ** 2) * legal_f).sum(dim=1) / action_count

            tp = _lean_rm_policy_tensor(target, legal)
            pp = _lean_rm_policy_tensor(pred, legal)
            tv = 0.5 * torch.abs(tp - pp).sum(dim=1)
            targ_arg = tp.masked_fill(~legal, -1.0).argmax(dim=1)
            pred_arg = pp.masked_fill(~legal, -1.0).argmax(dim=1)
            argmax = (targ_arg == pred_arg).float()
            target_has_pos = target.masked_fill(~legal, float("-inf")).max(dim=1).values > 0.0
            pred_has_pos = pred.masked_fill(~legal, float("-inf")).max(dim=1).values > 0.0
            branch_mismatch = (target_has_pos != pred_has_pos).float()

            target_value = (tp * target).sum(dim=1)
            model_value = (pp * target).sum(dim=1)
            legal_target_max = target.masked_fill(~legal, float("-inf")).max(dim=1).values
            signed_gap = (target_value - model_value) * CHIP_SCALE
            model_regret = (legal_target_max - model_value).clamp_min(0.0) * CHIP_SCALE

            for row, count in enumerate(group_counts):
                w = float(group_w[row].item())
                for threshold in MIN_GROUP_SIZES:
                    if int(count) < threshold:
                        continue
                    acc = threshold_acc[threshold]
                    acc["groups"] += 1
                    acc["items"] += int(count)
                    acc["weight"] += w
                    acc["within_mse_weighted_sum"] += float(within_t[row].item()) * w
                    acc["model_mse_weighted_sum"] += float(model_mse[row].item()) * w
                    acc["tv_weighted_sum"] += float(tv[row].item()) * w
                    acc["argmax_weighted_sum"] += float(argmax[row].item()) * w
                    acc["branch_mismatch_weighted_sum"] += float(branch_mismatch[row].item()) * w
                    acc["signed_gap_chips_weighted_sum"] += float(signed_gap[row].item()) * w
                    acc["model_regret_chips_weighted_sum"] += float(model_regret[row].item()) * w
                    for action in range(10):
                        acc["target_action_mass_weighted_sum"][action] += float(tp[row, action].item()) * w
                        acc["model_action_mass_weighted_sum"][action] += float(pp[row, action].item()) * w
                    acc["group_sizes"].append(int(count))
                    acc["iteration_spans"].append(int(iteration_spans[row]))

    out = {
        "street": STREET_NAMES[int(street_code)],
        "street_item_count": int(total_items),
        "street_training_weight": float(total_weight),
        "unique_input_count": int(unique_input_count),
        "duplicate_input_group_count": int(len(moments)),
        "thresholds": {
            str(size): _finalize_acc(threshold_acc[size], total_items=total_items, total_weight=total_weight)
            for size in MIN_GROUP_SIZES
            if threshold_acc[size]["groups"] > 0
        },
    }
    del moments, keys
    gc.collect()
    return out


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        raise SystemExit("batch/threads must be positive")
    solver_path = args.solver.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    torch.set_num_threads(int(args.threads))
    _selftest_lean_rm_policy_tensor()

    solver = SolverLibrary(solver_path)
    seed, config, completed, sampler, runtimes, history, finalized = load_checkpoint(checkpoint, solver=solver)
    if not finalized or int(completed) != int(config.iterations):
        raise SystemExit("expected finalized milestone checkpoint")
    if REPRESENTATION != "C0_V1_FROZEN_CONTROL":
        raise SystemExit("same-input audit currently requires frozen SPNNIV1 representation")

    for domain in DOMAINS:
        first = runtimes[domain].bundle.adv_mem.items[0]
        decoded = decode_spnniv1(first.observation)
        if int(decoded.categorical[1]) != _street_from_observation(first.observation):
            raise RuntimeError("SPNNIV1 street-byte offset parity failed")

    report: dict[str, Any] = {
        "schema": "SPINCORE_LT2_SAME_INPUT_TARGET_VARIANCE_V1",
        "checkpoint": str(checkpoint),
        "completed_iteration": int(completed),
        "seed": int(seed),
        "method": {
            "read_only": True,
            "no_roots": True,
            "no_optimizer_steps": True,
            "no_training_memory_writes": True,
            "group_key": "exact SPNNIV1 observation bytes + exact 10-action legal mask",
            "primary_weighting": "stored sample.weight proxy for the training objective",
            "decomposition": "sample-target MSE on duplicate inputs = within-same-input target variance + current-model MSE to weighted same-input mean target",
            "interpretation_limit": "within-input variance is empirical under the stored reservoir and can mix hidden/future chance, opponent sampling, and historical-policy drift; it is not an intrinsic game-theoretic variance floor",
            "minimum_group_sizes_reported": list(MIN_GROUP_SIZES),
            "batch_size": int(args.batch_size),
            "threads": int(args.threads),
        },
        "domains": {},
    }

    for domain in DOMAINS:
        runtime = runtimes[domain]
        domain_block = {
            "memory_items": int(len(runtime.bundle.adv_mem.items)),
            "memory_seen": int(runtime.bundle.adv_mem.seen),
            "streets": {},
        }
        print(f"=== SAME_INPUT domain={domain} ===", flush=True)
        for street_code, street_name in STREET_NAMES.items():
            block = _audit_street(runtime, street_code, int(args.batch_size))
            domain_block["streets"][street_name] = block
            t2 = block.get("thresholds", {}).get("2")
            if t2:
                w = t2["training_weighted_proxy"]
                print(
                    f"{domain} {street_name}: coverage={t2['covered_item_fraction']:.4f} "
                    f"groups={t2['group_count']} conditional_var_frac={w['conditional_variance_fraction']:.4f} "
                    f"model_mean_frac={w['conditional_mean_model_error_fraction']:.4f} "
                    f"tv={w['model_policy_tv_to_same_input_mean_policy']:.4f}",
                    flush=True,
                )
            else:
                print(f"{domain} {street_name}: no duplicate exact inputs", flush=True)
        report["domains"][domain] = domain_block

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("LT2_SAME_INPUT_TARGET_VARIANCE_COMPLETE", flush=True)
    print(f"report={args.report.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
