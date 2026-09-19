#!/usr/bin/env python3
from __future__ import annotations

"""Controlled fresh-refit audit on the fixed Jammer FAI structural cohort."""

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch

import audit_lt2_jammer_fai_broad_calibration as broad
import audit_lt2_jammer_fai_fold_shift_infoset as foldshift
import audit_lt2_stage_a_b_first_divergence as fd
from spincore.lean_functional_training import load_checkpoint
from spincore.solver import SolverLibrary

FORENSIC_SEEDS = (20260920, 20260921, 20260922, 20260923, 20260924, 20260925)
CHIP_SCALE = 1500.0


class StageView:
    def __init__(self, runtime):
        self.runtime = runtime


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--stage-b", type=Path, required=True)
    p.add_argument("--structural-report", type=Path, required=True)
    p.add_argument("--scenarios-per-seed", type=int, default=5000)
    p.add_argument("--anchors-per-shift-per-seed", type=int, default=32)
    p.add_argument("--budgets", default="100,400,1600")
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _mean_ci(values):
    xs = [float(x) for x in values]
    mean = float(statistics.fmean(xs)) if xs else 0.0
    sem = (
        float(statistics.stdev(xs) / math.sqrt(len(xs)))
        if len(xs) > 1
        else 0.0
    )
    return {
        "mean": mean,
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
        "n": len(xs),
    }


def _cluster_ci(rows, getter):
    by_seed = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(float(getter(row)))
    means = [
        float(statistics.fmean(values))
        for _, values in sorted(by_seed.items())
    ]
    return _mean_ci(means)


def _reconstruct_anchors(
    *,
    solver,
    stage_a,
    stage_b,
    scenarios_per_seed,
    anchors_per_group,
    source_rows,
):
    row_map = {}
    for row in source_rows:
        key = (
            int(row["seed"]),
            int(row["scenario_index"]),
            int(row["actor"]),
            str(row["fold_shift_group"]),
        )
        if key in row_map:
            raise RuntimeError(f"duplicate source row key {key}")
        row_map[key] = row

    selected = []
    candidate_counts = {}
    groups = ("B_MORE_FOLD", "B_LESS_FOLD")

    for seed in FORENSIC_SEEDS:
        pool = broad._collect_seed_common_fai(
            solver=solver,
            stage_a=stage_a,
            stage_b=stage_b,
            seed=int(seed),
            scenarios=int(scenarios_per_seed),
        )
        focused = {group: [] for group in groups}
        for anchor0 in pool:
            group, _fa, _fb, _delta = foldshift._classify_shift(
                stage_a, stage_b, anchor0
            )
            if tuple(int(x) for x in anchor0["legal"]) != (0, 1):
                continue
            if int(anchor0["common_public_action_count"]) != 1:
                continue
            if group not in focused:
                continue
            anchor = dict(anchor0)
            anchor["fold_shift_group"] = group
            focused[group].append(anchor)

        candidate_counts[str(seed)] = {
            group: len(focused[group]) for group in groups
        }

        for group_index, group in enumerate(groups):
            need = int(anchors_per_group)
            candidates = focused[group]
            if len(candidates) < need:
                raise RuntimeError(
                    f"seed {seed} group {group}: {len(candidates)} < {need}"
                )
            chooser = random.Random(
                fd._mix64(seed, 0x5712C7A1, group_index)
            )
            indices = sorted(chooser.sample(range(len(candidates)), need))
            selected.extend(candidates[i] for i in indices)

    selected.sort(
        key=lambda anchor: (
            int(anchor["seed"]),
            str(anchor["fold_shift_group"]),
            int(anchor["scenario_index"]),
            int(anchor["actor"]),
        )
    )

    for index, anchor in enumerate(selected):
        anchor["anchor_index"] = int(index)
        key = (
            int(anchor["seed"]),
            int(anchor["scenario_index"]),
            int(anchor["actor"]),
            str(anchor["fold_shift_group"]),
        )
        if key not in row_map:
            raise RuntimeError(
                f"reconstructed anchor missing from source report: {key}"
            )
        anchor["source_row"] = row_map[key]

    if len(selected) != 384:
        raise RuntimeError(
            f"expected 384 reconstructed anchors, got {len(selected)}"
        )

    return selected, candidate_counts


def _evaluate_pair(stage_a, stage_b, anchors):
    rows = []
    for anchor in anchors:
        q = torch.tensor(
            anchor["source_row"]["stage_a"]["q_canonical"],
            dtype=torch.float32,
        )
        metrics_a = broad._stage_metrics(stage=stage_a, anchor=anchor, q=q)
        metrics_b = broad._stage_metrics(stage=stage_b, anchor=anchor, q=q)
        rows.append({
            "seed": int(anchor["seed"]),
            "group": str(anchor["fold_shift_group"]),
            "policy_value_b_minus_a_chips": (
                float(metrics_b["policy_value_normalized"])
                - float(metrics_a["policy_value_normalized"])
            ) * CHIP_SCALE,
            "policy_regret_b_minus_a_chips": (
                float(metrics_b["policy_regret_chips"])
                - float(metrics_a["policy_regret_chips"])
            ),
            "gap_mse_b_minus_a": (
                float(metrics_b["canonical_action_gap_mse"])
                - float(metrics_a["canonical_action_gap_mse"])
            ),
            "class_error_b_minus_a": (
                float(metrics_b["class_error_mass"])
                - float(metrics_a["class_error_mass"])
            ),
            "fallback_a": (
                1.0 if metrics_a["fallback_all_nonpositive"] else 0.0
            ),
            "fallback_b": (
                1.0 if metrics_b["fallback_all_nonpositive"] else 0.0
            ),
            "regret_a": float(metrics_a["policy_regret_chips"]),
            "regret_b": float(metrics_b["policy_regret_chips"]),
            "best_match_a": 1.0 if metrics_a["best_action_match"] else 0.0,
            "best_match_b": 1.0 if metrics_b["best_action_match"] else 0.0,
        })

    summary = {}
    for group in ("B_MORE_FOLD", "B_LESS_FOLD"):
        selected = [row for row in rows if row["group"] == group]
        summary[group] = {
            "n": len(selected),
            "policy_value_b_minus_a_chips": {
                "seed_cluster_ci": _cluster_ci(
                    selected,
                    lambda row: row["policy_value_b_minus_a_chips"],
                )
            },
            "policy_regret_b_minus_a_chips": {
                "seed_cluster_ci": _cluster_ci(
                    selected,
                    lambda row: row["policy_regret_b_minus_a_chips"],
                )
            },
            "gap_mse_b_minus_a": {
                "seed_cluster_ci": _cluster_ci(
                    selected,
                    lambda row: row["gap_mse_b_minus_a"],
                )
            },
            "class_error_b_minus_a": {
                "seed_cluster_ci": _cluster_ci(
                    selected,
                    lambda row: row["class_error_b_minus_a"],
                )
            },
            "fallback_a_rate": float(
                statistics.fmean(row["fallback_a"] for row in selected)
            ),
            "fallback_b_rate": float(
                statistics.fmean(row["fallback_b"] for row in selected)
            ),
            "regret_a": {
                "seed_cluster_ci": _cluster_ci(
                    selected, lambda row: row["regret_a"]
                )
            },
            "regret_b": {
                "seed_cluster_ci": _cluster_ci(
                    selected, lambda row: row["regret_b"]
                )
            },
            "best_match_a_rate": float(
                statistics.fmean(row["best_match_a"] for row in selected)
            ),
            "best_match_b_rate": float(
                statistics.fmean(row["best_match_b"] for row in selected)
            ),
        }

    return summary


def _fit(
    runtime,
    *,
    init_seed,
    batch_seed,
    budget,
    batch_size,
    learning_rate,
):
    runtime.session.batch_mode = "vectorized"
    runtime.session.reset_advantage_network(
        init_seed=int(init_seed),
        lr=float(learning_rate),
    )
    runtime.bundle.batch_rng.seed(int(batch_seed))
    started = time.perf_counter()
    losses = runtime.session.train_advantage(
        steps=int(budget),
        batch_size=int(batch_size),
    )
    return {
        "seconds": float(time.perf_counter() - started),
        "loss_last": float(losses[-1]) if losses else None,
    }


def main() -> int:
    args = parse_args()
    budgets = [
        int(token)
        for token in str(args.budgets).split(",")
        if token.strip()
    ]
    if not budgets or any(budget <= 0 for budget in budgets):
        raise SystemExit("positive budgets required")
    if args.replicates <= 0 or args.threads <= 0:
        raise SystemExit("positive replicates/threads required")

    source = json.loads(
        args.structural_report.read_text(encoding="utf-8")
    )
    if (
        source.get("schema")
        != "SPINCORE_LT2_JAMMER_FAI_STRUCTURAL_INFOSET_CONFIRMATION_V1"
    ):
        raise SystemExit("wrong structural report schema")
    if source["method"]["future_holdout_seeds_touched"] is not False:
        raise SystemExit("structural report touched holdout unexpectedly")

    torch.set_num_threads(int(args.threads))
    solver = SolverLibrary(args.solver.resolve(strict=True))

    (
        seed_a,
        config_a,
        iteration_a,
        _sampler_a,
        runtimes_a,
        _history_a,
        _finalized_a,
    ) = load_checkpoint(args.stage_a.resolve(strict=True), solver=solver)
    (
        seed_b,
        config_b,
        iteration_b,
        _sampler_b,
        runtimes_b,
        _history_b,
        _finalized_b,
    ) = load_checkpoint(args.stage_b.resolve(strict=True), solver=solver)

    if int(iteration_a) != 3000 or int(iteration_b) != 7500:
        raise RuntimeError(
            f"unexpected checkpoint iterations A={iteration_a} B={iteration_b}"
        )

    runtime_a = runtimes_a["TRUE_HEADS_UP"]
    runtime_b = runtimes_b["TRUE_HEADS_UP"]
    stage_a = StageView(runtime_a)
    stage_b = StageView(runtime_b)

    anchors, candidate_counts = _reconstruct_anchors(
        solver=solver,
        stage_a=stage_a,
        stage_b=stage_b,
        scenarios_per_seed=int(args.scenarios_per_seed),
        anchors_per_group=int(args.anchors_per_shift_per_seed),
        source_rows=list(source["rows"]),
    )

    production = _evaluate_pair(stage_a, stage_b, anchors)
    reproduced = production["B_MORE_FOLD"][
        "policy_value_b_minus_a_chips"
    ]["seed_cluster_ci"]["mean"]
    expected = source["summary"]["groups"]["B_MORE_FOLD"][
        "policy_value_b_minus_a_chips"
    ]["seed_cluster_ci"]["mean"]
    if abs(float(reproduced) - float(expected)) > 2e-4:
        raise RuntimeError(
            f"production reproduction drift: {reproduced} vs {expected}"
        )

    batch_size = min(
        int(config_a.batch_size),
        int(config_b.batch_size),
    )
    learning_rate = float(config_a.learning_rate)
    if abs(float(config_b.learning_rate) - learning_rate) > 1e-15:
        raise RuntimeError("A/B learning-rate mismatch")

    trials = []
    for budget in budgets:
        for replicate in range(int(args.replicates)):
            init_seed = (
                fd._mix64(20260919, budget, replicate, 0xA11CE)
                & 0x7FFFFFFF
            )
            batch_seed = (
                fd._mix64(20260919, budget, replicate, 0xBA7C4)
                & 0x7FFFFFFF
            )

            print(
                f"REFIT budget={budget} "
                f"rep={replicate + 1}/{args.replicates} "
                f"init={init_seed} batch={batch_seed}",
                flush=True,
            )

            fit_a = _fit(
                runtime_a,
                init_seed=init_seed,
                batch_seed=batch_seed,
                budget=budget,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
            fit_b = _fit(
                runtime_b,
                init_seed=init_seed,
                batch_seed=batch_seed,
                budget=budget,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

            summary = _evaluate_pair(stage_a, stage_b, anchors)
            trials.append({
                "budget": int(budget),
                "replicate": int(replicate),
                "init_seed": int(init_seed),
                "batch_seed": int(batch_seed),
                "fit_a": fit_a,
                "fit_b": fit_b,
                "summary": summary,
            })

            value = summary["B_MORE_FOLD"][
                "policy_value_b_minus_a_chips"
            ]["seed_cluster_ci"]
            gap = summary["B_MORE_FOLD"][
                "gap_mse_b_minus_a"
            ]["seed_cluster_ci"]
            print(
                "  B_MORE_FOLD "
                f"B-A value={value['mean']:+.3f} "
                f"CI95=[{value['ci95_low']:+.3f},"
                f"{value['ci95_high']:+.3f}] "
                f"gapMSE={gap['mean']:+.6f}",
                flush=True,
            )

    by_budget = {}
    for budget in budgets:
        selected = [trial for trial in trials if trial["budget"] == budget]
        values = [
            trial["summary"]["B_MORE_FOLD"][
                "policy_value_b_minus_a_chips"
            ]["seed_cluster_ci"]["mean"]
            for trial in selected
        ]
        gaps = [
            trial["summary"]["B_MORE_FOLD"][
                "gap_mse_b_minus_a"
            ]["seed_cluster_ci"]["mean"]
            for trial in selected
        ]
        by_budget[str(budget)] = {
            "replicate_means_b_more_fold_policy_value_b_minus_a_chips":
                _mean_ci(values),
            "replicate_means_b_more_fold_gap_mse_b_minus_a":
                _mean_ci(gaps),
        }

    report = {
        "schema": "SPINCORE_LT2_JAMMER_FAI_CONTROLLED_REFIT_V1",
        "stage_a": {
            "checkpoint": str(args.stage_a.resolve()),
            "completed_iteration": int(iteration_a),
            "seed": int(seed_a),
            "adv_mem_items": len(runtime_a.bundle.adv_mem.items),
            "adv_mem_seen": int(runtime_a.bundle.adv_mem.seen),
        },
        "stage_b": {
            "checkpoint": str(args.stage_b.resolve()),
            "completed_iteration": int(iteration_b),
            "seed": int(seed_b),
            "adv_mem_items": len(runtime_b.bundle.adv_mem.items),
            "adv_mem_seen": int(runtime_b.bundle.adv_mem.seen),
        },
        "method": {
            "read_only_source_checkpoints": True,
            "new_training_roots": 0,
            "source_training_memory_writes": 0,
            "future_holdout_seeds_touched": False,
            "in_memory_optimizer_steps": int(
                sum(
                    2 * budget * int(args.replicates)
                    for budget in budgets
                )
            ),
            "budgets": budgets,
            "replicates": int(args.replicates),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "batch_mode": "vectorized",
            "same_init_and_batch_seed_across_A_B_within_trial": True,
            "cohort": (
                "fixed 384-anchor powered structural infoset cohort selected "
                "by original production A/B fold shift"
            ),
            "reference": (
                "reuse stored low-noise q_canonical from structural "
                "confirmation; no reference recomputation"
            ),
        },
        "candidate_counts_by_seed": candidate_counts,
        "production_reproduction": production,
        "trials": trials,
        "by_budget": by_budget,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("LT2_JAMMER_FAI_CONTROLLED_REFIT_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
