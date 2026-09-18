#!/usr/bin/env python3
from __future__ import annotations

"""Stage-A/B target-drift and model-tracking matrix on LT2 regression contexts.

Read-only diagnostic. Reuses the exact FAILURE/CONTROL selection design from the
cross-street future-chance audit, but now evaluates both preserved stages.

For every selected state:
  * same explicit hidden deal;
  * same future-board samples;
  * same target RNG seeds;
  * Stage-A conditional target process;
  * Stage-B conditional target process;
  * Stage-A current Advantage model;
  * Stage-B current Advantage model.

Raw Advantage vectors are canonicalized by subtracting the mean over legal
actions. This removes the stage-specific scalar baseline while preserving all
action-value gaps.

The audit separates:
  1. target nonstationarity: Stage-A conditional target -> Stage-B target;
  2. own-target approximation: model_A -> target_A and model_B -> target_B;
  3. tracking failure: model drift fails to follow target drift.

No training roots, optimizer steps, or memory writes.
"""

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch

import audit_lt2_cross_street_future_chance as cs
import audit_lt2_hu_preflop_conditional_resampling as cond
import audit_lt2_jammer_facing_allin_common_reference_v2 as common
import audit_lt2_jammer_facing_allin_target_overlay as jam
import audit_lt2_stage_a_b_first_divergence as fd
from audit_lt2_hu_preflop_target_estimator_budget import _mean_ci
from audit_lt2_checkpoint_fit import _selftest_lean_rm_policy_tensor
from spincore.solver import SolverLibrary

FORENSIC_SEEDS = cs.FORENSIC_SEEDS
GROUPS = cs.GROUPS
STATUS = cs.STATUS
CHIP_SCALE = 1500.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--stage-b", type=Path, required=True)
    p.add_argument("--scenarios-per-seed", type=int, default=5000)
    p.add_argument("--states-per-seed-status", type=int, default=2)
    p.add_argument("--future-boards", type=int, default=8)
    p.add_argument("--repeats-per-board", type=int, default=4)
    p.add_argument("--exact-opponent-levels", type=int, default=1)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _mse_legal(values: torch.Tensor, legal: torch.Tensor) -> float:
    vals = values[legal]
    if int(vals.numel()) <= 0:
        raise RuntimeError("empty legal action set")
    return float(torch.mean(vals * vals).item())


def _best_action(target: torch.Tensor, legal: torch.Tensor) -> int:
    masked = target.clone()
    masked[~legal] = float("-inf")
    return int(masked.argmax().item())


def _action_regret_chips(
    target: torch.Tensor,
    legal: torch.Tensor,
    action: int,
) -> float:
    if not bool(legal[int(action)].item()):
        return float("nan")
    best = float(target[legal].max().item())
    value = float(target[int(action)].item())
    return float(max(0.0, best - value) * CHIP_SCALE)


def _select_anchors(
    *,
    solver: SolverLibrary,
    stage_a: jam.StageModels,
    stage_b: jam.StageModels,
    scenarios_per_seed: int,
    states_per_seed_status: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}

    for seed in FORENSIC_SEEDS:
        pools = cs._collect_seed_candidates(
            solver=solver,
            stage_a=stage_a,
            stage_b=stage_b,
            seed=int(seed),
            scenarios=int(scenarios_per_seed),
        )
        counts[str(seed)] = {}
        for group in GROUPS:
            counts[str(seed)][group] = {}
            for status in STATUS:
                pool = pools[(group, status)]
                counts[str(seed)][group][status] = len(pool)
                need = int(states_per_seed_status)
                if len(pool) < need:
                    raise RuntimeError(
                        f"seed={seed} group={group} status={status} "
                        f"only {len(pool)} candidates; need {need}"
                    )
                chooser = random.Random(
                    fd._mix64(
                        seed,
                        GROUPS.index(group),
                        STATUS.index(status),
                        0x5E1EC7,
                    )
                )
                indices = sorted(chooser.sample(range(len(pool)), need))
                for idx in indices:
                    selected.append(pool[idx])

    for index, anchor in enumerate(selected):
        anchor["anchor_index"] = int(index)
    return selected, counts


def _conditional_reference_pair(
    *,
    solver: SolverLibrary,
    stage_a: jam.StageModels,
    stage_b: jam.StageModels,
    anchor: dict[str, Any],
    future_boards: int,
    repeats_per_board: int,
    exact_level: int,
) -> dict[str, Any]:
    legal = torch.tensor(anchor["legal_mask"], dtype=torch.bool)
    if int(legal.sum().item()) <= 0:
        raise RuntimeError("empty legal mask")

    ta_rows: list[torch.Tensor] = []
    tb_rows: list[torch.Tensor] = []
    nodes_a: list[int] = []
    nodes_b: list[int] = []

    for bpos in range(int(future_boards)):
        board_rng = random.Random(
            fd._mix64(
                anchor["seed"],
                anchor["scenario_index"],
                anchor["anchor_index"],
                bpos,
                0x7A2D21F7,
            )
        )
        board = cs._future_board(anchor, rng=board_rng)

        for rep in range(int(repeats_per_board)):
            rng_seed = fd._mix64(
                anchor["seed"],
                anchor["scenario_index"],
                anchor["anchor_index"],
                bpos,
                rep,
                0xA8B7C6D5,
            )
            ta_raw, na = cs._capture(
                solver=solver,
                runtime=stage_a.runtime,
                anchor=anchor,
                board=board,
                completed=int(stage_a.completed_iteration),
                exact_level=int(exact_level),
                rng_seed=int(rng_seed),
            )
            tb_raw, nb = cs._capture(
                solver=solver,
                runtime=stage_b.runtime,
                anchor=anchor,
                board=board,
                completed=int(stage_b.completed_iteration),
                exact_level=int(exact_level),
                rng_seed=int(rng_seed),
            )
            ta_rows.append(common._canonicalize_target(ta_raw, legal))
            tb_rows.append(common._canonicalize_target(tb_raw, legal))
            nodes_a.append(int(na))
            nodes_b.append(int(nb))

    ref_a = torch.stack(ta_rows).mean(dim=0)
    ref_b = torch.stack(tb_rows).mean(dim=0)

    model_a_raw = cond._model_raw(
        stage_a.runtime, anchor["observation"], tuple(anchor["legal_mask"])
    )
    model_b_raw = cond._model_raw(
        stage_b.runtime, anchor["observation"], tuple(anchor["legal_mask"])
    )
    model_a = common._canonicalize_target(model_a_raw, legal)
    model_b = common._canonicalize_target(model_b_raw, legal)

    target_delta = ref_b - ref_a
    model_delta = model_b - model_a

    own_a = _mse_legal(model_a - ref_a, legal)
    own_b = _mse_legal(model_b - ref_b, legal)
    target_drift = _mse_legal(target_delta, legal)
    model_drift = _mse_legal(model_delta, legal)
    tracking_error = _mse_legal(model_delta - target_delta, legal)
    cross_b_to_a = _mse_legal(model_b - ref_a, legal)
    cross_a_to_b = _mse_legal(model_a - ref_b, legal)

    best_a = _best_action(ref_a, legal)
    best_b = _best_action(ref_b, legal)
    model_best_a = _best_action(model_a, legal)
    model_best_b = _best_action(model_b, legal)

    return {
        "target_drift_mse": target_drift,
        "model_drift_mse": model_drift,
        "tracking_error_mse": tracking_error,
        "stage_a_own_target_model_mse": own_a,
        "stage_b_own_target_model_mse": own_b,
        "stage_b_minus_a_own_target_model_mse": float(own_b - own_a),
        "stage_b_model_to_stage_a_target_mse": cross_b_to_a,
        "stage_a_model_to_stage_b_target_mse": cross_a_to_b,
        "reference_best_action_a": int(best_a),
        "reference_best_action_b": int(best_b),
        "reference_best_action_changed": bool(best_a != best_b),
        "model_best_action_a": int(model_best_a),
        "model_best_action_b": int(model_best_b),
        "model_a_matches_own_reference_best": bool(model_best_a == best_a),
        "model_b_matches_own_reference_best": bool(model_best_b == best_b),
        "sampled_stage_a_action_regret_to_stage_a_target_chips": _action_regret_chips(
            ref_a, legal, int(anchor["a_slot"])
        ),
        "sampled_stage_b_action_regret_to_stage_b_target_chips": _action_regret_chips(
            ref_b, legal, int(anchor["b_slot"])
        ),
        "sampled_b_minus_a_own_target_action_regret_chips": float(
            _action_regret_chips(ref_b, legal, int(anchor["b_slot"]))
            - _action_regret_chips(ref_a, legal, int(anchor["a_slot"]))
        ),
        "reference_target_a": [float(x) for x in ref_a.tolist()],
        "reference_target_b": [float(x) for x in ref_b.tolist()],
        "canonical_model_a": [float(x) for x in model_a.tolist()],
        "canonical_model_b": [float(x) for x in model_b.tolist()],
        "mean_nodes_stage_a": float(statistics.fmean(nodes_a)),
        "mean_nodes_stage_b": float(statistics.fmean(nodes_b)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_fields = (
        "target_drift_mse",
        "model_drift_mse",
        "tracking_error_mse",
        "stage_a_own_target_model_mse",
        "stage_b_own_target_model_mse",
        "stage_b_minus_a_own_target_model_mse",
        "stage_b_model_to_stage_a_target_mse",
        "stage_a_model_to_stage_b_target_mse",
        "sampled_stage_a_action_regret_to_stage_a_target_chips",
        "sampled_stage_b_action_regret_to_stage_b_target_chips",
        "sampled_b_minus_a_own_target_action_regret_chips",
        "mean_nodes_stage_a",
        "mean_nodes_stage_b",
    )
    out = {
        "n": len(rows),
        "metrics": {
            field: _mean_ci([float(r[field]) for r in rows])
            for field in scalar_fields
        },
        "rates": {
            "reference_best_action_changed": float(
                statistics.fmean(
                    1.0 if r["reference_best_action_changed"] else 0.0
                    for r in rows
                )
            ),
            "model_a_matches_own_reference_best": float(
                statistics.fmean(
                    1.0 if r["model_a_matches_own_reference_best"] else 0.0
                    for r in rows
                )
            ),
            "model_b_matches_own_reference_best": float(
                statistics.fmean(
                    1.0 if r["model_b_matches_own_reference_best"] else 0.0
                    for r in rows
                )
            ),
        },
    }
    return out


def _independent_difference_ci(
    failure: dict[str, Any],
    control: dict[str, Any],
    field: str,
) -> dict[str, float]:
    f = failure["metrics"][field]
    c = control["metrics"][field]
    diff = float(f["mean"] - c["mean"])
    se = float((float(f["sem"]) ** 2 + float(c["sem"]) ** 2) ** 0.5)
    return {
        "mean_failure_minus_control": diff,
        "sem": se,
        "ci95_low": diff - 1.96 * se,
        "ci95_high": diff + 1.96 * se,
    }


def main() -> int:
    args = parse_args()
    if args.scenarios_per_seed <= 0 or args.states_per_seed_status <= 0:
        raise SystemExit("positive scenario/state counts required")
    if args.future_boards <= 0 or args.repeats_per_board <= 0:
        raise SystemExit("positive conditional-reference budget required")
    if args.exact_opponent_levels < 0:
        raise SystemExit("exact opponent level must be non-negative")
    if args.threads <= 0:
        raise SystemExit("positive thread count required")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(args.threads))
    _selftest_lean_rm_policy_tensor()

    solver = SolverLibrary(args.solver.resolve(strict=True))
    if not solver.explicit_deal_available:
        raise SystemExit("explicit-deal solver API required")

    stage_a_snapshot = args.report.parent / "stage_a_hu_models.pt"
    stage_b_snapshot = args.report.parent / "stage_b_hu_models.pt"
    meta_a = jam._extract_stage_snapshot(
        args.stage_a.resolve(strict=True), stage_a_snapshot
    )
    meta_b = jam._extract_stage_snapshot(
        args.stage_b.resolve(strict=True), stage_b_snapshot
    )
    stage_a = jam.StageModels(stage_a_snapshot, solver)
    stage_b = jam.StageModels(stage_b_snapshot, solver)

    selected, candidate_counts = _select_anchors(
        solver=solver,
        stage_a=stage_a,
        stage_b=stage_b,
        scenarios_per_seed=int(args.scenarios_per_seed),
        states_per_seed_status=int(args.states_per_seed_status),
    )

    rows = []
    for index, anchor in enumerate(selected):
        print(
            f"TARGET_DRIFT anchor={index+1}/{len(selected)} "
            f"group={anchor['group']} status={anchor['status']} "
            f"seed={anchor['seed']} scenario={anchor['scenario_index']} "
            f"transition={anchor['a_slot']}->{anchor['b_slot']}",
            flush=True,
        )
        metrics = _conditional_reference_pair(
            solver=solver,
            stage_a=stage_a,
            stage_b=stage_b,
            anchor=anchor,
            future_boards=int(args.future_boards),
            repeats_per_board=int(args.repeats_per_board),
            exact_level=int(args.exact_opponent_levels),
        )
        rows.append({
            "anchor_index": int(index),
            "group": str(anchor["group"]),
            "status": str(anchor["status"]),
            "seed": int(anchor["seed"]),
            "scenario_index": int(anchor["scenario_index"]),
            "street": int(anchor["street"]),
            "visible_board_count": int(anchor["visible_board_count"]),
            "a_slot": int(anchor["a_slot"]),
            "b_slot": int(anchor["b_slot"]),
            "a_class": str(anchor["a_class"]),
            "b_class": str(anchor["b_class"]),
            **metrics,
        })

    summaries: dict[str, Any] = {}
    comparison_fields = (
        "target_drift_mse",
        "stage_b_minus_a_own_target_model_mse",
        "tracking_error_mse",
        "sampled_b_minus_a_own_target_action_regret_chips",
    )
    for group in GROUPS:
        summaries[group] = {}
        for status in STATUS:
            rr = [
                r for r in rows
                if r["group"] == group and r["status"] == status
            ]
            summaries[group][status] = _aggregate(rr)

        summaries[group]["FAILURE_MINUS_CONTROL"] = {
            field: _independent_difference_ci(
                summaries[group]["FAILURE"],
                summaries[group]["CONTROL"],
                field,
            )
            for field in comparison_fields
        }

    report = {
        "schema": "SPINCORE_LT2_CROSS_STREET_TARGET_DRIFT_TRACKING_V1",
        "stage_a": {
            "checkpoint": str(args.stage_a.resolve()),
            "completed_iteration": int(meta_a["completed_iteration"]),
        },
        "stage_b": {
            "checkpoint": str(args.stage_b.resolve()),
            "completed_iteration": int(meta_b["completed_iteration"]),
        },
        "method": {
            "read_only": True,
            "training_memory_writes": 0,
            "optimizer_steps": 0,
            "new_training_roots": 0,
            "forensic_seeds": list(FORENSIC_SEEDS),
            "future_holdout_seeds_touched": False,
            "groups": list(GROUPS),
            "status": list(STATUS),
            "states_per_seed_status": int(args.states_per_seed_status),
            "selection_reused_from_cross_street_future_chance_v1": True,
            "hidden_hand": "actual dealt opponent hand held fixed",
            "future_board_sampling": (
                "same future-board samples are used for Stage A and Stage B on "
                "each anchor"
            ),
            "target_rng_pairing": (
                "same RNG seed supplied to Stage A and Stage B for each explicit "
                "deal/repeat"
            ),
            "conditional_reference": {
                "future_boards": int(args.future_boards),
                "repeats_per_board": int(args.repeats_per_board),
                "exact_opponent_levels": int(args.exact_opponent_levels),
            },
            "canonical_gauge": (
                "subtract legal-action mean from Advantage vectors; preserves "
                "action gaps and removes arbitrary/current-policy scalar baseline"
            ),
            "interpretation": {
                "target_drift_mse": (
                    "change in canonical conditional target from Stage A to Stage B"
                ),
                "own_target_model_mse": (
                    "function-approximation error of each stage to its own current "
                    "conditional target"
                ),
                "tracking_error_mse": (
                    "MSE of (model_B-model_A) - (target_B-target_A); high means "
                    "model drift failed to track target drift"
                ),
            },
        },
        "candidate_counts_by_seed": candidate_counts,
        "summaries": summaries,
        "rows": rows,
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== CROSS-STREET TARGET-DRIFT / MODEL-TRACKING MATRIX ===")
    for group in GROUPS:
        print(group)
        for status in STATUS:
            s = summaries[group][status]
            m = s["metrics"]
            print(
                f"  {status}: n={s['n']} "
                f"target_drift={m['target_drift_mse']['mean']:.6f} "
                f"Aerr={m['stage_a_own_target_model_mse']['mean']:.6f} "
                f"Berr={m['stage_b_own_target_model_mse']['mean']:.6f} "
                f"B-Aerr={m['stage_b_minus_a_own_target_model_mse']['mean']:+.6f} "
                f"tracking={m['tracking_error_mse']['mean']:.6f} "
                f"best_change={s['rates']['reference_best_action_changed']:.3f}"
            )
    print("LT2_CROSS_STREET_TARGET_DRIFT_TRACKING_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
