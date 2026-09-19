#!/usr/bin/env python3
from __future__ import annotations

"""Low-noise infoset audit conditioned only on Stage-B fold-mass shift direction.

The full-population actual-deal reconciliation established that the resolved
Jammer FAI loss is carried by states where Stage B folds more than Stage A.
This diagnostic asks whether that same sign appears under a low-noise infoset
reference rather than the realized hidden hand/board.

Selection is pre-reference and outcome-free:
  * common Stage-A/B Jammer FAI states only;
  * classify by production fold-mass shift, B_MORE_FOLD or B_LESS_FOLD;
  * do NOT inspect sampled FAI action, terminal outcome, or realized Q;
  * uniformly sample a fixed number per seed from each shift group.

Reference:
  * JAMMER action is hand-independent -> uniform compatible opponent hands;
  * uniform future boards;
  * common canonical Q-like action gaps;
  * exact0 after opponent is already all-in.

No training roots, optimizer steps, memory writes, or holdout seeds.
"""

import argparse
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch

import audit_lt2_hu_preflop_conditional_resampling as cond
import audit_lt2_jammer_fai_broad_calibration as broad
import audit_lt2_jammer_facing_allin_target_overlay as overlay
import audit_lt2_stage_a_b_first_divergence as fd
from audit_lt2_hu_preflop_target_estimator_budget import _mean_ci
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.solver import SolverLibrary

FORENSIC_SEEDS = (20260920, 20260921, 20260922, 20260923, 20260924, 20260925)
CHIP_SCALE = 1500.0
SHIFT_EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--stage-b", type=Path, required=True)
    p.add_argument("--scenarios-per-seed", type=int, default=5000)
    p.add_argument("--anchors-per-shift-per-seed", type=int, default=8)
    p.add_argument("--reference-hands", type=int, default=64)
    p.add_argument("--reference-boards-per-hand", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _sigma_from_anchor(stage, anchor):
    raw = cond._model_raw(
        stage.runtime,
        anchor["observation"],
        tuple(anchor["legal_mask"]),
    ).float()
    legal = tuple(int(x) for x in anchor["legal"])
    sigma = tuple(float(x) for x in lean_regret_matching_policy(raw.tolist(), legal))
    return raw, sigma


def _classify_shift(stage_a, stage_b, anchor):
    _, sigma_a = _sigma_from_anchor(stage_a, anchor)
    _, sigma_b = _sigma_from_anchor(stage_b, anchor)
    fold_a = float(sigma_a[0]) if 0 in anchor["legal"] else 0.0
    fold_b = float(sigma_b[0]) if 0 in anchor["legal"] else 0.0
    delta = fold_b - fold_a
    if delta > SHIFT_EPS:
        group = "B_MORE_FOLD"
    elif delta < -SHIFT_EPS:
        group = "B_LESS_FOLD"
    else:
        group = "NO_FOLD_SHIFT"
    return group, fold_a, fold_b, delta


def _cluster_ci(rows, getter):
    by_seed = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), []).append(float(getter(r)))
    seed_means = [float(statistics.fmean(v)) for _, v in sorted(by_seed.items())]
    return _mean_ci(seed_means)


def _anchor_ci(rows, getter):
    return _mean_ci([float(getter(r)) for r in rows])


def _group_summary(rows):
    classes = {}
    legal_counts = {}
    path_counts = {}
    for r in rows:
        oc = str(r["optimal_class"])
        classes[oc] = classes.get(oc, 0) + 1
        sig = ",".join(str(x) for x in r["legal"])
        legal_counts[sig] = legal_counts.get(sig, 0) + 1
        plen = str(r["common_public_action_count"])
        path_counts[plen] = path_counts.get(plen, 0) + 1

    fields = (
        "fold_mass_b_minus_a",
        "reference_fold_minus_continue_chips",
        "policy_value_b_minus_a_chips",
        "policy_regret_b_minus_a_chips",
        "class_error_b_minus_a",
        "canonical_gap_mse_b_minus_a",
        "raw_target_mse_b_minus_a",
    )
    out = {
        "n": len(rows),
        "reference_optimal_class_counts": classes,
        "legal_signature_counts": legal_counts,
        "common_public_action_count_counts": path_counts,
    }
    for field in fields:
        out[field] = {
            "anchor_ci": _anchor_ci(rows, lambda r, f=field: r[f]),
            "seed_cluster_ci": _cluster_ci(rows, lambda r, f=field: r[f]),
        }

    out["stage_a_policy_regret_chips"] = {
        "seed_cluster_ci": _cluster_ci(rows, lambda r: r["stage_a"]["policy_regret_chips"])
    }
    out["stage_b_policy_regret_chips"] = {
        "seed_cluster_ci": _cluster_ci(rows, lambda r: r["stage_b"]["policy_regret_chips"])
    }
    out["stage_a_class_error_mass"] = {
        "seed_cluster_ci": _cluster_ci(rows, lambda r: r["stage_a"]["class_error_mass"])
    }
    out["stage_b_class_error_mass"] = {
        "seed_cluster_ci": _cluster_ci(rows, lambda r: r["stage_b"]["class_error_mass"])
    }
    return out


def main() -> int:
    args = parse_args()
    if args.scenarios_per_seed <= 0 or args.anchors_per_shift_per_seed <= 0:
        raise SystemExit("positive scenarios/anchors required")
    if args.reference_hands <= 0 or args.reference_boards_per_hand <= 0:
        raise SystemExit("positive reference budget required")
    if args.threads <= 0:
        raise SystemExit("positive thread count required")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(args.threads))
    solver = SolverLibrary(args.solver.resolve(strict=True))

    snap_a = args.report.parent / "stage_a_hu_models.pt"
    snap_b = args.report.parent / "stage_b_hu_models.pt"
    meta_a = overlay._extract_stage_snapshot(args.stage_a.resolve(strict=True), snap_a)
    meta_b = overlay._extract_stage_snapshot(args.stage_b.resolve(strict=True), snap_b)
    stage_a = overlay.StageModels(snap_a, solver)
    stage_b = overlay.StageModels(snap_b, solver)

    selected = []
    candidate_counts = {}
    groups = ("B_MORE_FOLD", "B_LESS_FOLD")

    for seed in FORENSIC_SEEDS:
        pool = broad._collect_seed_common_fai(
            solver=solver,
            stage_a=stage_a,
            stage_b=stage_b,
            seed=int(seed),
            scenarios=int(args.scenarios_per_seed),
        )
        by_group = {g: [] for g in ("B_MORE_FOLD", "B_LESS_FOLD", "NO_FOLD_SHIFT")}
        for anchor in pool:
            group, fa, fb, delta = _classify_shift(stage_a, stage_b, anchor)
            anchor = dict(anchor)
            anchor["fold_shift_group"] = group
            anchor["fold_mass_a_pre_reference"] = float(fa)
            anchor["fold_mass_b_pre_reference"] = float(fb)
            anchor["fold_mass_b_minus_a_pre_reference"] = float(delta)
            by_group[group].append(anchor)

        candidate_counts[str(seed)] = {g: len(v) for g, v in by_group.items()}

        for gidx, group in enumerate(groups):
            candidates = by_group[group]
            need = int(args.anchors_per_shift_per_seed)
            if len(candidates) < need:
                raise RuntimeError(
                    f"seed {seed} group {group}: only {len(candidates)} candidates"
                )
            chooser = random.Random(fd._mix64(seed, 0xF01D5A17, gidx))
            indices = sorted(chooser.sample(range(len(candidates)), need))
            selected.extend(candidates[i] for i in indices)

    # Deterministic stable ordering before assigning anchor ids.
    selected.sort(
        key=lambda a: (
            int(a["seed"]),
            str(a["fold_shift_group"]),
            int(a["scenario_index"]),
            int(a["actor"]),
        )
    )
    for i, anchor in enumerate(selected):
        anchor["anchor_index"] = int(i)

    rows = []
    for i, anchor in enumerate(selected):
        print(
            f"FAI_FOLD_SHIFT anchor={i+1}/{len(selected)} "
            f"group={anchor['fold_shift_group']} seed={anchor['seed']} "
            f"scenario={anchor['scenario_index']} blind={anchor['blind']} "
            f"path_len={anchor['common_public_action_count']}",
            flush=True,
        )
        q, ref_meta = broad._common_reference(
            solver=solver,
            stage_a=stage_a,
            stage_b=stage_b,
            anchor=anchor,
            hands_n=int(args.reference_hands),
            boards_n=int(args.reference_boards_per_hand),
        )
        a = broad._stage_metrics(stage=stage_a, anchor=anchor, q=q)
        b = broad._stage_metrics(stage=stage_b, anchor=anchor, q=q)

        legal = tuple(int(x) for x in anchor["legal"])
        nonfold = [x for x in legal if x != 0]
        if not nonfold:
            raise RuntimeError("FAI anchor has no non-fold action")
        fold_q = float(q[0].item())
        continue_q = max(float(q[x].item()) for x in nonfold)
        nonfold_spread = (max(float(q[x].item()) for x in nonfold) -
                          min(float(q[x].item()) for x in nonfold)) * CHIP_SCALE
        if abs(nonfold_spread) > 1e-5:
            raise RuntimeError(
                f"non-fold value equivalence failed at anchor {i}: {nonfold_spread}"
            )

        value_delta = (
            float(b["policy_value_normalized"]) - float(a["policy_value_normalized"])
        ) * CHIP_SCALE
        regret_delta = float(b["policy_regret_chips"]) - float(a["policy_regret_chips"])

        # Same Q and same best value imply exact negative relation.
        if abs(value_delta + regret_delta) > 2e-4:
            raise RuntimeError(
                f"policy-value/regret identity failed at anchor {i}: "
                f"value_delta={value_delta} regret_delta={regret_delta}"
            )

        rows.append({
            "anchor_index": int(i),
            "seed": int(anchor["seed"]),
            "scenario_index": int(anchor["scenario_index"]),
            "blind": str(anchor["blind"]),
            "actor": int(anchor["actor"]),
            "fold_shift_group": str(anchor["fold_shift_group"]),
            "common_public_action_count": int(anchor["common_public_action_count"]),
            "legal": list(legal),
            "fold_mass_a_pre_reference": float(anchor["fold_mass_a_pre_reference"]),
            "fold_mass_b_pre_reference": float(anchor["fold_mass_b_pre_reference"]),
            "fold_mass_b_minus_a": float(anchor["fold_mass_b_minus_a_pre_reference"]),
            "reference": ref_meta,
            "reference_fold_minus_continue_chips": float(
                (fold_q - continue_q) * CHIP_SCALE
            ),
            "optimal_class": str(a["optimal_class"]),
            "policy_value_b_minus_a_chips": float(value_delta),
            "policy_regret_b_minus_a_chips": float(regret_delta),
            "class_error_b_minus_a": float(
                b["class_error_mass"] - a["class_error_mass"]
            ),
            "canonical_gap_mse_b_minus_a": float(
                b["canonical_action_gap_mse"] - a["canonical_action_gap_mse"]
            ),
            "raw_target_mse_b_minus_a": float(
                b["raw_target_mse"] - a["raw_target_mse"]
            ),
            "stage_a": a,
            "stage_b": b,
        })

    max_gap_delta = max(
        float(r["reference"]["max_stage_a_b_canonical_gap_delta"]) for r in rows
    )
    if max_gap_delta > 1e-7:
        raise RuntimeError(f"stage action-gap invariance failed: {max_gap_delta}")

    summary = {
        "n": len(rows),
        "groups": {
            group: _group_summary([r for r in rows if r["fold_shift_group"] == group])
            for group in groups
        },
    }

    report = {
        "schema": "SPINCORE_LT2_JAMMER_FAI_FOLD_SHIFT_INFOSET_V1",
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
            "new_training_roots": 0,
            "optimizer_steps": 0,
            "training_memory_writes": 0,
            "forensic_seeds": list(FORENSIC_SEEDS),
            "future_holdout_seeds_touched": False,
            "scenarios_per_seed": int(args.scenarios_per_seed),
            "anchors_per_shift_per_seed": int(args.anchors_per_shift_per_seed),
            "selection": (
                "common Jammer FAI states before FAI action; classify only from "
                "Stage-B-minus-A production fold-mass sign; uniform deterministic "
                "sample within B_MORE_FOLD and B_LESS_FOLD per seed; no sampled "
                "FAI action, terminal outcome, actual-deal Q, or low-noise Q used "
                "for selection"
            ),
            "reference": {
                "opponent_hand_posterior": (
                    "uniform compatible hands because JAMMER action is hand-independent"
                ),
                "hands": int(args.reference_hands),
                "boards_per_hand": int(args.reference_boards_per_hand),
                "exact_opponent_levels": 0,
                "canonical_gauge": "Q(a)-mean_legal(Q)",
            },
            "primary_metrics": [
                "policy_value_b_minus_a_chips",
                "policy_regret_b_minus_a_chips",
                "reference_fold_minus_continue_chips",
                "class_error_b_minus_a",
                "fold_mass_b_minus_a",
            ],
        },
        "candidate_counts_by_seed": candidate_counts,
        "max_stage_a_b_canonical_gap_delta": float(max_gap_delta),
        "summary": summary,
        "rows": rows,
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== JAMMER FAI FOLD-SHIFT LOW-NOISE INFOSET AUDIT ===")
    print(f"anchors={len(rows)} candidates_by_seed={candidate_counts}")
    for group in groups:
        s = summary["groups"][group]
        pv = s["policy_value_b_minus_a_chips"]["seed_cluster_ci"]
        fs = s["fold_mass_b_minus_a"]["seed_cluster_ci"]
        margin = s["reference_fold_minus_continue_chips"]["seed_cluster_ci"]
        print(
            f"{group}: n={s['n']} "
            f"policy_B-A={pv['mean']:+.3f} "
            f"CI95=[{pv['ci95_low']:+.3f},{pv['ci95_high']:+.3f}] "
            f"fold_B-A={fs['mean']:+.4f} "
            f"ref_fold_minus_continue={margin['mean']:+.2f}"
        )
        print(
            f"  optimal_classes={s['reference_optimal_class_counts']} "
            f"legal={s['legal_signature_counts']} "
            f"path_len={s['common_public_action_count_counts']}"
        )
    print("LT2_JAMMER_FAI_FOLD_SHIFT_INFOSET_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
