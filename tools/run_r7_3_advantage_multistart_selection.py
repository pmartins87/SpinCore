from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from spincore.deep_cfr import regret_matching_policy
from spincore.r7 import stratified_audit_indices
from spincore.solver import SolverLibrary

from run_r7_3_advantage_fit_sign_sensitivity import (
    DEFAULT_DECK_STREAM_SEED,
    _pair_metrics,
    _predict,
    collect_common_memory,
    train_replica,
)


REPLICA_SPECS = tuple((0x81000 + i * 0x101, 0xC1000 + i * 0x211) for i in range(8))


def _target_policy_tv(pred: torch.Tensor, samples) -> dict[str, float]:
    weighted_tv = 0.0
    total_weight = 0.0
    tvs = []
    support_equal_weight = 0.0
    for i, sample in enumerate(samples):
        legal = tuple(a for a, yes in enumerate(sample.legal) if yes)
        target_values = [float(x) for x in sample.target]
        pred_values = [float(x) for x in pred[i].tolist()]
        target_policy = regret_matching_policy(target_values, legal)
        pred_policy = regret_matching_policy(pred_values, legal)
        tv = 0.5 * sum(abs(a - b) for a, b in zip(target_policy, pred_policy))
        weight = float(sample.weight)
        weighted_tv += weight * tv
        total_weight += weight
        tvs.append(tv)
        target_support = tuple(target_values[a] > 0.0 for a in legal)
        pred_support = tuple(pred_values[a] > 0.0 for a in legal)
        if target_support == pred_support:
            support_equal_weight += weight
    tensor = torch.tensor(tvs, dtype=torch.float32)
    return {
        "target_policy_weighted_mean_tv": weighted_tv / max(total_weight, 1e-12),
        "target_policy_p50_tv": float(torch.quantile(tensor, torch.tensor(0.50))) if len(tvs) else math.inf,
        "target_policy_p95_tv": float(torch.quantile(tensor, torch.tensor(0.95))) if len(tvs) else math.inf,
        "target_positive_support_weighted_equal_fraction": support_equal_weight / max(total_weight, 1e-12),
    }


def _weighted_nrmse(pred: torch.Tensor, samples) -> float:
    target = torch.tensor([x.target for x in samples], dtype=torch.float32)
    legal = torch.tensor([x.legal for x in samples], dtype=torch.bool)
    weights = torch.tensor([x.weight for x in samples], dtype=torch.float32)
    mask = legal.float()
    count = mask.sum(1).clamp_min(1.0)
    per_sq = (((pred - target) ** 2) * mask).sum(1) / count
    per_energy = ((target**2) * mask).sum(1) / count
    w = weights / weights.mean().clamp_min(1e-12)
    mse = (per_sq * w).sum() / w.sum().clamp_min(1e-12)
    energy = (per_energy * w).sum() / w.sum().clamp_min(1e-12)
    return float(torch.sqrt(mse / energy.clamp_min(1e-12)))


def _select(reports, members):
    eligible = [i for i in members if float(reports[i]["eval_weighted_nrmse"]) <= 0.75]
    if not eligible:
        eligible = list(members)
    return min(eligible, key=lambda i: float(reports[i]["target_policy_weighted_mean_tv"]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Behavior-aware multistart selection for AdvantageNet")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_ADVANTAGE_MULTISTART_SELECTION_256.json"))
    ap.add_argument("--roots", type=int, default=256)
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_DECK_STREAM_SEED)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--chunk-steps", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=4096)
    ap.add_argument("--fit-target", type=float, default=0.50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=1024)
    ap.add_argument("--eval-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    args._solver_obj = solver
    started = time.time()
    memory, collection = collect_common_memory(
        solver=solver,
        roots=int(args.roots),
        deck_stream_seed=int(args.deck_stream_seed),
        reservoir_capacity=int(args.reservoir_capacity),
        device=args.device,
    )
    state = memory.state_dict()
    ids = stratified_audit_indices(len(memory.items), int(args.eval_size), 0x5E1EC7)
    samples = [memory.items[i] for i in ids]
    observations = [x.observation for x in samples]
    legal_masks = [tuple(int(v) for v in x.legal) for x in samples]

    predictions = []
    reports = []
    for init_seed, batch_seed in REPLICA_SPECS:
        bundle, report = train_replica(
            memory_state=state,
            init_seed=int(init_seed),
            batch_seed=int(batch_seed),
            args=args,
        )
        pred = _predict(bundle.advantage, observations, args.device)
        report["eval_weighted_nrmse"] = _weighted_nrmse(pred, samples)
        report.update(_target_policy_tv(pred, samples))
        predictions.append(pred)
        reports.append(report)

    group_a = tuple(range(4))
    group_b = tuple(range(4, 8))
    selected_a = _select(reports, group_a)
    selected_b = _select(reports, group_b)
    selected_pair = _pair_metrics(
        predictions[selected_a], predictions[selected_b], legal_masks
    )

    # Compare with the four fixed cross-group single-model pairs that use the
    # same training budget but no behavior-aware selection.
    unselected_pairs = []
    for i in range(4):
        metrics = _pair_metrics(predictions[i], predictions[i + 4], legal_masks)
        metrics["left"] = i
        metrics["right"] = i + 4
        unselected_pairs.append(metrics)
    baseline_mean = sum(float(x["regret_matching_mean_tv"]) for x in unselected_pairs) / 4.0
    baseline_p95 = sum(float(x["regret_matching_p95_tv"]) for x in unselected_pairs) / 4.0
    selected_mean = float(selected_pair["regret_matching_mean_tv"])
    selected_p95 = float(selected_pair["regret_matching_p95_tv"])

    payload = {
        "schema": "SPINCORE_R7_3_ADVANTAGE_MULTISTART_SELECTION_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "collection": collection,
        "same_memory_for_all_replicas": True,
        "replicas": reports,
        "selection": {
            "group_A_members": list(group_a),
            "group_B_members": list(group_b),
            "selected_A": int(selected_a),
            "selected_B": int(selected_b),
            "criterion": "MIN_TARGET_REGRET_POLICY_WEIGHTED_TV_SUBJECT_TO_FROZEN_NRMSE_GATE",
            "selected_pair": selected_pair,
        },
        "unselected_cross_group_pairs": unselected_pairs,
        "summary": {
            "unselected_mean_tv_average": float(baseline_mean),
            "unselected_p95_tv_average": float(baseline_p95),
            "selected_mean_tv": float(selected_mean),
            "selected_p95_tv": float(selected_p95),
            "selected_to_unselected_mean_ratio": selected_mean / max(baseline_mean, 1e-12),
            "selected_to_unselected_p95_ratio": selected_p95 / max(baseline_p95, 1e-12),
            "diagnosis": (
                "BEHAVIOR_AWARE_MULTISTART_SELECTION_MATERIAL"
                if min(
                    selected_mean / max(baseline_mean, 1e-12),
                    selected_p95 / max(baseline_p95, 1e-12),
                ) <= 0.75
                else "BEHAVIOR_AWARE_MULTISTART_SELECTION_NOT_MATERIAL_AT_SCREEN_SCALE"
            ),
        },
        "interpretation_note": (
            "Diagnostic only. Eight independent AdvantageNet fits see the exact same frozen "
            "memory. Each disjoint four-model group selects the gate-passing model whose hard "
            "regret-matching policy is closest to the memory's own target-regret policy. This "
            "tests whether behavior-aware restart selection can stabilize CFR without changing "
            "targets or the production regret map."
        ),
        "acceptance_gate_changed": False,
        "production_multistart_selection_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
