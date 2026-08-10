from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from spincore.solver import SolverLibrary
from spincore.r7 import stratified_audit_indices

from run_r7_3_advantage_fit_sign_sensitivity import (
    DEFAULT_DECK_STREAM_SEED,
    _pair_metrics,
    _predict,
    collect_common_memory,
    train_replica,
)


REPLICA_SPECS = tuple(
    (0x51000 + i * 0x101, 0xA1000 + i * 0x211) for i in range(8)
)


def _ensemble(predictions: list[torch.Tensor], members: tuple[int, ...]) -> torch.Tensor:
    return torch.stack([predictions[i] for i in members], dim=0).mean(dim=0)


def _weighted_nrmse(pred: torch.Tensor, samples) -> float:
    target = torch.tensor([x.target for x in samples], dtype=torch.float32)
    legal = torch.tensor([x.legal for x in samples], dtype=torch.bool)
    weights = torch.tensor([x.weight for x in samples], dtype=torch.float32)
    mask = legal.float()
    per_sq = (((pred - target) ** 2) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    per_energy = ((target**2) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    w = weights / weights.mean().clamp_min(1e-12)
    mse = (per_sq * w).sum() / w.sum().clamp_min(1e-12)
    energy = (per_energy * w).sum() / w.sum().clamp_min(1e-12)
    return float(torch.sqrt(mse / energy.clamp_min(1e-12)))


def _mean(rows, key):
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="R7.3 same-memory Advantage ensemble stability")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_ADVANTAGE_ENSEMBLE_STABILITY_256.json"))
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
    ids = stratified_audit_indices(len(memory.items), int(args.eval_size), 0xE85EAB1E)
    samples = [memory.items[i] for i in ids]
    observations = [x.observation for x in samples]
    legal_masks = [tuple(int(v) for v in x.legal) for x in samples]

    predictions = []
    replicas = []
    for init_seed, batch_seed in REPLICA_SPECS:
        bundle, report = train_replica(
            memory_state=state,
            init_seed=int(init_seed),
            batch_seed=int(batch_seed),
            args=args,
        )
        pred = _predict(bundle.advantage, observations, args.device)
        report["eval_weighted_nrmse"] = _weighted_nrmse(pred, samples)
        predictions.append(pred)
        replicas.append(report)

    # Disjoint ensembles keep the stability comparison honest: no shared model
    # appears on both sides of an ensemble-vs-ensemble comparison.
    comparisons = {
        "size_1": [
            ((0,), (4,)),
            ((1,), (5,)),
            ((2,), (6,)),
            ((3,), (7,)),
        ],
        "size_2": [
            ((0, 1), (4, 5)),
            ((2, 3), (6, 7)),
        ],
        "size_4": [
            ((0, 1, 2, 3), (4, 5, 6, 7)),
        ],
    }

    results = {}
    for label, pairs in comparisons.items():
        pair_rows = []
        for left, right in pairs:
            a = _ensemble(predictions, left)
            b = _ensemble(predictions, right)
            metrics = _pair_metrics(a, b, legal_masks)
            metrics["left_members"] = list(left)
            metrics["right_members"] = list(right)
            metrics["left_weighted_nrmse"] = _weighted_nrmse(a, samples)
            metrics["right_weighted_nrmse"] = _weighted_nrmse(b, samples)
            pair_rows.append(metrics)
        results[label] = {
            "ensemble_size": len(pairs[0][0]),
            "pairs": pair_rows,
            "mean_regret_matching_tv": _mean(pair_rows, "regret_matching_mean_tv"),
            "mean_p95_tv": _mean(pair_rows, "regret_matching_p95_tv"),
            "mean_positive_support_equal_fraction": _mean(pair_rows, "positive_support_equal_fraction"),
            "mean_left_nrmse": _mean(pair_rows, "left_weighted_nrmse"),
            "mean_right_nrmse": _mean(pair_rows, "right_weighted_nrmse"),
        }

    size1 = float(results["size_1"]["mean_regret_matching_tv"])
    size2 = float(results["size_2"]["mean_regret_matching_tv"])
    size4 = float(results["size_4"]["mean_regret_matching_tv"])
    p95_1 = float(results["size_1"]["mean_p95_tv"])
    p95_4 = float(results["size_4"]["mean_p95_tv"])

    payload = {
        "schema": "SPINCORE_R7_3_ADVANTAGE_ENSEMBLE_STABILITY_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "collection": collection,
        "same_memory_for_all_replicas": True,
        "replica_count": 8,
        "replicas": replicas,
        "ensemble_results": results,
        "summary": {
            "size1_mean_tv": size1,
            "size2_mean_tv": size2,
            "size4_mean_tv": size4,
            "size4_to_size1_mean_tv_ratio": size4 / max(size1, 1e-12),
            "size1_mean_p95_tv": p95_1,
            "size4_mean_p95_tv": p95_4,
            "size4_to_size1_p95_tv_ratio": p95_4 / max(p95_1, 1e-12),
            "diagnosis": (
                "ADVANTAGE_ENSEMBLING_MATERIAL_FOR_FIT_STABILITY"
                if min(size4 / max(size1, 1e-12), p95_4 / max(p95_1, 1e-12)) <= 0.75
                else "ADVANTAGE_ENSEMBLING_NOT_MATERIAL_AT_SCREEN_SCALE"
            ),
        },
        "interpretation_note": (
            "Eight independently initialized/trained AdvantageNets see the exact same frozen "
            "Advantage reservoir. Disjoint 1-, 2- and 4-model ensembles average raw predicted "
            "advantages before the unchanged production hard regret-matching map. The experiment "
            "tests whether model averaging suppresses function-approximation/sign variance. It "
            "does not change production behavior or acceptance gates."
        ),
        "acceptance_gate_changed": False,
        "production_ensemble_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
