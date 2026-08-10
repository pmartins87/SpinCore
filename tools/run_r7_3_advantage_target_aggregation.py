from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from spincore.r7 import stratified_audit_indices
from spincore.solver import SolverLibrary
from spincore_nn.reservoir import AdvantageSample, UniformReservoir

from run_r7_3_advantage_ensemble_stability import _weighted_nrmse
from run_r7_3_advantage_fit_sign_sensitivity import (
    DEFAULT_DECK_STREAM_SEED,
    REPLICA_SPECS,
    _pair_metrics,
    _predict,
    collect_common_memory,
    train_replica,
)


def aggregate_advantage_memory(memory: UniformReservoir) -> UniformReservoir:
    # For weighted squared error, replacing all samples with the same
    # (observation, legal-mask) by their weighted target mean and total weight
    # preserves the exact population minimizer:
    #   sum_i w_i ||f-y_i||^2 = W ||f-y_bar||^2 + constant.
    groups: dict[tuple[bytes, tuple[int, ...]], dict] = {}
    for sample in memory.items:
        key = (sample.observation, tuple(int(x) for x in sample.legal))
        row = groups.get(key)
        if row is None:
            row = {
                "weight": 0.0,
                "target_sum": [0.0] * len(sample.target),
                "max_iteration": int(sample.iteration),
            }
            groups[key] = row
        w = float(sample.weight)
        row["weight"] += w
        row["max_iteration"] = max(int(row["max_iteration"]), int(sample.iteration))
        for a, value in enumerate(sample.target):
            row["target_sum"][a] += w * float(value)

    aggregated = []
    for (observation, legal), row in groups.items():
        weight = max(float(row["weight"]), 1e-12)
        target = tuple(float(x) / weight for x in row["target_sum"])
        aggregated.append(
            AdvantageSample(
                observation,
                tuple(int(x) for x in legal),
                target,
                float(row["weight"]),
                int(row["max_iteration"]),
            )
        )

    # Reuse the tested serialization constructor, then replace only the logical
    # reservoir contents. The aggregated set is no larger than the original and
    # therefore remains within the original capacity.
    out = UniformReservoir.from_state_dict(memory.state_dict())
    out.items = aggregated
    out.seen = len(aggregated)
    return out


def _mean(rows, key):
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def run_variant(*, name: str, train_memory: UniformReservoir, eval_samples, legal_masks, args):
    state = train_memory.state_dict()
    predictions = []
    replicas = []
    for init_seed, batch_seed in REPLICA_SPECS:
        bundle, report = train_replica(
            memory_state=state,
            init_seed=int(init_seed),
            batch_seed=int(batch_seed),
            args=args,
        )
        pred = _predict(bundle.advantage, [x.observation for x in eval_samples], args.device)
        report["original_memory_weighted_nrmse"] = _weighted_nrmse(pred, eval_samples)
        predictions.append(pred)
        replicas.append(report)

    pairs = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            metrics = _pair_metrics(predictions[i], predictions[j], legal_masks)
            metrics["left"] = i
            metrics["right"] = j
            pairs.append(metrics)
    return {
        "name": name,
        "train_sample_count": len(train_memory.items),
        "train_seen": int(train_memory.seen),
        "replicas": replicas,
        "pairwise": pairs,
        "summary": {
            "pairwise_mean_tv": _mean(pairs, "regret_matching_mean_tv"),
            "pairwise_p95_tv": _mean(pairs, "regret_matching_p95_tv"),
            "positive_support_equal_fraction": _mean(pairs, "positive_support_equal_fraction"),
            "mean_original_memory_nrmse": _mean(replicas, "original_memory_weighted_nrmse"),
            "max_original_memory_nrmse": max(float(x["original_memory_weighted_nrmse"]) for x in replicas),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Exact weighted duplicate-target aggregation for AdvantageNet stability")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_ADVANTAGE_TARGET_AGGREGATION_256.json"))
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
    raw_memory, collection = collect_common_memory(
        solver=solver,
        roots=int(args.roots),
        deck_stream_seed=int(args.deck_stream_seed),
        reservoir_capacity=int(args.reservoir_capacity),
        device=args.device,
    )
    aggregated_memory = aggregate_advantage_memory(raw_memory)

    ids = stratified_audit_indices(len(raw_memory.items), int(args.eval_size), 0xA66E6A7E)
    eval_samples = [raw_memory.items[i] for i in ids]
    legal_masks = [tuple(int(v) for v in x.legal) for x in eval_samples]

    raw = run_variant(
        name="raw_samples",
        train_memory=raw_memory,
        eval_samples=eval_samples,
        legal_masks=legal_masks,
        args=args,
    )
    aggregated = run_variant(
        name="weighted_duplicate_aggregation",
        train_memory=aggregated_memory,
        eval_samples=eval_samples,
        legal_masks=legal_masks,
        args=args,
    )

    mean_ratio = float(aggregated["summary"]["pairwise_mean_tv"]) / max(
        float(raw["summary"]["pairwise_mean_tv"]), 1e-12
    )
    p95_ratio = float(aggregated["summary"]["pairwise_p95_tv"]) / max(
        float(raw["summary"]["pairwise_p95_tv"]), 1e-12
    )
    payload = {
        "schema": "SPINCORE_R7_3_ADVANTAGE_TARGET_AGGREGATION_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "collection": collection,
        "raw_sample_count": len(raw_memory.items),
        "aggregated_unique_count": len(aggregated_memory.items),
        "aggregation_ratio": len(aggregated_memory.items) / max(len(raw_memory.items), 1),
        "population_weighted_mse_minimizer_preserved": True,
        "raw": raw,
        "aggregated": aggregated,
        "summary": {
            "aggregated_to_raw_mean_tv_ratio": float(mean_ratio),
            "aggregated_to_raw_p95_tv_ratio": float(p95_ratio),
            "diagnosis": (
                "WEIGHTED_TARGET_AGGREGATION_MATERIAL_FOR_FIT_STABILITY"
                if min(mean_ratio, p95_ratio) <= 0.85
                else "WEIGHTED_TARGET_AGGREGATION_NOT_MATERIAL_AT_SCREEN_SCALE"
            ),
        },
        "interpretation_note": (
            "Diagnostic only. Duplicate samples with exactly the same observation/legal mask are "
            "replaced by one weighted-mean target with summed LCFR weight. This preserves the "
            "exact minimizer of the weighted MSE population objective, while changing stochastic "
            "minibatch dynamics and reducing contradictory duplicate draws. Evaluation NRMSE and "
            "hard regret-policy stability are measured on the original unaggregated memory."
        ),
        "acceptance_gate_changed": False,
        "production_memory_aggregation_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
