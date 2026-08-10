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
    _predict,
    collect_common_memory,
    train_replica,
)


REPLICA_SPECS = tuple((0x91000 + i * 0x101, 0xD1000 + i * 0x211) for i in range(8))
PAIRS = [((0, 1, 2, 3), (4, 5, 6, 7))]


def _q(xs, q):
    if not xs:
        return math.inf
    t = torch.tensor(xs, dtype=torch.float32)
    return float(torch.quantile(t, torch.tensor(float(q))))


def _member_policies(predictions, members, obs_index, legal_mask):
    legal = tuple(i for i, yes in enumerate(legal_mask) if yes)
    return [
        regret_matching_policy([float(x) for x in predictions[i][obs_index].tolist()], legal)
        for i in members
    ]


def _aggregate(policies, legal_mask, mode):
    legal = [i for i, yes in enumerate(legal_mask) if yes]
    out = [0.0] * 6
    for action in legal:
        xs = sorted(float(p[action]) for p in policies)
        if mode == "mean":
            value = sum(xs) / len(xs)
        elif mode == "median":
            n = len(xs)
            value = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
        elif mode == "trimmed_mean":
            core = xs[1:-1] if len(xs) >= 4 else xs
            value = sum(core) / len(core)
        else:
            raise ValueError(mode)
        out[action] = value
    total = sum(out[a] for a in legal)
    if total <= 0.0:
        value = 1.0 / len(legal)
        for action in legal:
            out[action] = value
    else:
        for action in legal:
            out[action] /= total
    return out


def _compare(predictions, left, right, legal_masks, mode):
    tvs = []
    for obs_index, legal_mask in enumerate(legal_masks):
        pa = _aggregate(_member_policies(predictions, left, obs_index, legal_mask), legal_mask, mode)
        pb = _aggregate(_member_policies(predictions, right, obs_index, legal_mask), legal_mask, mode)
        tvs.append(0.5 * sum(abs(float(x) - float(y)) for x, y in zip(pa, pb)))
    return {
        "mean_tv": sum(tvs) / max(len(tvs), 1),
        "p50_tv": _q(tvs, 0.50),
        "p95_tv": _q(tvs, 0.95),
        "max_tv": max(tvs) if tvs else math.inf,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare mean/median/trimmed aggregation of regret-matched Advantage ensemble policies")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_ADVANTAGE_ROBUST_POLICY_AGGREGATION_256.json"))
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
    ids = stratified_audit_indices(len(memory.items), int(args.eval_size), 0xA661E)
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
        predictions.append(_predict(bundle.advantage, observations, args.device))
        replicas.append(report)

    results = {}
    left, right = PAIRS[0]
    for mode in ("mean", "median", "trimmed_mean"):
        results[mode] = _compare(predictions, left, right, legal_masks, mode)
        results[mode]["left_members"] = list(left)
        results[mode]["right_members"] = list(right)

    mean = results["mean"]
    best_mode = min(("median", "trimmed_mean"), key=lambda m: (results[m]["p95_tv"], results[m]["mean_tv"]))
    best = results[best_mode]
    payload = {
        "schema": "SPINCORE_R7_3_ADVANTAGE_ROBUST_POLICY_AGGREGATION_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "collection": collection,
        "same_memory_for_all_replicas": True,
        "replica_count": 8,
        "ensemble_size_per_side": 4,
        "replicas": replicas,
        "results": results,
        "summary": {
            "mean_aggregation_mean_tv": float(mean["mean_tv"]),
            "mean_aggregation_p95_tv": float(mean["p95_tv"]),
            "best_robust_mode": best_mode,
            "best_robust_mean_tv": float(best["mean_tv"]),
            "best_robust_p95_tv": float(best["p95_tv"]),
            "best_to_mean_mean_ratio": float(best["mean_tv"] / max(mean["mean_tv"], 1e-12)),
            "best_to_mean_p95_ratio": float(best["p95_tv"] / max(mean["p95_tv"], 1e-12)),
            "diagnosis": (
                "ROBUST_POLICY_AGGREGATION_MATERIAL"
                if min(
                    best["mean_tv"] / max(mean["mean_tv"], 1e-12),
                    best["p95_tv"] / max(mean["p95_tv"], 1e-12),
                ) <= 0.85
                else "ROBUST_POLICY_AGGREGATION_NOT_MATERIAL"
            ),
        },
        "interpretation_note": (
            "Same-memory diagnostic only. Each member is first converted with the unchanged hard "
            "regret-matching map. Coordinatewise median and trimmed-mean aggregation are then compared "
            "with ordinary probability averaging using the same eight independently fitted networks. "
            "The purpose is to test whether rare member outliers dominate the policy-mixture p95 tail."
        ),
        "acceptance_gate_changed": False,
        "production_policy_mapping_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
