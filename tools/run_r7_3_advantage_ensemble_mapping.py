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


REPLICA_SPECS = tuple((0x71000 + i * 0x101, 0xB1000 + i * 0x211) for i in range(8))
GROUPS = {
    "size_1": [((0,), (4,)), ((1,), (5,)), ((2,), (6,)), ((3,), (7,))],
    "size_2": [((0, 1), (4, 5)), ((2, 3), (6, 7))],
    "size_4": [((0, 1, 2, 3), (4, 5, 6, 7))],
}


def _q(xs, q):
    if not xs:
        return math.inf
    t = torch.tensor(xs, dtype=torch.float32)
    return float(torch.quantile(t, torch.tensor(float(q))))


def _hard_policy(pred: torch.Tensor, legal_mask: tuple[int, ...]) -> list[float]:
    legal = tuple(i for i, yes in enumerate(legal_mask) if yes)
    return regret_matching_policy([float(x) for x in pred.tolist()], legal)


def _raw_average_policy(predictions, members, obs_index, legal_mask):
    raw = torch.stack([predictions[i][obs_index] for i in members], dim=0).mean(dim=0)
    return _hard_policy(raw, legal_mask)


def _policy_average(predictions, members, obs_index, legal_mask):
    policies = [_hard_policy(predictions[i][obs_index], legal_mask) for i in members]
    return [sum(p[a] for p in policies) / len(policies) for a in range(6)]


def _compare(predictions, left, right, legal_masks, mapping):
    tvs = []
    for obs_index, legal_mask in enumerate(legal_masks):
        if mapping == "raw_then_regret_matching":
            a = _raw_average_policy(predictions, left, obs_index, legal_mask)
            b = _raw_average_policy(predictions, right, obs_index, legal_mask)
        elif mapping == "regret_matching_then_policy_average":
            a = _policy_average(predictions, left, obs_index, legal_mask)
            b = _policy_average(predictions, right, obs_index, legal_mask)
        else:
            raise ValueError(mapping)
        tvs.append(0.5 * sum(abs(x - y) for x, y in zip(a, b)))
    return {
        "mean_tv": sum(tvs) / max(len(tvs), 1),
        "p50_tv": _q(tvs, 0.50),
        "p95_tv": _q(tvs, 0.95),
        "max_tv": max(tvs) if tvs else math.inf,
    }


def _avg(rows, key):
    return sum(float(x[key]) for x in rows) / max(len(rows), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare Advantage ensemble mapping orders")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_ADVANTAGE_ENSEMBLE_MAPPING_256.json"))
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
    ids = stratified_audit_indices(len(memory.items), int(args.eval_size), 0xE4A991)
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
    for mapping in ("raw_then_regret_matching", "regret_matching_then_policy_average"):
        by_size = {}
        for size, pairs in GROUPS.items():
            rows = []
            for left, right in pairs:
                row = _compare(predictions, left, right, legal_masks, mapping)
                row["left_members"] = list(left)
                row["right_members"] = list(right)
                rows.append(row)
            by_size[size] = {
                "pairs": rows,
                "mean_tv": _avg(rows, "mean_tv"),
                "mean_p95_tv": _avg(rows, "p95_tv"),
                "mean_max_tv": _avg(rows, "max_tv"),
            }
        results[mapping] = by_size

    raw4 = results["raw_then_regret_matching"]["size_4"]
    policy4 = results["regret_matching_then_policy_average"]["size_4"]
    payload = {
        "schema": "SPINCORE_R7_3_ADVANTAGE_ENSEMBLE_MAPPING_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "collection": collection,
        "same_memory_for_all_replicas": True,
        "replica_count": 8,
        "replicas": replicas,
        "results": results,
        "summary": {
            "raw_average_size4_mean_tv": float(raw4["mean_tv"]),
            "raw_average_size4_p95_tv": float(raw4["mean_p95_tv"]),
            "policy_mixture_size4_mean_tv": float(policy4["mean_tv"]),
            "policy_mixture_size4_p95_tv": float(policy4["mean_p95_tv"]),
            "policy_to_raw_size4_mean_ratio": float(policy4["mean_tv"] / max(raw4["mean_tv"], 1e-12)),
            "policy_to_raw_size4_p95_ratio": float(policy4["mean_p95_tv"] / max(raw4["mean_p95_tv"], 1e-12)),
            "diagnosis": (
                "POLICY_MIXTURE_ENSEMBLE_MAPPING_MATERIAL"
                if min(
                    policy4["mean_tv"] / max(raw4["mean_tv"], 1e-12),
                    policy4["mean_p95_tv"] / max(raw4["mean_p95_tv"], 1e-12),
                ) <= 0.80
                else "RAW_ADVANTAGE_AVERAGING_PREFERRED_OR_MAPPING_DIFFERENCE_SMALL"
            ),
        },
        "interpretation_note": (
            "Same eight independently fitted AdvantageNets and same frozen reservoir are used for "
            "both mappings. Raw-then-regret-matching averages predicted advantages first and then "
            "uses the unchanged hard map. Policy-mixture first applies hard regret matching to "
            "each member and then averages legal-action probabilities. This is a diagnostic of "
            "nonlinear sign/support amplification, not a production policy change."
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
