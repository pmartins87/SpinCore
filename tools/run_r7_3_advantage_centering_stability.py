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


REPLICA_SPECS = tuple((0xC3100 + i * 0x101, 0xD5100 + i * 0x211) for i in range(8))
PAIR_GROUPS = {
    "size_1": [((0,), (4,)), ((1,), (5,)), ((2,), (6,)), ((3,), (7,))],
    "size_2": [((0, 1), (4, 5)), ((2, 3), (6, 7))],
    "size_4": [((0, 1, 2, 3), (4, 5, 6, 7))],
}


def _q(xs, q):
    if not xs:
        return math.inf
    t = torch.tensor(xs, dtype=torch.float32)
    return float(torch.quantile(t, torch.tensor(float(q))))


def _legal_values(raw: torch.Tensor, legal_mask: tuple[int, ...]) -> list[float]:
    return [float(raw[a]) for a, yes in enumerate(legal_mask) if yes]


def _center(raw: torch.Tensor, legal_mask: tuple[int, ...], mode: str) -> list[float]:
    values = [float(x) for x in raw.tolist()]
    legal = [a for a, yes in enumerate(legal_mask) if yes]
    xs = [values[a] for a in legal]
    if not xs:
        raise ValueError("empty legal mask")
    if mode == "raw":
        offset = 0.0
    elif mode == "legal_mean":
        offset = sum(xs) / len(xs)
    elif mode == "legal_median":
        ys = sorted(xs)
        n = len(ys)
        offset = ys[n // 2] if n % 2 else 0.5 * (ys[n // 2 - 1] + ys[n // 2])
    elif mode == "legal_midrange":
        offset = 0.5 * (min(xs) + max(xs))
    else:
        raise ValueError(mode)
    return [values[a] - offset if a in legal else 0.0 for a in range(6)]


def _ensemble_raw(predictions, members, obs_index):
    return torch.stack([predictions[i][obs_index] for i in members], dim=0).mean(dim=0)


def _policy(predictions, members, obs_index, legal_mask, mode):
    raw = _ensemble_raw(predictions, members, obs_index)
    centered = _center(raw, legal_mask, mode)
    legal = tuple(a for a, yes in enumerate(legal_mask) if yes)
    return regret_matching_policy(centered, legal)


def _compare(predictions, left, right, legal_masks, mode):
    tvs = []
    common_offset_abs = []
    residual_rms = []
    for obs_index, legal_mask in enumerate(legal_masks):
        pa = _policy(predictions, left, obs_index, legal_mask, mode)
        pb = _policy(predictions, right, obs_index, legal_mask, mode)
        tvs.append(0.5 * sum(abs(float(x) - float(y)) for x, y in zip(pa, pb)))

        raw_a = _ensemble_raw(predictions, left, obs_index)
        raw_b = _ensemble_raw(predictions, right, obs_index)
        legal = [a for a, yes in enumerate(legal_mask) if yes]
        diffs = [float(raw_a[a] - raw_b[a]) for a in legal]
        common = sum(diffs) / len(diffs)
        common_offset_abs.append(abs(common))
        residual_rms.append(math.sqrt(sum((x - common) ** 2 for x in diffs) / len(diffs)))
    return {
        "mean_tv": sum(tvs) / max(len(tvs), 1),
        "p50_tv": _q(tvs, 0.50),
        "p95_tv": _q(tvs, 0.95),
        "max_tv": max(tvs) if tvs else math.inf,
        "mean_abs_common_mode_difference": sum(common_offset_abs) / max(len(common_offset_abs), 1),
        "mean_centered_residual_rms_difference": sum(residual_rms) / max(len(residual_rms), 1),
    }


def _avg(rows, key):
    return sum(float(x[key]) for x in rows) / max(len(rows), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Test legal-action common-mode centering before hard regret matching")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_ADVANTAGE_CENTERING_STABILITY_256.json"))
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
    ids = stratified_audit_indices(len(memory.items), int(args.eval_size), 0xC37E11)
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
    for mode in ("raw", "legal_mean", "legal_median", "legal_midrange"):
        by_size = {}
        for size, pairs in PAIR_GROUPS.items():
            rows = []
            for left, right in pairs:
                row = _compare(predictions, left, right, legal_masks, mode)
                row["left_members"] = list(left)
                row["right_members"] = list(right)
                rows.append(row)
            by_size[size] = {
                "pairs": rows,
                "mean_tv": _avg(rows, "mean_tv"),
                "mean_p95_tv": _avg(rows, "p95_tv"),
                "mean_max_tv": _avg(rows, "max_tv"),
                "mean_abs_common_mode_difference": _avg(rows, "mean_abs_common_mode_difference"),
                "mean_centered_residual_rms_difference": _avg(rows, "mean_centered_residual_rms_difference"),
            }
        results[mode] = by_size

    raw1 = results["raw"]["size_1"]
    best_mode = min(
        ("legal_mean", "legal_median", "legal_midrange"),
        key=lambda m: (results[m]["size_1"]["mean_p95_tv"], results[m]["size_1"]["mean_tv"]),
    )
    best1 = results[best_mode]["size_1"]
    payload = {
        "schema": "SPINCORE_R7_3_ADVANTAGE_CENTERING_STABILITY_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "collection": collection,
        "same_memory_for_all_replicas": True,
        "replica_count": 8,
        "replicas": replicas,
        "results": results,
        "summary": {
            "raw_size1_mean_tv": float(raw1["mean_tv"]),
            "raw_size1_p95_tv": float(raw1["mean_p95_tv"]),
            "best_centering_mode": best_mode,
            "best_size1_mean_tv": float(best1["mean_tv"]),
            "best_size1_p95_tv": float(best1["mean_p95_tv"]),
            "best_to_raw_mean_ratio": float(best1["mean_tv"] / max(raw1["mean_tv"], 1e-12)),
            "best_to_raw_p95_ratio": float(best1["mean_p95_tv"] / max(raw1["mean_p95_tv"], 1e-12)),
            "diagnosis": (
                "ADVANTAGE_COMMON_MODE_CENTERING_MATERIAL"
                if min(
                    best1["mean_tv"] / max(raw1["mean_tv"], 1e-12),
                    best1["mean_p95_tv"] / max(raw1["mean_p95_tv"], 1e-12),
                ) <= 0.80
                else "ADVANTAGE_COMMON_MODE_CENTERING_NOT_MATERIAL"
            ),
        },
        "interpretation_note": (
            "Diagnostic only. All fits use the same frozen Advantage memory. The experiment tests "
            "whether independent regressors mainly disagree by a legal-action common offset, which "
            "can flip hard-regret signs without representing relative action-value disagreement. "
            "Centering changes only the pre-regret-matching mapping in this diagnostic; production "
            "regret matching and gates are unchanged."
        ),
        "acceptance_gate_changed": False,
        "production_regret_mapping_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
