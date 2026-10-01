#!/usr/bin/env python3
from __future__ import annotations

"""Build the frozen fresh common-state bank for post-15315 rebuild stability.

The bank is evaluation-only. It is generated once from the validated teacher
trajectory using the preregistered seed/episode count and must never be used for
training or member selection.

PROJECT_CONTRACT_IDS:
TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,
VALID-025,VALID-031,VALID-035,VALID-036,VALID-037,SAFE-011
"""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
from audit_3h_semantic_sidecar_attribution_10105 import (
    decode_obs,
    private_semantics,
    rank,
    suit,
)
import post15315_collection as collect
from spincore.solver import SolverLibrary

PROTOCOL = ROOT / "contracts" / "post15315_multirebuild_preregistration.json"
EXPECTED_CHECKPOINT_SHA = distill.EXPECTED_SHA
ADV_SCHEMA = "SPINCORE_3H_SEMANTIC_LONG_ENSEMBLE_V1"
BANK_SCHEMA = "SPINCORE_POST15315_REBUILD_STABILITY_BANK_V1"
FINAL_ITERATION = 15315
FOLD = 0
ALL_IN = 9


def load_protocol(path: Path) -> dict:
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("schema") != "SPINCORE_POST15315_MULTIREBUILD_PREREG_V1":
        raise RuntimeError("wrong post-15315 preregistration schema")
    return p


def load_teacher(path: Path, expected_sha256: str):
    actual = distill.sha256(path.resolve(strict=True))
    if actual != str(expected_sha256):
        raise RuntimeError(
            f"teacher SHA mismatch: expected={expected_sha256} actual={actual}"
        )
    p = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if p.get("schema") != ADV_SCHEMA:
        raise RuntimeError("wrong teacher schema")
    if p.get("source_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("teacher source checkpoint mismatch")
    if int(p.get("completed_iteration", -1)) != FINAL_ITERATION:
        raise RuntimeError("teacher iteration mismatch")
    states = list(p.get("members") or [])
    if len(states) != 8:
        raise RuntimeError("expected eight teacher members")
    return [distill.load_semantic_advantage(s) for s in states], p, actual


def is_preflop_72o(obs: bytes) -> bool:
    hole, board, _numeric, _cat = decode_obs(obs)
    if board or len(hole) != 2:
        return False
    ranks = sorted((rank(hole[0]), rank(hole[1])))
    return bool(ranks == [2, 7] and suit(hole[0]) != suit(hole[1]))


def surface_flags(obs: bytes, legal) -> tuple[bool, bool, bool]:
    legal = tuple(bool(x) for x in legal)
    if len(legal) != 10:
        raise RuntimeError(f"expected ten-slot legal mask, got {len(legal)}")
    hole, board, _numeric, _cat = decode_obs(obs)
    ps = private_semantics(hole, board)
    pre72 = bool(is_preflop_72o(obs) and legal[ALL_IN])
    trips_fold = bool(
        len(board) >= 3 and int(ps["made"]) >= 3 and legal[FOLD]
    )
    hc_allin = bool(
        len(board) >= 3
        and bool(ps["high_card_no_draw"])
        and legal[ALL_IN]
    )
    return pre72, trips_fold, hc_allin


def samples_to_arrays(samples):
    if not samples:
        raise RuntimeError("empty stability bank")
    observations = np.stack(
        [np.frombuffer(bytes(s.observation), dtype=np.uint8) for s in samples],
        axis=0,
    )
    if observations.shape[1] != 126:
        raise RuntimeError(f"observation width drift: {observations.shape}")
    legal = np.asarray(
        [[bool(x) for x in s.legal] for s in samples],
        dtype=bool,
    )
    targets = np.asarray(
        [[float(x) for x in s.target] for s in samples],
        dtype=np.float32,
    )
    if legal.shape[1] != 10 or targets.shape[1] != 10:
        raise RuntimeError(
            f"native action carrier drift: legal={legal.shape} target={targets.shape}"
        )
    flags = [surface_flags(bytes(s.observation), s.legal) for s in samples]
    s72 = np.asarray([x[0] for x in flags], dtype=bool)
    trips = np.asarray([x[1] for x in flags], dtype=bool)
    hc = np.asarray([x[2] for x in flags], dtype=bool)
    return observations, legal, targets, s72, trips, hc


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--semantic-advantage", type=Path, required=True)
    ap.add_argument("--teacher-sha256", required=True)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--out-bank", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    protocol = load_protocol(args.protocol.resolve(strict=True))
    spec = protocol["shared_fresh_evaluation"]["stability_state_bank"]
    seed = int(spec["seed"])
    episodes = int(spec["forced_three_handed_episodes"])
    if seed != 153159001 or episodes != 80000:
        raise RuntimeError("frozen stability-bank seed/episode count drift")

    cp = args.checkpoint.resolve(strict=True)
    if distill.sha256(cp) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("source 10105 checkpoint SHA mismatch")

    torch.set_num_threads(int(args.threads))
    teacher, teacher_payload, teacher_sha = load_teacher(
        args.semantic_advantage, args.teacher_sha256
    )
    solver = SolverLibrary(args.solver.resolve(strict=True))

    train, hold, stats = collect.collect_fresh_split(
        solver,
        teacher,
        episodes,
        master_seed=seed,
        sample_iteration=FINAL_ITERATION,
        progress_prefix="POST15315_STABILITY_BANK",
    )
    # The split is not used for model selection; concatenation is a deterministic
    # storage convention only. Every collected state belongs to the same frozen
    # evaluation bank.
    samples = list(train) + list(hold)
    arrays = samples_to_arrays(samples)
    observations, legal, targets, s72, trips, hc = arrays

    coverage = {
        "preflop_72o_allin_opportunities": int(s72.sum()),
        "postflop_trips_plus_fold_legal": int(trips.sum()),
        "postflop_high_card_no_immediate_draw_allin_legal": int(hc.sum()),
    }
    minima = {k: int(v) for k, v in spec["required_surfaces"].items()}
    coverage_ok = {k: coverage[k] >= minima[k] for k in minima}
    if not all(coverage_ok.values()):
        raise RuntimeError(
            "frozen stability-bank surface coverage failed: "
            + json.dumps(
                {"coverage": coverage, "required": minima},
                sort_keys=True,
            )
        )

    protocol_sha = distill.sha256(args.protocol.resolve(strict=True))
    bank_meta = {
        "schema": BANK_SCHEMA,
        "protocol_sha256": protocol_sha,
        "teacher_sha256": teacher_sha,
        "teacher_schema": teacher_payload.get("schema"),
        "teacher_iteration": FINAL_ITERATION,
        "seed": seed,
        "forced_three_handed_episodes": episodes,
        "storage_order": "TRAIN_SPLIT_THEN_HOLD_SPLIT",
        "state_count": int(observations.shape[0]),
        "sample_digest_train": collect.sample_digest(train),
        "sample_digest_hold": collect.sample_digest(hold),
        "collection": collect.canonical_collection_stats(stats),
        "coverage": coverage,
        "coverage_minima": minima,
        "coverage_ok": coverage_ok,
        "data_use": spec["data_use"],
        "training_use_permitted": False,
        "member_selection_permitted": False,
    }
    meta_json = json.dumps(bank_meta, sort_keys=True).encode("utf-8")
    atomic_npz(
        args.out_bank,
        observations=observations,
        legal=legal,
        targets=targets,
        surface_72o=s72,
        surface_trips_fold=trips,
        surface_high_card_no_draw=hc,
        metadata_json=np.frombuffer(meta_json, dtype=np.uint8),
    )
    bank_sha = distill.sha256(args.out_bank.resolve(strict=True))

    report = {
        **bank_meta,
        "bank_path": str(args.out_bank.resolve()),
        "bank_sha256": bank_sha,
        "stability_bank_build_pass": True,
        "interpretation": (
            "This is the preregistered common state bank for rebuild-seed "
            "stability only. It is evaluation-only and cannot be used to train, "
            "weight, drop, or select rebuild members."
        ),
    }
    atomic_json(args.report, report)
    print("POST15315_REBUILD_STABILITY_BANK_PASS")
    print(f"bank={args.out_bank.resolve()}")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
