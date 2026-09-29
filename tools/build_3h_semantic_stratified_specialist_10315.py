#!/usr/bin/env python3
from __future__ import annotations

"""Build the frozen target-stratified strong-hand specialist from teacher @10315.

The architecture/hyperparameters are frozen from the 10115 lane that passed
four-seed confirmation and fresh-seed DC1 5k:
- route: postflop made_category >= TRIPS and Fold legal -> specialist;
- teacher: current semantic Advantage ensemble;
- specialist pool: fresh unique strong/Fold-legal teacher states;
- training mode: STRATIFIED;
- target strata: 70/15/15 low/mid/high teacher-Fold;
- optimizer budget: fixed 25 steps.

There is intentionally no new step/mode model selection at 10315.
The next gate is unchanged four-seed validation.
"""

import argparse
import json
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_semantic_stratified_specialist_moe_10115 as spec
from audit_3h_average_policy_semantic_continuation_10105 import V1SemanticPolicyNet
from spincore.solver import SolverLibrary

EXPECTED_SHA = distill.EXPECTED_SHA
ADV_SCHEMA = "SPINCORE_3H_SEMANTIC_LONG_ENSEMBLE_V1"
FULLPOOL_SCHEMA = "SPINCORE_3H_SEMANTIC_POSTLONG_FULLPOOL_TAIL_V1"
OUT_SCHEMA = "SPINCORE_3H_SEMANTIC_STRATIFIED_STRONG_SPECIALIST_MOE_V1"
FINAL_ITERATION = 10315
DOMAIN = "THREE_HANDED"
TRAIN_EPISODES = 300000
TRAIN_SEED = 20260929 ^ 0x53EC315
SELECTED_MODE = "STRATIFIED"
SELECTED_STEPS = 25


def load_adv(path: Path):
    p = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if p.get("schema") != ADV_SCHEMA:
        raise RuntimeError("wrong 10315 semantic Advantage schema")
    if p.get("source_checkpoint_sha256") != EXPECTED_SHA:
        raise RuntimeError("10315 semantic Advantage source mismatch")
    if int(p.get("completed_iteration", -1)) != FINAL_ITERATION:
        raise RuntimeError("10315 semantic Advantage iteration mismatch")
    states = list(p.get("members") or [])
    if len(states) != 8:
        raise RuntimeError("expected eight semantic Advantage members")
    return [distill.load_semantic_advantage(s) for s in states]


def load_fullpool(path: Path):
    p = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if p.get("schema") != FULLPOOL_SCHEMA:
        raise RuntimeError("wrong 10315 full-pool schema")
    if p.get("source_checkpoint_sha256") != EXPECTED_SHA:
        raise RuntimeError("10315 full-pool source mismatch")
    if int(p.get("semantic_completed_iteration", -1)) != FINAL_ITERATION:
        raise RuntimeError("10315 full-pool iteration mismatch")
    if int(p.get("selected_steps", -1)) != 500:
        raise RuntimeError("10315 full-pool step-budget mismatch")
    m = V1SemanticPolicyNet()
    m.load_state_dict(p["model_state"])
    m.eval()
    return m, p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--semantic-advantage", type=Path, required=True)
    ap.add_argument("--fullpool-tail", type=Path, required=True)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--out-model", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(int(args.threads))

    cp = args.checkpoint.resolve(strict=True)
    if distill.sha256(cp) != EXPECTED_SHA:
        raise RuntimeError("source 10105 checkpoint SHA mismatch")
    payload = torch.load(cp, map_location="cpu", weights_only=False)
    cfg = dict(payload.get("config") or {})

    adv = load_adv(args.semantic_advantage)
    fullpool, fullpool_payload = load_fullpool(args.fullpool_tail)
    solver = SolverLibrary(args.solver.resolve(strict=True))

    distill.MASTER_SEED = TRAIN_SEED
    a, b, collection = distill.collect_fresh(
        solver, adv, TRAIN_EPISODES
    )
    unique = spec.unique_strong(list(a) + list(b))
    strata = spec.stratum_counts(unique)

    if len(unique) < 1200:
        raise RuntimeError(
            f"10315 specialist unique strong pool too small: {len(unique)}"
        )
    if int(strata.get("high", 0)) < 30:
        raise RuntimeError(
            f"10315 high-target strong stratum too small: {strata}"
        )
    if int(strata.get("mid", 0)) < 10:
        raise RuntimeError(
            f"10315 mid-target strong stratum too small: {strata}"
        )

    specialist = spec.clone_model(fullpool)
    optimizer = torch.optim.Adam(
        specialist.parameters(),
        lr=float(cfg["learning_rate"]),
    )
    rng = random.Random(
        (distill.TRAIN_SEED ^ 0xA11CE) + 0x10001
    )
    spec.train_range(
        specialist,
        optimizer,
        unique,
        cfg,
        0,
        SELECTED_STEPS,
        rng,
        SELECTED_MODE,
    )
    specialist.eval()

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": OUT_SCHEMA,
            "source_checkpoint_sha256": EXPECTED_SHA,
            "semantic_completed_iteration": FINAL_ITERATION,
            "selected_steps": SELECTED_STEPS,
            "selected_training_mode": SELECTED_MODE,
            "specialist_train_episodes": TRAIN_EPISODES,
            "specialist_collection_seed": TRAIN_SEED,
            "specialist_unique_strong_states": len(unique),
            "specialist_unique_strong_strata": strata,
            "base_fullpool_schema": fullpool_payload.get("schema"),
            "base_model_state": {
                k: v.detach().cpu() for k, v in fullpool.state_dict().items()
            },
            "specialist_model_state": {
                k: v.detach().cpu() for k, v in specialist.state_dict().items()
            },
            "route_contract": (
                "postflop made_category>=TRIPS AND FOLD legal -> specialist; "
                "otherwise fullpool general"
            ),
            "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
        },
        args.out_model,
    )

    report = {
        "schema": "SPINCORE_3H_SEMANTIC_POSTLONG_STRATIFIED_SPECIALIST_BUILD_10315_V1",
        "scope": "RESEARCH_ONLY_FIXED_ARCHITECTURE_NO_MODEL_SELECTION",
        "source_checkpoint_sha256": EXPECTED_SHA,
        "semantic_completed_iteration": FINAL_ITERATION,
        "teacher_ensemble_sha256": distill.sha256(args.semantic_advantage),
        "fullpool_sha256": distill.sha256(args.fullpool_tail),
        "train_collection_seed": TRAIN_SEED,
        "train_collection": collection,
        "unique_strong_states": len(unique),
        "unique_strong_strata": strata,
        "selected_training_mode": SELECTED_MODE,
        "selected_steps": SELECTED_STEPS,
        "specialist_artifact": str(args.out_model.resolve()),
        "postlong_specialist_build_pass": True,
        "interpretation": (
            "This build freezes the previously validated specialist architecture "
            "instead of re-selecting mode or optimizer steps after seeing 10315. "
            "The model is not admitted until it passes the independent four-seed "
            "confirmation gate."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== 3H semantic postlong stratified specialist @10315 ===")
    print(f"unique_strong_states={len(unique)}")
    print("strata=" + json.dumps(strata, sort_keys=True))
    print(f"selected_training_mode={SELECTED_MODE}")
    print(f"selected_steps={SELECTED_STEPS}")
    print(f"report={args.report.resolve()}")
    print("3H_SEMANTIC_POSTLONG_STRATIFIED_SPECIALIST_10315_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
