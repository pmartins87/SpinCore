#!/usr/bin/env python3
from __future__ import annotations

"""Rebuild the 3H semantic AveragePolicy/full-pool tail from teacher @10315.

This is the precommitted post-long bridge. It does not continue the historical
AveragePolicy. It starts from the frozen finalized 10105 policy/optimizer,
generates fresh strategy targets from the 10315 semantic Advantage ensemble,
trains the already-validated 500-step V1+semantic AveragePolicy, then rebuilds
the full-pool strong-diversity base with an independent 180k-episode teacher
pool. A completely independent 50k-episode stream validates the result.

Research/development only. No production/DC2 promotion is authorized here.
"""

import argparse
import json
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_fresh_semantic_strategy_independent500_10105 as confirm
import audit_3h_semantic_tail_strong_hand_coverage_10115 as coverage
from audit_3h_average_policy_semantic_continuation_10105 import (
    semantic_vector_from_obs,
)
from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.training import train_step
from spincore.solver import SolverLibrary

EXPECTED_SHA = distill.EXPECTED_SHA
ADV_SCHEMA = "SPINCORE_3H_SEMANTIC_RESEARCH_ENSEMBLE_V1"
TAIL_SCHEMA = "SPINCORE_3H_SEMANTIC_POSTLONG_TAIL_POLICY_V1"
FULLPOOL_SCHEMA = "SPINCORE_3H_SEMANTIC_POSTLONG_FULLPOOL_TAIL_V1"
FINAL_ITERATION = 10315
DOMAIN = "THREE_HANDED"
FOLD = 0

BASE_TRAIN_EPISODES = 8000
BASE_TRAIN_SEED = 20260929 ^ 0x10315A
AUGMENT_EPISODES = 180000
AUGMENT_SEED = 20260929 ^ 0x10315F
EVAL_EPISODES = 50000
EVAL_SEED = 20260929 ^ 0x10315E
SELECTED_STEPS = 500


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
    return [distill.load_semantic_advantage(s) for s in states], p


def strong(sample):
    m = coverage.sample_meta(sample)
    return bool(
        int(m["street"]) > 0
        and int(m["made"]) >= 3
        and bool(m["fold_legal"])
    )


def skey(sample):
    return (
        bytes(sample.observation),
        tuple(bool(x) for x in sample.legal),
    )


def describe(samples):
    cats = {}
    streets = {}
    targets = []
    for s in samples:
        m = coverage.sample_meta(s)
        cats[str(int(m["made"]))] = cats.get(str(int(m["made"])), 0) + 1
        streets[str(int(m["street"]))] = streets.get(str(int(m["street"])), 0) + 1
        targets.append(float(s.target[FOLD]))
    return {
        "count": len(samples),
        "category_counts": cats,
        "street_counts": streets,
        "target_fold_mean": statistics.fmean(targets) if targets else None,
        "target_fold_max": max(targets) if targets else None,
    }


def train_semantic_only(model, optimizer, train, cfg):
    rng = random.Random(distill.TRAIN_SEED)
    bs = int(cfg["batch_size"])
    for step in range(SELECTED_STEPS):
        idx = rng.sample(range(len(train)), min(bs, len(train)))
        samples = [train[i] for i in idx]
        b, t, w = vectorized_batch(samples, "cpu")
        sb = dict(b)
        sb["semantic"] = torch.tensor(
            np.asarray(
                [semantic_vector_from_obs(s.observation) for s in samples],
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )
        train_step(model, optimizer, sb, t, w, "strategy")
        if (step + 1) % 100 == 0:
            print(
                f"POSTLONG_FULLPOOL_TRAIN {step + 1}/{SELECTED_STEPS}",
                flush=True,
            )


def strong_summary(v1, sem, samples):
    rows = coverage.strong_rows(coverage.predict_rows(v1, sem, samples))
    return coverage.subset_summary(rows), coverage.strata(rows)


def ratio(a, b):
    if b is None or float(b) == 0.0:
        return None
    return float(a) / float(b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--semantic-advantage", type=Path, required=True)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--out-tail", type=Path, required=True)
    ap.add_argument("--out-fullpool", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(int(args.threads))
    cp = args.checkpoint.resolve(strict=True)
    if distill.sha256(cp) != EXPECTED_SHA:
        raise RuntimeError("source 10105 checkpoint SHA mismatch")

    payload = torch.load(cp, map_location="cpu", weights_only=False)
    d3 = (payload.get("domains") or {}).get(DOMAIN) or {}
    cfg = dict(payload.get("config") or {})

    adv, adv_payload = load_adv(args.semantic_advantage)
    solver = SolverLibrary(args.solver.resolve(strict=True))

    # Fresh ordinary policy targets from the 10315 teacher.
    distill.MASTER_SEED = BASE_TRAIN_SEED
    base_train, base_discard, base_collection = distill.collect_fresh(
        solver, adv, BASE_TRAIN_EPISODES
    )
    if len(base_train) < 20000 or len(base_discard) < 5000:
        raise RuntimeError(
            f"10315 ordinary stream too small: train={len(base_train)} "
            f"discard={len(base_discard)}"
        )

    step0_v1, _, step0_sem, _ = confirm.initial_policy_pair(d3, cfg)
    v1, v1opt, tail, tailopt = confirm.initial_policy_pair(d3, cfg)
    confirm.train_500(v1, v1opt, tail, tailopt, base_train, cfg)

    # Independent novel strong-state pool for the full-pool general model.
    base_strong = [s for s in base_train if strong(s)]
    base_keys = {skey(s) for s in base_strong}

    distill.MASTER_SEED = AUGMENT_SEED
    aug_a, aug_b, augment_collection = distill.collect_fresh(
        solver, adv, AUGMENT_EPISODES
    )
    seen = set()
    novel = []
    for s in list(aug_a) + list(aug_b):
        if not strong(s):
            continue
        k = skey(s)
        if k in base_keys or k in seen:
            continue
        seen.add(k)
        novel.append(s)
    if len(novel) < 700:
        raise RuntimeError(
            f"10315 novel strong pool unexpectedly small: {len(novel)}"
        )

    full_train = list(base_train) + novel
    _, _, fullpool, fullopt = confirm.initial_policy_pair(d3, cfg)
    train_semantic_only(fullpool, fullopt, full_train, cfg)
    fullpool.eval()
    tail.eval()
    v1.eval()
    step0_v1.eval()
    step0_sem.eval()

    # Completely independent evaluation.
    distill.MASTER_SEED = EVAL_SEED
    ev_a, ev_b, eval_collection = distill.collect_fresh(
        solver, adv, EVAL_EPISODES
    )
    evaluation = list(ev_a) + list(ev_b)
    if len(evaluation) < 100000:
        raise RuntimeError(
            f"10315 independent evaluation too small: {len(evaluation)}"
        )

    step0 = distill.metrics(step0_v1, step0_sem, evaluation)
    tail_global = distill.metrics(v1, tail, evaluation)
    full_global = distill.metrics(v1, fullpool, evaluation)
    tail_strong, tail_strata = strong_summary(v1, tail, evaluation)
    full_strong, full_strata = strong_summary(v1, fullpool, evaluation)

    if int(tail_strong.get("count", 0)) < 200:
        raise RuntimeError(
            f"10315 independent strong-hand evaluation too small: "
            f"{tail_strong.get('count', 0)}"
        )

    tail_hc = tail_global["high_card_no_draw_allin"]
    full_hc = full_global["high_card_no_draw_allin"]

    tail_criteria = {
        "tail_semantic_ce_at_least_10pct_better_than_v1": bool(
            float(tail_global["semantic_weighted_ce"])
            <= 0.90 * float(tail_global["v1_weighted_ce"])
        ),
        "tail_semantic_tv_at_least_15pct_better_than_v1": bool(
            float(tail_global["semantic_weighted_tv"])
            <= 0.85 * float(tail_global["v1_weighted_tv"])
        ),
        "tail_high_card_abs_bias_at_least_50pct_better": bool(
            float(tail_hc["v1_abs_bias"]) > 0.0
            and float(tail_hc["semantic_abs_bias"])
            <= 0.50 * float(tail_hc["v1_abs_bias"])
        ),
        "tail_semantic_ce_at_least_10pct_better_than_step0": bool(
            float(tail_global["semantic_weighted_ce"])
            <= 0.90 * float(step0["semantic_weighted_ce"])
        ),
    }

    fullpool_criteria = {
        "fullpool_global_ce_not_worse_than_tail_by_over_2pct": bool(
            float(full_global["semantic_weighted_ce"])
            <= 1.02 * float(tail_global["semantic_weighted_ce"])
        ),
        "fullpool_global_tv_not_worse_than_tail_by_over_2pct": bool(
            float(full_global["semantic_weighted_tv"])
            <= 1.02 * float(tail_global["semantic_weighted_tv"])
        ),
        "fullpool_high_card_abs_bias_not_worse_than_tail_by_over_1pp": bool(
            float(full_hc["semantic_abs_bias"])
            <= float(tail_hc["semantic_abs_bias"]) + 0.01
        ),
        "fullpool_strong_fold_bias_not_worse_than_tail": bool(
            float(full_strong["semantic_tail_fold_abs_bias"])
            <= float(tail_strong["semantic_tail_fold_abs_bias"]) + 1e-12
        ),
    }

    passed = all(tail_criteria.values()) and all(fullpool_criteria.values())

    args.out_tail.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": TAIL_SCHEMA,
            "source_checkpoint_sha256": EXPECTED_SHA,
            "semantic_completed_iteration": FINAL_ITERATION,
            "selected_steps": SELECTED_STEPS,
            "train_collection_seed": BASE_TRAIN_SEED,
            "model_state": {
                k: v.detach().cpu() for k, v in tail.state_dict().items()
            },
            "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
        },
        args.out_tail,
    )
    torch.save(
        {
            "schema": FULLPOOL_SCHEMA,
            "source_checkpoint_sha256": EXPECTED_SHA,
            "semantic_completed_iteration": FINAL_ITERATION,
            "selected_steps": SELECTED_STEPS,
            "ordinary_train_samples": len(base_train),
            "ordinary_strong_samples": len(base_strong),
            "extra_unique_strong_states": len(novel),
            "model_state": {
                k: v.detach().cpu() for k, v in fullpool.state_dict().items()
            },
            "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
        },
        args.out_fullpool,
    )

    report = {
        "schema": "SPINCORE_3H_SEMANTIC_POSTLONG_POLICY_REBUILD_10315_V1",
        "scope": "RESEARCH_ONLY_POSTLONG_FRESH_TARGET_REBUILD",
        "source_checkpoint_sha256": EXPECTED_SHA,
        "semantic_completed_iteration": FINAL_ITERATION,
        "teacher_ensemble_sha256": distill.sha256(args.semantic_advantage),
        "teacher_schema": adv_payload.get("schema"),
        "base_train_seed": BASE_TRAIN_SEED,
        "augment_seed": AUGMENT_SEED,
        "eval_seed": EVAL_SEED,
        "base_collection": base_collection,
        "base_train_samples": len(base_train),
        "base_discarded_samples": len(base_discard),
        "base_strong": describe(base_strong),
        "augmentation_collection": augment_collection,
        "novel_strong": describe(novel),
        "augmented_train_samples": len(full_train),
        "independent_eval_collection": eval_collection,
        "independent_eval_samples": len(evaluation),
        "step0": step0,
        "ordinary_tail_global": tail_global,
        "ordinary_tail_strong": tail_strong,
        "ordinary_tail_strata": tail_strata,
        "fullpool_global": full_global,
        "fullpool_strong": full_strong,
        "fullpool_strata": full_strata,
        "effect_sizes": {
            "fullpool_over_tail_ce_ratio": ratio(
                full_global["semantic_weighted_ce"],
                tail_global["semantic_weighted_ce"],
            ),
            "fullpool_over_tail_tv_ratio": ratio(
                full_global["semantic_weighted_tv"],
                tail_global["semantic_weighted_tv"],
            ),
            "fullpool_over_tail_strong_bias_ratio": ratio(
                full_strong["semantic_tail_fold_abs_bias"],
                tail_strong["semantic_tail_fold_abs_bias"],
            ),
            "fullpool_minus_tail_hcdn_abs_bias": (
                float(full_hc["semantic_abs_bias"])
                - float(tail_hc["semantic_abs_bias"])
            ),
        },
        "tail_precommitted_criteria": tail_criteria,
        "fullpool_precommitted_criteria": fullpool_criteria,
        "postlong_policy_rebuild_pass": passed,
        "tail_artifact": str(args.out_tail.resolve()),
        "fullpool_artifact": str(args.out_fullpool.resolve()),
        "interpretation": (
            "PASS means the 10315 teacher can be freshly distilled into the "
            "already-frozen 500-step semantic AveragePolicy and the full-pool "
            "strong-diversity base without regressing the validated global "
            "surfaces. It is still research-only and must pass the unchanged "
            "stratified-specialist multi-seed gate before DC1."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== 3H semantic postlong policy rebuild @10315 ===")
    print("tail_criteria=" + json.dumps(tail_criteria, sort_keys=True))
    print("fullpool_criteria=" + json.dumps(fullpool_criteria, sort_keys=True))
    print(f"postlong_policy_rebuild_pass={passed}")
    print(f"report={args.report.resolve()}")
    print("3H_SEMANTIC_POSTLONG_POLICY_REBUILD_10315_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
