#!/usr/bin/env python3
from __future__ import annotations

"""Precommitted multi-seed confirmation of the rebuilt 10315 stratified specialist.

Why this gate exists:
- the target-stratified specialist failed its first 150k validation only on the
  rare legitimate high-Fold subset;
- a later, independently generated 200k stream (created for a different
  fold-logit-calibration experiment) showed the *unchanged* specialist satisfying
  all of the substantive specialist thresholds, while the new calibration itself
  failed two frozen gates.

That later observation is post-hoc for the unchanged specialist and therefore
cannot promote it.  This script resolves the ambiguity without changing any
model parameters.

It evaluates the rebuilt 10315 stratified-specialist artifact on FOUR completely
new deterministic 120k-episode streams plus their pooled union.  No training,
calibration, threshold selection, or DC1 data is used.

PASS authorizes one exact fixed-seed DC1 1k replay of the unchanged specialist.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_semantic_fold_logit_calibrated_specialist_10115 as cal
from audit_3h_average_policy_semantic_continuation_10105 import V1SemanticPolicyNet
from spincore.solver import SolverLibrary

EXPECTED_SHA=distill.EXPECTED_SHA
ADV_SCHEMA="SPINCORE_3H_SEMANTIC_LONG_ENSEMBLE_V1"
SPECIALIST_SCHEMA="SPINCORE_3H_SEMANTIC_STRATIFIED_STRONG_SPECIALIST_MOE_V1"
FINAL_ITERATION=10315
EPISODES_PER_SEED=120000
SEEDS=(
    20260929 ^ 0xD03101,
    20260929 ^ 0xD03102,
    20260929 ^ 0xD03103,
    20260929 ^ 0xD03104,
)


def load_adv(path:Path):
    p=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=ADV_SCHEMA:
        raise RuntimeError("wrong 10315 semantic Advantage schema")
    if p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError("10315 semantic Advantage source mismatch")
    if int(p.get("completed_iteration",-1))!=FINAL_ITERATION:
        raise RuntimeError("10315 semantic Advantage iteration mismatch")
    states=list(p.get("members") or [])
    if len(states)!=8:
        raise RuntimeError("expected eight Advantage members")
    return [distill.load_semantic_advantage(s) for s in states]


def load_specialist(path:Path):
    p=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=SPECIALIST_SCHEMA:
        raise RuntimeError("wrong stratified specialist schema")
    if p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError("specialist source mismatch")
    if int(p.get("semantic_completed_iteration",-1))!=FINAL_ITERATION:
        raise RuntimeError("specialist iteration mismatch")
    if int(p.get("selected_steps",-1)) not in (10,25,50,100,200):
        raise RuntimeError("specialist selected-step mismatch")

    base=V1SemanticPolicyNet()
    base.load_state_dict(p["base_model_state"])
    base.eval()
    specialist=V1SemanticPolicyNet()
    specialist.load_state_dict(p["specialist_model_state"])
    specialist.eval()
    return base,specialist,p


def metrics_for(base,specialist,samples):
    bp,t,w=cal.model_probs(base,samples)
    sp,t2,w2=cal.model_probs(specialist,samples)
    if not np.allclose(t,t2) or not np.allclose(w,w2):
        raise RuntimeError("base/specialist target-weight identity drift")
    return cal.summarize(bp,t,w,samples),cal.summarize(sp,t,w,samples)


def ratio(a,b):
    if b is None or float(b)==0.0:
        return None
    return float(a)/float(b)


def effect(base,spec):
    return {
        "ce_ratio":ratio(spec["weighted_ce"],base["weighted_ce"]),
        "tv_ratio":ratio(spec["weighted_tv"],base["weighted_tv"]),
        "fold_bias_ratio":ratio(spec["fold_abs_bias"],base["fold_abs_bias"]),
        "low_mean_ratio":ratio(spec["low_target_fold_mean"],base["low_target_fold_mean"]),
        "low_p95_ratio":ratio(spec["low_target_fold_p95"],base["low_target_fold_p95"]),
        "straight_flush_bias_ratio":ratio(
            spec["straight_flush_fold_abs_bias"],
            base["straight_flush_fold_abs_bias"],
        ),
        "river_bias_ratio":ratio(
            spec["river_fold_abs_bias"],
            base["river_fold_abs_bias"],
        ),
        "high_target_bias_delta":(
            float(spec["high_target_fold_abs_bias"])
            -float(base["high_target_fold_abs_bias"])
        ),
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage",type=Path,required=True)
    ap.add_argument("--stratified-specialist",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    args=ap.parse_args()
    torch.set_num_threads(int(args.threads))

    cp=args.checkpoint.resolve(strict=True)
    if distill.sha256(cp)!=EXPECTED_SHA:
        raise RuntimeError("source checkpoint SHA mismatch")
    # Load only as an integrity guard; the models carry their own architecture.
    _payload=torch.load(cp,map_location="cpu",weights_only=False)

    adv=load_adv(args.semantic_advantage)
    base,specialist,source=load_specialist(args.stratified_specialist)
    solver=SolverLibrary(args.solver.resolve(strict=True))

    seed_rows=[]
    pooled=[]
    for index,seed in enumerate(SEEDS,1):
        distill.MASTER_SEED=int(seed)
        a,b,collection=distill.collect_fresh(
            solver,adv,EPISODES_PER_SEED
        )
        samples=cal.unique_strong(list(a)+list(b))
        if len(samples)<600:
            raise RuntimeError(
                f"seed {index} unique strong sample too small: {len(samples)}"
            )
        base_m,spec_m=metrics_for(base,specialist,samples)
        row={
            "seed_index":index,
            "seed":int(seed),
            "episodes":EPISODES_PER_SEED,
            "collection":collection,
            "unique_strong_samples":len(samples),
            "base":base_m,
            "specialist":spec_m,
            "effect_sizes":effect(base_m,spec_m),
        }
        seed_rows.append(row)
        pooled.extend(samples)
        print(
            "MULTISEED_SPECIALIST "
            +json.dumps({
                "seed_index":index,
                "strong":len(samples),
                "high":spec_m["high_target_count"],
                "ce_ratio":row["effect_sizes"]["ce_ratio"],
                "tv_ratio":row["effect_sizes"]["tv_ratio"],
                "high_bias_delta":row["effect_sizes"]["high_target_bias_delta"],
            },sort_keys=True),
            flush=True,
        )

    # Seeds are disjoint by RNG construction; retain all states in the pooled
    # evidence even if two happen to encode the same observable state.
    base_pool,spec_pool=metrics_for(base,specialist,pooled)
    eff=effect(base_pool,spec_pool)

    per_seed_sample_ok=all(
        r["unique_strong_samples"]>=600
        and int(r["specialist"]["high_target_count"])>=20
        for r in seed_rows
    )
    per_seed_no_catastrophic_high_bias=all(
        float(r["effect_sizes"]["high_target_bias_delta"])<=0.05
        for r in seed_rows
    )
    per_seed_no_ce_regression=all(
        float(r["effect_sizes"]["ce_ratio"])<=1.0
        for r in seed_rows
    )

    criteria={
        "four_new_seed_streams_completed":len(seed_rows)==4,
        "each_seed_has_at_least_600_unique_strong_and_20_high_target":
            per_seed_sample_ok,
        "pooled_has_at_least_2400_strong_states":len(pooled)>=2400,
        "pooled_has_at_least_100_high_target_states":
            int(spec_pool["high_target_count"])>=100,
        "pooled_ce_at_least_10pct_better_than_fullpool":
            float(eff["ce_ratio"])<=0.90,
        "pooled_tv_at_least_10pct_better_than_fullpool":
            float(eff["tv_ratio"])<=0.90,
        "pooled_fold_bias_at_least_30pct_better":
            float(eff["fold_bias_ratio"])<=0.70,
        "pooled_low_target_mean_at_least_30pct_better":
            float(eff["low_mean_ratio"])<=0.70,
        "pooled_low_target_p95_at_least_30pct_better":
            float(eff["low_p95_ratio"])<=0.70,
        "pooled_straight_flush_bias_at_least_30pct_better":
            float(eff["straight_flush_bias_ratio"])<=0.70,
        "pooled_river_bias_at_least_30pct_better":
            float(eff["river_bias_ratio"])<=0.70,
        "pooled_high_target_bias_not_worse_by_over_1pp":
            float(eff["high_target_bias_delta"])<=0.01,
        "no_single_seed_high_target_bias_regression_over_5pp":
            per_seed_no_catastrophic_high_bias,
        "no_single_seed_ce_regression":
            per_seed_no_ce_regression,
    }
    passed=all(criteria.values())

    report={
        "schema":"SPINCORE_3H_SEMANTIC_STRATIFIED_SPECIALIST_MULTISEED_CONFIRMATION_10315_V1",
        "scope":"RESEARCH_ONLY_REBUILT_10315_MODEL_FOUR_NEW_SEEDS_NO_POSTHOC_SELECTION",
        "source_checkpoint_sha256":EXPECTED_SHA,
        "semantic_completed_iteration":FINAL_ITERATION,
        "specialist_schema":SPECIALIST_SCHEMA,
        "specialist_selected_training_mode":source.get("selected_training_mode"),
        "specialist_selected_steps":int(source["selected_steps"]),
        "episodes_per_seed":EPISODES_PER_SEED,
        "seeds":[int(x) for x in SEEDS],
        "seed_rows":seed_rows,
        "pooled_unique_strong_observations":len(pooled),
        "pooled_fullpool":base_pool,
        "pooled_specialist":spec_pool,
        "pooled_effect_sizes":eff,
        "precommitted_criteria":criteria,
        "semantic_stratified_specialist_multiseed_pass":passed,
        "interpretation":(
            "PASS means the rebuilt 10315 target-stratified specialist reproduces "
            "its broad gains across four entirely new deterministic streams and "
            "the rare legitimate high-Fold issue does not reproduce materially "
            "in pooled evidence. PASS authorizes the precommitted fresh-seed "
            "DC1 5k comparison only; no production/DC2/canonical-strength claim."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )

    print("=== stratified specialist multiseed confirmation ===")
    print("pooled_fullpool="+json.dumps(base_pool,sort_keys=True))
    print("pooled_specialist="+json.dumps(spec_pool,sort_keys=True))
    print("pooled_effect_sizes="+json.dumps(eff,sort_keys=True))
    print("criteria="+json.dumps(criteria,sort_keys=True))
    print(
        "semantic_stratified_specialist_multiseed_pass="
        +str(passed)
    )
    print(f"report={args.report.resolve()}")
    print("SEMANTIC_STRATIFIED_SPECIALIST_MULTISEED_CONFIRMATION_COMPLETE")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
