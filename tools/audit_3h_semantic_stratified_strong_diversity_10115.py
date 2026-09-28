#!/usr/bin/env python3
from __future__ import annotations

"""Second-stage stratified strong-hand diversity repair for canonical 10115.

The first diversity intervention (+256 novel strong states) materially improved
unseen strong-hand Fold calibration but still left four sampled strong-hand
folds in the fixed DC1 1k replay.  This gate increases *novel* support and
stratifies it toward the broad failure surfaces identified before looking at
this gate's evaluation set: river strong hands and straight/flush hands.

No DC1 states are used for training or selection here.

Primary candidate:
- exact canonical 22,726 ordinary strategy samples;
- +512 novel strong-hand teacher samples from a new 180k-episode pool;
- fixed quotas:
  * 128 river strong states;
  * 160 non-river straight/flush states;
  * 160 non-river trips states;
  * 64 remaining novel strong states;
- same frozen 500 optimizer steps.

Evaluation:
- a completely new 40k-episode teacher stream;
- compare canonical baseline, the prior +256 diversity candidate, and this
  stratified +512 candidate.

Research-only. No production/DC2 promotion.
"""

import argparse
import json
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_fresh_semantic_strategy_independent500_10105 as confirm
import audit_3h_semantic_tail_strong_hand_coverage_10115 as coverage
from audit_3h_average_policy_semantic_continuation_10105 import (
    V1SemanticPolicyNet,
    semantic_vector_from_obs,
)

from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.training import train_step
from spincore.solver import SolverLibrary

EXPECTED_SHA=distill.EXPECTED_SHA
ADV_SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_ENSEMBLE_V1"
TAIL_SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_TAIL_POLICY_V1"
DIV1_SCHEMA="SPINCORE_3H_SEMANTIC_STRONG_DIVERSITY_TAIL_CANDIDATE_V1"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
DOMAIN="THREE_HANDED"
FINAL_ITERATION=10115

BASE_TRAIN_SEED=20260927 ^ 0x10115A
BASE_TRAIN_EPISODES=8000
EXPECTED_BASE_TRAIN=22726
EXPECTED_BASE_HOLD=5701

AUGMENT_EPISODES=180000
AUGMENT_SEED=20260928 ^ 0x51A7F1
SELECT_SEED=20260928 ^ 0x5EED52

QUOTAS={
    "river":128,
    "nonriver_straight_flush":160,
    "nonriver_trips":160,
    "remaining":64,
}
EXTRA_STRONG=sum(QUOTAS.values())

EVAL_EPISODES=40000
EVAL_SEED=20260928 ^ 0xE7A152
SELECTED_STEPS=500
FOLD=0


def load_adv(path:Path):
    p=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=ADV_SCHEMA:
        raise RuntimeError("wrong canonical semantic Advantage schema")
    if p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError("canonical semantic Advantage source mismatch")
    if int(p.get("completed_iteration",-1))!=FINAL_ITERATION:
        raise RuntimeError("canonical semantic Advantage iteration mismatch")
    states=list(p.get("members") or [])
    if len(states)!=8:
        raise RuntimeError("canonical semantic Advantage ensemble-size drift")
    return [distill.load_semantic_advantage(s) for s in states]


def load_model(path:Path,allowed_schema:str):
    p=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=allowed_schema:
        raise RuntimeError(
            f"wrong model schema {p.get('schema')!r}; expected {allowed_schema!r}"
        )
    if p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError("model source mismatch")
    completed=int(p.get(
        "completed_iteration",p.get("semantic_completed_iteration",-1)
    ))
    if completed!=FINAL_ITERATION:
        raise RuntimeError("model iteration mismatch")
    if int(p.get("selected_steps",-1))!=SELECTED_STEPS:
        raise RuntimeError("model step-budget mismatch")
    m=V1SemanticPolicyNet()
    m.load_state_dict(p["model_state"])
    m.eval()
    return m,p


def strong_sample(sample)->bool:
    m=coverage.sample_meta(sample)
    return bool(
        int(m["street"])>0
        and int(m["made"])>=3
        and bool(m["fold_legal"])
    )


def key(sample):
    return (bytes(sample.observation),tuple(bool(x) for x in sample.legal))


def describe(samples):
    cats={}
    streets={}
    targets=[]
    for s in samples:
        m=coverage.sample_meta(s)
        cats[str(int(m["made"]))]=cats.get(str(int(m["made"])),0)+1
        streets[str(int(m["street"]))]=streets.get(str(int(m["street"])),0)+1
        targets.append(float(s.target[FOLD]))
    return {
        "count":len(samples),
        "category_counts":cats,
        "street_counts":streets,
        "target_fold_mean":statistics.fmean(targets) if targets else None,
        "target_fold_max":max(targets) if targets else None,
    }


def train_semantic(model,opt,train,cfg):
    rng=random.Random(distill.TRAIN_SEED)
    bs=int(cfg["batch_size"])
    for step in range(SELECTED_STEPS):
        idx=rng.sample(range(len(train)),min(bs,len(train)))
        samples=[train[i] for i in idx]
        b,t,w=vectorized_batch(samples,"cpu")
        sb=dict(b)
        sb["semantic"]=torch.tensor(
            np.asarray(
                [semantic_vector_from_obs(s.observation) for s in samples],
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )
        train_step(model,opt,sb,t,w,"strategy")
        if (step+1)%100==0:
            print(f"STRATIFIED_DIVERSITY_TRAIN {step+1}/{SELECTED_STEPS}",flush=True)


def max_state_diff(model,state):
    cur=model.state_dict()
    if set(cur)!=set(state):
        raise RuntimeError("canonical tail state-key drift")
    return max(
        float((cur[k].detach().cpu()-state[k].detach().cpu()).abs().max().item())
        for k in cur
    )


def eval_strong(v1,sem,samples):
    rows=coverage.predict_rows(v1,sem,samples)
    strong=coverage.strong_rows(rows)
    return {
        "overall":coverage.subset_summary(strong),
        "straight_or_flush":coverage.subset_summary(
            [r for r in strong if int(r["made"]) in (4,5)]
        ),
        "river":coverage.subset_summary(
            [r for r in strong if int(r["street"])==3]
        ),
        "strata":coverage.strata(strong),
    }


def select_stratified(candidates):
    rng=random.Random(SELECT_SEED)
    remaining=list(candidates)
    rng.shuffle(remaining)

    def take(name,predicate,n):
        chosen=[]
        keep=[]
        for s in remaining:
            if len(chosen)<n and predicate(coverage.sample_meta(s)):
                chosen.append(s)
            else:
                keep.append(s)
        if len(chosen)!=n:
            raise RuntimeError(
                f"stratum {name} too small: {len(chosen)} < {n}"
            )
        remaining[:] = keep
        return chosen

    river=take(
        "river",lambda m:int(m["street"])==3,QUOTAS["river"]
    )
    sf=take(
        "nonriver_straight_flush",
        lambda m:int(m["street"]) in (1,2) and int(m["made"]) in (4,5),
        QUOTAS["nonriver_straight_flush"],
    )
    trips=take(
        "nonriver_trips",
        lambda m:int(m["street"]) in (1,2) and int(m["made"])==3,
        QUOTAS["nonriver_trips"],
    )
    if len(remaining)<QUOTAS["remaining"]:
        raise RuntimeError("remaining novel strong pool too small")
    tail=remaining[:QUOTAS["remaining"]]
    selected=river+sf+trips+tail
    if len(selected)!=EXTRA_STRONG:
        raise RuntimeError("stratified selection size drift")
    return selected,{
        "river":describe(river),
        "nonriver_straight_flush":describe(sf),
        "nonriver_trips":describe(trips),
        "remaining":describe(tail),
    }


def ratio(a,b):
    if b is None or float(b)==0:
        return None
    return float(a)/float(b)


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage",type=Path,required=True)
    ap.add_argument("--canonical-tail",type=Path,required=True)
    ap.add_argument("--diversity-v1-tail",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--out-model",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    args=ap.parse_args()
    torch.set_num_threads(int(args.threads))

    cp=args.checkpoint.resolve(strict=True)
    if distill.sha256(cp)!=EXPECTED_SHA:
        raise RuntimeError("source checkpoint SHA mismatch")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    d3=(payload.get("domains") or {}).get(DOMAIN) or {}
    cfg=dict(payload.get("config") or {})

    models=load_adv(args.semantic_advantage)
    canonical_tail,canonical_payload=load_model(args.canonical_tail,TAIL_SCHEMA)
    diversity_v1,_=load_model(args.diversity_v1_tail,DIV1_SCHEMA)
    solver=SolverLibrary(args.solver.resolve(strict=True))

    distill.MASTER_SEED=BASE_TRAIN_SEED
    base_train,base_hold,base_collection=distill.collect_fresh(
        solver,models,BASE_TRAIN_EPISODES
    )
    if len(base_train)!=EXPECTED_BASE_TRAIN or len(base_hold)!=EXPECTED_BASE_HOLD:
        raise RuntimeError("canonical base stream identity drift")
    base_strong=[s for s in base_train if strong_sample(s)]
    if len(base_strong)!=44:
        raise RuntimeError(f"canonical base strong-count drift: {len(base_strong)}")

    # Exact baseline reproduction guard.
    base_v1,base_v1opt,base_sem,base_semopt=confirm.initial_policy_pair(d3,cfg)
    confirm.train_500(base_v1,base_v1opt,base_sem,base_semopt,base_train,cfg)
    repro=max_state_diff(base_sem,canonical_payload["model_state"])
    if repro>2e-6:
        raise RuntimeError(f"canonical tail reproduction drift: {repro}")
    print(f"STRATIFIED_BASELINE_REPRO_PASS max_state_diff={repro:.9g}",flush=True)

    # Large independent teacher pool.
    distill.MASTER_SEED=AUGMENT_SEED
    aug_a,aug_b,aug_collection=distill.collect_fresh(
        solver,models,AUGMENT_EPISODES
    )
    aug_all=list(aug_a)+list(aug_b)
    base_keys={key(s) for s in base_strong}
    seen=set()
    novel=[]
    for s in aug_all:
        if not strong_sample(s):
            continue
        k=key(s)
        if k in base_keys or k in seen:
            continue
        seen.add(k)
        novel.append(s)

    strat_extra,strata=select_stratified(novel)
    train=list(base_train)+strat_extra

    _v,_vo,strat,strat_opt=confirm.initial_policy_pair(d3,cfg)
    train_semantic(strat,strat_opt,train,cfg)

    # New untouched evaluation.
    distill.MASTER_SEED=EVAL_SEED
    ev_a,ev_b,ev_collection=distill.collect_fresh(
        solver,models,EVAL_EPISODES
    )
    evaluation=list(ev_a)+list(ev_b)
    strong_eval=[s for s in evaluation if strong_sample(s)]
    if len(strong_eval)<150:
        raise RuntimeError(f"independent strong eval too small: {len(strong_eval)}")

    base_global=distill.metrics(base_v1,base_sem,evaluation)
    div1_global=distill.metrics(base_v1,diversity_v1,evaluation)
    strat_global=distill.metrics(base_v1,strat,evaluation)

    base_strong_eval=eval_strong(base_v1,base_sem,strong_eval)
    div1_strong_eval=eval_strong(base_v1,diversity_v1,strong_eval)
    strat_strong_eval=eval_strong(base_v1,strat,strong_eval)

    def bias(x,part="overall"):
        return float(x[part]["semantic_tail_fold_abs_bias"])

    b=bias(base_strong_eval)
    d=bias(div1_strong_eval)
    s=bias(strat_strong_eval)
    bsf=bias(base_strong_eval,"straight_or_flush")
    dsf=bias(div1_strong_eval,"straight_or_flush")
    ssf=bias(strat_strong_eval,"straight_or_flush")
    br=bias(base_strong_eval,"river")
    dr=bias(div1_strong_eval,"river")
    sr=bias(strat_strong_eval,"river")

    base_ce=float(base_global["semantic_weighted_ce"])
    strat_ce=float(strat_global["semantic_weighted_ce"])
    base_tv=float(base_global["semantic_weighted_tv"])
    strat_tv=float(strat_global["semantic_weighted_tv"])
    base_hc=float(base_global["high_card_no_draw_allin"]["semantic_abs_bias"])
    strat_hc=float(strat_global["high_card_no_draw_allin"]["semantic_abs_bias"])

    criteria={
        "new_eval_has_at_least_150_strong_states":len(strong_eval)>=150,
        "stratified_strong_bias_at_least_20pct_better_than_diversity_v1":(
            d>0 and s<=0.80*d
        ),
        "stratified_straight_flush_bias_at_least_20pct_better_than_diversity_v1":(
            dsf>0 and ssf<=0.80*dsf
        ),
        "stratified_river_bias_at_least_20pct_better_than_diversity_v1":(
            dr>0 and sr<=0.80*dr
        ),
        "stratified_strong_bias_at_least_60pct_better_than_canonical":(
            b>0 and s<=0.40*b
        ),
        "global_ce_not_worse_than_canonical_by_over_2pct":(
            strat_ce<=1.02*base_ce
        ),
        "global_tv_not_worse_than_canonical_by_over_2pct":(
            strat_tv<=1.02*base_tv
        ),
        "high_card_no_draw_abs_bias_not_worse_by_over_1pp":(
            strat_hc<=base_hc+0.01
        ),
    }
    passed=all(criteria.values())

    args.out_model.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema":"SPINCORE_3H_SEMANTIC_STRATIFIED_STRONG_DIVERSITY_TAIL_CANDIDATE_V1",
        "source_checkpoint_sha256":EXPECTED_SHA,
        "semantic_completed_iteration":FINAL_ITERATION,
        "selected_steps":SELECTED_STEPS,
        "extra_unique_strong_states":EXTRA_STRONG,
        "selection_quotas":QUOTAS,
        "model_state":{k:v.detach().cpu() for k,v in strat.state_dict().items()},
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    },args.out_model)

    report={
        "schema":"SPINCORE_3H_SEMANTIC_STRATIFIED_STRONG_DIVERSITY_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_STRATIFIED_DIVERSITY_REPAIR",
        "source_checkpoint_sha256":EXPECTED_SHA,
        "semantic_completed_iteration":FINAL_ITERATION,
        "canonical_tail_reproduction_max_state_diff":repro,
        "base_collection":base_collection,
        "base_strong":describe(base_strong),
        "augmentation_collection":aug_collection,
        "augmentation_pool_samples":len(aug_all),
        "novel_strong_candidates":len(novel),
        "selection_quotas":QUOTAS,
        "selected_strata":strata,
        "selected_extra":describe(strat_extra),
        "augmented_train_samples":len(train),
        "independent_eval_collection":ev_collection,
        "independent_eval_samples":len(evaluation),
        "independent_strong_samples":len(strong_eval),
        "canonical":{
            "global":base_global,
            "strong":base_strong_eval,
        },
        "diversity_v1":{
            "global":div1_global,
            "strong":div1_strong_eval,
        },
        "stratified_diversity":{
            "global":strat_global,
            "strong":strat_strong_eval,
        },
        "effect_sizes":{
            "stratified_vs_canonical_strong_bias_ratio":ratio(s,b),
            "stratified_vs_diversity_v1_strong_bias_ratio":ratio(s,d),
            "stratified_vs_diversity_v1_straight_flush_bias_ratio":ratio(ssf,dsf),
            "stratified_vs_diversity_v1_river_bias_ratio":ratio(sr,dr),
            "stratified_vs_canonical_global_ce_ratio":ratio(strat_ce,base_ce),
            "stratified_vs_canonical_global_tv_ratio":ratio(strat_tv,base_tv),
            "stratified_minus_canonical_hcdn_abs_bias":strat_hc-base_hc,
        },
        "precommitted_criteria":criteria,
        "semantic_stratified_diversity_repair_pass":passed,
        "interpretation":(
            "PASS means a larger stratified novel strong-hand support further "
            "improves unseen strong-hand calibration over the already-passed +256 "
            "diversity candidate while preserving global fit. PASS authorizes one "
            "exact DC1 1k replay; it does not authorize production, DC2 or 5k."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== stratified strong diversity repair ===")
    print("canonical_strong="+json.dumps(base_strong_eval["overall"],sort_keys=True))
    print("diversity_v1_strong="+json.dumps(div1_strong_eval["overall"],sort_keys=True))
    print("stratified_strong="+json.dumps(strat_strong_eval["overall"],sort_keys=True))
    print("criteria="+json.dumps(criteria,sort_keys=True))
    print(f"semantic_stratified_diversity_repair_pass={passed}")
    print(f"report={args.report.resolve()}")
    print("SEMANTIC_STRATIFIED_STRONG_DIVERSITY_REPAIR_COMPLETE")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
