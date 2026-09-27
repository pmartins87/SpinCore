#!/usr/bin/env python3
from __future__ import annotations

"""Audit why semantic-10115 AveragePolicy folds strong made hands.

The DC1 attribution gate already established:
- semantic Advantage ENS8 Fold = 0 on all 12 strong-hand DC1 opportunities;
- semantic tail AveragePolicy sometimes assigns enormous Fold probability.

This audit regenerates the exact 10115 fresh strategy-training stream and a new
independent fresh-target stream.  It measures:
- how many trips-or-better / Fold-legal samples exist;
- their Advantage-generated Fold targets;
- finalized 10105 V1 and semantic-10115 tail predictions;
- exact exposure of those training samples under the frozen 500-step uniform
  minibatch stream.

No model is trained or modified here.
"""

import argparse
import copy
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
from audit_3h_average_policy_semantic_continuation_10105 import (
    V1SemanticPolicyNet,
    semantic_vector_from_obs,
)
from audit_3h_semantic_sidecar_attribution_10105 import (
    decode_obs,
    private_semantics,
)

from spincore.lean_functional_training import load_checkpoint
from spincore_nn.action_models import make_policy_action_model
from spincore_nn.lean_batch import vectorized_batch
from spincore.solver import SolverLibrary

EXPECTED_SHA=distill.EXPECTED_SHA
ADV_SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_ENSEMBLE_V1"
POLICY_SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_TAIL_POLICY_V1"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
DOMAIN="THREE_HANDED"
FINAL_ITERATION=10115
TRAIN_EPISODES=8000
EVAL_EPISODES=30000
TRAIN_SEED=20260927 ^ 0x10115A
INDEPENDENT_SEED=20260927 ^ 0x57A0C0
FOLD=0
CATEGORY_NAMES={
    3:"TRIPS",4:"STRAIGHT",5:"FLUSH",6:"FULL_HOUSE",
    7:"QUADS",8:"STRAIGHT_FLUSH",
}


def load_adv(path:Path):
    p=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=ADV_SCHEMA:
        raise RuntimeError("wrong semantic Advantage artifact")
    if p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError("semantic Advantage source mismatch")
    if int(p.get("completed_iteration",-1))!=FINAL_ITERATION:
        raise RuntimeError("semantic Advantage iteration mismatch")
    states=list(p.get("members") or [])
    if len(states)!=8:
        raise RuntimeError("semantic Advantage ensemble-size drift")
    return [distill.load_semantic_advantage(s) for s in states]


def load_policy(path:Path):
    p=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=POLICY_SCHEMA:
        raise RuntimeError("wrong semantic tail-policy artifact")
    if p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError("semantic tail-policy source mismatch")
    if int(p.get("completed_iteration",-1))!=FINAL_ITERATION:
        raise RuntimeError("semantic tail-policy iteration mismatch")
    if int(p.get("selected_steps",-1))!=500:
        raise RuntimeError("semantic tail-policy step-budget mismatch")
    m=V1SemanticPolicyNet()
    m.load_state_dict(p["model_state"])
    m.eval()
    return m


def sample_meta(sample):
    hole,board,numeric,cat=decode_obs(sample.observation)
    ps=private_semantics(hole,board)
    street=int(cat[1])
    return {
        "made":int(ps["made"]),
        "street":street,
        "fold_legal":bool(sample.legal[FOLD]),
        "facing":bool(float(numeric[1])>1e-9),
        "pot":float(numeric[0]),
        "to_call":float(numeric[1]),
    }


def predict_rows(v1,sem,samples,batch_size=4096):
    rows=[]
    v1.eval(); sem.eval()
    for start in range(0,len(samples),batch_size):
        chunk=samples[start:start+batch_size]
        b,t,w=vectorized_batch(chunk,"cpu")
        sb=dict(b)
        sb["semantic"]=torch.tensor(
            np.asarray(
                [semantic_vector_from_obs(s.observation) for s in chunk],
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )
        with torch.no_grad():
            pv=torch.softmax(
                v1(b).masked_fill(~b["legal"],-1e9),dim=-1
            ).cpu().numpy()
            ps=torch.softmax(
                sem(sb).masked_fill(~sb["legal"],-1e9),dim=-1
            ).cpu().numpy()
        tt=t.cpu().numpy()
        ww=w.cpu().numpy()
        for i,s in enumerate(chunk):
            m=sample_meta(s)
            m.update({
                "target_fold":float(tt[i,FOLD]) if m["fold_legal"] else 0.0,
                "v1_fold":float(pv[i,FOLD]) if m["fold_legal"] else 0.0,
                "semantic_tail_fold":float(ps[i,FOLD]) if m["fold_legal"] else 0.0,
                "weight":float(ww[i]),
            })
            rows.append(m)
    return rows


def weighted_mean(rows,key):
    if not rows:
        return None
    den=sum(float(r["weight"]) for r in rows)
    return sum(float(r[key])*float(r["weight"]) for r in rows)/den if den else None


def subset_summary(rows):
    if not rows:
        return {"count":0}
    tgt=[float(r["target_fold"]) for r in rows]
    v1=[float(r["v1_fold"]) for r in rows]
    sem=[float(r["semantic_tail_fold"]) for r in rows]
    return {
        "count":len(rows),
        "target_fold_weighted_mean":weighted_mean(rows,"target_fold"),
        "v1_fold_weighted_mean":weighted_mean(rows,"v1_fold"),
        "semantic_tail_fold_weighted_mean":weighted_mean(rows,"semantic_tail_fold"),
        "semantic_tail_fold_abs_bias":abs(
            weighted_mean(rows,"semantic_tail_fold")-weighted_mean(rows,"target_fold")
        ),
        "target_fold_max":max(tgt),
        "semantic_tail_fold_median":statistics.median(sem),
        "semantic_tail_fold_p90":float(np.quantile(np.asarray(sem),0.90)),
        "semantic_tail_fold_p95":float(np.quantile(np.asarray(sem),0.95)),
        "semantic_tail_fold_max":max(sem),
        "target_count_ge_0_10":sum(x>=0.10 for x in tgt),
        "tail_count_ge_0_10":sum(x>=0.10 for x in sem),
        "tail_count_ge_0_25":sum(x>=0.25 for x in sem),
        "tail_count_ge_0_50":sum(x>=0.50 for x in sem),
        "tail_count_ge_0_90":sum(x>=0.90 for x in sem),
        "sum_expected_target_folds":sum(tgt),
        "sum_expected_tail_folds":sum(sem),
    }


def strong_rows(rows):
    return [
        r for r in rows
        if int(r["street"])>0 and int(r["made"])>=3 and bool(r["fold_legal"])
    ]


def strata(rows):
    out={}
    for made in sorted({int(r["made"]) for r in rows}):
        if made<3:
            continue
        rr=[r for r in rows if int(r["made"])==made]
        out["category_"+CATEGORY_NAMES.get(made,str(made))]=subset_summary(rr)
    for street in (1,2,3):
        rr=[r for r in rows if int(r["street"])==street]
        if rr:
            out["street_"+str(street)]=subset_summary(rr)
    return out


def exact_500_exposure(train,source_config):
    strong_indices={
        i for i,s in enumerate(train)
        if (
            (lambda m: int(m["street"])>0 and int(m["made"])>=3 and bool(m["fold_legal"]))
            (sample_meta(s))
        )
    }
    rng=random.Random(distill.TRAIN_SEED)
    bs=int(source_config.batch_size)
    counts={i:0 for i in strong_indices}
    total=0
    for _ in range(500):
        idx=rng.sample(range(len(train)),min(bs,len(train)))
        for i in idx:
            if i in counts:
                counts[i]+=1
                total+=1
    vals=list(counts.values())
    return {
        "strong_training_samples":len(strong_indices),
        "total_strong_sample_draws_across_500_steps":total,
        "mean_draws_per_strong_sample":statistics.fmean(vals) if vals else 0.0,
        "min_draws_per_strong_sample":min(vals) if vals else 0,
        "median_draws_per_strong_sample":statistics.median(vals) if vals else 0.0,
        "max_draws_per_strong_sample":max(vals) if vals else 0,
        "strong_samples_never_drawn":sum(x==0 for x in vals),
        "total_training_draws":500*min(bs,len(train)),
    }


def diagnosis(train_summary,eval_summary,exposure):
    train_target=float(train_summary.get("target_fold_weighted_mean") or 0.0)
    train_pred=float(train_summary.get("semantic_tail_fold_weighted_mean") or 0.0)
    eval_target=float(eval_summary.get("target_fold_weighted_mean") or 0.0)
    eval_pred=float(eval_summary.get("semantic_tail_fold_weighted_mean") or 0.0)
    if train_target>0.10 or eval_target>0.10:
        return "TARGET_DISTRIBUTION_CONTAINS_MATERIAL_STRONG_HAND_FOLD"
    if train_pred>0.15:
        return "TAIL_UNDERFITS_STRONG_HAND_TARGETS_ON_ITS_OWN_TRAIN_DISTRIBUTION"
    if eval_pred>0.15:
        return "TAIL_FITS_TRAIN_BUT_GENERALIZES_POORLY_TO_NEW_STRONG_HAND_STATES"
    if int(exposure["strong_samples_never_drawn"])>0:
        return "MINIBATCH_EXPOSURE_GAP"
    return "NO_BROAD_FRESH_STREAM_FAILURE_REPRODUCED_DC1_IS_DISTRIBUTION_SHIFT"


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage",type=Path,required=True)
    ap.add_argument("--semantic-policy",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    args=ap.parse_args()
    torch.set_num_threads(int(args.threads))

    if distill.sha256(args.checkpoint.resolve(strict=True))!=EXPECTED_SHA:
        raise RuntimeError("source checkpoint SHA mismatch")

    solver=SolverLibrary(args.solver.resolve(strict=True))
    _seed,source_config,completed,_sampler,runtimes,_history,_finalized=load_checkpoint(
        args.checkpoint.resolve(strict=True),solver=solver
    )
    if int(completed)!=10105:
        raise RuntimeError("source checkpoint iteration drift")

    models=load_adv(args.semantic_advantage)
    sem=load_policy(args.semantic_policy)

    _,v1=make_policy_action_model(REPRESENTATION,device="cpu",seed=0)
    v1.load_state_dict(runtimes[DOMAIN].bundle.policy.state_dict())
    v1.eval()

    distill.MASTER_SEED=TRAIN_SEED
    train,discarded,train_collection=distill.collect_fresh(
        solver,models,TRAIN_EPISODES
    )
    if len(train)!=22746 or len(discarded)!=5672:
        raise RuntimeError(
            f"10115 training-stream identity drift train={len(train)} holdout={len(discarded)}"
        )

    train_rows=predict_rows(v1,sem,train)
    train_strong=strong_rows(train_rows)
    exposure=exact_500_exposure(train,source_config)

    distill.MASTER_SEED=INDEPENDENT_SEED
    eval_a,eval_b,eval_collection=distill.collect_fresh(
        solver,models,EVAL_EPISODES
    )
    evaluation=list(eval_a)+list(eval_b)
    eval_rows=predict_rows(v1,sem,evaluation)
    eval_strong=strong_rows(eval_rows)

    train_summary=subset_summary(train_strong)
    eval_summary=subset_summary(eval_strong)
    print(
        "STRONG_HAND_COVERAGE_COUNTS "
        + json.dumps(
            {
                "train_strong":len(train_strong),
                "independent_strong":len(eval_strong),
                "independent_episodes":EVAL_EPISODES,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if len(eval_strong)<100:
        raise RuntimeError(
            f"independent strong-hand sample too small: {len(eval_strong)} "
            f"from {EVAL_EPISODES} episodes"
        )

    diagnosis_code=diagnosis(train_summary,eval_summary,exposure)

    report={
        "schema":"SPINCORE_3H_SEMANTIC_TAIL_STRONG_HAND_COVERAGE_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_NO_MODEL_CHANGE",
        "source_checkpoint_sha256":EXPECTED_SHA,
        "semantic_completed_iteration":FINAL_ITERATION,
        "train_collection":train_collection,
        "discarded_train_holdout_samples":len(discarded),
        "independent_eval_collection":eval_collection,
        "independent_eval_samples":len(evaluation),
        "strong_hand_contract":"postflop made_category>=TRIPS and FOLD legal",
        "train_strong":train_summary,
        "train_strata":strata(train_strong),
        "independent_strong":eval_summary,
        "independent_strata":strata(eval_strong),
        "frozen_500_uniform_minibatch_exposure":exposure,
        "diagnosis_code":diagnosis_code,
        "interpretation":{
            "TARGET_DISTRIBUTION_CONTAINS_MATERIAL_STRONG_HAND_FOLD":
                "The fresh semantic Advantage target distribution itself contains material Fold mass; investigate the Advantage/CFR target source before changing AveragePolicy sampling.",
            "TAIL_UNDERFITS_STRONG_HAND_TARGETS_ON_ITS_OWN_TRAIN_DISTRIBUTION":
                "The tail AveragePolicy assigns material Fold mass even on strong-hand samples from its own training distribution despite low target Fold mass; class imbalance/objective allocation is the primary suspect.",
            "TAIL_FITS_TRAIN_BUT_GENERALIZES_POORLY_TO_NEW_STRONG_HAND_STATES":
                "The tail policy fits training strong hands but fails on new ones; fresh-target diversity/coverage is the primary suspect.",
            "MINIBATCH_EXPOSURE_GAP":
                "Some strong-hand training samples were never drawn by the frozen 500-step minibatch stream.",
            "NO_BROAD_FRESH_STREAM_FAILURE_REPRODUCED_DC1_IS_DISTRIBUTION_SHIFT":
                "Broad fresh streams do not reproduce the DC1 failure; the fixed DC1 states are a narrower distribution-shift/generalization issue.",
        }[diagnosis_code],
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )

    print("=== 10115 semantic tail strong-hand coverage audit ===")
    print("train_strong="+json.dumps(train_summary,sort_keys=True))
    print("independent_strong="+json.dumps(eval_summary,sort_keys=True))
    print("exposure="+json.dumps(exposure,sort_keys=True))
    print(f"diagnosis_code={diagnosis_code}")
    print(f"report={args.report.resolve()}")
    print("SEMANTIC_TAIL_STRONG_HAND_COVERAGE_AUDIT_COMPLETE")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
