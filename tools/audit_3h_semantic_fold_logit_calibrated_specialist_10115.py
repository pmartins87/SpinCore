#!/usr/bin/env python3
from __future__ import annotations

"""Calibrate Fold odds of the target-stratified strong specialist.

The stratified specialist passes every frozen independent criterion except one:
legitimate high-target-Fold states remain under-confident.  At the same time,
low-target strong states are still over-Folded.  That pattern is consistent with
probability compression toward the middle.

This gate does NOT retrain poker strategy or alter teacher targets.  It applies
a two-parameter affine calibration only to the specialist's Fold log-odds on the
already-defined strong/Fold-legal semantic stratum:

    z = logit(P(Fold))
    z' = scale * z + bias
    P'(Fold) = sigmoid(z')

The remaining non-Fold probability mass is redistributed between Call and
All-in in their original learned ratio.

A fresh 200k-episode stream selects (scale,bias) on a fixed grid.  A completely
separate 200k-episode stream validates the selected transform.  The full-pool
general model remains untouched outside the strong specialist route.

Research-only. PASS authorizes one exact fixed-seed DC1 1k replay.
"""

import argparse
import json
import math
from pathlib import Path
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
from spincore.solver import SolverLibrary

EXPECTED_SHA=distill.EXPECTED_SHA
ADV_SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_ENSEMBLE_V1"
SOURCE_SCHEMA="SPINCORE_3H_SEMANTIC_STRATIFIED_STRONG_SPECIALIST_MOE_V1"
OUT_SCHEMA="SPINCORE_3H_SEMANTIC_FOLD_LOGIT_CALIBRATED_STRONG_SPECIALIST_MOE_V1"
FINAL_ITERATION=10115
DOMAIN="THREE_HANDED"
CAL_EPISODES=200000
CAL_SEED=20260928 ^ 0xCA1B4A
EVAL_EPISODES=200000
EVAL_SEED=20260928 ^ 0xE2A115
SCALES=(1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.5,4.0)
BIASES=tuple(round(-1.50+0.25*i,2) for i in range(13))
LOW_MAX=0.05
HIGH_MIN=0.50
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
        raise RuntimeError("expected eight Advantage members")
    return [distill.load_semantic_advantage(s) for s in states]


def load_source(path:Path):
    p=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=SOURCE_SCHEMA:
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


def strong(sample):
    m=coverage.sample_meta(sample)
    return bool(
        int(m["street"])>0
        and int(m["made"])>=3
        and bool(m["fold_legal"])
    )


def skey(sample):
    return (
        bytes(sample.observation),
        tuple(bool(x) for x in sample.legal),
    )


def unique_strong(samples):
    out=[]
    seen=set()
    for s in samples:
        if not strong(s):
            continue
        key=skey(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def model_probs(model,samples):
    parts=[]
    targets=[]
    weights=[]
    for start in range(0,len(samples),4096):
        chunk=samples[start:start+4096]
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
            p=model.probabilities(sb).detach().cpu().numpy()
        parts.append(p)
        targets.append(t.detach().cpu().numpy())
        weights.append(w.detach().cpu().numpy())
    return (
        np.concatenate(parts,axis=0).astype(np.float64),
        np.concatenate(targets,axis=0).astype(np.float64),
        np.concatenate(weights,axis=0).astype(np.float64),
    )


def calibrate_probs(probs,scale,bias):
    out=np.array(probs,copy=True,dtype=np.float64)
    pf=np.clip(out[:,FOLD],1e-8,1.0-1e-8)
    logit=np.log(pf)-np.log1p(-pf)
    q=1.0/(1.0+np.exp(-(float(scale)*logit+float(bias))))
    non=out[:,1:]
    non_sum=np.clip(non.sum(axis=1),1e-12,None)
    non=non*((1.0-q)/non_sum)[:,None]
    out[:,FOLD]=q
    out[:,1:]=non
    out/=np.clip(out.sum(axis=1),1e-12,None)[:,None]
    return out


def weighted_mean(values,weights):
    den=float(weights.sum())
    if den<=0:
        raise RuntimeError("nonpositive metric weight")
    return float(np.dot(values,weights)/den)


def weighted_quantile(values,weights,q):
    order=np.argsort(values)
    v=np.asarray(values)[order]
    w=np.asarray(weights)[order]
    cw=np.cumsum(w)
    if cw[-1]<=0:
        raise RuntimeError("nonpositive quantile weight")
    pos=float(q)*cw[-1]
    idx=int(np.searchsorted(cw,pos,side="left"))
    idx=min(max(idx,0),len(v)-1)
    return float(v[idx])


def subset_mask(samples,predicate):
    return np.asarray(
        [bool(predicate(coverage.sample_meta(s),s)) for s in samples],
        dtype=bool,
    )


def summarize(probs,targets,weights,samples):
    eps=1e-12
    ce_rows=-(targets*np.log(np.clip(probs,eps,1.0))).sum(axis=1)
    tv_rows=0.5*np.abs(probs-targets).sum(axis=1)

    fold_pred=probs[:,FOLD]
    fold_target=targets[:,FOLD]
    overall=np.ones(len(samples),dtype=bool)
    low=fold_target<=LOW_MAX
    high=fold_target>=HIGH_MIN
    sf=subset_mask(
        samples,
        lambda m,s:int(m["made"]) in (4,5),
    )
    river=subset_mask(
        samples,
        lambda m,s:int(m["street"])==3,
    )

    def fold_bias(mask):
        if not np.any(mask):
            return None
        p=weighted_mean(fold_pred[mask],weights[mask])
        t=weighted_mean(fold_target[mask],weights[mask])
        return abs(p-t)

    def fold_mean(mask):
        if not np.any(mask):
            return None
        return weighted_mean(fold_pred[mask],weights[mask])

    return {
        "weighted_ce":weighted_mean(ce_rows,weights),
        "weighted_tv":weighted_mean(tv_rows,weights),
        "fold_abs_bias":fold_bias(overall),
        "straight_flush_fold_abs_bias":fold_bias(sf),
        "river_fold_abs_bias":fold_bias(river),
        "low_target_count":int(low.sum()),
        "low_target_fold_mean":fold_mean(low),
        "low_target_fold_p95":(
            weighted_quantile(fold_pred[low],weights[low],0.95)
            if np.any(low) else None
        ),
        "high_target_count":int(high.sum()),
        "high_target_fold_abs_bias":fold_bias(high),
        "high_target_target_mean":(
            weighted_mean(fold_target[high],weights[high])
            if np.any(high) else None
        ),
        "high_target_pred_mean":fold_mean(high),
    }


def feasible(base,row):
    return bool(
        row["weighted_ce"]<=0.90*base["weighted_ce"]
        and row["weighted_tv"]<=0.90*base["weighted_tv"]
        and row["fold_abs_bias"]<=0.60*base["fold_abs_bias"]
        and row["low_target_fold_mean"]<=0.60*base["low_target_fold_mean"]
        and row["low_target_fold_p95"]<=0.60*base["low_target_fold_p95"]
        and row["high_target_fold_abs_bias"]<=base["high_target_fold_abs_bias"]
    )


def ratio(a,b):
    if b is None or float(b)==0.0:
        return None
    return float(a)/float(b)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage",type=Path,required=True)
    ap.add_argument("--stratified-specialist",type=Path,required=True)
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

    adv=load_adv(args.semantic_advantage)
    base,specialist,source=load_source(args.stratified_specialist)
    solver=SolverLibrary(args.solver.resolve(strict=True))

    distill.MASTER_SEED=CAL_SEED
    ca,cb,cal_collection=distill.collect_fresh(
        solver,adv,CAL_EPISODES
    )
    cal=unique_strong(list(ca)+list(cb))
    if len(cal)<900:
        raise RuntimeError(
            f"calibration unique strong sample too small: {len(cal)}"
        )

    base_p,targets,weights=model_probs(base,cal)
    spec_p,targets2,weights2=model_probs(specialist,cal)
    if not np.allclose(targets,targets2) or not np.allclose(weights,weights2):
        raise RuntimeError("calibration target/weight identity drift")

    base_cal=summarize(base_p,targets,weights,cal)
    specialist_cal=summarize(spec_p,targets,weights,cal)

    rows=[]
    feasible_rows=[]
    for scale in SCALES:
        for bias in BIASES:
            p=calibrate_probs(spec_p,scale,bias)
            row={
                "fold_logit_scale":float(scale),
                "fold_logit_bias":float(bias),
                **summarize(p,targets,weights,cal),
            }
            row["calibration_feasible"]=feasible(base_cal,row)
            rows.append(row)
            if row["calibration_feasible"]:
                feasible_rows.append(row)

    if feasible_rows:
        selected=min(
            feasible_rows,
            key=lambda x:(
                x["weighted_ce"],
                x["weighted_tv"],
                abs(x["fold_logit_scale"]-1.0),
                abs(x["fold_logit_bias"]),
            ),
        )
        selection_feasible=True
    else:
        selected=min(
            rows,
            key=lambda x:(
                x["weighted_ce"],
                x["high_target_fold_abs_bias"],
            ),
        )
        selection_feasible=False

    scale=float(selected["fold_logit_scale"])
    bias=float(selected["fold_logit_bias"])

    distill.MASTER_SEED=EVAL_SEED
    ea,eb,eval_collection=distill.collect_fresh(
        solver,adv,EVAL_EPISODES
    )
    evaluation=unique_strong(list(ea)+list(eb))
    if len(evaluation)<900:
        raise RuntimeError(
            f"independent unique strong eval too small: {len(evaluation)}"
        )

    base_ep,tgt,w=model_probs(base,evaluation)
    spec_ep,tgt2,w2=model_probs(specialist,evaluation)
    if not np.allclose(tgt,tgt2) or not np.allclose(w,w2):
        raise RuntimeError("evaluation target/weight identity drift")
    cal_ep=calibrate_probs(spec_ep,scale,bias)

    base_eval=summarize(base_ep,tgt,w,evaluation)
    specialist_eval=summarize(spec_ep,tgt,w,evaluation)
    calibrated_eval=summarize(cal_ep,tgt,w,evaluation)

    criteria={
        "calibration_selection_had_feasible_candidate":
            selection_feasible,
        "new_eval_has_at_least_900_unique_strong_states":
            len(evaluation)>=900,
        "new_eval_has_at_least_40_high_target_fold_states":
            calibrated_eval["high_target_count"]>=40,
        "calibrated_strong_ce_at_least_10pct_better_than_fullpool":
            calibrated_eval["weighted_ce"]<=0.90*base_eval["weighted_ce"],
        "calibrated_strong_tv_at_least_10pct_better_than_fullpool":
            calibrated_eval["weighted_tv"]<=0.90*base_eval["weighted_tv"],
        "calibrated_fold_bias_at_least_40pct_better_than_fullpool":
            calibrated_eval["fold_abs_bias"]<=0.60*base_eval["fold_abs_bias"],
        "calibrated_low_target_fold_mean_at_least_40pct_better":
            calibrated_eval["low_target_fold_mean"]<=0.60*base_eval["low_target_fold_mean"],
        "calibrated_low_target_fold_p95_at_least_40pct_better":
            calibrated_eval["low_target_fold_p95"]<=0.60*base_eval["low_target_fold_p95"],
        "calibrated_straight_flush_bias_at_least_40pct_better":
            calibrated_eval["straight_flush_fold_abs_bias"]<=0.60*base_eval["straight_flush_fold_abs_bias"],
        "calibrated_river_bias_at_least_40pct_better":
            calibrated_eval["river_fold_abs_bias"]<=0.60*base_eval["river_fold_abs_bias"],
        "legitimate_high_fold_bias_not_worse_than_fullpool":
            calibrated_eval["high_target_fold_abs_bias"]<=base_eval["high_target_fold_abs_bias"],
    }
    passed=all(criteria.values())

    args.out_model.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema":OUT_SCHEMA,
        "source_checkpoint_sha256":EXPECTED_SHA,
        "semantic_completed_iteration":FINAL_ITERATION,
        "selected_steps":int(source["selected_steps"]),
        "selected_training_mode":source.get("selected_training_mode"),
        "fold_logit_scale":scale,
        "fold_logit_bias":bias,
        "base_model_state":{
            k:v.detach().cpu()
            for k,v in base.state_dict().items()
        },
        "specialist_model_state":{
            k:v.detach().cpu()
            for k,v in specialist.state_dict().items()
        },
        "route_contract":(
            "postflop made_category>=TRIPS AND FOLD legal -> "
            "Fold-logit-calibrated stratified specialist; otherwise "
            "fullpool general"
        ),
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    },args.out_model)

    report={
        "schema":"SPINCORE_3H_SEMANTIC_FOLD_LOGIT_CALIBRATED_STRONG_SPECIALIST_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_POSTHOC_FOLD_ODDS_CALIBRATION_NO_ACTION_HARDCODE",
        "source_checkpoint_sha256":EXPECTED_SHA,
        "calibration_collection":cal_collection,
        "calibration_unique_strong_samples":len(cal),
        "calibration_fullpool":base_cal,
        "calibration_uncalibrated_specialist":specialist_cal,
        "calibration_grid_size":len(rows),
        "calibration_feasible_count":len(feasible_rows),
        "selected_calibration":selected,
        "selected_fold_logit_scale":scale,
        "selected_fold_logit_bias":bias,
        "independent_eval_collection":eval_collection,
        "independent_unique_strong_samples":len(evaluation),
        "fullpool_general":base_eval,
        "uncalibrated_stratified_specialist":specialist_eval,
        "fold_logit_calibrated_specialist":calibrated_eval,
        "effect_sizes":{
            "calibrated_vs_fullpool_ce_ratio":
                ratio(calibrated_eval["weighted_ce"],base_eval["weighted_ce"]),
            "calibrated_vs_fullpool_tv_ratio":
                ratio(calibrated_eval["weighted_tv"],base_eval["weighted_tv"]),
            "calibrated_vs_fullpool_fold_bias_ratio":
                ratio(calibrated_eval["fold_abs_bias"],base_eval["fold_abs_bias"]),
            "calibrated_vs_fullpool_low_mean_ratio":
                ratio(calibrated_eval["low_target_fold_mean"],base_eval["low_target_fold_mean"]),
            "calibrated_vs_fullpool_low_p95_ratio":
                ratio(calibrated_eval["low_target_fold_p95"],base_eval["low_target_fold_p95"]),
            "calibrated_vs_fullpool_sf_bias_ratio":
                ratio(calibrated_eval["straight_flush_fold_abs_bias"],base_eval["straight_flush_fold_abs_bias"]),
            "calibrated_vs_fullpool_river_bias_ratio":
                ratio(calibrated_eval["river_fold_abs_bias"],base_eval["river_fold_abs_bias"]),
            "calibrated_minus_fullpool_high_bias":
                float(calibrated_eval["high_target_fold_abs_bias"])
                -float(base_eval["high_target_fold_abs_bias"]),
        },
        "precommitted_criteria":criteria,
        "semantic_fold_logit_calibrated_specialist_pass":passed,
        "interpretation":(
            "PASS means a two-parameter calibration of the learned specialist's "
            "Fold odds corrects both low-target over-Folding and legitimate "
            "high-target under-confidence on a fully independent stream, while "
            "preserving the learned Call/All-in ratio. PASS authorizes one exact "
            "DC1 1k replay only; no production/DC2/5k claim."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )

    print("=== fold-logit calibrated strong specialist ===")
    print(f"selected_scale={scale}")
    print(f"selected_bias={bias}")
    print(f"calibration_feasible_count={len(feasible_rows)}")
    print("fullpool="+json.dumps(base_eval,sort_keys=True))
    print("uncalibrated="+json.dumps(specialist_eval,sort_keys=True))
    print("calibrated="+json.dumps(calibrated_eval,sort_keys=True))
    print("criteria="+json.dumps(criteria,sort_keys=True))
    print(
        "semantic_fold_logit_calibrated_specialist_pass="
        +str(passed)
    )
    print(f"report={args.report.resolve()}")
    print("SEMANTIC_FOLD_LOGIT_CALIBRATED_SPECIALIST_AUDIT_COMPLETE")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
