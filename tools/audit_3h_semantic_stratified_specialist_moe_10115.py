#!/usr/bin/env python3
from __future__ import annotations

"""Train a target-stratified learned specialist for the rare strong-hand stratum.

The confidence-gated router failed because it could not simultaneously preserve
rare legitimate high-Fold teacher states and improve overall strong-hand CE
enough.  This gate addresses the underlying class imbalance directly in
specialist training while keeping the inference route simple and unchanged:
postflop made_category >= TRIPS and Fold legal -> specialist.

No action is hard-coded.  The specialist learns the complete teacher action
distribution.

Two training arms share the exact same fresh teacher pool, initialization and
milestone budgets:
- UNIFORM: natural sampling over unique strong states;
- STRATIFIED: target-aware minibatches with fixed 70/15/15 proportions for
  low/mid/high teacher Fold strata.

Selection is performed on a deterministic stratified 80/20 split from a fresh
300k-episode pool.  The chosen arm+step must satisfy calibration constraints.
The final model is retrained from the same full-pool initialization on all
collected unique strong states.  Final validation uses a completely untouched
150k-episode stream.

Research-only. PASS authorizes one exact fixed-seed DC1 1k replay.
"""

import argparse
import copy
import json
import math
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
FULLPOOL_SCHEMA="SPINCORE_3H_SEMANTIC_FULLPOOL_STRONG_DIVERSITY_TAIL_CANDIDATE_V1"
OUT_SCHEMA="SPINCORE_3H_SEMANTIC_STRATIFIED_STRONG_SPECIALIST_MOE_V1"
FINAL_ITERATION=10115
DOMAIN="THREE_HANDED"
TRAIN_EPISODES=300000
TRAIN_SEED=20260928 ^ 0x57A711
SPLIT_SEED=20260928 ^ 0x5A17ED
MILESTONES=(10,25,50,100,200)
EVAL_EPISODES=150000
EVAL_SEED=20260928 ^ 0xE150A5
LOW_MAX=0.05
HIGH_MIN=0.50
STRATIFIED_PROPORTIONS=(0.70,0.15,0.15)
FOLD=0

_D3=None
_CFG=None


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


def load_fullpool(path:Path):
    p=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=FULLPOOL_SCHEMA:
        raise RuntimeError("wrong full-pool schema")
    if p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError("full-pool source mismatch")
    if int(p.get("semantic_completed_iteration",-1))!=FINAL_ITERATION:
        raise RuntimeError("full-pool iteration mismatch")
    if int(p.get("selected_steps",-1))!=500:
        raise RuntimeError("full-pool selected-step mismatch")
    m=V1SemanticPolicyNet()
    m.load_state_dict(p["model_state"])
    m.eval()
    return m,p


def clone_model(model):
    out=V1SemanticPolicyNet(model.cfg)
    out.load_state_dict(copy.deepcopy(model.state_dict()))
    return out


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
        k=skey(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def stratum(sample):
    fold=float(sample.target[FOLD])
    if fold<=LOW_MAX:
        return "low"
    if fold>=HIGH_MIN:
        return "high"
    return "mid"


def stratum_counts(samples):
    out={"low":0,"mid":0,"high":0}
    for s in samples:
        out[stratum(s)]+=1
    return out


def stratified_split(samples):
    groups={"low":[],"mid":[],"high":[]}
    for s in samples:
        groups[stratum(s)].append(s)

    rng=random.Random(SPLIT_SEED)
    train=[]
    select=[]
    counts={}
    for name in ("low","mid","high"):
        rows=list(groups[name])
        rng.shuffle(rows)
        if len(rows)<10:
            raise RuntimeError(f"strong stratum {name} too small: {len(rows)}")
        cut=max(1,min(len(rows)-1,int(round(0.80*len(rows)))))
        train.extend(rows[:cut])
        select.extend(rows[cut:])
        counts[name]={
            "total":len(rows),
            "train":cut,
            "select":len(rows)-cut,
        }

    rng.shuffle(train)
    rng.shuffle(select)
    return train,select,counts


def semantic_batch(samples):
    b,t,w=vectorized_batch(samples,"cpu")
    sb=dict(b)
    sb["semantic"]=torch.tensor(
        np.asarray(
            [semantic_vector_from_obs(s.observation) for s in samples],
            dtype=np.float32,
        ),
        dtype=torch.float32,
    )
    return sb,t,w


def draw_uniform(samples,batch_size,rng):
    return [
        samples[i]
        for i in rng.sample(
            range(len(samples)),
            min(batch_size,len(samples)),
        )
    ]


def draw_with_replacement(pool,n,rng):
    if not pool:
        raise RuntimeError("cannot draw from empty specialist stratum")
    return [pool[rng.randrange(len(pool))] for _ in range(n)]


def draw_stratified(groups,batch_size,rng):
    low_n=int(round(batch_size*STRATIFIED_PROPORTIONS[0]))
    mid_n=int(round(batch_size*STRATIFIED_PROPORTIONS[1]))
    high_n=batch_size-low_n-mid_n
    rows=[]
    rows.extend(draw_with_replacement(groups["low"],low_n,rng))
    rows.extend(draw_with_replacement(groups["mid"],mid_n,rng))
    rows.extend(draw_with_replacement(groups["high"],high_n,rng))
    rng.shuffle(rows)
    return rows


def train_range(model,opt,train,cfg,start,end,rng,mode):
    bs=int(cfg["batch_size"])
    groups=None
    if mode=="STRATIFIED":
        groups={"low":[],"mid":[],"high":[]}
        for s in train:
            groups[stratum(s)].append(s)

    for _step in range(start,end):
        if mode=="UNIFORM":
            ss=draw_uniform(train,bs,rng)
        elif mode=="STRATIFIED":
            ss=draw_stratified(groups,bs,rng)
        else:
            raise RuntimeError(f"unknown train mode {mode}")
        sb,t,w=semantic_batch(ss)
        train_step(model,opt,sb,t,w,"strategy")
    return end


def dummy_v1():
    v1,_,_,_=confirm.initial_policy_pair(_D3,_CFG)
    return v1


def predict_rows(model,samples):
    return coverage.strong_rows(
        coverage.predict_rows(dummy_v1(),model,samples)
    )


def low_target_tail(rows):
    rr=[r for r in rows if float(r["target_fold"])<=LOW_MAX]
    if not rr:
        return {"count":0}
    vals=[float(r["semantic_tail_fold"]) for r in rr]
    return {
        "count":len(rr),
        "pred_fold_mean":statistics.fmean(vals),
        "pred_fold_median":statistics.median(vals),
        "pred_fold_p90":float(np.quantile(np.asarray(vals),0.90)),
        "pred_fold_p95":float(np.quantile(np.asarray(vals),0.95)),
        "pred_fold_max":max(vals),
    }


def high_target(rows):
    rr=[r for r in rows if float(r["target_fold"])>=HIGH_MIN]
    if not rr:
        return {"count":0,"abs_bias":None}
    tgt=statistics.fmean(float(r["target_fold"]) for r in rr)
    pred=statistics.fmean(float(r["semantic_tail_fold"]) for r in rr)
    return {
        "count":len(rr),
        "target_mean":tgt,
        "pred_mean":pred,
        "abs_bias":abs(pred-tgt),
    }


def subset(rows,predicate):
    return coverage.subset_summary([r for r in rows if predicate(r)])


def eval_pack(model,samples):
    m=distill.metrics(dummy_v1(),model,samples)
    rows=predict_rows(model,samples)
    return {
        "weighted_ce":float(m["semantic_weighted_ce"]),
        "weighted_tv":float(m["semantic_weighted_tv"]),
        "fold_overall":coverage.subset_summary(rows),
        "fold_straight_flush":subset(
            rows,lambda r:int(r["made"]) in (4,5)
        ),
        "fold_river":subset(
            rows,lambda r:int(r["street"])==3
        ),
        "low_target_fold_tail":low_target_tail(rows),
        "high_target_fold":high_target(rows),
    }


def selection_feasible(base,row):
    high_base=base["high_target_fold"]["abs_bias"]
    high_row=row["high_target_fold"]["abs_bias"]
    return bool(
        row["weighted_ce"]<=base["weighted_ce"]
        and row["fold_overall"]["semantic_tail_fold_abs_bias"]
            <=0.80*base["fold_overall"]["semantic_tail_fold_abs_bias"]
        and row["low_target_fold_tail"]["pred_fold_mean"]
            <=0.75*base["low_target_fold_tail"]["pred_fold_mean"]
        and row["low_target_fold_tail"]["pred_fold_p95"]
            <=0.75*base["low_target_fold_tail"]["pred_fold_p95"]
        and high_base is not None
        and high_row is not None
        and high_row<=high_base+0.01
    )


def ratio(a,b):
    if b is None or float(b)==0.0:
        return None
    return float(a)/float(b)


def main():
    global _D3,_CFG
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage",type=Path,required=True)
    ap.add_argument("--fullpool-tail",type=Path,required=True)
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
    _D3=(payload.get("domains") or {}).get(DOMAIN) or {}
    _CFG=dict(payload.get("config") or {})

    adv=load_adv(args.semantic_advantage)
    fullpool,_full_payload=load_fullpool(args.fullpool_tail)
    solver=SolverLibrary(args.solver.resolve(strict=True))

    # New independent specialist pool.
    distill.MASTER_SEED=TRAIN_SEED
    ta,tb,train_collection=distill.collect_fresh(
        solver,adv,TRAIN_EPISODES
    )
    unique=unique_strong(list(ta)+list(tb))
    if len(unique)<1200:
        raise RuntimeError(
            f"specialist unique strong pool too small: {len(unique)}"
        )

    train,select,split_counts=stratified_split(unique)
    if split_counts["high"]["select"]<10:
        raise RuntimeError(
            "specialist selection high-target sample too small"
        )

    base_select=eval_pack(fullpool,select)
    rows=[]
    state_cache={}

    for mode in ("UNIFORM","STRATIFIED"):
        model=clone_model(fullpool)
        opt=torch.optim.Adam(
            model.parameters(),
            lr=float(_CFG["learning_rate"]),
        )
        rng=random.Random(
            (distill.TRAIN_SEED ^ 0xA11CE)
            + (0 if mode=="UNIFORM" else 0x10001)
        )
        current=0
        for milestone in MILESTONES:
            current=train_range(
                model,opt,train,_CFG,current,milestone,rng,mode
            )
            pack=eval_pack(model,select)
            row={
                "mode":mode,
                "steps":milestone,
                **pack,
            }
            row["selection_feasible"]=selection_feasible(
                base_select,row
            )
            rows.append(row)
            state_cache[(mode,milestone)]=copy.deepcopy(
                model.state_dict()
            )
            print(
                "STRATIFIED_SPECIALIST_MILESTONE "
                +json.dumps({
                    "mode":mode,
                    "steps":milestone,
                    "ce":row["weighted_ce"],
                    "tv":row["weighted_tv"],
                    "fold_bias":row["fold_overall"]["semantic_tail_fold_abs_bias"],
                    "low_mean":row["low_target_fold_tail"]["pred_fold_mean"],
                    "low_p95":row["low_target_fold_tail"]["pred_fold_p95"],
                    "high_bias":row["high_target_fold"]["abs_bias"],
                    "feasible":row["selection_feasible"],
                },sort_keys=True),
                flush=True,
            )

    feasible=[r for r in rows if r["selection_feasible"]]
    if not feasible:
        raise RuntimeError(
            "no specialist mode/step satisfies frozen selection constraints"
        )

    selected=min(
        feasible,
        key=lambda r:(
            r["weighted_ce"],
            r["weighted_tv"],
            0 if r["mode"]=="STRATIFIED" else 1,
            r["steps"],
        ),
    )
    selected_mode=str(selected["mode"])
    selected_steps=int(selected["steps"])

    # Retrain from fullpool init on all unique strong states.
    specialist=clone_model(fullpool)
    specialist_opt=torch.optim.Adam(
        specialist.parameters(),
        lr=float(_CFG["learning_rate"]),
    )
    rng=random.Random(
        (distill.TRAIN_SEED ^ 0xA11CE)
        + (0 if selected_mode=="UNIFORM" else 0x10001)
    )
    train_range(
        specialist,
        specialist_opt,
        unique,
        _CFG,
        0,
        selected_steps,
        rng,
        selected_mode,
    )
    specialist.eval()

    # Completely untouched independent validation.
    distill.MASTER_SEED=EVAL_SEED
    ea,eb,eval_collection=distill.collect_fresh(
        solver,adv,EVAL_EPISODES
    )
    evaluation=unique_strong(list(ea)+list(eb))
    if len(evaluation)<700:
        raise RuntimeError(
            f"independent unique strong eval too small: {len(evaluation)}"
        )

    base_eval=eval_pack(fullpool,evaluation)
    spec_eval=eval_pack(specialist,evaluation)

    high_count=int(spec_eval["high_target_fold"]["count"])
    bce=float(base_eval["weighted_ce"])
    sce=float(spec_eval["weighted_ce"])
    btv=float(base_eval["weighted_tv"])
    stv=float(spec_eval["weighted_tv"])
    bbias=float(
        base_eval["fold_overall"]["semantic_tail_fold_abs_bias"]
    )
    sbias=float(
        spec_eval["fold_overall"]["semantic_tail_fold_abs_bias"]
    )
    bmean=float(
        base_eval["low_target_fold_tail"]["pred_fold_mean"]
    )
    smean=float(
        spec_eval["low_target_fold_tail"]["pred_fold_mean"]
    )
    bp95=float(
        base_eval["low_target_fold_tail"]["pred_fold_p95"]
    )
    sp95=float(
        spec_eval["low_target_fold_tail"]["pred_fold_p95"]
    )
    bsf=float(
        base_eval["fold_straight_flush"]["semantic_tail_fold_abs_bias"]
    )
    ssf=float(
        spec_eval["fold_straight_flush"]["semantic_tail_fold_abs_bias"]
    )
    briver=float(
        base_eval["fold_river"]["semantic_tail_fold_abs_bias"]
    )
    sriver=float(
        spec_eval["fold_river"]["semantic_tail_fold_abs_bias"]
    )
    bh=base_eval["high_target_fold"]["abs_bias"]
    sh=spec_eval["high_target_fold"]["abs_bias"]

    criteria={
        "new_eval_has_at_least_700_unique_strong_states":
            len(evaluation)>=700,
        "new_eval_has_at_least_20_high_target_fold_states":
            high_count>=20,
        "specialist_strong_ce_at_least_10pct_better":
            sce<=0.90*bce,
        "specialist_strong_tv_at_least_10pct_better":
            stv<=0.90*btv,
        "specialist_fold_abs_bias_at_least_30pct_better":
            sbias<=0.70*bbias,
        "specialist_low_target_fold_mean_at_least_30pct_better":
            smean<=0.70*bmean,
        "specialist_low_target_fold_p95_at_least_30pct_better":
            sp95<=0.70*bp95,
        "specialist_straight_flush_bias_at_least_30pct_better":
            ssf<=0.70*bsf,
        "specialist_river_bias_at_least_30pct_better":
            sriver<=0.70*briver,
        "legitimate_high_fold_bias_not_worse_by_over_1pp":
            (
                bh is not None
                and sh is not None
                and sh<=bh+0.01
            ),
    }
    passed=all(criteria.values())

    args.out_model.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema":OUT_SCHEMA,
        "source_checkpoint_sha256":EXPECTED_SHA,
        "semantic_completed_iteration":FINAL_ITERATION,
        "selected_steps":selected_steps,
        "selected_training_mode":selected_mode,
        "specialist_train_episodes":TRAIN_EPISODES,
        "specialist_unique_strong_states":len(unique),
        "base_model_state":{
            k:v.detach().cpu()
            for k,v in fullpool.state_dict().items()
        },
        "specialist_model_state":{
            k:v.detach().cpu()
            for k,v in specialist.state_dict().items()
        },
        "route_contract":(
            "postflop made_category>=TRIPS AND FOLD legal -> specialist; "
            "otherwise fullpool general"
        ),
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    },args.out_model)

    report={
        "schema":"SPINCORE_3H_SEMANTIC_STRATIFIED_STRONG_SPECIALIST_MOE_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_TARGET_STRATIFIED_SPECIALIST_NO_ACTION_HARDCODE",
        "source_checkpoint_sha256":EXPECTED_SHA,
        "train_collection":train_collection,
        "unique_strong_pool":len(unique),
        "unique_strong_strata":stratum_counts(unique),
        "split_counts":split_counts,
        "specialist_train_samples":len(train),
        "specialist_selection_samples":len(select),
        "fullpool_selection":base_select,
        "candidate_grid":rows,
        "selected_training_mode":selected_mode,
        "selected_steps":selected_steps,
        "selected_selection_row":selected,
        "independent_eval_collection":eval_collection,
        "independent_unique_strong_samples":len(evaluation),
        "independent_eval_strata":stratum_counts(evaluation),
        "fullpool_general":base_eval,
        "stratified_specialist":spec_eval,
        "effect_sizes":{
            "ce_ratio":ratio(sce,bce),
            "tv_ratio":ratio(stv,btv),
            "fold_bias_ratio":ratio(sbias,bbias),
            "low_target_mean_ratio":ratio(smean,bmean),
            "low_target_p95_ratio":ratio(sp95,bp95),
            "straight_flush_bias_ratio":ratio(ssf,bsf),
            "river_bias_ratio":ratio(sriver,briver),
            "high_target_bias_delta":(
                None if bh is None or sh is None
                else float(sh)-float(bh)
            ),
        },
        "precommitted_criteria":criteria,
        "semantic_stratified_specialist_moe_pass":passed,
        "interpretation":(
            "PASS means target-stratified training repairs the rare strong "
            "AveragePolicy surface on a fully independent stream while preserving "
            "legitimate high-Fold teacher states. PASS authorizes one exact DC1 "
            "1k replay only; no production/DC2/5k claim."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )

    print("=== target-stratified strong specialist ===")
    print(f"selected_training_mode={selected_mode}")
    print(f"selected_steps={selected_steps}")
    print("fullpool="+json.dumps(base_eval,sort_keys=True))
    print("specialist="+json.dumps(spec_eval,sort_keys=True))
    print("criteria="+json.dumps(criteria,sort_keys=True))
    print(
        "semantic_stratified_specialist_moe_pass="
        +str(passed)
    )
    print(f"report={args.report.resolve()}")
    print("SEMANTIC_STRATIFIED_SPECIALIST_MOE_AUDIT_COMPLETE")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
