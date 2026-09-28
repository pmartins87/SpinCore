#!/usr/bin/env python3
from __future__ import annotations

"""Strong-hand specialist Mixture-of-Experts gate for canonical 10115.

Motivation:
- the general semantic AveragePolicy improved strongly with 1,039 novel strong
  states, but the fixed DC1 1k still contains two sampled trips-or-better folds;
- continuing to add rare examples to the same global network shows diminishing
  returns and can oscillate locally;
- the remaining problem is confined to a very sparse semantic stratum.

This experiment keeps the validated full-pool general policy unchanged for all
ordinary states and trains a specialist policy only for:
    postflop made_category >= TRIPS and FOLD legal.

The specialist learns the complete teacher action distribution.  It does NOT
hard-code "never fold" or alter Advantage/CFR targets.

Training/selection:
- regenerate a new independent 120k-episode canonical-teacher stream;
- collect unique strong Fold-legal states;
- deterministic 80/20 specialist train/selection split;
- initialize specialist from the full-pool AveragePolicy;
- select among 10/25/50/100 fine-tuning steps by lowest weighted CE on the
  specialist selection split;
- retrain from the same full-pool initialization on all collected strong states
  for the selected budget.

Validation:
- completely new 80k-episode stream;
- compare full-pool general vs specialist on strong states, low-target-fold
  tails, river, straight/flush, and legitimate high-Fold targets.

Research only. PASS authorizes one exact DC1 1k MoE replay.
"""

import argparse, copy, json, random, statistics, sys
from pathlib import Path
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
OUT_SCHEMA="SPINCORE_3H_SEMANTIC_STRONG_SPECIALIST_MOE_V1"
FINAL_ITERATION=10115
DOMAIN="THREE_HANDED"
TRAIN_EPISODES=120000
TRAIN_SEED=20260928 ^ 0x5A3C1A
SPLIT_SEED=20260928 ^ 0x5A117
MILESTONES=(10,25,50,100)
EVAL_EPISODES=80000
EVAL_SEED=20260928 ^ 0xE80E51
FOLD=0


def load_adv(path):
    p=torch.load(Path(path).resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=ADV_SCHEMA or p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError("canonical Advantage artifact mismatch")
    if int(p.get("completed_iteration",-1))!=FINAL_ITERATION:
        raise RuntimeError("canonical Advantage iteration mismatch")
    states=list(p.get("members") or [])
    if len(states)!=8:
        raise RuntimeError("expected eight Advantage members")
    return [distill.load_semantic_advantage(s) for s in states]


def load_fullpool(path):
    p=torch.load(Path(path).resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=FULLPOOL_SCHEMA or p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError("full-pool policy artifact mismatch")
    if int(p.get("semantic_completed_iteration",-1))!=FINAL_ITERATION:
        raise RuntimeError("full-pool iteration mismatch")
    if int(p.get("selected_steps",-1))!=500:
        raise RuntimeError("full-pool base step-budget mismatch")
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
    return int(m["street"])>0 and int(m["made"])>=3 and bool(m["fold_legal"])


def key(sample):
    return (bytes(sample.observation),tuple(bool(x) for x in sample.legal))


def unique_strong(samples):
    out=[]; seen=set()
    for s in samples:
        if not strong(s):
            continue
        k=key(s)
        if k in seen:
            continue
        seen.add(k); out.append(s)
    return out


def train_steps(model,opt,samples,cfg,start_step,end_step,rng):
    bs=int(cfg["batch_size"])
    for step in range(start_step,end_step):
        idx=rng.sample(range(len(samples)),min(bs,len(samples)))
        ss=[samples[i] for i in idx]
        b,t,w=vectorized_batch(ss,"cpu")
        sb=dict(b)
        sb["semantic"]=torch.tensor(
            np.asarray(
                [semantic_vector_from_obs(s.observation) for s in ss],
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )
        train_step(model,opt,sb,t,w,"strategy")
    return end_step


def semantic_metrics(model,samples):
    # distill.metrics needs a V1 reference only for paired reporting; its
    # semantic fields are the metrics we use here.
    dummy_v1,_,_,_=confirm.initial_policy_pair(_D3,_CFG)
    return distill.metrics(dummy_v1,model,samples)


def fold_rows(model,samples):
    dummy_v1,_,_,_=confirm.initial_policy_pair(_D3,_CFG)
    return coverage.strong_rows(coverage.predict_rows(dummy_v1,model,samples))


def subset(rows,pred):
    rr=[r for r in rows if pred(r)]
    return coverage.subset_summary(rr)


def low_target_tail(rows):
    rr=[r for r in rows if float(r["target_fold"])<=0.05]
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


def high_target_bias(rows):
    rr=[r for r in rows if float(r["target_fold"])>=0.50]
    if not rr:
        return {"count":0,"abs_bias":None}
    t=statistics.fmean(float(r["target_fold"]) for r in rr)
    p=statistics.fmean(float(r["semantic_tail_fold"]) for r in rr)
    return {"count":len(rr),"target_mean":t,"pred_mean":p,"abs_bias":abs(p-t)}


def eval_pack(model,samples):
    rows=fold_rows(model,samples)
    return {
        "strong_metrics":semantic_metrics(model,samples),
        "fold_overall":coverage.subset_summary(rows),
        "fold_straight_flush":subset(rows,lambda r:int(r["made"]) in (4,5)),
        "fold_river":subset(rows,lambda r:int(r["street"])==3),
        "low_target_fold_tail":low_target_tail(rows),
        "high_target_fold":high_target_bias(rows),
    }


def ratio(a,b):
    return None if b is None or float(b)==0 else float(a)/float(b)


_D3=None
_CFG=None


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
        raise RuntimeError("checkpoint SHA mismatch")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    _D3=(payload.get("domains") or {}).get(DOMAIN) or {}
    _CFG=dict(payload.get("config") or {})

    adv=load_adv(args.semantic_advantage)
    fullpool,full_payload=load_fullpool(args.fullpool_tail)
    solver=SolverLibrary(args.solver.resolve(strict=True))

    # New independent specialist collection.
    distill.MASTER_SEED=TRAIN_SEED
    ta,tb,train_collection=distill.collect_fresh(solver,adv,TRAIN_EPISODES)
    unique=unique_strong(list(ta)+list(tb))
    if len(unique)<500:
        raise RuntimeError(f"specialist strong pool too small: {len(unique)}")

    order=list(range(len(unique)))
    random.Random(SPLIT_SEED).shuffle(order)
    cut=max(1,int(round(0.80*len(order))))
    train=[unique[i] for i in order[:cut]]
    select=[unique[i] for i in order[cut:]]
    if len(select)<100:
        raise RuntimeError(f"specialist selection split too small: {len(select)}")

    model=clone_model(fullpool)
    opt=torch.optim.Adam(model.parameters(),lr=float(_CFG["learning_rate"]))
    rng=random.Random(distill.TRAIN_SEED ^ 0x5EEC1A)
    milestone_rows=[]
    current=0
    states={}
    for m in MILESTONES:
        current=train_steps(model,opt,train,_CFG,current,m,rng)
        metrics=semantic_metrics(model,select)
        rows=fold_rows(model,select)
        tail=low_target_tail(rows)
        row={
            "steps":m,
            "selection_weighted_ce":float(metrics["semantic_weighted_ce"]),
            "selection_weighted_tv":float(metrics["semantic_weighted_tv"]),
            "selection_low_target_fold_tail":tail,
        }
        milestone_rows.append(row)
        states[m]=copy.deepcopy(model.state_dict())
        print("SPECIALIST_MILESTONE "+json.dumps(row,sort_keys=True),flush=True)

    selected=min(
        milestone_rows,
        key=lambda r:(r["selection_weighted_ce"],r["selection_weighted_tv"],r["steps"]),
    )
    selected_steps=int(selected["steps"])

    # Retrain selected budget on the entire unique strong pool.
    specialist=clone_model(fullpool)
    specialist_opt=torch.optim.Adam(
        specialist.parameters(),lr=float(_CFG["learning_rate"])
    )
    rng2=random.Random(distill.TRAIN_SEED ^ 0x5EEC1A)
    train_steps(
        specialist,specialist_opt,unique,_CFG,0,selected_steps,rng2
    )
    specialist.eval()

    # Completely new validation stream.
    distill.MASTER_SEED=EVAL_SEED
    ea,eb,eval_collection=distill.collect_fresh(solver,adv,EVAL_EPISODES)
    evaluation=list(ea)+list(eb)
    strong_eval=unique_strong(evaluation)
    if len(strong_eval)<400:
        raise RuntimeError(f"independent strong eval too small: {len(strong_eval)}")

    base=eval_pack(fullpool,strong_eval)
    spec=eval_pack(specialist,strong_eval)

    bce=float(base["strong_metrics"]["semantic_weighted_ce"])
    sce=float(spec["strong_metrics"]["semantic_weighted_ce"])
    btv=float(base["strong_metrics"]["semantic_weighted_tv"])
    stv=float(spec["strong_metrics"]["semantic_weighted_tv"])
    bbias=float(base["fold_overall"]["semantic_tail_fold_abs_bias"])
    sbias=float(spec["fold_overall"]["semantic_tail_fold_abs_bias"])
    bsf=float(base["fold_straight_flush"]["semantic_tail_fold_abs_bias"])
    ssf=float(spec["fold_straight_flush"]["semantic_tail_fold_abs_bias"])
    br=float(base["fold_river"]["semantic_tail_fold_abs_bias"])
    sr=float(spec["fold_river"]["semantic_tail_fold_abs_bias"])
    bmean=float(base["low_target_fold_tail"]["pred_fold_mean"])
    smean=float(spec["low_target_fold_tail"]["pred_fold_mean"])
    bp95=float(base["low_target_fold_tail"]["pred_fold_p95"])
    sp95=float(spec["low_target_fold_tail"]["pred_fold_p95"])
    bh=base["high_target_fold"].get("abs_bias")
    sh=spec["high_target_fold"].get("abs_bias")

    criteria={
        "new_eval_has_at_least_400_unique_strong_states":len(strong_eval)>=400,
        "specialist_strong_ce_at_least_20pct_better":sce<=0.80*bce,
        "specialist_strong_tv_at_least_20pct_better":stv<=0.80*btv,
        "specialist_fold_abs_bias_at_least_30pct_better":sbias<=0.70*bbias,
        "specialist_low_target_fold_mean_at_least_40pct_better":smean<=0.60*bmean,
        "specialist_low_target_fold_p95_at_least_30pct_better":sp95<=0.70*bp95,
        "specialist_straight_flush_bias_at_least_30pct_better":ssf<=0.70*bsf,
        "specialist_river_bias_at_least_30pct_better":sr<=0.70*br,
        "legitimate_high_fold_target_bias_not_worse_by_over_2pp":(
            bh is not None and sh is not None and sh<=bh+0.02
        ),
    }
    passed=all(criteria.values())

    args.out_model.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema":OUT_SCHEMA,
        "source_checkpoint_sha256":EXPECTED_SHA,
        "semantic_completed_iteration":FINAL_ITERATION,
        "selected_steps":selected_steps,
        "specialist_train_episodes":TRAIN_EPISODES,
        "specialist_unique_strong_states":len(unique),
        "base_model_state":{
            k:v.detach().cpu() for k,v in fullpool.state_dict().items()
        },
        "specialist_model_state":{
            k:v.detach().cpu() for k,v in specialist.state_dict().items()
        },
        "route_contract":"postflop made_category>=TRIPS AND FOLD legal -> specialist; otherwise fullpool general",
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    },args.out_model)

    report={
        "schema":"SPINCORE_3H_SEMANTIC_STRONG_SPECIALIST_MOE_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_LEARNED_STRONG_SPECIALIST_NO_ACTION_HARDCODE",
        "source_checkpoint_sha256":EXPECTED_SHA,
        "train_collection":train_collection,
        "unique_strong_pool":len(unique),
        "specialist_train_samples":len(train),
        "specialist_selection_samples":len(select),
        "milestones":milestone_rows,
        "selected_steps":selected_steps,
        "independent_eval_collection":eval_collection,
        "independent_eval_samples":len(evaluation),
        "independent_unique_strong_samples":len(strong_eval),
        "fullpool_general":base,
        "strong_specialist":spec,
        "effect_sizes":{
            "specialist_vs_fullpool_strong_ce_ratio":ratio(sce,bce),
            "specialist_vs_fullpool_strong_tv_ratio":ratio(stv,btv),
            "specialist_vs_fullpool_fold_bias_ratio":ratio(sbias,bbias),
            "specialist_vs_fullpool_low_target_fold_mean_ratio":ratio(smean,bmean),
            "specialist_vs_fullpool_low_target_fold_p95_ratio":ratio(sp95,bp95),
            "specialist_vs_fullpool_straight_flush_bias_ratio":ratio(ssf,bsf),
            "specialist_vs_fullpool_river_bias_ratio":ratio(sr,br),
        },
        "precommitted_criteria":criteria,
        "semantic_strong_specialist_moe_pass":passed,
        "interpretation":(
            "PASS means a learned specialist for the rare strong/Fold-legal "
            "semantic stratum improves teacher-policy distillation on an untouched "
            "independent stream without hard-coding poker actions. PASS authorizes "
            "one exact DC1 1k MoE replay only."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("=== strong specialist MoE audit ===")
    print("selected_steps="+str(selected_steps))
    print("fullpool="+json.dumps(base,sort_keys=True))
    print("specialist="+json.dumps(spec,sort_keys=True))
    print("criteria="+json.dumps(criteria,sort_keys=True))
    print(f"semantic_strong_specialist_moe_pass={passed}")
    print(f"report={args.report.resolve()}")
    print("SEMANTIC_STRONG_SPECIALIST_MOE_AUDIT_COMPLETE")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
