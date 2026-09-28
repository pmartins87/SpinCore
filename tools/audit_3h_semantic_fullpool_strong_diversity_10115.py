#!/usr/bin/env python3
from __future__ import annotations

"""Use every novel strong-hand state from the validated 180k teacher pool.

The +512 stratified candidate improved the independent strong-hand surface and
reduced fixed-DC1 sampled strong folds from 6 -> 3, but still missed the frozen
<= baseline-count gate.  This experiment does not target those DC1 states.
Instead it uses the complete independently generated novel strong-hand pool
that already underpinned the stratified experiment.

Training:
- exact canonical 22,726 ordinary strategy samples;
- every unique novel postflop trips-or-better / Fold-legal state from the same
  deterministic 180,000-episode teacher pool (expected 1,039 states);
- same 500 optimizer steps and same 10105 initialization.

Evaluation:
- completely new 50,000-episode teacher stream;
- compare canonical tail, +512 stratified tail, and full-pool tail.

Research-only.
"""

import argparse, json, random, statistics, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_fresh_semantic_strategy_independent500_10105 as confirm
import audit_3h_semantic_tail_strong_hand_coverage_10115 as coverage
from audit_3h_average_policy_semantic_continuation_10105 import V1SemanticPolicyNet, semantic_vector_from_obs
from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.training import train_step
from spincore.solver import SolverLibrary

EXPECTED_SHA=distill.EXPECTED_SHA
ADV_SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_ENSEMBLE_V1"
TAIL_SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_TAIL_POLICY_V1"
STRAT_SCHEMA="SPINCORE_3H_SEMANTIC_STRATIFIED_STRONG_DIVERSITY_TAIL_CANDIDATE_V1"
OUT_SCHEMA="SPINCORE_3H_SEMANTIC_FULLPOOL_STRONG_DIVERSITY_TAIL_CANDIDATE_V1"
DOMAIN="THREE_HANDED"
FINAL_ITERATION=10115
BASE_TRAIN_SEED=20260927 ^ 0x10115A
BASE_TRAIN_EPISODES=8000
AUGMENT_EPISODES=180000
AUGMENT_SEED=20260928 ^ 0x51A7F1
EVAL_EPISODES=50000
EVAL_SEED=20260928 ^ 0xF011E5
SELECTED_STEPS=500
FOLD=0


def load_adv(path):
    p=torch.load(Path(path).resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=ADV_SCHEMA or p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError("canonical Advantage artifact mismatch")
    if int(p.get("completed_iteration",-1))!=FINAL_ITERATION:
        raise RuntimeError("canonical Advantage iteration mismatch")
    states=list(p.get("members") or [])
    if len(states)!=8: raise RuntimeError("expected eight Advantage members")
    return [distill.load_semantic_advantage(s) for s in states]


def load_semantic(path,schema):
    p=torch.load(Path(path).resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=schema or p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError(f"semantic artifact mismatch for {schema}")
    it=int(p.get("completed_iteration",p.get("semantic_completed_iteration",-1)))
    if it!=FINAL_ITERATION or int(p.get("selected_steps",-1))!=SELECTED_STEPS:
        raise RuntimeError("semantic artifact iteration/step mismatch")
    m=V1SemanticPolicyNet(); m.load_state_dict(p["model_state"]); m.eval()
    return m,p


def strong(s):
    m=coverage.sample_meta(s)
    return int(m["street"])>0 and int(m["made"])>=3 and bool(m["fold_legal"])


def skey(s):
    return (bytes(s.observation),tuple(bool(x) for x in s.legal))


def describe(samples):
    cats={}; streets={}; tg=[]
    for s in samples:
        m=coverage.sample_meta(s)
        cats[str(int(m["made"]))]=cats.get(str(int(m["made"])),0)+1
        streets[str(int(m["street"]))]=streets.get(str(int(m["street"])),0)+1
        tg.append(float(s.target[FOLD]))
    return {
        "count":len(samples),"category_counts":cats,"street_counts":streets,
        "target_fold_mean":statistics.fmean(tg) if tg else None,
        "target_fold_max":max(tg) if tg else None,
    }


def train_sem(model,opt,train,cfg):
    rng=random.Random(distill.TRAIN_SEED); bs=int(cfg["batch_size"])
    for step in range(SELECTED_STEPS):
        idx=rng.sample(range(len(train)),min(bs,len(train)))
        ss=[train[i] for i in idx]
        b,t,w=vectorized_batch(ss,"cpu")
        sb=dict(b)
        sb["semantic"]=torch.tensor(np.asarray(
            [semantic_vector_from_obs(s.observation) for s in ss],dtype=np.float32
        ),dtype=torch.float32)
        train_step(model,opt,sb,t,w,"strategy")
        if (step+1)%100==0:
            print(f"FULLPOOL_DIVERSITY_TRAIN {step+1}/{SELECTED_STEPS}",flush=True)


def eval_strong(v1,sem,samples):
    rows=coverage.strong_rows(coverage.predict_rows(v1,sem,samples))
    return {
        "overall":coverage.subset_summary(rows),
        "straight_or_flush":coverage.subset_summary([r for r in rows if int(r["made"]) in (4,5)]),
        "river":coverage.subset_summary([r for r in rows if int(r["street"])==3]),
        "strata":coverage.strata(rows),
    }


def ratio(a,b):
    return None if b is None or float(b)==0 else float(a)/float(b)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage",type=Path,required=True)
    ap.add_argument("--canonical-tail",type=Path,required=True)
    ap.add_argument("--stratified-tail",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--out-model",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    args=ap.parse_args(); torch.set_num_threads(args.threads)

    cp=args.checkpoint.resolve(strict=True)
    if distill.sha256(cp)!=EXPECTED_SHA: raise RuntimeError("checkpoint SHA mismatch")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    d3=(payload.get("domains") or {}).get(DOMAIN) or {}
    cfg=dict(payload.get("config") or {})

    adv=load_adv(args.semantic_advantage)
    canonical,canonical_payload=load_semantic(args.canonical_tail,TAIL_SCHEMA)
    stratified,_=load_semantic(args.stratified_tail,STRAT_SCHEMA)
    solver=SolverLibrary(args.solver.resolve(strict=True))

    distill.MASTER_SEED=BASE_TRAIN_SEED
    base_train,base_hold,base_collection=distill.collect_fresh(solver,adv,BASE_TRAIN_EPISODES)
    if len(base_train)!=22726 or len(base_hold)!=5701:
        raise RuntimeError("canonical base stream identity drift")
    base_strong=[s for s in base_train if strong(s)]
    if len(base_strong)!=44: raise RuntimeError("canonical base strong-count drift")

    base_v1,base_v1opt,base_sem,base_semopt=confirm.initial_policy_pair(d3,cfg)
    confirm.train_500(base_v1,base_v1opt,base_sem,base_semopt,base_train,cfg)
    maxdiff=max(float((base_sem.state_dict()[k].detach().cpu()-canonical_payload["model_state"][k].detach().cpu()).abs().max().item()) for k in base_sem.state_dict())
    if maxdiff>2e-6: raise RuntimeError(f"canonical reproduction drift {maxdiff}")

    distill.MASTER_SEED=AUGMENT_SEED
    aa,ab,aug_collection=distill.collect_fresh(solver,adv,AUGMENT_EPISODES)
    base_keys={skey(s) for s in base_strong}; seen=set(); novel=[]
    for s in list(aa)+list(ab):
        if not strong(s): continue
        k=skey(s)
        if k in base_keys or k in seen: continue
        seen.add(k); novel.append(s)
    if len(novel)!=1039:
        raise RuntimeError(f"novel-pool identity drift {len(novel)} != 1039")

    train=list(base_train)+novel
    _v,_vo,full,fullopt=confirm.initial_policy_pair(d3,cfg)
    train_sem(full,fullopt,train,cfg)

    distill.MASTER_SEED=EVAL_SEED
    ea,eb,eval_collection=distill.collect_fresh(solver,adv,EVAL_EPISODES)
    evaluation=list(ea)+list(eb)
    strong_eval=[s for s in evaluation if strong(s)]
    if len(strong_eval)<220:
        raise RuntimeError(f"independent strong eval too small: {len(strong_eval)}")

    base_global=distill.metrics(base_v1,base_sem,evaluation)
    strat_global=distill.metrics(base_v1,stratified,evaluation)
    full_global=distill.metrics(base_v1,full,evaluation)
    base_s=eval_strong(base_v1,base_sem,strong_eval)
    strat_s=eval_strong(base_v1,stratified,strong_eval)
    full_s=eval_strong(base_v1,full,strong_eval)

    def bias(x,part="overall"): return float(x[part]["semantic_tail_fold_abs_bias"])
    b=bias(base_s); s=bias(strat_s); f=bias(full_s)
    bsf=bias(base_s,"straight_or_flush"); ssf=bias(strat_s,"straight_or_flush"); fsf=bias(full_s,"straight_or_flush")
    br=bias(base_s,"river"); sr=bias(strat_s,"river"); fr=bias(full_s,"river")
    bce=float(base_global["semantic_weighted_ce"]); fce=float(full_global["semantic_weighted_ce"])
    btv=float(base_global["semantic_weighted_tv"]); ftv=float(full_global["semantic_weighted_tv"])
    bh=float(base_global["high_card_no_draw_allin"]["semantic_abs_bias"]); fh=float(full_global["high_card_no_draw_allin"]["semantic_abs_bias"])

    criteria={
        "new_eval_has_at_least_220_strong_states":len(strong_eval)>=220,
        "fullpool_strong_bias_at_least_15pct_better_than_stratified":s>0 and f<=0.85*s,
        "fullpool_straight_flush_bias_at_least_15pct_better_than_stratified":ssf>0 and fsf<=0.85*ssf,
        "fullpool_river_bias_at_least_15pct_better_than_stratified":sr>0 and fr<=0.85*sr,
        "fullpool_strong_bias_at_least_70pct_better_than_canonical":b>0 and f<=0.30*b,
        "global_ce_not_worse_than_canonical_by_over_2pct":fce<=1.02*bce,
        "global_tv_not_worse_than_canonical_by_over_2pct":ftv<=1.02*btv,
        "high_card_no_draw_abs_bias_not_worse_by_over_1pp":fh<=bh+0.01,
    }
    passed=all(criteria.values())

    args.out_model.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema":OUT_SCHEMA,"source_checkpoint_sha256":EXPECTED_SHA,
        "semantic_completed_iteration":FINAL_ITERATION,"selected_steps":SELECTED_STEPS,
        "extra_unique_strong_states":len(novel),
        "model_state":{k:v.detach().cpu() for k,v in full.state_dict().items()},
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    },args.out_model)

    report={
        "schema":"SPINCORE_3H_SEMANTIC_FULLPOOL_STRONG_DIVERSITY_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_FULL_NOVEL_STRONG_POOL",
        "canonical_tail_reproduction_max_state_diff":maxdiff,
        "base_collection":base_collection,"base_strong":describe(base_strong),
        "augmentation_collection":aug_collection,"novel_strong_candidates":len(novel),
        "fullpool_extra":describe(novel),"augmented_train_samples":len(train),
        "independent_eval_collection":eval_collection,
        "independent_eval_samples":len(evaluation),"independent_strong_samples":len(strong_eval),
        "canonical":{"global":base_global,"strong":base_s},
        "stratified_512":{"global":strat_global,"strong":strat_s},
        "fullpool_1039":{"global":full_global,"strong":full_s},
        "effect_sizes":{
            "fullpool_vs_stratified_strong_bias_ratio":ratio(f,s),
            "fullpool_vs_stratified_straight_flush_bias_ratio":ratio(fsf,ssf),
            "fullpool_vs_stratified_river_bias_ratio":ratio(fr,sr),
            "fullpool_vs_canonical_strong_bias_ratio":ratio(f,b),
            "fullpool_vs_canonical_global_ce_ratio":ratio(fce,bce),
            "fullpool_vs_canonical_global_tv_ratio":ratio(ftv,btv),
            "fullpool_minus_canonical_hcdn_abs_bias":fh-bh,
        },
        "precommitted_criteria":criteria,
        "semantic_fullpool_diversity_repair_pass":passed,
        "interpretation":"PASS authorizes one exact fixed-seed DC1 1k replay of the full-pool candidate. It does not authorize production, DC2 or 5k.",
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("=== full-pool strong diversity audit ===")
    print("canonical="+json.dumps(base_s["overall"],sort_keys=True))
    print("stratified="+json.dumps(strat_s["overall"],sort_keys=True))
    print("fullpool="+json.dumps(full_s["overall"],sort_keys=True))
    print("criteria="+json.dumps(criteria,sort_keys=True))
    print(f"semantic_fullpool_diversity_repair_pass={passed}")
    print(f"report={args.report.resolve()}")
    print("SEMANTIC_FULLPOOL_STRONG_DIVERSITY_AUDIT_COMPLETE")
    return 0

if __name__=="__main__": raise SystemExit(main())
