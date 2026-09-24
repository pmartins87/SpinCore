#!/usr/bin/env python3
from __future__ import annotations

"""Audit the finalized 10105 3H AveragePolicy fit against its own strategy reservoir.

This diagnostic distinguishes three different objects that must not be conflated:
1) stored historical strategy targets in the Algorithm-R strategy reservoir;
2) the finalized 3H AveragePolicy network used by DC1/deployment;
3) the current iteration-10105 3H Advantage policy.

No model is trained and no checkpoint state is modified.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

from spincore.lean_action_policy import lean_regret_matching_policy
from spincore_nn.action_models import make_advantage_action_model, make_policy_action_model
from spincore_nn.lean_batch import vectorized_batch
import audit_lt3_10105_trip_local_geometry as local

EXPECTED_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
DOMAIN="THREE_HANDED"
SAMPLE_SEED=20260924


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def _weighted_mean(values,weights):
    if not values:
        return None
    sw=sum(float(x) for x in weights)
    return (
        sum(float(v)*float(w) for v,w in zip(values,weights))/sw
        if sw>0 else statistics.fmean(values)
    )


def _metrics(samples,policy,advantage,batch_size:int=4096)->dict[str,Any]:
    if not samples:
        return {
            "n":0,
            "target_vs_average_policy_tv_mean":None,
            "target_vs_average_policy_tv_iteration_weighted_mean":None,
            "target_vs_average_policy_argmax_mismatch_rate":None,
            "target_fold_mean":None,
            "average_policy_fold_mean":None,
            "current_advantage_fold_mean":None,
            "average_policy_vs_current_advantage_tv_mean":None,
        }

    tv_fit=[]
    tv_current=[]
    mismatch=[]
    target_fold=[]
    pred_fold=[]
    current_fold=[]
    weights=[]

    policy.eval(); advantage.eval()
    for start in range(0,len(samples),int(batch_size)):
        chunk=samples[start:start+int(batch_size)]
        batch,target_t,weight_t=vectorized_batch(chunk,"cpu")
        with torch.no_grad():
            logits=policy(batch).masked_fill(~batch["legal"],-1e9)
            pred=torch.softmax(logits,dim=-1).detach().cpu()
            raw=advantage(batch).detach().cpu()
        target=target_t.detach().cpu()
        legal=batch["legal"].detach().cpu()

        for i,s in enumerate(chunk):
            legal_actions=tuple(j for j,v in enumerate(legal[i].tolist()) if bool(v))
            cur=lean_regret_matching_policy(raw[i].tolist(),legal_actions)
            tgt=[float(x) for x in target[i].tolist()]
            pr=[float(x) for x in pred[i].tolist()]
            cur=[float(x) for x in cur]
            tv_fit.append(0.5*sum(abs(tgt[a]-pr[a]) for a in legal_actions))
            tv_current.append(0.5*sum(abs(pr[a]-cur[a]) for a in legal_actions))
            ta=max(legal_actions,key=lambda a:tgt[a])
            pa=max(legal_actions,key=lambda a:pr[a])
            mismatch.append(1.0 if ta!=pa else 0.0)
            weights.append(float(s.weight))
            if 0 in legal_actions:
                target_fold.append(tgt[0])
                pred_fold.append(pr[0])
                current_fold.append(cur[0])

    return {
        "n":len(samples),
        "target_vs_average_policy_tv_mean":statistics.fmean(tv_fit),
        "target_vs_average_policy_tv_median":statistics.median(tv_fit),
        "target_vs_average_policy_tv_p95":sorted(tv_fit)[max(0,math.ceil(0.95*len(tv_fit))-1)],
        "target_vs_average_policy_tv_iteration_weighted_mean":_weighted_mean(tv_fit,weights),
        "target_vs_average_policy_argmax_mismatch_rate":statistics.fmean(mismatch),
        "average_policy_vs_current_advantage_tv_mean":statistics.fmean(tv_current),
        "average_policy_vs_current_advantage_tv_median":statistics.median(tv_current),
        "average_policy_vs_current_advantage_tv_p95":sorted(tv_current)[max(0,math.ceil(0.95*len(tv_current))-1)],
        "fold_legal_n":len(target_fold),
        "target_fold_mean":statistics.fmean(target_fold) if target_fold else None,
        "average_policy_fold_mean":statistics.fmean(pred_fold) if pred_fold else None,
        "current_advantage_fold_mean":statistics.fmean(current_fold) if current_fold else None,
        "target_fold_mae_average_policy":(
            statistics.fmean(abs(a-b) for a,b in zip(target_fold,pred_fold))
            if target_fold else None
        ),
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--global-sample",type=int,default=50000)
    args=ap.parse_args()

    cp=args.checkpoint.resolve(strict=True)
    actual=sha256(cp)
    if actual!=EXPECTED_SHA:
        raise SystemExit(f"checkpoint SHA mismatch: {actual}")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    if int(payload.get("completed_iteration",-1))!=10105:
        raise SystemExit("expected iteration 10105")

    cfg=dict(payload.get("config") or {})
    d3=(payload.get("domains") or {}).get(DOMAIN) or {}
    pol_mem=d3.get("pol_mem") or {}
    items=list(pol_mem.get("items") or [])
    if not items:
        raise SystemExit("empty 3H strategy reservoir")

    _,policy=make_policy_action_model(REPRESENTATION,device="cpu",seed=0)
    policy.load_state_dict(d3["policy"])
    _,advantage=make_advantage_action_model(REPRESENTATION,device="cpu",seed=0)
    advantage.load_state_dict(d3["advantage"])

    rng=random.Random(SAMPLE_SEED)
    n=min(int(args.global_sample),len(items))
    global_idx=rng.sample(range(len(items)),n)
    global_samples=[items[i] for i in global_idx]

    # Reuse the corrected V2 morphology function. Only fold-legal flop paired-board
    # trips are included, matching the previous local-geometry audit.
    morph=[]
    subsets={name:[] for name in (
        "A_flop_paired_board_trips_facing_action",
        "B_three_seat_topology_one_opponent_folded",
        "C_halfpot_price",
        "D_halfpot_stack_5_10bb",
        "E_near_target_pot_geometry",
        "F_target_dealer_rel",
        "G_target_geometry_Q_kicker",
        "H_Q8_on_884_rank_pattern",
    )}

    # local.build_subsets expects a sample list and performs the same corrected
    # target topology semantics. Scan once and then collect sample identities.
    local_sets=local.build_subsets(items)
    for name in subsets:
        # build_subsets returns feature dicts, not samples. Reconstruct membership
        # deterministically by applying the equivalent predicates below.
        pass

    def feat(sample):
        f=local.features(sample)
        return f if f is not None and f["fold_legal"] else None

    target_status=(0,1,0)
    def top(f):
        return f["topology_live_count"]==3 and f["statuses"]==target_status

    for s in items:
        f=feat(s)
        if f is None:
            continue
        subsets["A_flop_paired_board_trips_facing_action"].append(s)
        if not top(f):
            continue
        subsets["B_three_seat_topology_one_opponent_folded"].append(s)
        if f["to_call_over_pot"] is None or not (0.45<=f["to_call_over_pot"]<=0.55):
            continue
        subsets["C_halfpot_price"].append(s)
        if not (5.0<=f["hero_stack_bb"]<=10.0):
            continue
        subsets["D_halfpot_stack_5_10bb"].append(s)
        near=(3.5<=f["pot_bb"]<=4.5 and 1.75<=f["to_call_bb"]<=2.25 and 1.75<=f["current_bet_bb"]<=2.25)
        if not near:
            continue
        subsets["E_near_target_pot_geometry"].append(s)
        if f["dealer_rel"]!=1:
            continue
        subsets["F_target_dealer_rel"].append(s)
        if f["kicker"]!=12:
            continue
        subsets["G_target_geometry_Q_kicker"].append(s)
        if f["trip_rank"]==8 and f["singleton_board_rank"]==4:
            subsets["H_Q8_on_884_rank_pattern"].append(s)

    counters=dict(d3.get("counters") or {})
    report={
        "schema":"SPINCORE_LT3_10105_AVERAGE_POLICY_FIT_AUDIT_V1",
        "checkpoint_sha256":actual,
        "completed_iteration":10105,
        "scope":"DIAGNOSTIC_ONLY_NO_TRAINING",
        "config":cfg,
        "three_handed_counters":counters,
        "strategy_reservoir":{
            "capacity":int(pol_mem.get("capacity",len(items))),
            "retained_items":len(items),
            "seen":int(pol_mem.get("seen",0)),
        },
        "global_uniform_reservoir_sample":{
            "sample_seed":SAMPLE_SEED,
            "sample_size":n,
            "metrics":_metrics(global_samples,policy,advantage),
        },
        "trip_local_subsets":{
            name:_metrics(samples,policy,advantage)
            for name,samples in subsets.items()
        },
        "interpretation":(
            "Target-vs-AveragePolicy metrics test how well the finalized AveragePolicy fits "
            "its own historical strategy-memory targets. AveragePolicy-vs-current-Advantage "
            "metrics are diagnostic policy drift only; current Advantage is not automatically "
            "a valid replacement for the time-averaged Deep-CFR strategy."
        ),
    }

    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    g=report["global_uniform_reservoir_sample"]["metrics"]
    print("=== LT3 10105 3H AveragePolicy fit audit ===")
    print("config_policy_steps="+str(cfg.get("policy_steps")))
    print("config_batch_size="+str(cfg.get("batch_size")))
    print("counter_policy_optimizer_steps="+str(counters.get("policy_optimizer_steps")))
    print("strategy_seen="+str(report["strategy_reservoir"]["seen"]))
    print("global_fit="+json.dumps(g,sort_keys=True))
    for name,row in report["trip_local_subsets"].items():
        print(name+"="+json.dumps(row,sort_keys=True))
    print("report="+str(args.report.resolve()))
    print("LT3_10105_AVERAGE_POLICY_FIT_AUDIT_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
