#!/usr/bin/env python3
from __future__ import annotations

"""Audit historical-target conflict and suit-label fragmentation in 10105 3H strategy memory.

Two keys are studied over the complete retained THREE_HANDED strategy reservoir:
- EXACT_V1: exact SPNNIV1 observation bytes + universal legal mask.
- SUIT_CANONICAL_V1: same state after canonicalizing only physical suit labels
  in the seven V1 card slots, plus the same legal mask.

For a deterministic fixed sample of repeated-state groups, the audit compares
the iteration-weighted empirical conditional target mean against the finalized
AveragePolicy.  It also measures within-group historical target disagreement.
For suit-canonical groups it measures whether the V1 policy itself changes when
only absolute suit labels differ.

This is diagnostic only. It does not train or mutate the checkpoint.
"""

import argparse
import gc
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

from spincore_nn.action_models import make_policy_action_model
from spincore_nn.lean_batch import vectorized_batch

EXPECTED_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
DOMAIN="THREE_HANDED"
SEED=20260925


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def key_bytes(sample,mode:str)->bytes:
    raw=bytearray(sample.observation)
    if len(raw)!=126 or raw[:8]!=b"SPNNIV1\x00":
        raise ValueError("bad SPNNIV1 observation")
    if mode=="SUIT_CANONICAL_V1":
        suit_map={}
        next_suit=0
        for pos in range(8,15):
            token=int(raw[pos])
            if token==0:
                continue
            card_id=token-1
            rank_index=card_id//4
            suit=card_id%4
            if suit not in suit_map:
                suit_map[suit]=next_suit
                next_suit+=1
            raw[pos]=rank_index*4+suit_map[suit]+1
    elif mode!="EXACT_V1":
        raise ValueError(mode)
    return bytes(raw)+bytes(int(x) for x in sample.legal)


def key_hash(sample,mode:str)->int:
    digest=hashlib.blake2b(key_bytes(sample,mode),digest_size=8,person=b"SCV1KEY").digest()
    return int.from_bytes(digest,"little",signed=False)


def entropy(p):
    return -sum(float(x)*math.log(max(float(x),1e-15)) for x in p if float(x)>0.0)


def tv(a,b):
    return 0.5*sum(abs(float(x)-float(y)) for x,y in zip(a,b))


def policy_predictions(model,samples,batch_size=4096):
    out=[]
    model.eval()
    for start in range(0,len(samples),batch_size):
        chunk=samples[start:start+batch_size]
        batch,_,_=vectorized_batch(chunk,"cpu")
        with torch.no_grad():
            logits=model(batch).masked_fill(~batch["legal"],-1e9)
            probs=torch.softmax(logits,dim=-1).detach().cpu().tolist()
        out.extend(tuple(float(x) for x in row) for row in probs)
    return out


def audit_mode(items,model,mode:str,selected_group_cap:int,reps_per_group:int):
    counts={}
    for sample in items:
        h=key_hash(sample,mode)
        counts[h]=counts.get(h,0)+1

    duplicate_hashes=[h for h,c in counts.items() if c>=2]
    repeated_items=sum(counts[h] for h in duplicate_hashes)
    rng=random.Random(SEED ^ (0xE11A if mode=="EXACT_V1" else 0x5A17))
    if len(duplicate_hashes)>selected_group_cap:
        selected=set(rng.sample(duplicate_hashes,selected_group_cap))
    else:
        selected=set(duplicate_hashes)

    aggregates={}
    for h in selected:
        aggregates[h]={
            "count":0,
            "weight":0.0,
            "target_sum":[0.0]*10,
            "argmax_counts":[0]*10,
            "min_iteration":10**9,
            "max_iteration":-1,
            "reps":[],
        }

    for sample in items:
        h=key_hash(sample,mode)
        row=aggregates.get(h)
        if row is None:
            continue
        w=float(sample.weight)
        target=tuple(float(x) for x in sample.target)
        row["count"]+=1
        row["weight"]+=w
        for a,x in enumerate(target):
            row["target_sum"][a]+=w*x
        legal=[a for a,v in enumerate(sample.legal) if bool(v)]
        if legal:
            arg=max(legal,key=lambda a:target[a])
            row["argmax_counts"][arg]+=1
        it=int(sample.iteration)
        row["min_iteration"]=min(row["min_iteration"],it)
        row["max_iteration"]=max(row["max_iteration"],it)
        if len(row["reps"])<reps_per_group:
            row["reps"].append(sample)

    reps=[]
    rep_owner=[]
    means={}
    for h,row in aggregates.items():
        sw=row["weight"]
        if sw<=0:
            continue
        mean=tuple(x/sw for x in row["target_sum"])
        means[h]=mean
        for sample in row["reps"]:
            reps.append(sample)
            rep_owner.append(h)

    preds=policy_predictions(model,reps) if reps else []
    pred_by_group={h:[] for h in means}
    target_conflict=[]
    for sample,h,pred in zip(reps,rep_owner,preds):
        pred_by_group[h].append(pred)
        target_conflict.append(tv(sample.target,means[h]))

    group_fit=[]
    group_argmax_mismatch=[]
    group_target_entropy=[]
    historical_argmax_disagreement=[]
    iteration_spans=[]
    suit_policy_variation=[]

    for h,row in aggregates.items():
        mean=means.get(h)
        pp=pred_by_group.get(h) or []
        if mean is None or not pp:
            continue
        mean_pred=tuple(statistics.fmean(p[a] for p in pp) for a in range(10))
        group_fit.append(tv(mean,mean_pred))
        legal=[a for a,v in enumerate(row["reps"][0].legal) if bool(v)]
        if legal:
            group_argmax_mismatch.append(
                float(max(legal,key=lambda a:mean[a])!=max(legal,key=lambda a:mean_pred[a]))
            )
        group_target_entropy.append(entropy(mean))
        max_arg=max(row["argmax_counts"]) if row["argmax_counts"] else 0
        historical_argmax_disagreement.append(
            1.0-(max_arg/row["count"] if row["count"] else 0.0)
        )
        iteration_spans.append(row["max_iteration"]-row["min_iteration"])
        if len(pp)>=2:
            avg=mean_pred
            suit_policy_variation.extend(tv(p,avg) for p in pp)

    hist={
        "1":0,"2":0,"3_4":0,"5_9":0,"10_plus":0
    }
    max_mult=0
    for c in counts.values():
        max_mult=max(max_mult,c)
        if c==1: hist["1"]+=1
        elif c==2: hist["2"]+=1
        elif c<=4: hist["3_4"]+=1
        elif c<=9: hist["5_9"]+=1
        else: hist["10_plus"]+=1

    result={
        "mode":mode,
        "retained_items":len(items),
        "unique_hashed_keys":len(counts),
        "duplicate_groups":len(duplicate_hashes),
        "repeated_items":repeated_items,
        "repeated_item_fraction":repeated_items/len(items) if items else 0.0,
        "max_multiplicity":max_mult,
        "multiplicity_group_histogram":hist,
        "selected_duplicate_groups":len(means),
        "representatives_evaluated":len(reps),
        "within_group_sample_target_to_conditional_mean_tv_mean":(
            statistics.fmean(target_conflict) if target_conflict else None
        ),
        "conditional_mean_target_vs_average_policy_tv_mean":(
            statistics.fmean(group_fit) if group_fit else None
        ),
        "conditional_mean_target_vs_average_policy_argmax_mismatch_rate":(
            statistics.fmean(group_argmax_mismatch) if group_argmax_mismatch else None
        ),
        "conditional_mean_target_entropy_mean_nats":(
            statistics.fmean(group_target_entropy) if group_target_entropy else None
        ),
        "historical_sample_argmax_disagreement_within_group_mean":(
            statistics.fmean(historical_argmax_disagreement)
            if historical_argmax_disagreement else None
        ),
        "iteration_span_mean":statistics.fmean(iteration_spans) if iteration_spans else None,
        "iteration_span_median":statistics.median(iteration_spans) if iteration_spans else None,
        "policy_prediction_tv_to_group_prediction_mean":(
            statistics.fmean(suit_policy_variation) if suit_policy_variation else 0.0
        ),
        "hash_note":"64-bit BLAKE2b keys; collision probability is negligible at this scale but not mathematically zero.",
    }
    del counts,duplicate_hashes,aggregates,means,pred_by_group,reps,preds
    gc.collect()
    return result


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--selected-groups",type=int,default=20000)
    ap.add_argument("--reps-per-group",type=int,default=4)
    args=ap.parse_args()

    cp=args.checkpoint.resolve(strict=True)
    actual=sha256(cp)
    if actual!=EXPECTED_SHA:
        raise SystemExit(f"checkpoint SHA mismatch: {actual}")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    if int(payload.get("completed_iteration",-1))!=10105:
        raise SystemExit("expected iteration 10105")
    d3=(payload.get("domains") or {}).get(DOMAIN) or {}
    mem=d3.get("pol_mem") or {}
    items=list(mem.get("items") or [])
    if not items:
        raise SystemExit("empty strategy reservoir")

    _,model=make_policy_action_model(REPRESENTATION,device="cpu",seed=0)
    model.load_state_dict(d3["policy"])

    results=[]
    for mode in ("EXACT_V1","SUIT_CANONICAL_V1"):
        print(f"AUDIT_MODE_START {mode}",flush=True)
        row=audit_mode(
            items,model,mode,
            selected_group_cap=int(args.selected_groups),
            reps_per_group=int(args.reps_per_group),
        )
        results.append(row)
        print("AUDIT_MODE_RESULT "+json.dumps(row,sort_keys=True),flush=True)

    report={
        "schema":"SPINCORE_LT3_10105_STRATEGY_TARGET_CONFLICT_AUDIT_V1",
        "checkpoint_sha256":actual,
        "completed_iteration":10105,
        "scope":"DIAGNOSTIC_ONLY_NO_TRAINING",
        "strategy_reservoir_retained":len(items),
        "strategy_reservoir_seen":int(mem.get("seen",0)),
        "selected_group_cap":int(args.selected_groups),
        "representatives_per_group_cap":int(args.reps_per_group),
        "modes":results,
        "interpretation":(
            "EXACT_V1 quantifies historical target conflict for genuinely identical frozen V1 observations. "
            "If sample-target-to-conditional-mean TV is large while conditional-mean-to-AveragePolicy TV is "
            "small, much of the earlier per-sample fit error is irreducible historical target conflict rather "
            "than optimizer underfit. SUIT_CANONICAL_V1 additionally tests whether splitting equivalent states "
            "by absolute suit labels fragments coverage or causes policy inconsistency. This audit cannot by "
            "itself distinguish temporal nonstationarity from hidden history information already discarded by V1."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"report={args.report.resolve()}")
    print("LT3_10105_STRATEGY_TARGET_CONFLICT_AUDIT_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
