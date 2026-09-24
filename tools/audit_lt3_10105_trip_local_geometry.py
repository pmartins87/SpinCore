#!/usr/bin/env python3
from __future__ import annotations

"""Local 10105 coverage audit around the observed Q8o/884 trips fold.

This is a diagnostic only.  It reads the frozen 3H Algorithm-R reservoirs and
measures progressively tighter neighborhoods of the observed benchmark state.
No training state is modified.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from spincore_nn.codec import decode_spnniv1

EXPECTED_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
DOMAIN="THREE_HANDED"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def rank(card:int)->int:
    return 2+int(card)//4


def straight_high(ranks):
    s=set(ranks)
    for hi in range(14,4,-1):
        if all(r in s for r in range(hi-4,hi+1)):
            return hi
    return 5 if {14,5,4,3,2}.issubset(s) else 0


def eval5(cards):
    ranks=[rank(c) for c in cards]
    suits=[int(c)%4 for c in cards]
    counts={r:ranks.count(r) for r in set(ranks)}
    groups=sorted(((n,r) for r,n in counts.items()),reverse=True)
    flush=len(set(suits))==1
    st=straight_high(ranks)
    if flush and st:return 8,(st,)
    if groups[0][0]==4:
        q=groups[0][1];return 7,(q,max(r for r in ranks if r!=q))
    trips=sorted((r for r,n in counts.items() if n==3),reverse=True)
    pairs=sorted((r for r,n in counts.items() if n>=2),reverse=True)
    if trips:
        ps=[r for r in pairs if r!=trips[0]]
        if ps:return 6,(trips[0],ps[0])
    if flush:return 5,tuple(sorted(ranks,reverse=True))
    if st:return 4,(st,)
    if trips:return 3,(trips[0],)
    ep=sorted((r for r,n in counts.items() if n==2),reverse=True)
    if len(ep)>=2:return 2,(ep[0],ep[1])
    if len(ep)==1:return 1,(ep[0],)
    return 0,()


def features(sample):
    d=decode_spnniv1(sample.observation)
    enc=tuple(int(x) for x in d.cards)
    if any(x==0 for x in enc[:2]):
        return None
    hole=tuple(x-1 for x in enc[:2])
    board=tuple(x-1 for x in enc[2:] if x)
    if len(board)!=3:
        return None
    cat,_=eval5(hole+board)
    if cat!=3:
        return None
    br=[rank(c) for c in board]
    hr=[rank(c) for c in hole]
    pair_ranks=[r for r in set(br) if br.count(r)==2]
    if len(pair_ranks)!=1 or pair_ranks[0] not in hr or hr[0]==hr[1]:
        return None
    trip_rank=pair_ranks[0]
    kicker=max(r for r in hr if r!=trip_rank)
    singleton=next(r for r in br if r!=trip_rank)
    n=tuple(float(x) for x in d.numeric)
    c=tuple(int(x) for x in d.categorical)
    pot_bb=n[0]; call_bb=n[1]
    return {
        "trip_rank":trip_rank,
        "kicker":kicker,
        "singleton_board_rank":singleton,
        "pot_bb":pot_bb,
        "to_call_bb":call_bb,
        "to_call_over_pot":(call_bb/pot_bb if pot_bb>0 else None),
        "current_bet_bb":n[2],
        "hero_stack_bb":n[3],
        "dealer_rel":c[2],
        "live_count":c[3],
        "fold_legal":bool(sample.legal[0]),
        "fold_target":float(sample.target[0]) if len(sample.target)>0 else None,
        "iteration":int(sample.iteration),
        "history":tuple(int(x) for x in d.history[:d.history_len]),
    }


def wilson(k,n,z=1.96):
    if n<=0:return 0.0,0.0
    p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return max(0.0,center-half),min(1.0,center+half)


def summarize(rows,reservoir_n,full_seen,*,strategy):
    k=len(rows);lo,hi=wilson(k,reservoir_n)
    out={
        "reservoir_count":k,
        "reservoir_n":reservoir_n,
        "full_seen":full_seen,
        "estimated_full_stream_count":(k/reservoir_n)*full_seen if reservoir_n else 0.0,
        "estimated_full_stream_count_wilson95":[lo*full_seen,hi*full_seen],
    }
    if strategy:
        folds=[r["fold_target"] for r in rows]
        weights=[float(r["iteration"]) for r in rows]
        out.update({
            "fold_target_mean":statistics.fmean(folds) if folds else None,
            "fold_target_median":statistics.median(folds) if folds else None,
            "fold_target_iteration_weighted_mean":(
                sum(p*w for p,w in zip(folds,weights))/sum(weights)
                if folds and sum(weights)>0 else None
            ),
            "fold_target_max":max(folds) if folds else None,
            "fold_target_ge_01_count":sum(p>=0.01 for p in folds),
            "fold_target_ge_05_count":sum(p>=0.05 for p in folds),
            "fold_target_ge_10_count":sum(p>=0.10 for p in folds),
        })
        cohorts={}
        for name,pred in (
            ("LE_8100",lambda it:it<=8100),
            ("8101_9105",lambda it:8101<=it<=9105),
            ("9106_10105",lambda it:9106<=it<=10105),
        ):
            vals=[r["fold_target"] for r in rows if pred(r["iteration"])]
            cohorts[name]={
                "count":len(vals),
                "fold_target_mean":statistics.fmean(vals) if vals else None,
                "fold_target_median":statistics.median(vals) if vals else None,
                "fold_target_ge_05_count":sum(p>=0.05 for p in vals),
                "fold_target_ge_10_count":sum(p>=0.10 for p in vals),
            }
        out["iteration_cohorts"]=cohorts
    return out


def build_subsets(items):
    rows=[f for s in items if (f:=features(s)) is not None and f["fold_legal"]]
    def sel(pred):return [r for r in rows if pred(r)]
    return {
        "A_flop_paired_board_trips_facing_action":rows,
        "B_live_count_2":sel(lambda r:r["live_count"]==2),
        "C_halfpot_price":sel(lambda r:r["live_count"]==2 and r["to_call_over_pot"] is not None and 0.45<=r["to_call_over_pot"]<=0.55),
        "D_halfpot_stack_5_10bb":sel(lambda r:r["live_count"]==2 and r["to_call_over_pot"] is not None and 0.45<=r["to_call_over_pot"]<=0.55 and 5.0<=r["hero_stack_bb"]<=10.0),
        "E_near_target_pot_geometry":sel(lambda r:r["live_count"]==2 and 5.0<=r["hero_stack_bb"]<=10.0 and 3.5<=r["pot_bb"]<=4.5 and 1.75<=r["to_call_bb"]<=2.25 and 1.75<=r["current_bet_bb"]<=2.25),
        "F_target_dealer_rel":sel(lambda r:r["live_count"]==2 and r["dealer_rel"]==1 and 5.0<=r["hero_stack_bb"]<=10.0 and 3.5<=r["pot_bb"]<=4.5 and 1.75<=r["to_call_bb"]<=2.25 and 1.75<=r["current_bet_bb"]<=2.25),
        "G_target_geometry_Q_kicker":sel(lambda r:r["live_count"]==2 and r["dealer_rel"]==1 and r["kicker"]==12 and 5.0<=r["hero_stack_bb"]<=10.0 and 3.5<=r["pot_bb"]<=4.5 and 1.75<=r["to_call_bb"]<=2.25 and 1.75<=r["current_bet_bb"]<=2.25),
        "H_Q8_on_884_rank_pattern":sel(lambda r:r["live_count"]==2 and r["dealer_rel"]==1 and r["trip_rank"]==8 and r["kicker"]==12 and r["singleton_board_rank"]==4 and 5.0<=r["hero_stack_bb"]<=10.0 and 3.5<=r["pot_bb"]<=4.5 and 1.75<=r["to_call_bb"]<=2.25 and 1.75<=r["current_bet_bb"]<=2.25),
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    args=ap.parse_args()
    cp=args.checkpoint.resolve(strict=True)
    actual=sha256(cp)
    if actual!=EXPECTED_SHA:
        raise SystemExit(f"checkpoint SHA mismatch: {actual}")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    if int(payload.get("completed_iteration",-1))!=10105:
        raise SystemExit("expected iteration 10105")
    d=(payload.get("domains") or {}).get(DOMAIN) or {}
    result={
        "schema":"SPINCORE_LT3_10105_TRIP_LOCAL_GEOMETRY_AUDIT_V1",
        "checkpoint_sha256":actual,
        "target_state":{
            "cards":"Qs8d / 8s8c4c",
            "pot_bb":4.0,
            "to_call_bb":2.0,
            "current_bet_bb":2.0,
            "hero_stack_bb":7.133333333333334,
            "dealer_rel":1,
            "live_count":2,
            "observed_current_policy_fold_probability":0.07327636331319809,
        },
        "strategy":{},
        "advantage":{},
        "note":"Counts are decision samples, not unique poker hands. Full-stream counts are Algorithm-R prevalence estimates.",
    }
    for kind,key,strategy in (("strategy","pol_mem",True),("advantage","adv_mem",False)):
        mem=d.get(key) or {}
        items=list(mem.get("items") or [])
        subsets=build_subsets(items)
        result[kind]={
            name:summarize(rows,len(items),int(mem.get("seen",0)),strategy=strategy)
            for name,rows in subsets.items()
        }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("=== LT3 10105 trips local-geometry audit ===")
    for name,row in result["strategy"].items():
        print(name, "count="+str(row["reservoir_count"]),
              "fold_mean="+str(row.get("fold_target_mean")),
              "recent="+json.dumps(row.get("iteration_cohorts",{}).get("9106_10105",{}),sort_keys=True))
    print("report="+str(args.report.resolve()))
    print("LT3_10105_TRIP_LOCAL_GEOMETRY_AUDIT_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
