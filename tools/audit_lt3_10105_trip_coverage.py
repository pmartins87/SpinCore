#!/usr/bin/env python3
from __future__ import annotations

"""Estimate 10105 training coverage for trips from the frozen reservoirs.

The checkpoint keeps uniform Algorithm-R reservoirs.  The reservoir contents are
therefore an unbiased sample of the full stream of strategy/advantage samples.
This audit does not pretend to recover the exact historical count of trips; it
reports the observed reservoir count plus a binomial/Wilson estimate of the
corresponding count in the full seen stream.

SPNNIV1 card encoding is 0=padding and 1..52=card_id+1.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Iterable

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))

from spincore_nn.codec import decode_spnniv1

EXPECTED_SCHEMA="SPINCORE_LEAN_FUNCTIONAL_TRAINING_V1"
EXPECTED_REPRESENTATION="C0_V1_FROZEN_CONTROL"
EXPECTED_ACTION_CANDIDATE="LEGACY_7_ACTION_BASELINE_V1"
DOMAIN="THREE_HANDED"
CATEGORY_NAMES={
    0:"HIGH_CARD",1:"PAIR",2:"TWO_PAIR",3:"TRIPS",4:"STRAIGHT",
    5:"FLUSH",6:"FULL_HOUSE",7:"QUADS",8:"STRAIGHT_FLUSH",
}


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def card_rank(card_id:int)->int:
    return 2+int(card_id)//4


def card_suit(card_id:int)->int:
    return int(card_id)%4


def straight_high(ranks:Iterable[int])->int:
    values=set(int(x) for x in ranks)
    for hi in range(14,4,-1):
        if all(r in values for r in range(hi-4,hi+1)):
            return hi
    return 5 if {14,5,4,3,2}.issubset(values) else 0


def evaluate_five(cards:tuple[int,...])->tuple[int,tuple[int,...]]:
    ranks=[card_rank(c) for c in cards]
    suits=[card_suit(c) for c in cards]
    counts={r:ranks.count(r) for r in set(ranks)}
    groups=sorted(((n,r) for r,n in counts.items()),reverse=True)
    flush=len(set(suits))==1
    straight=straight_high(ranks)
    if flush and straight:
        return 8,(straight,)
    if groups[0][0]==4:
        quad=groups[0][1]
        return 7,(quad,max(r for r in ranks if r!=quad))
    trips=sorted((r for r,n in counts.items() if n==3),reverse=True)
    pairs=sorted((r for r,n in counts.items() if n>=2),reverse=True)
    if trips:
        pair_candidates=[r for r in pairs if r!=trips[0]]
        if pair_candidates:
            return 6,(trips[0],pair_candidates[0])
    if flush:
        return 5,tuple(sorted(ranks,reverse=True))
    if straight:
        return 4,(straight,)
    if trips:
        kickers=sorted((r for r in ranks if r!=trips[0]),reverse=True)[:2]
        return 3,(trips[0],*kickers)
    exact_pairs=sorted((r for r,n in counts.items() if n==2),reverse=True)
    if len(exact_pairs)>=2:
        p1,p2=exact_pairs[:2]
        kicker=max(r for r in ranks if r not in (p1,p2))
        return 2,(p1,p2,kicker)
    if len(exact_pairs)==1:
        p=exact_pairs[0]
        kickers=sorted((r for r in ranks if r!=p),reverse=True)[:3]
        return 1,(p,*kickers)
    return 0,tuple(sorted(ranks,reverse=True))


def evaluate_visible(cards:tuple[int,...])->tuple[int,tuple[int,...]]:
    from itertools import combinations
    return max(evaluate_five(tuple(c)) for c in combinations(cards,5))


def decode_cards(observation:bytes):
    d=decode_spnniv1(observation)
    encoded=tuple(int(x) for x in d.cards)
    if any(x<0 or x>52 for x in encoded):
        raise ValueError("SPNNIV1 card encoding outside 0..52")
    hole_enc=encoded[:2]
    board_enc=encoded[2:]
    if any(x==0 for x in hole_enc):
        return None
    hole=tuple(x-1 for x in hole_enc)
    board=tuple(x-1 for x in board_enc if x)
    if len(board) not in (0,3,4,5):
        raise ValueError(f"unexpected visible board-card count: {len(board)}")
    return hole,board


def classify(observation:bytes):
    decoded=decode_cards(observation)
    if decoded is None:
        return None
    hole,board=decoded
    if not board:
        return {"street":"PREFLOP","category":None,"paired_board_trips_one_hole":False}
    cat,_=evaluate_visible(tuple(hole)+tuple(board))
    street={3:"FLOP",4:"TURN",5:"RIVER"}[len(board)]
    board_ranks=[card_rank(c) for c in board]
    hole_ranks=[card_rank(c) for c in hole]
    pair_ranks=[r for r in set(board_ranks) if board_ranks.count(r)==2]
    paired_board_trips_one_hole=bool(
        cat==3
        and len(pair_ranks)==1
        and pair_ranks[0] in hole_ranks
        and hole_ranks[0]!=hole_ranks[1]
    )
    return {
        "street":street,
        "category":CATEGORY_NAMES[cat],
        "paired_board_trips_one_hole":paired_board_trips_one_hole,
    }


def wilson(k:int,n:int,z:float=1.96):
    if n<=0:
        return 0.0,0.0
    p=k/n
    den=1.0+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return max(0.0,center-half),min(1.0,center+half)


def distribution_summary(items,seen:int,*,strategy:bool):
    categories=Counter()
    streets=Counter()
    trip_rows=[]
    special_rows=[]
    invalid=0
    for sample in items:
        try:
            info=classify(sample.observation)
        except Exception:
            invalid+=1
            continue
        if info is None:
            invalid+=1
            continue
        streets[info["street"]]+=1
        if info["category"] is not None:
            categories[info["category"]]+=1
        if info["category"]=="TRIPS":
            trip_rows.append(sample)
            if info["paired_board_trips_one_hole"]:
                special_rows.append(sample)

    n=len(items)
    def prevalence(rows):
        k=len(rows)
        lo,hi=wilson(k,n)
        p=(k/n) if n else 0.0
        return {
            "reservoir_count":k,
            "reservoir_n":n,
            "reservoir_fraction":p,
            "wilson95_fraction":[lo,hi],
            "full_seen":int(seen),
            "estimated_full_stream_count":p*seen,
            "estimated_full_stream_count_wilson95":[lo*seen,hi*seen],
        }

    out={
        "reservoir_items":n,
        "seen":int(seen),
        "invalid_or_unclassified":invalid,
        "street_counts":dict(sorted(streets.items())),
        "made_hand_counts":dict(sorted(categories.items())),
        "trips":prevalence(trip_rows),
        "paired_board_trips_one_hole":prevalence(special_rows),
    }

    if strategy:
        def target_stats(rows):
            facing=[s for s in rows if len(s.legal)>0 and bool(s.legal[0])]
            folds=[float(s.target[0]) for s in facing]
            weights=[float(s.weight) for s in facing]
            weighted=(
                sum(p*w for p,w in zip(folds,weights))/sum(weights)
                if folds and sum(weights)>0 else None
            )
            return {
                "fold_legal_count":len(facing),
                "fold_target_mean":statistics.fmean(folds) if folds else None,
                "fold_target_iteration_weighted_mean":weighted,
                "fold_target_median":statistics.median(folds) if folds else None,
                "fold_target_max":max(folds) if folds else None,
                "fold_target_gt_01_count":sum(p>0.01 for p in folds),
                "fold_target_ge_05_count":sum(p>=0.05 for p in folds),
                "fold_target_ge_10_count":sum(p>=0.10 for p in folds),
            }
        out["trips_strategy_targets"]=target_stats(trip_rows)
        out["paired_board_trips_one_hole_strategy_targets"]=target_stats(special_rows)
    return out


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    args=ap.parse_args()

    cp=args.checkpoint.resolve(strict=True)
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    if payload.get("schema")!=EXPECTED_SCHEMA:
        raise SystemExit("wrong checkpoint schema")
    if payload.get("representation")!=EXPECTED_REPRESENTATION:
        raise SystemExit("wrong checkpoint representation")
    if payload.get("action_candidate")!=EXPECTED_ACTION_CANDIDATE:
        raise SystemExit("wrong checkpoint action candidate")
    if int(payload.get("completed_iteration",-1))!=10105:
        raise SystemExit("coverage audit is pinned to iteration 10105")

    domain=(payload.get("domains") or {}).get(DOMAIN) or {}
    pol=domain.get("pol_mem") or {}
    adv=domain.get("adv_mem") or {}
    report={
        "schema":"SPINCORE_LT3_10105_TRIP_COVERAGE_AUDIT_V1",
        "checkpoint":str(cp),
        "checkpoint_sha256":sha256(cp),
        "completed_iteration":10105,
        "domain":DOMAIN,
        "interpretation":(
            "Algorithm-R reservoirs are uniform samples of their complete seen streams. "
            "Full-stream category counts are estimates from reservoir prevalence, not exact counts."
        ),
        "strategy_reservoir":distribution_summary(
            list(pol.get("items") or []),int(pol.get("seen",0)),strategy=True
        ),
        "advantage_reservoir":distribution_summary(
            list(adv.get("items") or []),int(adv.get("seen",0)),strategy=False
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    s=report["strategy_reservoir"]
    a=report["advantage_reservoir"]
    print("=== LT3 10105 trips coverage audit ===")
    print(f"checkpoint_sha256={report['checkpoint_sha256']}")
    print(
        "strategy reservoir: "
        f"items={s['reservoir_items']} seen={s['seen']} "
        f"trips={s['trips']['reservoir_count']} "
        f"paired_board_trips_one_hole={s['paired_board_trips_one_hole']['reservoir_count']}"
    )
    print(
        "strategy estimated full-stream trips="
        f"{s['trips']['estimated_full_stream_count']:.1f} "
        f"CI95=[{s['trips']['estimated_full_stream_count_wilson95'][0]:.1f},"
        f"{s['trips']['estimated_full_stream_count_wilson95'][1]:.1f}]"
    )
    print(
        "strategy paired-board trips fold target: "
        +json.dumps(s["paired_board_trips_one_hole_strategy_targets"],sort_keys=True)
    )
    print(
        "advantage reservoir: "
        f"items={a['reservoir_items']} seen={a['seen']} "
        f"trips={a['trips']['reservoir_count']} "
        f"paired_board_trips_one_hole={a['paired_board_trips_one_hole']['reservoir_count']}"
    )
    print(
        "advantage estimated full-stream trips="
        f"{a['trips']['estimated_full_stream_count']:.1f} "
        f"CI95=[{a['trips']['estimated_full_stream_count_wilson95'][0]:.1f},"
        f"{a['trips']['estimated_full_stream_count_wilson95'][1]:.1f}]"
    )
    print(f"report={args.report.resolve()}")
    print("LT3_10105_TRIP_COVERAGE_AUDIT_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
