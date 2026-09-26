#!/usr/bin/env python3
from __future__ import annotations

"""Diagnose 3H ALL_IN subgroup bias with a lightweight semantic sidecar.

The frozen 10105 3H Advantage reservoir and the already-built controlled-split
1600-step eight-model probe are reused.  A deterministic 250k calibration
subset is drawn from the 1.95M training side of the existing split.  Three
weighted ridge calibrators are fit to the stored ALL_IN target:

  A) RAW_AFFINE       : ensemble raw ALL_IN only;
  B) CONTEXT          : raw + street/facing/pot geometry;
  C) SEMANTIC_SIDECAR : CONTEXT + hand/board semantic features derived only
                        from the existing SPNNIV1 card tokens.

All three are evaluated on the original untouched 50k holdout.  This is a
diagnostic attribution experiment, not a deployable strategy patch.
"""

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import struct
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))

from spincore_nn.action_models import make_advantage_action_model
from spincore_nn.lean_batch import vectorized_batch

EXPECTED_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
PROBE_SCHEMA="SPINCORE_3H_ADVANTAGE_CONTROLLED_SPLIT_PROBE_V1"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
DOMAIN="THREE_HANDED"
BUDGET=1600
ALL_IN=9
CAL_SEED=20260926 ^ 0x5EAA17C
DEFAULT_CAL=250000

STRAIGHTS=(
    {14,2,3,4,5},
    {2,3,4,5,6},
    {3,4,5,6,7},
    {4,5,6,7,8},
    {5,6,7,8,9},
    {6,7,8,9,10},
    {7,8,9,10,11},
    {8,9,10,11,12},
    {9,10,11,12,13},
    {10,11,12,13,14},
)

def rank(card:int)->int:
    return 2+int(card)//4

def suit(card:int)->int:
    return int(card)%4

def decode_obs(obs:bytes):
    if len(obs)!=126 or obs[:8]!=b"SPNNIV1\x00":
        raise ValueError("bad SPNNIV1 observation")
    cards=tuple(int(x) for x in obs[8:15])
    numeric=tuple(float(x) for x in struct.unpack_from("<16f",obs,15))
    cat=tuple(int(x) for x in obs[79:87])
    hole=tuple(x-1 for x in cards[:2] if x)
    board=tuple(x-1 for x in cards[2:] if x)
    return hole,board,numeric,cat

def has_straight(ranks)->bool:
    r=set(ranks)
    return any(t.issubset(r) for t in STRAIGHTS)

def made_category(hole,board)->int:
    cards=tuple(hole)+tuple(board)
    ranks=[rank(c) for c in cards]
    suits=[suit(c) for c in cards]
    rc={r:ranks.count(r) for r in set(ranks)}
    sc={s:suits.count(s) for s in set(suits)}
    # 8 straight flush, 7 quads, 6 full house, 5 flush, 4 straight,
    # 3 trips, 2 two pair, 1 pair, 0 high card.
    for s,n in sc.items():
        if n>=5 and has_straight([rank(c) for c in cards if suit(c)==s]):
            return 8
    counts=sorted(rc.values(),reverse=True)
    if counts and counts[0]>=4:return 7
    triples=sum(v>=3 for v in rc.values())
    pairs=sum(v>=2 for v in rc.values())
    if triples>=1 and (pairs>=2 or triples>=2):return 6
    if any(n>=5 for n in sc.values()):return 5
    if has_straight(ranks):return 4
    if counts and counts[0]>=3:return 3
    if sum(v>=2 for v in rc.values())>=2:return 2
    if sum(v>=2 for v in rc.values())>=1:return 1
    return 0

def private_semantics(hole,board):
    if len(hole)!=2:
        return {
            "made":0,"pair_relation":0,"pocket_pair":0,"overcards":0,
            "flush_draw":0,"straight_draw":0,"backdoor_flush":0,
            "backdoor_straight":0,"high_card_no_draw":0,
        }
    visible=len(board)
    made=made_category(hole,board) if visible>=3 else 0
    board_ranks=[rank(c) for c in board]
    bcounts={r:board_ranks.count(r) for r in set(board_ranks)}
    board_high=max(board_ranks) if board_ranks else 0
    h0,h1=rank(hole[0]),rank(hole[1])
    pocket=int(h0==h1)
    m0=int(bcounts.get(h0,0)>0)
    m1=int(bcounts.get(h1,0)>0)
    over=int(h0>board_high)+int(h1>board_high) if board_high else 0

    # PairRelationV2-compatible ids.
    pair_rel=0
    distinct=sorted(set(board_ranks),reverse=True)
    board_paired=any(v>=2 for v in bcounts.values())
    if pocket:
        if m0:pair_rel=3
        elif h0>board_high:pair_rel=1
        else:pair_rel=2
    elif m0 and m1:
        pair_rel=7
    elif m0 or m1:
        mr=h0 if m0 else h1
        pos=distinct.index(mr)
        pair_rel=4 if pos==0 else (5 if pos==1 else 6)
    elif board_paired:
        pair_rel=8

    cards=tuple(hole)+tuple(board)
    suits=[suit(c) for c in cards]
    combined_suits={s:suits.count(s) for s in set(suits)}
    flush_draw=0
    if visible<5:
        for s,n in combined_suits.items():
            if n==4 and (suit(hole[0])==s or suit(hole[1])==s):
                flush_draw=1;break
    backdoor_flush=0
    if visible==3 and not flush_draw:
        for s,n in combined_suits.items():
            if n==3 and (suit(hole[0])==s or suit(hole[1])==s):
                backdoor_flush=1;break

    ranks=set(rank(c) for c in cards)
    straight_draw=0
    missing_count=0
    if visible<5 and not has_straight(ranks):
        missing=set()
        board_set=set(board_ranks)
        for window in STRAIGHTS:
            if len(window & ranks)==4:
                hero_adds=any(rank(c) in window and rank(c) not in board_set for c in hole)
                if hero_adds:
                    missing.update(window-ranks)
        missing_count=len(missing)
        straight_draw=int(missing_count>0)
    backdoor_straight=0
    if visible==3 and not has_straight(ranks) and not straight_draw:
        board_set=set(board_ranks)
        for window in STRAIGHTS:
            if len(window & ranks)==3 and any(
                rank(c) in window and rank(c) not in board_set for c in hole
            ):
                backdoor_straight=1;break

    # Match audit_3h_allin_holdout_calibration_10105.py exactly.
    # On flop/turn, reject one-card-to-flush / one-card-to-straight states.
    # On river there is no future card, so a four-suit or four-rank texture is
    # still "no immediate draw" as long as the made category remains HIGH_CARD.
    max_combined_suit=max(combined_suits.values()) if combined_suits else 0
    immediate_straight_draw=any(len(window & ranks)>=4 for window in STRAIGHTS)
    if visible>=5:
        no_immediate_draw=True
    else:
        no_immediate_draw=(max_combined_suit<4 and not immediate_straight_draw)
    hcdn=int(visible>=3 and made==0 and no_immediate_draw)
    return {
        "made":made,"pair_relation":pair_rel,"pocket_pair":pocket,
        "overcards":over,"flush_draw":flush_draw,"straight_draw":straight_draw,
        "backdoor_flush":backdoor_flush,"backdoor_straight":backdoor_straight,
        "high_card_no_draw":hcdn,
    }

def board_semantics(board):
    if not board:
        return {"paired":0,"max_suit":0.0,"straight_occ":0.0,"broadway":0.0}
    ranks=[rank(c) for c in board]
    suits=[suit(c) for c in board]
    paired=int(len(set(ranks))<len(ranks))
    max_s=max(suits.count(s) for s in set(suits))
    rset=set(ranks)
    occ=max(len(w & rset) for w in STRAIGHTS)
    broad=sum(r>=10 for r in ranks)
    return {
        "paired":paired,
        "max_suit":max_s/5.0,
        "straight_occ":occ/5.0,
        "broadway":broad/5.0,
    }

def safe_ratio(a,b):
    return float(a)/float(b) if abs(float(b))>1e-9 else 0.0

def feature_sets(sample,pred):
    hole,board,n,c=decode_obs(sample.observation)
    street=int(c[1])
    facing=float(n[1]>1e-9)
    pot=max(float(n[0]),1e-6)
    ps=private_semantics(hole,board)
    bs=board_semantics(board)

    raw=[
        1.0,float(pred),float(pred)*float(pred),
    ]
    context=raw + [
        float(street==0),float(street==1),float(street==2),float(street==3),
        facing,
        max(-5.0,min(5.0,safe_ratio(n[1],pot))),
        max(-5.0,min(5.0,safe_ratio(n[2],pot))),
        max(0.0,min(20.0,safe_ratio(n[3],pot)))/20.0,
        sum(bool(x) for x in sample.legal)/10.0,
    ]
    sem=context.copy()
    sem += [float(ps["made"]==k) for k in range(9)]
    sem += [float(ps["pair_relation"]==k) for k in range(9)]
    sem += [float(ps["overcards"]==k) for k in range(3)]
    sem += [
        float(ps["pocket_pair"]),
        float(ps["flush_draw"]),
        float(ps["straight_draw"]),
        float(ps["backdoor_flush"]),
        float(ps["backdoor_straight"]),
        float(ps["high_card_no_draw"]),
        float(bs["paired"]),
        float(bs["max_suit"]),
        float(bs["straight_occ"]),
        float(bs["broadway"]),
        float(pred)*float(ps["high_card_no_draw"]),
        float(pred)*float(street==3),
        float(ps["high_card_no_draw"])*float(street==1),
        float(ps["high_card_no_draw"])*float(street==2),
        float(ps["high_card_no_draw"])*float(street==3),
        float(ps["high_card_no_draw"])*facing,
    ]
    return np.asarray(raw,dtype=np.float64),np.asarray(context,dtype=np.float64),np.asarray(sem,dtype=np.float64),ps

def load_model(state):
    _,m=make_advantage_action_model(REPRESENTATION,device="cpu",seed=0)
    m.load_state_dict(state);m.eval()
    return m

def ridge_fit(rows,dim,lam):
    xtx=np.zeros((dim,dim),dtype=np.float64)
    xty=np.zeros(dim,dtype=np.float64)
    for x,y,w in rows:
        ww=float(w)
        xtx += ww*np.outer(x,x)
        xty += ww*x*float(y)
    scale=max(np.trace(xtx)/max(dim,1),1.0)
    reg=float(lam)*scale
    eye=np.eye(dim,dtype=np.float64)
    eye[0,0]=0.0
    return np.linalg.solve(xtx+reg*eye,xty)

class Metrics:
    def __init__(self):
        self.n=0;self.w=0.0;self.err=0.0;self.bias=0.0;self.pred=0.0;self.targ=0.0
    def add(self,p,t,w):
        p=float(p);t=float(t);w=float(w)
        self.n+=1;self.w+=w
        e=p-t
        self.err+=w*e*e
        self.bias+=w*e
        self.pred+=w*p
        self.targ+=w*t
    def out(self):
        return {
            "count":self.n,
            "weight_sum":self.w,
            "prediction_weighted_mean":self.pred/self.w if self.w else None,
            "target_weighted_mean":self.targ/self.w if self.w else None,
            "weighted_bias_prediction_minus_target":self.bias/self.w if self.w else None,
            "weighted_mse":self.err/self.w if self.w else None,
            "weighted_rmse":math.sqrt(self.err/self.w) if self.w else None,
        }

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--calibration-size",type=int,default=DEFAULT_CAL)
    ap.add_argument("--batch-size",type=int,default=4096)
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--ridge",type=float,default=1e-8)
    args=ap.parse_args()
    torch.set_num_threads(int(args.threads))

    import hashlib
    cp=args.checkpoint.resolve(strict=True)
    h=hashlib.sha256()
    with cp.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    if h.hexdigest()!=EXPECTED_SHA:raise SystemExit("checkpoint SHA mismatch")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    mem=((payload.get("domains") or {}).get(DOMAIN) or {}).get("adv_mem") or {}
    items=list(mem.get("items") or [])
    if len(items)!=2_000_000:raise SystemExit("expected saturated 2M reservoir")

    probe=torch.load(args.probe.resolve(strict=True),map_location="cpu",weights_only=False)
    if probe.get("schema")!=PROBE_SCHEMA:raise SystemExit("wrong probe schema")
    states=list(probe["snapshots"][str(BUDGET)])
    if len(states)!=8:raise SystemExit("expected eight 1600 snapshots")
    models=[load_model(s) for s in states]

    holdout_n=int(probe["holdout_size"]);split_seed=int(probe["holdout_seed"])
    split_rng=random.Random(split_seed)
    holdout_idx=set(split_rng.sample(range(len(items)),holdout_n))
    train_idx=[i for i in range(len(items)) if i not in holdout_idx]
    cal_n=min(int(args.calibration_size),len(train_idx))
    cal_rng=random.Random(CAL_SEED)
    cal_idx=cal_rng.sample(train_idx,cal_n)
    holdout=[items[i] for i in sorted(holdout_idx)]
    calibration=[items[i] for i in cal_idx]

    def predictions(samples):
        out=[]
        bs=int(args.batch_size)
        with torch.no_grad():
            for start in range(0,len(samples),bs):
                chunk=samples[start:start+bs]
                batch,target,weights=vectorized_batch(chunk,"cpu")
                raw=torch.stack([m(batch).float() for m in models],dim=0).mean(dim=0)
                for j,s in enumerate(chunk):
                    if not bool(s.legal[ALL_IN]):continue
                    p=float(raw[j,ALL_IN].cpu())
                    y=float(target[j,ALL_IN].cpu())
                    w=float(weights[j].cpu())
                    xr,xc,xs,ps=feature_sets(s,p)
                    out.append((xr,xc,xs,y,w,ps))
        return out

    cal_rows=predictions(calibration)
    if len(cal_rows)<1000:raise RuntimeError("too few ALL_IN-legal calibration rows")
    dims=[len(cal_rows[0][0]),len(cal_rows[0][1]),len(cal_rows[0][2])]
    betas=[
        ridge_fit(((r[i],r[3],r[4]) for r in cal_rows),dims[i],args.ridge)
        for i in range(3)
    ]
    cal_allin_legal_rows=len(cal_rows)
    del cal_rows

    hold_rows=predictions(holdout)
    names=("RAW_AFFINE","CONTEXT","SEMANTIC_SIDECAR")
    metrics={name:Metrics() for name in ("BASE_RAW",)+names}
    hc={name:Metrics() for name in ("BASE_RAW",)+names}
    hc_ctx={}

    for xr,xc,xs,y,w,ps in hold_rows:
        base=float(xr[1])
        preds=[float(xr@betas[0]),float(xc@betas[1]),float(xs@betas[2])]
        metrics["BASE_RAW"].add(base,y,w)
        for name,p in zip(names,preds):metrics[name].add(p,y,w)
        if ps["high_card_no_draw"]:
            hc["BASE_RAW"].add(base,y,w)
            for name,p in zip(names,preds):hc[name].add(p,y,w)

    mout={k:v.out() for k,v in metrics.items()}
    hout={k:v.out() for k,v in hc.items()}
    if int(hout["BASE_RAW"]["count"]) != 6639:
        raise RuntimeError(
            "high-card/no-draw population drift: expected 6639 rows from the frozen "
            "untouched-holdout calibration contract, got "
            + str(hout["BASE_RAW"]["count"])
        )

    base_bias=abs(hout["BASE_RAW"]["weighted_bias_prediction_minus_target"])
    sem_bias=abs(hout["SEMANTIC_SIDECAR"]["weighted_bias_prediction_minus_target"])
    base_mse=mout["BASE_RAW"]["weighted_mse"]
    sem_global=mout["SEMANTIC_SIDECAR"]["weighted_mse"]
    hc_base_mse=hout["BASE_RAW"]["weighted_mse"]
    hc_sem_mse=hout["SEMANTIC_SIDECAR"]["weighted_mse"]

    criteria={
        "semantic_reduces_abs_high_card_bias_by_at_least_50pct":bool(
            base_bias>0 and sem_bias<=0.5*base_bias
        ),
        "semantic_global_mse_not_worse_than_base_by_over_2pct":bool(
            sem_global<=1.02*base_mse
        ),
        "semantic_high_card_mse_not_worse_than_base":bool(
            hc_sem_mse<=hc_base_mse
        ),
    }
    passed=all(criteria.values())

    out={
        "schema":"SPINCORE_3H_SEMANTIC_SIDECAR_ATTRIBUTION_V2",
        "scope":"DIAGNOSTIC_ONLY_POSTHOC_LINEAR_CALIBRATION",
        "checkpoint_sha256":EXPECTED_SHA,
        "probe_schema":PROBE_SCHEMA,
        "budget":BUDGET,
        "replicas":8,
        "holdout_size":holdout_n,
        "calibration_seed":CAL_SEED,
        "calibration_requested":int(args.calibration_size),
        "calibration_allin_legal_rows":cal_allin_legal_rows,
        "feature_dimensions":{
            "RAW_AFFINE":dims[0],"CONTEXT":dims[1],"SEMANTIC_SIDECAR":dims[2]
        },
        "ridge":float(args.ridge),
        "all_allin_legal":mout,
        "postflop_high_card_no_draw":hout,
        "coefficients":{
            "RAW_AFFINE":betas[0].tolist(),
            "CONTEXT":betas[1].tolist(),
            "SEMANTIC_SIDECAR":betas[2].tolist(),
        },
        "precommitted_diagnostic_criteria":criteria,
        "semantic_sidecar_attribution_pass":passed,
        "population_contract":(
            "High-card/no-immediate-draw population matches "
            "audit_3h_allin_holdout_calibration_10105.py: made HIGH_CARD; on flop/turn "
            "exclude four-suit and four-of-five straight-window states; on river do not "
            "exclude four-suit/four-rank textures because no future draw remains."
        ),
        "interpretation":(
            "V2 fixes V1's river population mismatch. This gate asks whether state/hand semantics explain the high-card/no-draw "
            "ALL_IN bias on an untouched holdout beyond a raw affine or generic context "
            "calibration. PASS supports a targeted V1+semantic representation experiment; "
            "FAIL means explicit semantics in this lightweight form are insufficient and "
            "the next diagnosis should target objective/target noise or richer sequence "
            "representation. These calibrators are not deployment policies."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("=== 3H semantic sidecar attribution ===")
    print("global="+json.dumps(mout,sort_keys=True))
    print("high_card_no_draw="+json.dumps(hout,sort_keys=True))
    print("criteria="+json.dumps(criteria,sort_keys=True))
    print(f"semantic_sidecar_attribution_pass={passed}")
    print(f"report={args.report.resolve()}")
    print("3H_SEMANTIC_SIDECAR_ATTRIBUTION_PASS")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
