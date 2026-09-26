#!/usr/bin/env python3
from __future__ import annotations

"""Calibrate the 1600-step 3H ALL_IN Advantage head on its untouched holdout.

The same fixed 50k holdout excluded from every controlled-split replica is
reconstructed from the frozen 10105 reservoir.  The eight 1600-step snapshots
are evaluated without training.

The report focuses on whether small positive raw ALL_IN predictions in the
+0.003..+0.020 anomaly range correspond to positive or negative stored targets
on genuinely unseen data, globally and inside postflop HIGH_CARD /
no-immediate-draw states.

No CFR roots, optimizer steps, checkpoint mutation or strategy patch occurs.
"""

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import struct
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))

from spincore_nn.action_models import make_advantage_action_model
from spincore_nn.lean_batch import vectorized_batch

EXPECTED_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
PROBE_SCHEMA="SPINCORE_3H_ADVANTAGE_CONTROLLED_SPLIT_PROBE_V1"
GAP_SCHEMA="SPINCORE_3H_HIGH_CARD_MODEL_TARGET_GAP_AUDIT_V1"
SCHEMA="SPINCORE_3H_ALLIN_HOLDOUT_CALIBRATION_V1"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
DOMAIN="THREE_HANDED"
BUDGET=1600
ALL_IN=9
BIN_EDGES=(-float("inf"),0.0,0.0025,0.005,0.010,0.020,float("inf"))
BIN_NAMES=(
    "NEGATIVE",
    "ZERO_TO_0P0025",
    "0P0025_TO_0P005",
    "0P005_TO_0P010",
    "0P010_TO_0P020",
    "GE_0P020",
)


def rank(card:int)->int:
    return 2+int(card)//4


def suit(card:int)->int:
    return int(card)%4


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


def decode_obs(obs:bytes):
    if len(obs)!=126 or obs[:8]!=b"SPNNIV1\x00":
        raise ValueError("bad SPNNIV1 observation")
    cards=tuple(int(x) for x in obs[8:15])
    numeric=struct.unpack_from("<16f",obs,15)
    cat=tuple(int(x) for x in obs[79:87])
    hole=tuple(x-1 for x in cards[:2] if x)
    board=tuple(x-1 for x in cards[2:] if x)
    return hole,board,tuple(float(x) for x in numeric),cat


def high_card_no_immediate_draw(hole,board):
    if len(hole)!=2 or len(board)<3:
        return False
    cards=tuple(hole)+tuple(board)
    ranks=[rank(c) for c in cards]
    if len(set(ranks))!=len(ranks):
        return False
    rset=set(ranks)
    if any(t.issubset(rset) for t in STRAIGHTS):
        return False
    suits=[suit(c) for c in cards]
    max_s=max(suits.count(s) for s in set(suits))
    if max_s>=5:
        return False
    if len(board)>=5:
        return True
    if max_s>=4:
        return False
    if any(len(t & rset)>=4 for t in STRAIGHTS):
        return False
    return True


def prediction_bin(value:float)->str:
    x=float(value)
    for i,name in enumerate(BIN_NAMES):
        lo,hi=BIN_EDGES[i],BIN_EDGES[i+1]
        if lo<=x<hi:
            return name
    raise RuntimeError("prediction bin failure")


def street_name(street:int)->str:
    return {1:"FLOP",2:"TURN",3:"RIVER"}.get(int(street),str(int(street)))


class Acc:
    def __init__(self):
        self.n=0
        self.w=0.0
        self.pred=0.0
        self.pred_w=0.0
        self.target=0.0
        self.target_w=0.0
        self.pos=0
        self.pos_w=0.0
        self.neg=0
        self.neg_w=0.0
        self.sq_w=0.0
        self.abs_w=0.0

    def add(self,pred,target,weight):
        p=float(pred);t=float(target);w=float(weight)
        self.n+=1
        self.w+=w
        self.pred+=p
        self.pred_w+=p*w
        self.target+=t
        self.target_w+=t*w
        if t>0:
            self.pos+=1;self.pos_w+=w
        elif t<0:
            self.neg+=1;self.neg_w+=w
        e=p-t
        self.sq_w+=e*e*w
        self.abs_w+=abs(e)*w

    def out(self):
        if not self.n:
            return {
                "count":0,
                "weight_sum":0.0,
                "prediction_mean":None,
                "prediction_weighted_mean":None,
                "target_mean":None,
                "target_weighted_mean":None,
                "target_positive_fraction":None,
                "target_positive_weighted_fraction":None,
                "target_negative_fraction":None,
                "weighted_residual_bias_prediction_minus_target":None,
                "weighted_mse":None,
                "weighted_rmse":None,
                "weighted_mae":None,
            }
        mse=self.sq_w/self.w if self.w else None
        return {
            "count":self.n,
            "weight_sum":self.w,
            "prediction_mean":self.pred/self.n,
            "prediction_weighted_mean":self.pred_w/self.w if self.w else None,
            "target_mean":self.target/self.n,
            "target_weighted_mean":self.target_w/self.w if self.w else None,
            "target_positive_fraction":self.pos/self.n,
            "target_positive_weighted_fraction":self.pos_w/self.w if self.w else None,
            "target_negative_fraction":self.neg/self.n,
            "weighted_residual_bias_prediction_minus_target":(
                (self.pred_w-self.target_w)/self.w if self.w else None
            ),
            "weighted_mse":mse,
            "weighted_rmse":math.sqrt(mse) if mse is not None else None,
            "weighted_mae":self.abs_w/self.w if self.w else None,
        }


def load_model(state):
    _,m=make_advantage_action_model(REPRESENTATION,device="cpu",seed=0)
    m.load_state_dict(state)
    m.eval()
    return m


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--gap-report",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--batch-size",type=int,default=4096)
    args=ap.parse_args()

    torch.set_num_threads(int(args.threads))

    cp=args.checkpoint.resolve(strict=True)
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    import hashlib
    h=hashlib.sha256()
    with cp.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    actual=h.hexdigest()
    if actual!=EXPECTED_SHA:
        raise SystemExit(f"checkpoint SHA mismatch: {actual}")

    d3=(payload.get("domains") or {}).get(DOMAIN) or {}
    mem=d3.get("adv_mem") or {}
    items=list(mem.get("items") or [])
    if len(items)!=2_000_000:
        raise SystemExit(f"expected 2M retained 3H Advantage samples, got {len(items)}")

    probe=torch.load(args.probe.resolve(strict=True),map_location="cpu",weights_only=False)
    if probe.get("schema")!=PROBE_SCHEMA:
        raise SystemExit(f"expected probe schema {PROBE_SCHEMA}")
    if str(probe.get("source_checkpoint_sha256"))!=EXPECTED_SHA:
        raise SystemExit("probe checkpoint mismatch")
    states=list(probe["snapshots"][str(BUDGET)])
    if len(states)!=8:
        raise SystemExit("expected eight 1600-step replicas")
    models=[load_model(s) for s in states]

    holdout_n=int(probe["holdout_size"])
    holdout_seed=int(probe["holdout_seed"])
    rng=random.Random(holdout_seed)
    holdout_idx=sorted(rng.sample(range(len(items)),holdout_n))
    holdout=[items[i] for i in holdout_idx]

    gap=json.loads(args.gap_report.resolve(strict=True).read_text(encoding="utf-8"))
    if gap.get("schema")!=GAP_SCHEMA:
        raise SystemExit(f"expected gap schema {GAP_SCHEMA}")

    overall=Acc()
    hc=Acc()
    overall_bins={name:Acc() for name in BIN_NAMES}
    hc_bins={name:Acc() for name in BIN_NAMES}
    context_bins={}

    batch_size=int(args.batch_size)
    with torch.no_grad():
        for start in range(0,len(holdout),batch_size):
            chunk=holdout[start:start+batch_size]
            batch,target,weights=vectorized_batch(chunk,"cpu")
            outputs=[m(batch).float() for m in models]
            ens=torch.stack(outputs,dim=0).mean(dim=0).detach().cpu()
            target=target.detach().cpu()
            weights=weights.detach().cpu()

            for j,sample in enumerate(chunk):
                if not bool(sample.legal[ALL_IN]):
                    continue
                pred=float(ens[j,ALL_IN])
                targ=float(target[j,ALL_IN])
                w=float(weights[j])
                b=prediction_bin(pred)
                overall.add(pred,targ,w)
                overall_bins[b].add(pred,targ,w)

                hole,board,n,c=decode_obs(sample.observation)
                street=int(c[1])
                if int(c[0])!=0 or street==0:
                    continue
                if not high_card_no_immediate_draw(hole,board):
                    continue

                hc.add(pred,targ,w)
                hc_bins[b].add(pred,targ,w)
                facing="FACING_ACTION" if float(n[1])>1e-9 else "CHECKED_TO"
                ctx=f"{street_name(street)}_{facing}"
                context_bins.setdefault(ctx,{name:Acc() for name in BIN_NAMES})
                context_bins[ctx][b].add(pred,targ,w)

    overall_out=overall.out()
    hc_out=hc.out()
    overall_bins_out={k:v.out() for k,v in overall_bins.items()}
    hc_bins_out={k:v.out() for k,v in hc_bins.items()}
    context_out={
        ctx:{k:v.out() for k,v in bins.items()}
        for ctx,bins in sorted(context_bins.items())
    }

    exact=[]
    negative_holdout_bin=0
    negative_context_bin=0
    for row in gap["states"]:
        p=float(row["comparison"]["raw_all_in_advantage"])
        b=prediction_bin(p)
        street=int(row["street"])
        facing=str(row["facing_class"])
        ctx=f"{street_name(street)}_{facing}"
        hbin=hc_bins_out[b]
        cbin=context_out.get(ctx,{}).get(b)
        if (
            hbin["target_weighted_mean"] is not None
            and hbin["target_weighted_mean"]<0.0
        ):
            negative_holdout_bin+=1
        if (
            cbin is not None
            and cbin["target_weighted_mean"] is not None
            and cbin["target_weighted_mean"]<0.0
        ):
            negative_context_bin+=1
        exact.append({
            "key":row["key"],
            "hole_labels":row["hole_labels"],
            "board_labels":row["board_labels"],
            "street":street,
            "facing_class":facing,
            "raw_all_in_advantage":p,
            "raw_ensemble_all_in_policy":float(row["comparison"]["raw_ensemble_all_in_policy"]),
            "prediction_bin":b,
            "high_card_holdout_bin":hbin,
            "street_facing_holdout_bin":cbin,
            "local_E_weighted_target_mean":row["local_E"]["all_in_target_iteration_weighted_mean"],
            "local_E_recent_target_mean":row["local_E"]["recent_all_in_target_mean"],
        })

    positive_bins=[
        name for name in BIN_NAMES if name!="NEGATIVE"
    ]
    hc_positive_bin_summary={
        name:{
            "count":hc_bins_out[name]["count"],
            "prediction_weighted_mean":hc_bins_out[name]["prediction_weighted_mean"],
            "target_weighted_mean":hc_bins_out[name]["target_weighted_mean"],
            "target_positive_weighted_fraction":hc_bins_out[name]["target_positive_weighted_fraction"],
            "weighted_residual_bias_prediction_minus_target":hc_bins_out[name]["weighted_residual_bias_prediction_minus_target"],
        }
        for name in positive_bins
    }

    out={
        "schema":SCHEMA,
        "scope":"DIAGNOSTIC_ONLY_UNTOUCHED_50K_HOLDOUT_NO_TRAINING",
        "checkpoint_sha256":actual,
        "probe_schema":PROBE_SCHEMA,
        "gap_schema":GAP_SCHEMA,
        "budget":BUDGET,
        "replicas":len(models),
        "holdout_size":holdout_n,
        "holdout_seed":holdout_seed,
        "bin_edges":["-inf",0.0,0.0025,0.005,0.010,0.020,"+inf"],
        "all_allin_legal":overall_out,
        "postflop_high_card_no_draw_allin_legal":hc_out,
        "all_allin_legal_bins":overall_bins_out,
        "postflop_high_card_no_draw_bins":hc_bins_out,
        "postflop_high_card_no_draw_street_facing_bins":context_out,
        "positive_high_card_bin_summary":hc_positive_bin_summary,
        "exact_flagged_states":exact,
        "aggregate":{
            "exact_states":len(exact),
            "exact_states_whose_high_card_holdout_prediction_bin_has_negative_weighted_target_mean":negative_holdout_bin,
            "exact_states_whose_street_facing_holdout_prediction_bin_has_negative_weighted_target_mean":negative_context_bin,
        },
        "interpretation":(
            "This is calibration of the raw ALL_IN Advantage prediction, not a poker-EV "
            "proof. A positive-prediction bin with a positive held-out conditional target "
            "mean supports broad regression calibration in that prediction range. A zero "
            "or negative held-out conditional target mean indicates sign miscalibration "
            "for that population. Regret matching can still amplify small positive raw "
            "values even when the raw head is broadly calibrated."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== 3H ALL_IN holdout calibration @1600 ===")
    print("all="+json.dumps(overall_out,sort_keys=True))
    print("high_card_no_draw="+json.dumps(hc_out,sort_keys=True))
    for name in BIN_NAMES:
        print(f"HC_BIN {name} "+json.dumps(hc_bins_out[name],sort_keys=True))
    print("exact_bin_negative="+str(negative_holdout_bin)+"/"+str(len(exact)))
    print("exact_context_bin_negative="+str(negative_context_bin)+"/"+str(len(exact)))
    print(f"report={args.report.resolve()}")
    print("3H_ALLIN_HOLDOUT_CALIBRATION_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
