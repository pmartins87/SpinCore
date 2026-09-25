#!/usr/bin/env python3
from __future__ import annotations

"""Clean-holdout validation of frozen 10105 3H Advantage budget snapshots.

Uses the already-built 100/200/400 diagnostic probe.  For every replica, the
exact Python-random minibatch index stream is reconstructed from the saved
batch_seed.  A common holdout is sampled only from reservoir indices that were
never used by ANY of the eight replicas through 400 steps.

This separates "members disagree" from "members are still improving on unseen
Advantage targets".  No training and no checkpoint mutation occur.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))

from spincore_nn.action_models import make_advantage_action_model
from spincore_nn.lean_batch import vectorized_batch

EXPECTED_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
PROBE_SCHEMA="SPINCORE_3H_ADVANTAGE_BUDGET_PROBE_V1"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
BUDGETS=(100,200,400)
REPLICAS=8
HOLDOUT_SEED=20260925 ^ 0x401D0A7


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def load_model(state):
    _,model=make_advantage_action_model(REPRESENTATION,device="cpu",seed=0)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_models(models,samples,batch_size=4096):
    if not samples:
        raise RuntimeError("empty holdout")
    per_member_sum=[0.0 for _ in models]
    per_member_weight=[0.0 for _ in models]
    ens_sum=0.0
    ens_weight=0.0
    unweighted_member_sum=[0.0 for _ in models]
    unweighted_ens_sum=0.0
    n=0

    with torch.no_grad():
        for start in range(0,len(samples),batch_size):
            chunk=samples[start:start+batch_size]
            batch,target,weights=vectorized_batch(chunk,"cpu")
            legal=batch["legal"].float()
            denom=legal.sum(1).clamp_min(1.0)
            target=target.float()
            weights=weights.float()
            norm_w=weights/weights.mean().clamp_min(1e-12)

            outputs=[m(batch).float() for m in models]
            for j,out in enumerate(outputs):
                per=(((out-target)**2)*legal).sum(1)/denom
                per_member_sum[j]+=float((per*norm_w).sum().cpu())
                per_member_weight[j]+=float(norm_w.sum().cpu())
                unweighted_member_sum[j]+=float(per.sum().cpu())

            ens=torch.stack(outputs,dim=0).mean(dim=0)
            per_ens=(((ens-target)**2)*legal).sum(1)/denom
            ens_sum+=float((per_ens*norm_w).sum().cpu())
            ens_weight+=float(norm_w.sum().cpu())
            unweighted_ens_sum+=float(per_ens.sum().cpu())
            n+=len(chunk)

    member_weighted=[a/b for a,b in zip(per_member_sum,per_member_weight)]
    member_unweighted=[x/n for x in unweighted_member_sum]
    return {
        "n":n,
        "member_weighted_mse":member_weighted,
        "member_weighted_mse_mean":statistics.fmean(member_weighted),
        "member_weighted_mse_median":statistics.median(member_weighted),
        "member_weighted_mse_pstdev":statistics.pstdev(member_weighted),
        "raw_ensemble_weighted_mse":ens_sum/ens_weight,
        "member_unweighted_mse_mean":statistics.fmean(member_unweighted),
        "raw_ensemble_unweighted_mse":unweighted_ens_sum/n,
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--holdout-size",type=int,default=50000)
    args=ap.parse_args()

    cp=args.checkpoint.resolve(strict=True)
    actual=sha256(cp)
    if actual!=EXPECTED_SHA:
        raise SystemExit(f"checkpoint SHA mismatch: {actual}")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    if int(payload.get("completed_iteration",-1))!=10105:
        raise SystemExit("expected iteration 10105")
    d3=(payload.get("domains") or {}).get("THREE_HANDED") or {}
    mem=d3.get("adv_mem") or {}
    items=list(mem.get("items") or [])
    n_items=len(items)
    if not items:
        raise SystemExit("empty 3H Advantage reservoir")

    probe=torch.load(args.probe.resolve(strict=True),map_location="cpu",weights_only=False)
    if probe.get("schema")!=PROBE_SCHEMA:
        raise SystemExit("wrong probe schema")
    if probe.get("source_checkpoint_sha256")!=actual:
        raise SystemExit("probe/checkpoint SHA mismatch")
    if int(probe.get("replicas",-1))!=REPLICAS:
        raise SystemExit("replica count mismatch")
    if tuple(int(x) for x in probe.get("budgets") or [])!=BUDGETS:
        raise SystemExit("budget mismatch")
    batch_size=int(probe["batch_size"])
    if batch_size<=0:
        raise SystemExit("bad batch size")

    # Mark every reservoir index touched by every replica through 400 steps.
    used=bytearray(n_items)
    used_count=0
    for row in probe.get("member_meta") or []:
        rng=random.Random(int(row["batch_seed"]))
        for _ in range(max(BUDGETS)):
            idxs=rng.sample(range(n_items),min(batch_size,n_items))
            for idx in idxs:
                if used[idx]==0:
                    used[idx]=1
                    used_count+=1

    untouched=[i for i,v in enumerate(used) if v==0]
    if len(untouched)<int(args.holdout_size):
        raise RuntimeError(
            f"insufficient untouched indices: {len(untouched)} < {args.holdout_size}"
        )
    rng=random.Random(HOLDOUT_SEED)
    holdout_idx=rng.sample(untouched,int(args.holdout_size))
    holdout=[items[i] for i in holdout_idx]

    results={}
    for budget in BUDGETS:
        states=list(probe["snapshots"][str(budget)])
        if len(states)!=REPLICAS:
            raise RuntimeError("snapshot member count mismatch")
        models=[load_model(state) for state in states]
        metrics=evaluate_models(models,holdout)
        results[str(budget)]=metrics
        print(
            f"HOLDOUT_BUDGET_{budget} "
            + json.dumps({
                "member_weighted_mse_mean":metrics["member_weighted_mse_mean"],
                "member_weighted_mse_pstdev":metrics["member_weighted_mse_pstdev"],
                "raw_ensemble_weighted_mse":metrics["raw_ensemble_weighted_mse"],
            },sort_keys=True),
            flush=True,
        )
        del models

    report={
        "schema":"SPINCORE_3H_ADVANTAGE_CLEAN_HOLDOUT_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_NO_TRAINING",
        "checkpoint_sha256":actual,
        "probe_schema":PROBE_SCHEMA,
        "advantage_reservoir_retained":n_items,
        "advantage_reservoir_seen":int(mem.get("seen",0)),
        "replicas":REPLICAS,
        "batch_size":batch_size,
        "max_budget":max(BUDGETS),
        "union_training_indices_through_400":used_count,
        "union_training_fraction_through_400":used_count/n_items,
        "untouched_indices_through_400":len(untouched),
        "holdout_size":len(holdout),
        "holdout_seed":HOLDOUT_SEED,
        "budgets":results,
        "interpretation":(
            "The holdout contains only Advantage-reservoir entries not sampled by any of the "
            "eight diagnostic replicas through 400 steps. Falling clean-holdout MSE with budget "
            "is evidence that larger fit budgets continue to improve unseen target regression, "
            "even if policy argmax instability remains high. Flat/worsening MSE means more of "
            "the same optimization is not extracting materially more generalizable signal."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"report={args.report.resolve()}")
    print("3H_ADVANTAGE_CLEAN_HOLDOUT_AUDIT_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
