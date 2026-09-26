#!/usr/bin/env python3
from __future__ import annotations

"""Controlled-split 3H Advantage budget curve on frozen 10105 memory.

A fixed 50k validation split is removed from the 2M retained 3H Advantage
reservoir before any diagnostic fitting. Eight fresh replicas are then trained
on the identical remaining train pool, with matched deterministic seeds and
snapshots at 100/200/400/800/1600 steps.

This is a supervised diagnostic of the stored Advantage regression problem.
It generates no CFR roots and mutates no source checkpoint.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))

from spincore_nn.action_models import make_advantage_action_model
from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.training import train_step

EXPECTED_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
SCHEMA="SPINCORE_3H_ADVANTAGE_CONTROLLED_SPLIT_PROBE_V1"
DOMAIN="THREE_HANDED"
BUDGETS=(100,200,400,800,1600)
REPLICAS=8
SEED_BASE=20260926
HOLDOUT_SEED=20260926 ^ 0x50A17


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def mix64(*values:int)->int:
    x=0x9E3779B97F4A7C15
    mask=(1<<64)-1
    for value in values:
        y=int(value)&mask
        x^=(y+0x9E3779B97F4A7C15+((x<<6)&mask)+(x>>2))&mask
        x&=mask
    return x


def clone_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def load_model(state):
    _,m=make_advantage_action_model(REPRESENTATION,device="cpu",seed=0)
    m.load_state_dict(state)
    m.eval()
    return m


def evaluate_states(states,samples,batch_size=4096):
    models=[load_model(s) for s in states]
    member_num=[0.0]*len(models)
    member_den=[0.0]*len(models)
    ens_num=0.0
    ens_den=0.0
    n=0
    with torch.no_grad():
        for start in range(0,len(samples),batch_size):
            chunk=samples[start:start+batch_size]
            batch,target,weights=vectorized_batch(chunk,"cpu")
            target=target.float()
            weights=weights.float()
            legal=batch["legal"].float()
            denom=legal.sum(1).clamp_min(1.0)
            outputs=[m(batch).float() for m in models]
            for j,out in enumerate(outputs):
                per=(((out-target)**2)*legal).sum(1)/denom
                member_num[j]+=float((per*weights).sum().cpu())
                member_den[j]+=float(weights.sum().cpu())
            ens=torch.stack(outputs,dim=0).mean(dim=0)
            per=(((ens-target)**2)*legal).sum(1)/denom
            ens_num+=float((per*weights).sum().cpu())
            ens_den+=float(weights.sum().cpu())
            n+=len(chunk)
    member=[x/y for x,y in zip(member_num,member_den)]
    del models
    return {
        "n":n,
        "member_weighted_mse":member,
        "member_weighted_mse_mean":statistics.fmean(member),
        "member_weighted_mse_median":statistics.median(member),
        "member_weighted_mse_pstdev":statistics.pstdev(member),
        "raw_ensemble_weighted_mse":ens_num/ens_den,
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--holdout-size",type=int,default=50000)
    ap.add_argument("--threads",type=int,default=8)
    args=ap.parse_args()

    cp=args.checkpoint.resolve(strict=True)
    actual=sha256(cp)
    if actual!=EXPECTED_SHA:
        raise SystemExit(f"checkpoint SHA mismatch: {actual}")
    torch.set_num_threads(int(args.threads))

    payload=torch.load(cp,map_location="cpu",weights_only=False)
    if int(payload.get("completed_iteration",-1))!=10105:
        raise SystemExit("expected iteration 10105")
    cfg=dict(payload.get("config") or {})
    d3=(payload.get("domains") or {}).get(DOMAIN) or {}
    mem=d3.get("adv_mem") or {}
    items=list(mem.get("items") or [])
    n_items=len(items)
    if n_items!=2_000_000:
        raise SystemExit(f"expected saturated 2M reservoir, got {n_items}")

    holdout_n=int(args.holdout_size)
    if not (0<holdout_n<n_items):
        raise SystemExit("bad holdout size")
    split_rng=random.Random(HOLDOUT_SEED)
    holdout_idx=set(split_rng.sample(range(n_items),holdout_n))
    train_idx=[i for i in range(n_items) if i not in holdout_idx]
    holdout=[items[i] for i in sorted(holdout_idx)]

    batch_size=int(cfg["batch_size"])
    lr=float(cfg["learning_rate"])
    snapshots={str(b):[] for b in BUDGETS}
    member_meta=[]

    for rep in range(REPLICAS):
        init_seed=int(mix64(SEED_BASE,rep,0x3A11)&0x7fffffff)
        batch_seed=int(mix64(SEED_BASE,rep,0xBA7C)&0x7fffffff)
        _,model=make_advantage_action_model(REPRESENTATION,device="cpu",seed=init_seed)
        opt=torch.optim.Adam(model.parameters(),lr=lr)
        rng=random.Random(batch_seed)
        completed=0
        row={"replica":rep,"init_seed":init_seed,"batch_seed":batch_seed,"segments":{}}
        for budget in BUDGETS:
            started=time.perf_counter()
            losses=[]
            while completed<int(budget):
                pos=rng.sample(range(len(train_idx)),min(batch_size,len(train_idx)))
                samples=[items[train_idx[i]] for i in pos]
                batch,target,weights=vectorized_batch(samples,"cpu")
                losses.append(train_step(model,opt,batch,target,weights,"advantage"))
                completed+=1
            elapsed=time.perf_counter()-started
            snapshots[str(budget)].append(clone_state(model))
            row["segments"][str(budget)]={
                "cumulative_steps":int(budget),
                "segment_seconds":float(elapsed),
                "loss_last":float(losses[-1]) if losses else None,
            }
            print(
                f"CONTROLLED_REPLICA rep={rep+1}/{REPLICAS} budget={budget} "
                f"loss={row['segments'][str(budget)]['loss_last']:.8f} "
                f"seconds={elapsed:.3f}",
                flush=True,
            )
        member_meta.append(row)

    validation={}
    for budget in BUDGETS:
        validation[str(budget)]=evaluate_states(snapshots[str(budget)],holdout)
        print(
            f"CONTROLLED_HOLDOUT budget={budget} "
            + json.dumps(validation[str(budget)],sort_keys=True),
            flush=True,
        )

    out={
        "schema":SCHEMA,
        "scope":"DIAGNOSTIC_ONLY_FIXED_TRAIN_HOLDOUT_SPLIT_NO_CFR_ROOTS",
        "source_checkpoint_sha256":actual,
        "completed_iteration":10105,
        "representation":REPRESENTATION,
        "domain":DOMAIN,
        "reservoir_retained":n_items,
        "reservoir_seen":int(mem.get("seen",0)),
        "holdout_seed":HOLDOUT_SEED,
        "holdout_size":holdout_n,
        "train_size":len(train_idx),
        "budgets":list(BUDGETS),
        "replicas":REPLICAS,
        "batch_size":batch_size,
        "learning_rate":lr,
        "snapshots":snapshots,
        "member_meta":member_meta,
        "validation":validation,
        "semantics":"same fixed 50k holdout excluded from every replica at every budget",
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(out,args.out)
    print(f"probe={args.out.resolve()}")
    print(f"probe_bytes={args.out.stat().st_size}")
    print("3H_ADVANTAGE_CONTROLLED_SPLIT_PROBE_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
