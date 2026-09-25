#!/usr/bin/env python3
from __future__ import annotations

"""Fit matched 3H Advantage replicas to 100/200/400 steps on frozen 10105 memory.

Diagnostic only. The frozen checkpoint is read-only, no CFR roots are generated,
and no reservoir item is written. Each replica uses one deterministic init and
one deterministic minibatch stream; snapshots at 100, 200 and 400 steps therefore
measure budget scaling without changing the underlying training sample stream.
"""

import argparse
import hashlib
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))

from spincore.lean_functional_training import load_checkpoint
from spincore.solver import SolverLibrary

DOMAIN="THREE_HANDED"
EXPECTED_ITERATION=10105
EXPECTED_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
SCHEMA="SPINCORE_3H_ADVANTAGE_BUDGET_PROBE_V1"
BUDGETS=(100,200,400)
REPLICAS=8
SEED_BASE=20260925


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


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    args=ap.parse_args()

    cp=args.checkpoint.resolve(strict=True)
    actual=sha256(cp)
    if actual!=EXPECTED_SHA:
        raise SystemExit(f"checkpoint SHA mismatch: {actual}")
    torch.set_num_threads(int(args.threads))

    solver=SolverLibrary(args.solver.resolve(strict=True))
    seed,config,iteration,_sampler,runtimes,_history,finalized=load_checkpoint(cp,solver=solver)
    if int(iteration)!=EXPECTED_ITERATION or not finalized:
        raise RuntimeError("source must be finalized 10105 checkpoint")

    runtime=runtimes[DOMAIN]
    runtime.session.batch_mode="vectorized"
    native_steps=int(config.advantage_steps_for_domain(DOMAIN))
    if native_steps!=100:
        raise RuntimeError(f"expected native 3H fresh100 contract, got {native_steps}")
    batch_size=int(config.batch_size)
    lr=float(config.learning_rate)

    before_adv_seen=int(runtime.bundle.adv_mem.seen)
    before_adv_items=len(runtime.bundle.adv_mem.items)
    before_pol_seen=int(runtime.bundle.pol_mem.seen)
    current=clone_state(runtime.bundle.advantage)

    snapshots={str(b):[] for b in BUDGETS}
    meta=[]
    for rep in range(REPLICAS):
        init_seed=int(mix64(SEED_BASE,rep,0x3A11)&0x7fffffff)
        batch_seed=int(mix64(SEED_BASE,rep,0xBA7C)&0x7fffffff)
        runtime.session.reset_advantage_network(init_seed=init_seed,lr=lr)
        runtime.bundle.batch_rng.seed(batch_seed)

        row={"replica":rep,"init_seed":init_seed,"batch_seed":batch_seed,"snapshots":{}}
        completed=0
        for budget in BUDGETS:
            delta=int(budget-completed)
            started=time.perf_counter()
            losses=runtime.session.train_advantage(steps=delta,batch_size=batch_size)
            elapsed=time.perf_counter()-started
            if not losses:
                raise RuntimeError("empty fit")
            snapshots[str(budget)].append(clone_state(runtime.bundle.advantage))
            row["snapshots"][str(budget)]={
                "cumulative_steps":int(budget),
                "delta_steps":int(delta),
                "segment_seconds":float(elapsed),
                "loss_last":float(losses[-1]),
            }
            completed=budget
        meta.append(row)
        print(
            f"3H_BUDGET_REPLICA rep={rep+1}/{REPLICAS} "
            + " ".join(
                f"b{b}_loss={row['snapshots'][str(b)]['loss_last']:.8f}"
                for b in BUDGETS
            ),
            flush=True,
        )

    if int(runtime.bundle.adv_mem.seen)!=before_adv_seen or len(runtime.bundle.adv_mem.items)!=before_adv_items:
        raise RuntimeError("Advantage reservoir mutation detected")
    if int(runtime.bundle.pol_mem.seen)!=before_pol_seen:
        raise RuntimeError("strategy reservoir mutation detected")

    payload={
        "schema":SCHEMA,
        "scope":"DIAGNOSTIC_ONLY_FROZEN_10105_MEMORY",
        "source_checkpoint_sha256":actual,
        "completed_iteration":int(iteration),
        "training_seed":int(seed),
        "domain":DOMAIN,
        "representation":"C0_V1_FROZEN_CONTROL",
        "native_budget":native_steps,
        "budgets":list(BUDGETS),
        "replicas":REPLICAS,
        "batch_size":batch_size,
        "learning_rate":lr,
        "advantage_memory":{"retained_items":before_adv_items,"seen":before_adv_seen},
        "current_checkpoint_member":current,
        "snapshots":snapshots,
        "member_meta":meta,
        "semantics":"same init+batch stream per replica; cumulative snapshots at 100/200/400",
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,args.out)
    print(f"probe={args.out.resolve()}")
    print(f"probe_bytes={args.out.stat().st_size}")
    print("3H_ADVANTAGE_BUDGET_PROBE_BUILD_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
