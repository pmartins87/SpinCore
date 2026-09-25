#!/usr/bin/env python3
from __future__ import annotations

"""Fit eight independent 3H Advantage replicas from the frozen 10105 reservoir.

Diagnostic only:
- source checkpoint is read-only;
- no CFR roots are generated;
- no reservoir samples are added;
- each replica starts from a fresh deterministic initialization;
- each replica uses the checkpoint's exact 3H Advantage budget/batch/lr;
- the final checkpoint's actual current 3H Advantage model is preserved as a
  separate reference member.

The resulting artifact is inference-only and is not a promoted strategy.
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
SCHEMA="SPINCORE_3H_ENS8_DIAGNOSTIC_PROBE_V1"
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
    if args.threads<=0:
        raise SystemExit("--threads must be positive")

    torch.set_num_threads(int(args.threads))
    solver=SolverLibrary(args.solver.resolve(strict=True))
    seed,config,iteration,_sampler,runtimes,_history,finalized=load_checkpoint(cp,solver=solver)
    if int(iteration)!=EXPECTED_ITERATION:
        raise RuntimeError(f"expected iteration {EXPECTED_ITERATION}, got {iteration}")
    if not finalized:
        raise RuntimeError("source checkpoint must be finalized")

    runtime=runtimes[DOMAIN]
    runtime.session.batch_mode="vectorized"
    steps=int(config.advantage_steps_for_domain(DOMAIN))
    batch_size=int(config.batch_size)
    lr=float(config.learning_rate)
    if steps<=0:
        raise RuntimeError("3H Advantage budget must be positive")

    current_state=clone_state(runtime.bundle.advantage)
    before_seen=int(runtime.bundle.adv_mem.seen)
    before_items=len(runtime.bundle.adv_mem.items)
    before_strategy_seen=int(runtime.bundle.pol_mem.seen)

    members=[]
    meta=[]
    for rep in range(REPLICAS):
        init_seed=int(mix64(SEED_BASE,rep,0x3A11)&0x7fffffff)
        batch_seed=int(mix64(SEED_BASE,rep,0xBA7C)&0x7fffffff)
        runtime.session.reset_advantage_network(init_seed=init_seed,lr=lr)
        runtime.bundle.batch_rng.seed(batch_seed)
        started=time.perf_counter()
        losses=runtime.session.train_advantage(steps=steps,batch_size=batch_size)
        elapsed=time.perf_counter()-started
        if not losses:
            raise RuntimeError("empty 3H replica fit")
        state=clone_state(runtime.bundle.advantage)
        members.append(state)
        row={
            "replica":rep,
            "init_seed":init_seed,
            "batch_seed":batch_seed,
            "steps":steps,
            "batch_size":batch_size,
            "learning_rate":lr,
            "fit_seconds":float(elapsed),
            "loss_last":float(losses[-1]),
        }
        meta.append(row)
        print(
            f"3H_REPLICA rep={rep+1}/{REPLICAS} "
            f"loss={row['loss_last']:.8f} seconds={elapsed:.3f}",
            flush=True,
        )

    if int(runtime.bundle.adv_mem.seen)!=before_seen or len(runtime.bundle.adv_mem.items)!=before_items:
        raise RuntimeError("diagnostic unexpectedly mutated 3H Advantage reservoir")
    if int(runtime.bundle.pol_mem.seen)!=before_strategy_seen:
        raise RuntimeError("diagnostic unexpectedly mutated 3H strategy reservoir")

    payload={
        "schema":SCHEMA,
        "scope":"DIAGNOSTIC_ONLY_NO_CFR_ROOTS_NO_SOURCE_MUTATION",
        "source_checkpoint_sha256":actual,
        "completed_iteration":int(iteration),
        "training_seed":int(seed),
        "domain":DOMAIN,
        "representation":"C0_V1_FROZEN_CONTROL",
        "source_finalized":bool(finalized),
        "advantage_memory":{
            "retained_items":before_items,
            "seen":before_seen,
            "capacity":int(runtime.bundle.adv_mem.capacity),
        },
        "fit_contract":{
            "replicas":REPLICAS,
            "steps_per_replica":steps,
            "batch_size":batch_size,
            "learning_rate":lr,
            "semantics":"fresh independent fit on exact same frozen 3H Advantage reservoir",
        },
        "current_checkpoint_member":current_state,
        "members":members,
        "member_meta":meta,
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,args.out)
    print(f"probe={args.out.resolve()}")
    print(f"probe_bytes={args.out.stat().st_size}")
    print("3H_ENS8_DIAGNOSTIC_PROBE_BUILD_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
