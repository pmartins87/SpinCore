#!/usr/bin/env python3
from __future__ import annotations

"""Process-parallel HU ENS8 fresh-fit helper and parity benchmark support.

The implementation preserves the existing ENS8_A member seed contract and the
exact fresh400 training semantics.  Member jobs are isolated processes so four
8-thread fits can execute concurrently on a 32-thread Ryzen without sharing
mutable PyTorch model/optimizer state.
"""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_lt2_stage_a_b_first_divergence as fd
from spincore_nn.action_models import make_advantage_action_model
from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.reservoir import UniformReservoir
from spincore_nn.training import train_step

REPRESENTATION="C0_V1_FROZEN_CONTROL"
ENSEMBLE_SIZE=8
MEMBER_STEPS=400


def member_seeds(member:int)->tuple[int,int]:
    if not 0<=int(member)<ENSEMBLE_SIZE:
        raise ValueError("ensemble member out of range")
    init_seed=fd._mix64(20260920,int(member),0xE115E)&0x7FFFFFFF
    batch_seed=fd._mix64(20260920,int(member),0xEBA7C4)&0x7FFFFFFF
    return int(init_seed),int(batch_seed)


def _clone_state(model)->dict[str,torch.Tensor]:
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def make_snapshot(runtime,config,path:Path)->dict[str,Any]:
    payload={
        "schema":"SPINCORE_LT3_HU_PARALLEL_FIT_INPUT_V1",
        "representation":runtime.bundle.selected_representation,
        "config":runtime.bundle.config.to_dict(),
        "adv_mem":runtime.bundle.adv_mem.state_dict(),
        "optimizer_defaults":dict(runtime.bundle.adv_opt.defaults),
        "learning_rate":float(config.learning_rate),
        "batch_size":int(config.batch_size),
        "member_steps":MEMBER_STEPS,
    }
    path.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    torch.save(payload,path)
    return {
        "snapshot_seconds":float(time.perf_counter()-started),
        "snapshot_bytes":int(path.stat().st_size),
    }


def worker_fit(snapshot_path:Path,member:int,threads:int,output_path:Path)->None:
    torch.set_num_threads(int(threads))
    payload=torch.load(snapshot_path,map_location="cpu",weights_only=False)
    if payload.get("schema")!="SPINCORE_LT3_HU_PARALLEL_FIT_INPUT_V1":
        raise RuntimeError("wrong parallel-fit snapshot schema")
    if payload.get("representation")!=REPRESENTATION:
        raise RuntimeError("parallel fit requires frozen SPNNIV1 representation")
    if int(payload.get("member_steps",-1))!=MEMBER_STEPS:
        raise RuntimeError("parallel fit member-step drift")

    memory=UniformReservoir.from_state_dict(payload["adv_mem"])
    init_seed,batch_seed=member_seeds(member)
    cfg,model=make_advantage_action_model(
        REPRESENTATION,device="cpu",seed=init_seed
    )
    if cfg.to_dict()!=dict(payload["config"]):
        raise RuntimeError("parallel fit model config drift")

    defaults=dict(payload["optimizer_defaults"])
    defaults["lr"]=float(payload["learning_rate"])
    optimizer=torch.optim.Adam(model.parameters(),**defaults)
    rng=random.Random(batch_seed)
    batch_size=int(payload["batch_size"])

    losses=[]
    started=time.perf_counter()
    for _ in range(MEMBER_STEPS):
        samples=memory.sample(min(batch_size,len(memory.items)),rng)
        batch,target,weights=vectorized_batch(samples,"cpu")
        losses.append(train_step(model,optimizer,batch,target,weights,"advantage"))
    elapsed=float(time.perf_counter()-started)

    out={
        "schema":"SPINCORE_LT3_HU_PARALLEL_FIT_MEMBER_V1",
        "member":int(member),
        "init_seed":int(init_seed),
        "batch_seed":int(batch_seed),
        "threads":int(threads),
        "steps":MEMBER_STEPS,
        "fit_seconds":elapsed,
        "losses":[float(x) for x in losses],
        "state":_clone_state(model),
        "optimizer_state":optimizer.state_dict(),
    }
    tmp=output_path.with_suffix(output_path.suffix+".tmp")
    torch.save(out,tmp)
    os.replace(tmp,output_path)


def _run_worker_subprocess(
    *,
    python:Path,
    snapshot:Path,
    member:int,
    threads:int,
    output:Path,
)->dict[str,Any]:
    env=dict(os.environ)
    env["OMP_NUM_THREADS"]=str(int(threads))
    env["MKL_NUM_THREADS"]=str(int(threads))
    cmd=[
        str(python),str(Path(__file__).resolve()),
        "--worker",
        "--snapshot",str(snapshot),
        "--member",str(int(member)),
        "--threads",str(int(threads)),
        "--output",str(output),
    ]
    started=time.perf_counter()
    proc=subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    elapsed=float(time.perf_counter()-started)
    if proc.returncode!=0:
        raise RuntimeError(
            f"parallel member {member} failed rc={proc.returncode}\n{proc.stdout}"
        )
    payload=torch.load(output,map_location="cpu",weights_only=False)
    if int(payload.get("member",-1))!=int(member):
        raise RuntimeError("parallel member output identity drift")
    payload["process_wall_seconds"]=elapsed
    return payload


def parallel_fit(
    runtime,
    config,
    *,
    python:Path,
    work_dir:Path,
    concurrency:int=4,
    threads_per_member:int=8,
)->tuple[tuple[dict[str,torch.Tensor],...],list[dict[str,Any]],float,dict[str,Any]]:
    if concurrency<=0 or threads_per_member<=0:
        raise ValueError("positive concurrency/threads required")
    work_dir.mkdir(parents=True,exist_ok=True)
    snapshot=work_dir/"hu_fit_input.pt"
    snapshot_meta=make_snapshot(runtime,config,snapshot)

    started=time.perf_counter()
    results=[None]*ENSEMBLE_SIZE
    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        futures={}
        for member in range(ENSEMBLE_SIZE):
            output=work_dir/f"member_{member}.pt"
            fut=pool.submit(
                _run_worker_subprocess,
                python=python,
                snapshot=snapshot,
                member=member,
                threads=threads_per_member,
                output=output,
            )
            futures[fut]=member
        for fut in as_completed(futures):
            member=futures[fut]
            results[member]=fut.result()
    fit_wall=float(time.perf_counter()-started)

    states=tuple(result["state"] for result in results)
    member_meta=[
        {
            "member":int(result["member"]),
            "init_seed":int(result["init_seed"]),
            "batch_seed":int(result["batch_seed"]),
            "steps":int(result["steps"]),
            "fit_seconds":float(result["fit_seconds"]),
            "process_wall_seconds":float(result["process_wall_seconds"]),
            "loss_last":float(result["losses"][-1]),
        }
        for result in results
    ]
    meta={
        **snapshot_meta,
        "fit_wall_seconds":fit_wall,
        "concurrency":int(concurrency),
        "threads_per_member":int(threads_per_member),
    }
    return states,member_meta,fit_wall,meta


def install_parallel_result(runtime,states,member_outputs_meta,*,config)->None:
    """Recreate the authoritative post-fit runtime semantics.

    The ordinary runtime retains member 7 as bundle.advantage, exactly like the
    historical sequential _fit_hu_ensemble path.  The caller then installs the
    full ensemble behavior from all returned states.
    """
    last=ENSEMBLE_SIZE-1
    init_seed,_=member_seeds(last)
    runtime.session.reset_advantage_network(
        init_seed=init_seed,
        lr=float(config.learning_rate),
    )
    runtime.bundle.advantage.load_state_dict(states[last])
    # The next iteration resets the network before fitting.  Preserve counters
    # exactly; optimizer state does not affect policy inference and is discarded
    # at the next reset.
    # One reset above occurred locally; account for the remaining seven resets.
    runtime.bundle.counters["advantage_resets"] += ENSEMBLE_SIZE-1
    runtime.bundle.counters["adv_optimizer_steps"] += ENSEMBLE_SIZE*MEMBER_STEPS
    runtime.bundle.counters["advantage_ready"]=1


def tensor_states_equal(a,b)->bool:
    if set(a)!=set(b):
        return False
    return all(torch.equal(a[k],b[k]) for k in a)


def state_digest(state:dict[str,torch.Tensor])->str:
    h=hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8")+b"\0")
        tensor=state[key].detach().cpu().contiguous()
        h.update(str(tensor.dtype).encode("ascii")+b"\0")
        h.update(str(tuple(tensor.shape)).encode("ascii")+b"\0")
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",action="store_true")
    p.add_argument("--snapshot",type=Path)
    p.add_argument("--member",type=int)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--output",type=Path)
    return p.parse_args()


def main()->int:
    args=parse_args()
    if not args.worker:
        raise SystemExit("this module is intended for --worker subprocess use")
    if args.snapshot is None or args.member is None or args.output is None:
        raise SystemExit("worker requires --snapshot --member --output")
    worker_fit(args.snapshot,args.member,args.threads,args.output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
