#!/usr/bin/env python3
from __future__ import annotations

"""Shared mechanics for LT3 sequential/parallel ENS8 continuation gates."""

import copy
import hashlib
import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

from spincore.lean_action_policy import LeanEnsembleActionAdvantagePolicy
from spincore.lean_concurrent_iteration import _collect_policy_from_episodes,_presample_plan
from spincore.lean_functional_training import DOMAINS,_advantage_reset_seed
from spincore.lean_parallel import ParallelRootExecutor

import lt3_hu_ens8_parallel_fit as par
import run_lt2_hu_ens8_online_pilot as seq

DOMAIN_HU="TRUE_HEADS_UP"
DOMAIN_3H="THREE_HANDED"
ENSEMBLE_SIZE=8
MEMBER_STEPS=400
ALLOWED_ENSEMBLE_SCHEMAS={
    "SPINCORE_LT2_HU_ENS8_CURRENT_STATE_V1",
    "SPINCORE_LT3_HU_ENS8_CURRENT_STATE_V1",
}


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def load_ensemble_behavior(runtime,path:Path,expected_iteration:int):
    payload=torch.load(Path(path),map_location="cpu",weights_only=False)
    if payload.get("schema") not in ALLOWED_ENSEMBLE_SCHEMAS:
        raise RuntimeError(f"unsupported ensemble schema: {payload.get('schema')!r}")
    if int(payload.get("completed_iteration",-1))!=int(expected_iteration):
        raise RuntimeError("ensemble/checkpoint iteration mismatch")
    if int(payload.get("ensemble_size",-1))!=ENSEMBLE_SIZE:
        raise RuntimeError("ensemble size drift")
    if int(payload.get("member_steps",-1))!=MEMBER_STEPS:
        raise RuntimeError("ensemble member-step drift")
    members=payload.get("members")
    if not isinstance(members,(list,tuple)) or len(members)!=ENSEMBLE_SIZE:
        raise RuntimeError("ensemble members payload drift")
    states=tuple({
        key:value.detach().cpu().clone()
        for key,value in state.items()
    } for state in members)
    par.install_ensemble_behavior(runtime,states)
    return states,list(payload.get("member_meta") or []),str(payload.get("schema"))


def _update_digest(h,obj)->None:
    if obj is None:
        h.update(b"N")
    elif isinstance(obj,bool):
        h.update(b"B1" if obj else b"B0")
    elif isinstance(obj,int):
        h.update(b"I"+str(obj).encode()+b";")
    elif isinstance(obj,float):
        h.update(b"F"+struct.pack("<d",float(obj)))
    elif isinstance(obj,str):
        data=obj.encode("utf-8")
        h.update(b"S"+str(len(data)).encode()+b":"+data)
    elif isinstance(obj,(bytes,bytearray)):
        data=bytes(obj)
        h.update(b"Y"+str(len(data)).encode()+b":"+data)
    elif isinstance(obj,torch.Tensor):
        t=obj.detach().cpu().contiguous()
        h.update(b"T"+str(t.dtype).encode()+b":"+str(tuple(t.shape)).encode()+b":")
        h.update(t.numpy().tobytes())
    elif isinstance(obj,np.ndarray):
        a=np.ascontiguousarray(obj)
        h.update(b"A"+str(a.dtype).encode()+b":"+str(a.shape).encode()+b":")
        h.update(a.tobytes())
    elif isinstance(obj,np.generic):
        _update_digest(h,obj.item())
    elif isinstance(obj,dict):
        h.update(b"D")
        for key in sorted(obj,key=lambda x:repr(x)):
            _update_digest(h,key)
            _update_digest(h,obj[key])
        h.update(b"d")
    elif isinstance(obj,(list,tuple)):
        h.update(b"L")
        for value in obj:
            _update_digest(h,value)
        h.update(b"l")
    else:
        h.update(b"R"+repr(obj).encode("utf-8"))


def digest_object(obj)->str:
    h=hashlib.sha256()
    _update_digest(h,obj)
    return h.hexdigest()


def digest_model(model)->str:
    return par.state_digest({
        k:v.detach().cpu()
        for k,v in model.state_dict().items()
    })


def digest_sample(sample)->str:
    return digest_object({
        "observation":sample.observation,
        "legal":tuple(int(x) for x in sample.legal),
        "target":tuple(float(x) for x in sample.target),
        "weight":float(sample.weight),
        "iteration":int(sample.iteration),
    })


class ReservoirWriteAudit:
    def __init__(self):
        self.rows=[]

    def __call__(self,index:int,sample)->None:
        self.rows.append((int(index),digest_sample(sample)))

    def snapshot(self)->dict[str,Any]:
        return {
            "count":len(self.rows),
            "digest":digest_object(self.rows),
        }


def attach_write_audits(runtimes,*,hu_mirror=None):
    audits={}
    for domain in DOMAINS:
        runtime=runtimes[domain]
        for kind,memory in (
            ("adv",runtime.bundle.adv_mem),
            ("pol",runtime.bundle.pol_mem),
        ):
            audit=ReservoirWriteAudit()
            audits[f"{domain}:{kind}"]=audit
            if domain==DOMAIN_HU and kind=="adv" and hu_mirror is not None:
                hu_mirror.bind_authoritative(memory,extra_observer=audit)
            else:
                memory.set_write_observer(audit)
    return audits


def semantic_fingerprint(*,sampler,runtimes,hu_states,audits):
    domains={}
    for domain in DOMAINS:
        b=runtimes[domain].bundle
        domains[domain]={
            "advantage_model":digest_model(b.advantage),
            "policy_model":digest_model(b.policy),
            "adv_optimizer":digest_object(b.adv_opt.state_dict()),
            "policy_optimizer":digest_object(b.pol_opt.state_dict()),
            "batch_rng":digest_object(b.batch_rng.getstate()),
            "counters":dict(b.counters),
            "adv_mem_seen":int(b.adv_mem.seen),
            "adv_mem_rng":digest_object(b.adv_mem.rng.getstate()),
            "pol_mem_seen":int(b.pol_mem.seen),
            "pol_mem_rng":digest_object(b.pol_mem.rng.getstate()),
            "adv_writes":audits[f"{domain}:adv"].snapshot(),
            "pol_writes":audits[f"{domain}:pol"].snapshot(),
        }
    return {
        "sampler_rng":digest_object(sampler.rng.bit_generator.state),
        "domains":domains,
        "hu_ensemble_members":[par.state_digest(s) for s in hu_states],
    }


def run_one_iteration(
    *,
    seed:int,
    iteration:int,
    config,
    sampler,
    runtimes,
    root_executor:ParallelRootExecutor,
    hu_states,
    fit_mode:str,
    parallel_fitter=None,
):
    if fit_mode not in {"sequential","parallel"}:
        raise ValueError("fit_mode must be sequential or parallel")
    if fit_mode=="parallel" and parallel_fitter is None:
        raise ValueError("parallel fitter required")

    iter_started=time.perf_counter()
    plans=_presample_plan(
        seed=int(seed),
        iteration=int(iteration),
        config=config,
        sampler=sampler,
    )
    report={"iteration":int(iteration),"domains":{},"hu_ensemble_size":ENSEMBLE_SIZE}
    base={}

    for domain in DOMAINS:
        runtime=runtimes[domain]
        plan=plans[domain]
        base[domain]={
            "nodes_before":int(runtime.bundle.counters["nodes"]),
            "adv_before":int(runtime.bundle.adv_mem.seen),
            "pol_before":int(runtime.bundle.pol_mem.seen),
        }
        stats=root_executor.collect(
            domain=domain,
            bundle=runtime.bundle,
            iteration=int(iteration),
            exact_opponent_levels=int(config.exact_opponent_levels),
            jobs=plan["jobs"],
            hu_preflop_board_average_k=1,
            ensemble_model_states=(hu_states if domain==DOMAIN_HU else None),
        )
        base[domain]["tree_seconds"]=float(stats["seconds"])
        base[domain]["root_behavior_ensemble_size"]=int(stats.get("ensemble_size",1))

    r3=runtimes[DOMAIN_3H]
    fit3_started=time.perf_counter()
    r3.session.reset_advantage_network(
        init_seed=_advantage_reset_seed(seed,DOMAIN_3H,iteration),
        lr=float(config.learning_rate),
    )
    losses3=r3.session.train_advantage(
        steps=int(config.advantage_steps_for_domain(DOMAIN_3H)),
        batch_size=int(config.batch_size),
    )
    fit3_seconds=float(time.perf_counter()-fit3_started)

    hu_runtime=runtimes[DOMAIN_HU]
    if fit_mode=="sequential":
        _models,hu_states,hu_member_meta,hu_fit_seconds=seq._fit_hu_ensemble(
            hu_runtime,config,count_optimizer_steps=True
        )
    else:
        hu_states,hu_member_meta,hu_fit_seconds,last_optimizer=parallel_fitter.fit()
        par.install_parallel_result(
            hu_runtime,hu_states,last_optimizer,config=config
        )
        par.install_ensemble_behavior(hu_runtime,hu_states)

    for domain in DOMAINS:
        runtime=runtimes[domain]
        plan=plans[domain]
        policy_report=_collect_policy_from_episodes(
            seed=int(seed),
            iteration=int(iteration),
            domain=domain,
            episodes=plan["policy_episodes"],
            runtime=runtime,
        )
        before=base[domain]
        if domain==DOMAIN_3H:
            fit_seconds=fit3_seconds
            optimizer_steps=int(config.advantage_steps_for_domain(domain))
            loss_last=float(losses3[-1])
            extra={"ensemble_size":1}
        else:
            fit_seconds=float(hu_fit_seconds)
            optimizer_steps=ENSEMBLE_SIZE*MEMBER_STEPS
            loss_last=float(sum(x["loss_last"] for x in hu_member_meta)/len(hu_member_meta))
            extra={
                "ensemble_size":ENSEMBLE_SIZE,
                "member_steps":MEMBER_STEPS,
                "member_loss_last":[float(x["loss_last"]) for x in hu_member_meta],
                "member_fit_seconds":[float(x["fit_seconds"]) for x in hu_member_meta],
                "fit_mode":fit_mode,
            }
        report["domains"][domain]={
            "roots":int(plan["roots"]),
            "nodes":int(runtime.bundle.counters["nodes"])-int(before["nodes_before"]),
            "advantage_samples":int(runtime.bundle.adv_mem.seen)-int(before["adv_before"]),
            "strategy_samples":int(runtime.bundle.pol_mem.seen)-int(before["pol_before"]),
            "tree_seconds":float(before["tree_seconds"]),
            "seconds_per_root":float(before["tree_seconds"]/plan["roots"]),
            "root_behavior_ensemble_size":int(before["root_behavior_ensemble_size"]),
            "advantage_fit_seconds":float(fit_seconds),
            "advantage_optimizer_steps_total_this_iteration":int(optimizer_steps),
            "advantage_loss_last":loss_last,
            "sampled_policy":policy_report,
            **extra,
        }

    report["wall_seconds"]=float(time.perf_counter()-iter_started)
    return report,hu_states,hu_member_meta
