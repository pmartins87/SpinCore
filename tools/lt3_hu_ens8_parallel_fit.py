#!/usr/bin/env python3
from __future__ import annotations

"""Memory-safe process-parallel HU ENS8 fresh fitting.

The original draft copied the full 2M-sample Python reservoir into every
subprocess.  That is not acceptable on the 64-GB Ryzen / WSL memory envelope.
This implementation instead builds one compact, persistent mmap mirror of the
HU Advantage reservoir.  Worker processes open the mirror read-only and sample
indices with the exact historical Python-random stream.

Important contracts:
- same ENS8_A member init/batch seeds;
- same 400 optimizer steps per member;
- same batch size and sample order;
- same vectorized SPNNIV1 tensor construction semantics;
- no mutation of source checkpoints;
- process workers never deserialize the full Python reservoir.

For a long run the mirror is built once, then updated only at reservoir slots
actually replaced by new samples.  The benchmark uses the same infrastructure.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import resource
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_lt2_stage_a_b_first_divergence as fd
from spincore_nn.action_models import make_advantage_action_model
from spincore_nn.training import train_step

REPRESENTATION="C0_V1_FROZEN_CONTROL"
ENSEMBLE_SIZE=8
MEMBER_STEPS=400
PACKED_SCHEMA="SPINCORE_LT3_PACKED_ADV_RESERVOIR_V1"

_PACKED=None


def member_seeds(member:int)->tuple[int,int]:
    if not 0<=int(member)<ENSEMBLE_SIZE:
        raise ValueError("ensemble member out of range")
    init_seed=fd._mix64(20260920,int(member),0xE115E)&0x7FFFFFFF
    batch_seed=fd._mix64(20260920,int(member),0xEBA7C4)&0x7FFFFFFF
    return int(init_seed),int(batch_seed)


def _clone_state(model)->dict[str,torch.Tensor]:
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def _manifest_files(root:Path)->dict[str,Path]:
    return {
        "observations":root/"observations.u8",
        "legal":root/"legal.u8",
        "targets":root/"targets.f32",
        "weights":root/"weights.f32",
    }


class PackedAdvantageReservoir:
    """Compact mmap mirror of the authoritative Python reservoir."""

    def __init__(self,manifest_path:Path,*,mode:str="r"):
        self.manifest_path=Path(manifest_path).resolve()
        meta=json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if meta.get("schema")!=PACKED_SCHEMA:
            raise RuntimeError("wrong packed-reservoir schema")
        self.meta=meta
        self.count=int(meta["count"])
        self.root=self.manifest_path.parent
        files=_manifest_files(self.root)
        self.observations=np.memmap(
            files["observations"],dtype=np.uint8,mode=mode,shape=(self.count,126)
        )
        self.legal=np.memmap(
            files["legal"],dtype=np.uint8,mode=mode,shape=(self.count,10)
        )
        self.targets=np.memmap(
            files["targets"],dtype=np.float32,mode=mode,shape=(self.count,10)
        )
        self.weights=np.memmap(
            files["weights"],dtype=np.float32,mode=mode,shape=(self.count,)
        )

    @classmethod
    def build(cls,memory,root:Path,*,chunk_size:int=8192):
        root=Path(root).resolve()
        root.mkdir(parents=True,exist_ok=True)
        count=len(memory.items)
        if count<=0:
            raise ValueError("cannot pack empty reservoir")
        files=_manifest_files(root)
        obs=np.memmap(files["observations"],dtype=np.uint8,mode="w+",shape=(count,126))
        legal=np.memmap(files["legal"],dtype=np.uint8,mode="w+",shape=(count,10))
        targets=np.memmap(files["targets"],dtype=np.float32,mode="w+",shape=(count,10))
        weights=np.memmap(files["weights"],dtype=np.float32,mode="w+",shape=(count,))

        started=time.perf_counter()
        for start in range(0,count,int(chunk_size)):
            end=min(count,start+int(chunk_size))
            chunk=memory.items[start:end]
            observations=[s.observation for s in chunk]
            if any(len(x)!=126 or x[:8]!=b"SPNNIV1\x00" for x in observations):
                raise RuntimeError("bad SPNNIV1 observation in reservoir")
            raw=np.frombuffer(b"".join(observations),dtype=np.uint8).reshape(-1,126)
            obs[start:end]=raw
            legal[start:end]=np.asarray([s.legal for s in chunk],dtype=np.uint8)
            targets[start:end]=np.asarray([s.target for s in chunk],dtype=np.float32)
            weights[start:end]=np.asarray([s.weight for s in chunk],dtype=np.float32)

        for mm in (obs,legal,targets,weights):
            mm.flush()
        elapsed=float(time.perf_counter()-started)
        del obs,legal,targets,weights

        meta={
            "schema":PACKED_SCHEMA,
            "count":int(count),
            "capacity":int(memory.capacity),
            "seen":int(memory.seen),
            "observation_bytes":126,
            "action_slots":10,
            "build_seconds":elapsed,
            "files":{k:v.name for k,v in files.items()},
            "bytes":int(sum(v.stat().st_size for v in files.values())),
        }
        manifest=root/"manifest.json"
        manifest.write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        return cls(manifest,mode="r+"),meta

    def _write_row(self,index:int,sample)->None:
        i=int(index)
        if not 0<=i<self.count:
            raise IndexError("packed reservoir index out of range")
        obs=sample.observation
        if len(obs)!=126 or obs[:8]!=b"SPNNIV1\x00":
            raise ValueError("bad SPNNIV1 observation")
        self.observations[i]=np.frombuffer(obs,dtype=np.uint8)
        self.legal[i]=np.asarray(sample.legal,dtype=np.uint8)
        self.targets[i]=np.asarray(sample.target,dtype=np.float32)
        self.weights[i]=np.float32(sample.weight)

    def update(self,index:int,sample)->None:
        self._write_row(index,sample)

    def flush(self)->None:
        for mm in (self.observations,self.legal,self.targets,self.weights):
            mm.flush()

    def close(self)->None:
        self.flush()
        del self.observations,self.legal,self.targets,self.weights

    def batch(self,indices:list[int]):
        idx=np.asarray(indices,dtype=np.int64)
        raw=np.asarray(self.observations[idx],dtype=np.uint8)
        if np.any(raw[:,93]>32):
            raise ValueError("bad history length in packed reservoir")

        numeric=raw[:,15:79].copy().view("<f4").reshape(-1,16)
        batch={
            "cards":torch.from_numpy(np.array(raw[:,8:15],dtype=np.int64,copy=True,order="C")),
            "numeric":torch.from_numpy(np.array(numeric,dtype=np.float32,copy=True,order="C")),
            "categorical":torch.from_numpy(np.array(raw[:,79:87],dtype=np.int64,copy=True,order="C")),
            "legal":torch.from_numpy(np.array(self.legal[idx],dtype=np.bool_,copy=True,order="C")),
            "history_len":torch.from_numpy(np.array(raw[:,93],dtype=np.int64,copy=True,order="C")),
            "history":torch.from_numpy(np.array(raw[:,94:126],dtype=np.int64,copy=True,order="C")),
        }
        target=torch.from_numpy(np.array(self.targets[idx],dtype=np.float32,copy=True,order="C"))
        weights=torch.from_numpy(np.array(self.weights[idx],dtype=np.float32,copy=True,order="C"))
        return batch,target,weights


def _worker_init(manifest_path:str,threads:int)->None:
    global _PACKED
    threads=int(threads)
    os.environ["OMP_NUM_THREADS"]=str(threads)
    os.environ["MKL_NUM_THREADS"]=str(threads)
    os.environ["OPENBLAS_NUM_THREADS"]=str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _PACKED=PackedAdvantageReservoir(Path(manifest_path),mode="r")


def _worker_ping()->dict[str,int]:
    if _PACKED is None:
        raise RuntimeError("packed worker not initialized")
    return {
        "pid":int(os.getpid()),
        "count":int(_PACKED.count),
        "maxrss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _worker_fit(task:tuple[int,dict[str,Any]])->dict[str,Any]:
    if _PACKED is None:
        raise RuntimeError("packed worker not initialized")
    member,contract=task
    member=int(member)
    init_seed,batch_seed=member_seeds(member)

    cfg,model=make_advantage_action_model(
        REPRESENTATION,device="cpu",seed=int(init_seed)
    )
    if cfg.to_dict()!=dict(contract["model_config"]):
        raise RuntimeError("parallel fit model config drift")

    defaults=dict(contract["optimizer_defaults"])
    defaults["lr"]=float(contract["learning_rate"])
    optimizer=torch.optim.Adam(model.parameters(),**defaults)

    rng=random.Random(int(batch_seed))
    batch_size=min(int(contract["batch_size"]),int(_PACKED.count))
    losses=[]
    started=time.perf_counter()
    population=range(int(_PACKED.count))
    for _ in range(MEMBER_STEPS):
        indices=rng.sample(population,batch_size)
        batch,target,weights=_PACKED.batch(indices)
        losses.append(train_step(model,optimizer,batch,target,weights,"advantage"))

    return {
        "member":member,
        "init_seed":int(init_seed),
        "batch_seed":int(batch_seed),
        "steps":MEMBER_STEPS,
        "fit_seconds":float(time.perf_counter()-started),
        "losses":[float(x) for x in losses],
        "loss_last":float(losses[-1]),
        "state":_clone_state(model),
        "maxrss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "pid":int(os.getpid()),
    }


class ParallelEnsembleFitter:
    """Persistent worker pool; safe to reuse across training iterations."""

    def __init__(
        self,
        *,
        manifest_path:Path,
        contract:dict[str,Any],
        concurrency:int,
        threads_per_member:int,
    ):
        if int(concurrency)<=0 or int(threads_per_member)<=0:
            raise ValueError("positive concurrency/threads required")
        self.manifest_path=Path(manifest_path).resolve()
        self.contract=dict(contract)
        self.concurrency=int(concurrency)
        self.threads_per_member=int(threads_per_member)

        ctx=mp.get_context("spawn")
        started=time.perf_counter()
        self.pool=ProcessPoolExecutor(
            max_workers=self.concurrency,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(str(self.manifest_path),self.threads_per_member),
        )
        # Force all workers to spawn before timing a fit.
        pings=list(self.pool.map(lambda_placeholder, range(self.concurrency)))
        self.startup_seconds=float(time.perf_counter()-started)
        self.worker_pings=pings

    def fit(self):
        started=time.perf_counter()
        tasks=[(member,self.contract) for member in range(ENSEMBLE_SIZE)]
        results=list(self.pool.map(_worker_fit,tasks,chunksize=1))
        wall=float(time.perf_counter()-started)
        results.sort(key=lambda x:int(x["member"]))
        states=tuple(r["state"] for r in results)
        meta=[
            {
                "member":int(r["member"]),
                "init_seed":int(r["init_seed"]),
                "batch_seed":int(r["batch_seed"]),
                "steps":int(r["steps"]),
                "fit_seconds":float(r["fit_seconds"]),
                "loss_last":float(r["loss_last"]),
                "maxrss_kib":int(r["maxrss_kib"]),
                "pid":int(r["pid"]),
            }
            for r in results
        ]
        return states,meta,wall

    def close(self)->None:
        self.pool.shutdown(wait=True,cancel_futures=False)

    def __enter__(self):
        return self

    def __exit__(self,*_exc):
        self.close()


def lambda_placeholder(_x):
    return _worker_ping()


def make_fit_contract(runtime,config)->dict[str,Any]:
    return {
        "model_config":runtime.bundle.config.to_dict(),
        "optimizer_defaults":dict(runtime.bundle.adv_opt.defaults),
        "learning_rate":float(config.learning_rate),
        "batch_size":int(config.batch_size),
        "member_steps":MEMBER_STEPS,
    }


def install_parallel_result(runtime,states,*,config)->None:
    """Install the same authoritative post-fit state as sequential ENS8.

    Sequential ENS8 leaves member 7 in bundle.advantage and counts eight resets
    plus 8x400 optimizer steps.  The full current behavior is installed by the
    caller from all eight returned member states.
    """
    last=ENSEMBLE_SIZE-1
    init_seed,_=member_seeds(last)
    runtime.session.reset_advantage_network(
        init_seed=int(init_seed),
        lr=float(config.learning_rate),
    )
    runtime.bundle.advantage.load_state_dict(states[last])
    runtime.bundle.counters["advantage_resets"]+=ENSEMBLE_SIZE-1
    runtime.bundle.counters["adv_optimizer_steps"]+=ENSEMBLE_SIZE*MEMBER_STEPS
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
    p.add_argument("--manifest",type=Path)
    p.add_argument("--member",type=int)
    p.add_argument("--threads",type=int,default=8)
    return p.parse_args()


def main()->int:
    # Kept only as a small manual worker smoke entry point.
    args=parse_args()
    if args.manifest is None or args.member is None:
        raise SystemExit("manual worker requires --manifest --member")
    _worker_init(str(args.manifest),args.threads)
    contract=json.loads((args.manifest.parent/"fit_contract.json").read_text())
    out=_worker_fit((args.member,contract))
    print(json.dumps({
        "member":out["member"],
        "loss_last":out["loss_last"],
        "fit_seconds":out["fit_seconds"],
        "digest":state_digest(out["state"]),
    },sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
