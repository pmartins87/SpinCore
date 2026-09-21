#!/usr/bin/env python3
from __future__ import annotations

"""Export the frozen LT2 hybrid PyTorch deployment bundle to a native C++ format."""

import argparse
import hashlib
import struct
from pathlib import Path

import torch

MAGIC=b"SCLT2D1\x00"
VERSION=1
ACTIONS=10
ENSEMBLE=8
KIND_3H_POLICY=1
KIND_HU_ADVANTAGE=2

EXPECTED_KEYS=(
    "card_emb.weight",
    "cat_emb.weight",
    "hist_emb.weight",
    "gru.weight_ih_l0",
    "gru.weight_hh_l0",
    "gru.bias_ih_l0",
    "gru.bias_hh_l0",
    "body.0.weight",
    "body.0.bias",
    "body.2.weight",
    "body.2.bias",
    "head.weight",
    "head.bias",
)


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def _fixed_ascii(value:str,n:int)->bytes:
    raw=value.encode("ascii")
    if len(raw)>n:
        raise ValueError("metadata string too long")
    return raw+b"\x00"*(n-len(raw))


def _write_model(f,kind:int,index:int,state:dict[str,torch.Tensor])->None:
    keys=tuple(state.keys())
    if keys!=EXPECTED_KEYS:
        raise RuntimeError(f"unexpected model state_dict keys/order: {keys}")
    f.write(struct.pack("<III",int(kind),int(index),len(keys)))
    for name in keys:
        tensor=state[name].detach().cpu().contiguous().to(dtype=torch.float32)
        encoded=name.encode("utf-8")
        if len(encoded)>65535:
            raise RuntimeError("tensor name too long")
        dims=tuple(int(x) for x in tensor.shape)
        f.write(struct.pack("<H",len(encoded)))
        f.write(encoded)
        f.write(struct.pack("<I",len(dims)))
        for d in dims:
            f.write(struct.pack("<I",d))
        f.write(struct.pack("<Q",tensor.numel()))
        f.write(tensor.numpy().tobytes(order="C"))


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()

    source=args.source.resolve(strict=True)
    payload=torch.load(source,map_location="cpu",weights_only=False)
    if payload.get("schema")!="SPINCORE_LT2_HYBRID_DEPLOYMENT_V1":
        raise SystemExit("wrong hybrid deployment schema")
    if int(payload.get("completed_iteration",-1))!=8100:
        raise SystemExit("expected frozen iteration 8100")
    if payload.get("holdout_status")!="PASS":
        raise SystemExit("deployment source lacks holdout PASS")

    d3=(payload.get("domains") or {}).get("THREE_HANDED") or {}
    dhu=(payload.get("domains") or {}).get("TRUE_HEADS_UP") or {}
    if d3.get("mode")!="AVERAGE_POLICY":
        raise SystemExit("3H mode drift")
    if dhu.get("mode")!="ADVANTAGE_ENSEMBLE_RAW_MEAN_RM":
        raise SystemExit("HU mode drift")
    members=list(dhu.get("members") or [])
    if int(dhu.get("ensemble_size",-1))!=ENSEMBLE or len(members)!=ENSEMBLE:
        raise SystemExit("HU ensemble size drift")

    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII",VERSION,ACTIONS,ENSEMBLE,1+ENSEMBLE))
        f.write(_fixed_ascii(sha256(source),64))
        f.write(_fixed_ascii(str(payload["source_checkpoint_sha256"]),64))
        f.write(_fixed_ascii(str(payload["source_ensemble_sha256"]),64))
        _write_model(f,KIND_3H_POLICY,0,d3["policy"])
        for index,state in enumerate(members):
            _write_model(f,KIND_HU_ADVANTAGE,index,state)

    print("LT2_CPP_DEPLOYMENT_EXPORTED")
    print(f"source_sha256={sha256(source)}")
    print(f"out={args.out.resolve()}")
    print(f"out_sha256={sha256(args.out.resolve())}")
    print(f"out_bytes={args.out.stat().st_size}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
