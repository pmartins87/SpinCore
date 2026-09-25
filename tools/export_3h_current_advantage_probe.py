#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import torch

EXPECTED_SCHEMA="SPINCORE_LEAN_FUNCTIONAL_TRAINING_V1"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
OUT_SCHEMA="SPINCORE_3H_CURRENT_ADVANTAGE_PROBE_V1"

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    cp=args.checkpoint.resolve(strict=True)
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    if payload.get("schema")!=EXPECTED_SCHEMA:
        raise SystemExit("wrong checkpoint schema")
    if payload.get("representation")!=REPRESENTATION:
        raise SystemExit("wrong representation")
    if int(payload.get("completed_iteration",-1))!=10105:
        raise SystemExit("expected iteration 10105")
    d3=(payload.get("domains") or {}).get("THREE_HANDED") or {}
    if "advantage" not in d3:
        raise SystemExit("missing 3H advantage state")
    out={
        "schema":OUT_SCHEMA,
        "representation":REPRESENTATION,
        "completed_iteration":10105,
        "source_checkpoint_sha256":sha256(cp),
        "advantage":d3["advantage"],
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(out,args.out)
    print(f"probe={args.out.resolve()}")
    print(f"probe_bytes={args.out.stat().st_size}")
    print("3H_CURRENT_ADVANTAGE_PROBE_EXPORT_PASS")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
