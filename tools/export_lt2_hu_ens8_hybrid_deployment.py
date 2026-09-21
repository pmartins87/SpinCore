#!/usr/bin/env python3
from __future__ import annotations

"""Export the frozen LT2 hybrid deployment bundle.

THREE_HANDED comes from finalized AveragePolicy in iteration-8100 checkpoint.
TRUE_HEADS_UP comes from the mandatory iteration-8100 ENS8 sidecar.
"""

import argparse
import hashlib
from pathlib import Path
import torch

from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_hybrid_deployment_agent import (
    DEPLOYMENT_SCHEMA,
    MODE_AVERAGE_POLICY,
    MODE_HU_ENS8,
    REPRESENTATION,
)

CHECKPOINT_SCHEMA="SPINCORE_LEAN_FUNCTIONAL_TRAINING_V1"
ENSEMBLE_SCHEMA="SPINCORE_LT2_HU_ENS8_CURRENT_STATE_V1"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--ensemble",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()

    checkpoint=args.checkpoint.resolve(strict=True)
    ensemble=args.ensemble.resolve(strict=True)
    cp=torch.load(checkpoint,map_location="cpu",weights_only=False)
    ens=torch.load(ensemble,map_location="cpu",weights_only=False)

    if cp.get("schema")!=CHECKPOINT_SCHEMA:
        raise SystemExit("wrong training checkpoint schema")
    if cp.get("representation")!=REPRESENTATION:
        raise SystemExit("checkpoint representation drift")
    if cp.get("action_candidate")!=FIRST_RELEASE_ACTION_SPEC.candidate_id:
        raise SystemExit("checkpoint action candidate drift")
    if not bool(cp.get("finalized")):
        raise SystemExit("checkpoint must be finalized")
    if int(cp.get("completed_iteration",-1))!=8100:
        raise SystemExit("expected iteration-8100 checkpoint")

    if ens.get("schema")!=ENSEMBLE_SCHEMA:
        raise SystemExit("wrong HU ensemble schema")
    if int(ens.get("completed_iteration",-1))!=8100:
        raise SystemExit("expected iteration-8100 HU ensemble")
    members=list(ens.get("members") or [])
    if int(ens.get("ensemble_size",-1))!=8 or len(members)!=8:
        raise SystemExit("validated candidate requires exactly 8 HU members")

    d3=(cp.get("domains") or {}).get("THREE_HANDED") or {}
    if "policy" not in d3:
        raise SystemExit("checkpoint missing THREE_HANDED policy")

    bundle={
        "schema":DEPLOYMENT_SCHEMA,
        "representation":REPRESENTATION,
        "action_candidate":FIRST_RELEASE_ACTION_SPEC.candidate_id,
        "completed_iteration":8100,
        "source_checkpoint":str(checkpoint),
        "source_checkpoint_sha256":sha256(checkpoint),
        "source_ensemble":str(ensemble),
        "source_ensemble_sha256":sha256(ensemble),
        "candidate":"THREE_HANDED AveragePolicy + TRUE_HEADS_UP current ENS8@8100",
        "holdout_status":"PASS",
        "domains":{
            "THREE_HANDED":{
                "mode":MODE_AVERAGE_POLICY,
                "policy":d3["policy"],
            },
            "TRUE_HEADS_UP":{
                "mode":MODE_HU_ENS8,
                "ensemble_size":8,
                "member_steps":int(ens.get("member_steps",400)),
                "semantics":ens.get("semantics"),
                "members":members,
                "member_meta":ens.get("member_meta"),
            },
        },
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(bundle,args.out)
    print("LT2_HYBRID_DEPLOYMENT_BUNDLE_EXPORTED")
    print(f"out={args.out.resolve()}")
    print(f"out_sha256={sha256(args.out.resolve())}")
    print(f"out_bytes={args.out.stat().st_size}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
