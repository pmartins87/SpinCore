#!/usr/bin/env python3
from __future__ import annotations

"""Finalize AveragePolicy on an immutable raw LT3 milestone copy.

The source checkpoint is read-only. This tool loads the raw training state,
fits AveragePolicy exactly once on the in-memory copy, and writes a separate
finalized checkpoint suitable for LeanFunctionalAgent inference.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))

from spincore.lean_functional_training import finalize, load_checkpoint, save_checkpoint
from spincore.solver import SolverLibrary


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--source-checkpoint",type=Path,required=True)
    p.add_argument("--expected-source-sha256",type=str,required=True)
    p.add_argument("--expected-iteration",type=int,default=8600)
    p.add_argument("--output-checkpoint",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--threads",type=int,default=8)
    return p.parse_args()


def main()->int:
    args=parse_args()
    solver_path=args.solver.resolve(strict=True)
    source=args.source_checkpoint.resolve(strict=True)
    before=sha256(source)
    if before!=args.expected_source_sha256:
        raise RuntimeError(f"raw source hash mismatch: expected={args.expected_source_sha256} actual={before}")

    torch.set_num_threads(int(args.threads))
    solver=SolverLibrary(solver_path)
    seed,config,iteration,sampler,runtimes,history,source_finalized=load_checkpoint(source,solver=solver)
    if int(iteration)!=int(args.expected_iteration):
        raise RuntimeError(f"expected iteration {args.expected_iteration}, got {iteration}")
    if source_finalized:
        raise RuntimeError("source milestone is already finalized; expected immutable raw milestone")

    started=time.perf_counter()
    final=finalize(config=config,runtimes=runtimes)
    finalize_seconds=float(time.perf_counter()-started)

    out=args.output_checkpoint.resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    save_checkpoint(
        out,
        seed=int(seed),
        config=config,
        completed_iteration=int(iteration),
        sampler=sampler,
        runtimes=runtimes,
        history=history,
        finalized=True,
    )

    after=sha256(source)
    if after!=before:
        raise RuntimeError("raw milestone changed during derived finalization")

    _seed,_config,check_iter,_sampler,_runtimes,_history,check_finalized=load_checkpoint(out,solver=solver)
    if int(check_iter)!=int(iteration) or not check_finalized:
        raise RuntimeError("derived finalized checkpoint postvalidation failed")

    payload={
        "schema":"SPINCORE_LT3_DERIVED_MILESTONE_FINALIZATION_V1",
        "status":"PASS",
        "source_checkpoint":str(source),
        "source_checkpoint_sha256":before,
        "source_unchanged":True,
        "source_finalized":False,
        "iteration":int(iteration),
        "output_checkpoint":str(out),
        "output_checkpoint_sha256":sha256(out),
        "output_finalized":True,
        "finalize_seconds":finalize_seconds,
        "final":final,
        "holdout_touched":False,
        "new_training_roots":0,
        "warning":"Derived evaluation artifact only; raw milestone remains canonical and immutable.",
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("LT3_DERIVED_MILESTONE_FINALIZATION_PASS")
    print(f"source_sha256={before}")
    print(f"output_sha256={payload['output_checkpoint_sha256']}")
    print(f"report={args.report.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
