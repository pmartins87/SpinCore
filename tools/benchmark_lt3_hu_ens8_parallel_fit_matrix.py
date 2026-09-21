#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

from spincore.lean_functional_training import load_checkpoint
from spincore.solver import SolverLibrary
import run_lt2_hu_ens8_online_pilot as seq
import lt3_hu_ens8_parallel_fit as par

SOURCE_ITERATION=8100
DOMAIN_HU="TRUE_HEADS_UP"


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--source-checkpoint",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--parent-threads",type=int,default=8)
    return p.parse_args()


def load_hu(source:Path,solver):
    _,config,completed,_,runtimes,_,_=load_checkpoint(source,solver=solver)
    if int(completed)!=SOURCE_ITERATION:
        raise RuntimeError(f"expected source {SOURCE_ITERATION}, got {completed}")
    hu=runtimes[DOMAIN_HU]
    hu.session.batch_mode="vectorized"
    return config,hu


def main()->int:
    args=parse_args()
    torch.set_num_threads(args.parent_threads)
    solver=SolverLibrary(args.solver.resolve(strict=True))
    source=args.source_checkpoint.resolve(strict=True)

    config,hu=load_hu(source,solver)
    before=dict(hu.bundle.counters)
    started=time.perf_counter()
    _models,seq_states,seq_meta,_seq_internal=seq._fit_hu_ensemble(
        hu,config,count_optimizer_steps=False
    )
    sequential_wall=float(time.perf_counter()-started)
    hu.bundle.counters.clear()
    hu.bundle.counters.update(before)

    candidates=[
        {"name":"2x8","concurrency":2,"threads":8},
        {"name":"4x8","concurrency":4,"threads":8},
        {"name":"4x4","concurrency":4,"threads":4},
        {"name":"8x4","concurrency":8,"threads":4},
        {"name":"8x2","concurrency":8,"threads":2},
    ]

    rows=[]
    best=None
    for cand in candidates:
        config_i,hu_i=load_hu(source,solver)
        work=args.report.parent/f"work_{cand['name']}"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True,exist_ok=True)

        states,meta,fit_wall,extra=par.parallel_fit(
            hu_i,config_i,
            python=Path(sys.executable),
            work_dir=work,
            concurrency=int(cand["concurrency"]),
            threads_per_member=int(cand["threads"]),
        )
        total=float(extra["snapshot_seconds"])+float(fit_wall)
        exact_states=all(
            par.tensor_states_equal(seq_states[m],states[m])
            for m in range(par.ENSEMBLE_SIZE)
        )
        exact_loss=all(
            float(seq_meta[m]["loss_last"])==float(meta[m]["loss_last"])
            for m in range(par.ENSEMBLE_SIZE)
        )
        speedup=sequential_wall/total if total>0 else 0.0
        row={
            "name":cand["name"],
            "concurrency":int(cand["concurrency"]),
            "threads_per_member":int(cand["threads"]),
            "logical_threads_requested":int(cand["concurrency"])*int(cand["threads"]),
            "parallel_fit_wall_seconds":float(fit_wall),
            "snapshot_seconds":float(extra["snapshot_seconds"]),
            "parallel_total_seconds":float(total),
            "speedup":float(speedup),
            "all_member_states_exact":bool(exact_states),
            "all_member_loss_last_exact":bool(exact_loss),
            "snapshot_bytes":int(extra["snapshot_bytes"]),
        }
        rows.append(row)
        if exact_states and exact_loss:
            if best is None or row["parallel_total_seconds"]<best["parallel_total_seconds"]:
                best=row

        shutil.rmtree(work,ignore_errors=True)

    if best is None:
        verdict="FAIL_NO_EXACT_PARALLEL_CONFIGURATION"
    elif float(best["speedup"])<1.25:
        verdict="FAIL_NO_MATERIAL_SPEEDUP"
    else:
        verdict="PASS"

    report={
        "schema":"SPINCORE_LT3_HU_ENS8_PARALLEL_FIT_MATRIX_V1",
        "verdict":verdict,
        "source_iteration":SOURCE_ITERATION,
        "ensemble_size":par.ENSEMBLE_SIZE,
        "member_steps":par.MEMBER_STEPS,
        "sequential_threads":int(args.parent_threads),
        "sequential_wall_seconds":float(sequential_wall),
        "candidates":rows,
        "best_exact_candidate":best,
        "source_checkpoint_read_only":True,
        "training_roots":0,
        "holdout_touched":False,
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== LT3 HU ENS8 PARALLEL FIT MATRIX ===")
    print(f"sequential_wall_seconds={sequential_wall:.3f}")
    for row in rows:
        print(
            f"{row['name']}: total={row['parallel_total_seconds']:.3f}s "
            f"speedup={row['speedup']:.3f}x "
            f"state_exact={row['all_member_states_exact']} "
            f"loss_exact={row['all_member_loss_last_exact']}"
        )
    print("best_exact_candidate="+("NONE" if best is None else best["name"]))
    if best is not None:
        print(f"best_speedup={best['speedup']:.3f}x")
    print(f"VERDICT={verdict}")
    print("LT3_HU_ENS8_PARALLEL_FIT_MATRIX_COMPLETE")
    return 0 if verdict=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
