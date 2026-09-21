#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import tempfile
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
    p.add_argument("--concurrency",type=int,default=4)
    p.add_argument("--threads-per-member",type=int,default=8)
    p.add_argument("--parent-threads",type=int,default=8)
    return p.parse_args()


def main()->int:
    args=parse_args()
    torch.set_num_threads(args.parent_threads)
    solver=SolverLibrary(args.solver.resolve(strict=True))
    source=args.source_checkpoint.resolve(strict=True)

    seed,config,completed,sampler,runtimes,history,finalized=load_checkpoint(
        source,solver=solver
    )
    if int(completed)!=SOURCE_ITERATION:
        raise RuntimeError(f"expected source {SOURCE_ITERATION}, got {completed}")
    hu=runtimes[DOMAIN_HU]
    hu.session.batch_mode="vectorized"

    before_counters=dict(hu.bundle.counters)
    started=time.perf_counter()
    _models,seq_states,seq_meta,seq_seconds=seq._fit_hu_ensemble(
        hu,config,count_optimizer_steps=False
    )
    seq_wall=float(time.perf_counter()-started)
    hu.bundle.counters.clear()
    hu.bundle.counters.update(before_counters)

    # Reload pristine source runtime so parallel snapshot sees the same reservoir
    # and optimizer-default contract as the sequential reference.
    _,config2,completed2,_,runtimes2,_,_=load_checkpoint(source,solver=solver)
    hu2=runtimes2[DOMAIN_HU]
    hu2.session.batch_mode="vectorized"

    work=args.report.parent/"lt3_hu_parallel_fit_benchmark_work"
    if work.exists():
        import shutil
        shutil.rmtree(work)
    work.mkdir(parents=True,exist_ok=True)

    par_states,par_meta,par_seconds,extra=par.parallel_fit(
        hu2,config2,
        python=Path(sys.executable),
        work_dir=work,
        concurrency=args.concurrency,
        threads_per_member=args.threads_per_member,
    )

    member_rows=[]
    exact=True
    loss_exact=True
    for member in range(par.ENSEMBLE_SIZE):
        state_equal=par.tensor_states_equal(seq_states[member],par_states[member])
        loss_equal=float(seq_meta[member]["loss_last"])==float(par_meta[member]["loss_last"])
        exact=exact and state_equal
        loss_exact=loss_exact and loss_equal
        member_rows.append({
            "member":member,
            "state_exact":bool(state_equal),
            "loss_last_exact":bool(loss_equal),
            "sequential_digest":par.state_digest(seq_states[member]),
            "parallel_digest":par.state_digest(par_states[member]),
            "sequential_fit_seconds":float(seq_meta[member]["fit_seconds"]),
            "parallel_worker_fit_seconds":float(par_meta[member]["fit_seconds"]),
        })

    speedup=seq_wall/par_seconds if par_seconds>0 else 0.0
    verdict=(
        "PASS"
        if exact and loss_exact and speedup>=1.25
        else "FAIL"
    )
    report={
        "schema":"SPINCORE_LT3_HU_ENS8_PARALLEL_FIT_BENCHMARK_V1",
        "verdict":verdict,
        "source_iteration":SOURCE_ITERATION,
        "ensemble_size":par.ENSEMBLE_SIZE,
        "member_steps":par.MEMBER_STEPS,
        "sequential_threads":int(args.parent_threads),
        "parallel_concurrency":int(args.concurrency),
        "parallel_threads_per_member":int(args.threads_per_member),
        "sequential_wall_seconds":float(seq_wall),
        "parallel_wall_seconds":float(par_seconds),
        "speedup":float(speedup),
        "snapshot_seconds":float(extra["snapshot_seconds"]),
        "snapshot_bytes":int(extra["snapshot_bytes"]),
        "all_member_states_exact":bool(exact),
        "all_member_loss_last_exact":bool(loss_exact),
        "members":member_rows,
        "source_checkpoint_read_only":True,
        "training_roots":0,
        "holdout_touched":False,
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== LT3 HU ENS8 PARALLEL FIT BENCHMARK ===")
    print(f"sequential_wall_seconds={seq_wall:.3f}")
    print(f"parallel_wall_seconds={par_seconds:.3f}")
    print(f"speedup={speedup:.3f}x")
    print(f"state_exact={exact} loss_exact={loss_exact}")
    print(f"snapshot_mib={extra['snapshot_bytes']/1024/1024:.2f}")
    print(f"VERDICT={verdict}")
    print("LT3_HU_ENS8_PARALLEL_FIT_BENCHMARK_COMPLETE")
    return 0 if verdict=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
