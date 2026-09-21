#!/usr/bin/env python3
from __future__ import annotations

"""Exact-parity / throughput matrix for memory-safe HU ENS8 fitting."""

import argparse
import gc
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
from spincore_nn.reservoir import UniformReservoir
import run_lt2_hu_ens8_online_pilot as seq
import lt3_hu_ens8_parallel_fit as par

SOURCE_ITERATION=8100
DOMAIN_HU="TRUE_HEADS_UP"
DOMAIN_3H="THREE_HANDED"


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--source-checkpoint",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--parent-threads",type=int,default=8)
    return p.parse_args()


def main()->int:
    args=parse_args()
    torch.set_num_threads(int(args.parent_threads))
    solver=SolverLibrary(args.solver.resolve(strict=True))
    source=args.source_checkpoint.resolve(strict=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)

    print("MATRIX_LOAD_SOURCE_BEGIN",flush=True)
    seed,config,completed,sampler,runtimes,history,finalized=load_checkpoint(
        source,solver=solver
    )
    if int(completed)!=SOURCE_ITERATION:
        raise RuntimeError(f"expected source {SOURCE_ITERATION}, got {completed}")

    # Keep only the HU objects required by canonical fresh fitting.  The old
    # matrix accidentally retained/reloaded multiple full 2M-reservoir
    # checkpoints in one process.
    hu=runtimes[DOMAIN_HU]
    hu.session.batch_mode="vectorized"
    del runtimes[DOMAIN_3H]
    del sampler,history
    gc.collect()
    print(
        f"MATRIX_SOURCE_READY hu_adv_len={len(hu.bundle.adv_mem.items)} "
        f"hu_adv_seen={hu.bundle.adv_mem.seen}",
        flush=True,
    )

    packed_dir=args.report.parent/"packed_hu_adv"
    if packed_dir.exists():
        shutil.rmtree(packed_dir)
    print("MATRIX_PACKED_MIRROR_BUILD_BEGIN",flush=True)
    mirror,packed_meta=par.PackedAdvantageReservoir.build(
        hu.bundle.adv_mem,packed_dir
    )
    contract=par.make_fit_contract(hu,config)
    (packed_dir/"fit_contract.json").write_text(
        json.dumps(contract,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(
        f"MATRIX_PACKED_MIRROR_BUILD_PASS seconds={packed_meta['build_seconds']:.3f} "
        f"mib={packed_meta['bytes']/1024/1024:.2f}",
        flush=True,
    )

    print("MATRIX_SEQUENTIAL_REFERENCE_BEGIN",flush=True)
    before=dict(hu.bundle.counters)
    started=time.perf_counter()
    _models,seq_states,seq_meta,_seq_internal=seq._fit_hu_ensemble(
        hu,config,count_optimizer_steps=False
    )
    sequential_wall=float(time.perf_counter()-started)
    hu.bundle.counters.clear()
    hu.bundle.counters.update(before)
    print(f"MATRIX_SEQUENTIAL_REFERENCE_PASS seconds={sequential_wall:.3f}",flush=True)

    # The packed mirror is now authoritative for the benchmark candidates.
    # Drop the huge Python reservoir and the unused HU policy reservoir before
    # spawning any fit workers.
    hu.bundle.adv_mem=UniformReservoir(1,0)
    hu.session.bundle.adv_mem=hu.bundle.adv_mem
    hu.session.collector.advantage_memory=hu.bundle.adv_mem
    hu.bundle.pol_mem=UniformReservoir(1,1)
    hu.session.bundle.pol_mem=hu.bundle.pol_mem
    hu.session.collector.strategy_memory=hu.bundle.pol_mem
    gc.collect()

    candidates=[
        # Same 8 threads/member as the canonical reference: these are the most
        # likely to preserve bit-exact model states.
        {"name":"2x8","concurrency":2,"threads":8},
        {"name":"4x8","concurrency":4,"threads":8},
        {"name":"8x8","concurrency":8,"threads":8},
        # Lower-thread layouts are measured too, but exact state parity remains
        # mandatory before they can be selected.
        {"name":"4x4","concurrency":4,"threads":4},
        {"name":"8x4","concurrency":8,"threads":4},
        {"name":"8x2","concurrency":8,"threads":2},
    ]

    rows=[]
    best=None
    for cand in candidates:
        name=cand["name"]
        print(f"MATRIX_CANDIDATE_BEGIN {name}",flush=True)
        with par.ParallelEnsembleFitter(
            manifest_path=mirror.manifest_path,
            contract=contract,
            concurrency=int(cand["concurrency"]),
            threads_per_member=int(cand["threads"]),
        ) as fitter:
            states,meta,fit_wall,_last_optimizer=fitter.fit()
            startup=float(fitter.startup_seconds)
            pings=list(fitter.worker_pings)

        exact_states=all(
            par.tensor_states_equal(seq_states[m],states[m])
            for m in range(par.ENSEMBLE_SIZE)
        )
        exact_loss=all(
            float(seq_meta[m]["loss_last"])==float(meta[m]["loss_last"])
            for m in range(par.ENSEMBLE_SIZE)
        )
        speedup=sequential_wall/fit_wall if fit_wall>0 else 0.0
        row={
            "name":name,
            "concurrency":int(cand["concurrency"]),
            "threads_per_member":int(cand["threads"]),
            "logical_threads_requested":int(cand["concurrency"])*int(cand["threads"]),
            "pool_startup_seconds_one_time":startup,
            "parallel_fit_wall_seconds":float(fit_wall),
            "speedup_steady_state":float(speedup),
            "all_member_states_exact":bool(exact_states),
            "all_member_loss_last_exact":bool(exact_loss),
            "worker_maxrss_kib_max":max(int(x["maxrss_kib"]) for x in meta),
            "worker_pings":pings,
            "members":[
                {
                    "member":m,
                    "state_exact":bool(par.tensor_states_equal(seq_states[m],states[m])),
                    "loss_last_exact":bool(
                        float(seq_meta[m]["loss_last"])==float(meta[m]["loss_last"])
                    ),
                    "sequential_digest":par.state_digest(seq_states[m]),
                    "parallel_digest":par.state_digest(states[m]),
                    "sequential_loss_last":float(seq_meta[m]["loss_last"]),
                    "parallel_loss_last":float(meta[m]["loss_last"]),
                    "parallel_fit_seconds":float(meta[m]["fit_seconds"]),
                    "worker_maxrss_kib":int(meta[m]["maxrss_kib"]),
                }
                for m in range(par.ENSEMBLE_SIZE)
            ],
        }
        rows.append(row)
        print(
            f"MATRIX_CANDIDATE_RESULT {name} fit={fit_wall:.3f}s "
            f"speedup={speedup:.3f}x state_exact={exact_states} "
            f"loss_exact={exact_loss}",
            flush=True,
        )
        if exact_states and exact_loss:
            if best is None or row["parallel_fit_wall_seconds"]<best["parallel_fit_wall_seconds"]:
                best=row

    if best is None:
        verdict="FAIL_NO_EXACT_PARALLEL_CONFIGURATION"
    elif float(best["speedup_steady_state"])<1.25:
        verdict="FAIL_NO_MATERIAL_SPEEDUP"
    else:
        verdict="PASS"

    report={
        "schema":"SPINCORE_LT3_HU_ENS8_PARALLEL_FIT_MATRIX_V2",
        "verdict":verdict,
        "source_iteration":SOURCE_ITERATION,
        "ensemble_size":par.ENSEMBLE_SIZE,
        "member_steps":par.MEMBER_STEPS,
        "sequential_threads":int(args.parent_threads),
        "sequential_wall_seconds":float(sequential_wall),
        "packed_reservoir":{
            **packed_meta,
            "persistent_mirror":True,
            "per_iteration_full_reservoir_copy":False,
        },
        "candidates":rows,
        "best_exact_candidate":best,
        "source_checkpoint_read_only":True,
        "training_roots":0,
        "holdout_touched":False,
        "old_v1_matrix_status":"SUPERSEDED_MEMORY_UNSAFE",
    }
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    mirror.close()

    print("=== LT3 HU ENS8 PARALLEL FIT MATRIX V2 ===",flush=True)
    print(f"sequential_wall_seconds={sequential_wall:.3f}",flush=True)
    for row in rows:
        print(
            f"{row['name']}: fit={row['parallel_fit_wall_seconds']:.3f}s "
            f"speedup={row['speedup_steady_state']:.3f}x "
            f"state_exact={row['all_member_states_exact']} "
            f"loss_exact={row['all_member_loss_last_exact']}",
            flush=True,
        )
    print("best_exact_candidate="+("NONE" if best is None else best["name"]),flush=True)
    if best is not None:
        print(f"best_speedup={best['speedup_steady_state']:.3f}x",flush=True)
    print(f"VERDICT={verdict}",flush=True)
    print("LT3_HU_ENS8_PARALLEL_FIT_MATRIX_COMPLETE",flush=True)
    return 0 if verdict=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
