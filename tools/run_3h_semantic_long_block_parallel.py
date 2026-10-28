#!/usr/bin/env python3
from __future__ import annotations

"""Parallel-fit wrapper for generic 3H Semantic ENS8 long blocks.

This wrapper does not redefine the training algorithm. It imports the generic
serial carrier and replaces only pilot.fit_ensemble with the exact-parity
process-parallel fitter already validated for the long semantic 3H lane.
"""

import atexit
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import run_3h_semantic_long_block as serial_long
import run_3h_semantic_online_pilot_10105 as pilot
import lt3_3h_semantic_parallel_fit as parallel_fit

_FITTER=None
_PACKED=None
_PACK_META=None
_BUILD_SECONDS=0.0


def _settings():
    try:
        concurrency=int(os.environ["SPINCORE_SEMANTIC_FIT_CONCURRENCY"])
        threads=int(os.environ["SPINCORE_SEMANTIC_THREADS_PER_MEMBER"])
    except KeyError as exc:
        raise RuntimeError("parallel semantic profile is not frozen in environment") from exc
    pack_dir=Path(os.environ["SPINCORE_SEMANTIC_PACK_DIR"])
    if concurrency<=0 or threads<=0 or concurrency*threads>32:
        raise RuntimeError("invalid frozen semantic parallel profile")
    return concurrency,threads,pack_dir


def _close():
    global _FITTER,_PACKED
    if _FITTER is not None:
        _FITTER.close()
        _FITTER=None
    if _PACKED is not None:
        _PACKED.close()
        _PACKED=None


atexit.register(_close)


def _parallel_fit_ensemble(
    memory,
    sem_rows,
    train_pool,
    member_meta,
    *,
    lr:float,
    batch_size:int,
    iteration:int,
):
    global _FITTER,_PACKED,_PACK_META,_BUILD_SECONDS

    concurrency,threads,pack_dir=_settings()
    if int(pilot.MEMBER_STEPS)!=1600:
        raise RuntimeError("semantic production member-step contract drift")

    if _FITTER is None:
        started=time.perf_counter()
        _PACKED,_PACK_META=parallel_fit.PackedSemanticAdvantageReservoir.build(
            memory,sem_rows,train_pool,pack_dir
        )
        _PACKED.bind_authoritative(
            memory,sem_rows,semantic_fn=pilot.shadow.semantic_vector
        )
        _PACKED.flush()
        contract=parallel_fit.make_fit_contract(
            member_meta,
            learning_rate=float(lr),
            batch_size=int(batch_size),
            member_steps=int(pilot.MEMBER_STEPS),
        )
        _FITTER=parallel_fit.ParallelSemanticEnsembleFitter(
            manifest_path=_PACKED.manifest_path,
            contract=contract,
            concurrency=concurrency,
            threads_per_member=threads,
        )
        _BUILD_SECONDS=float(time.perf_counter()-started)
        print(
            "SEMANTIC_GENERIC_PARALLEL_INIT "
            +json.dumps({
                "concurrency":concurrency,
                "threads_per_member":threads,
                "declared_threads":concurrency*threads,
                "pack_bytes":int(_PACK_META["bytes"]),
                "pack_build_seconds":float(_PACK_META["build_seconds"]),
                "pool_startup_seconds":float(_FITTER.startup_seconds),
                "init_total_seconds":_BUILD_SECONDS,
                "worker_pids":[int(row["pid"]) for row in _FITTER.worker_pings],
            },sort_keys=True),
            flush=True,
        )
    else:
        expected=[
            (int(row["init_seed"]),int(row["batch_seed"])) for row in member_meta
        ]
        actual=[
            (int(row["init_seed"]),int(row["batch_seed"]))
            for row in _FITTER.contract["member_meta"]
        ]
        if actual!=expected:
            raise RuntimeError("semantic member seed contract changed mid-run")
        if float(_FITTER.contract["learning_rate"])!=float(lr):
            raise RuntimeError("semantic learning-rate drift")
        if int(_FITTER.contract["batch_size"])!=int(batch_size):
            raise RuntimeError("semantic batch-size drift")

    _PACKED.flush()
    started=time.perf_counter()
    rows=_FITTER.fit()
    wall=float(time.perf_counter()-started)

    states=[row["state"] for row in rows]
    models=[serial_long.distill.load_semantic_advantage(state) for state in states]
    reports=[]
    for row in rows:
        reports.append({
            "member":int(row["member"]),
            "init_seed":int(row["init_seed"]),
            "batch_seed":int(row["batch_seed"]),
            "steps":int(row["steps"]),
            "loss_last":float(row["loss_last"]),
            "fit_seconds":float(row["fit_seconds"]),
            "pid":int(row["pid"]),
            "maxrss_kib":int(row["maxrss_kib"]),
        })
    print(
        "SEMANTIC_GENERIC_PARALLEL_FIT "
        +json.dumps({
            "iteration":int(iteration),
            "concurrency":int(_FITTER.concurrency),
            "threads_per_member":int(_FITTER.threads_per_member),
            "fit_wall_seconds":wall,
            "member_fit_seconds_sum":float(sum(float(row["fit_seconds"]) for row in rows)),
            "packed_write_updates":int(_PACKED.write_updates),
        },sort_keys=True),
        flush=True,
    )
    return models,states,reports,wall


def main()->int:
    pilot.fit_ensemble=_parallel_fit_ensemble
    try:
        return int(serial_long.main())
    finally:
        _close()


if __name__=="__main__":
    raise SystemExit(main())
