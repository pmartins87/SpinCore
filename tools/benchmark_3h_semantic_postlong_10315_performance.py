#!/usr/bin/env python3
from __future__ import annotations

"""Target-Ryzen performance/parity gate for the exact postlong-10315 pipeline.

PROJECT_CONTRACT_IDS:
GOV-023,PERF-001,PERF-002,PERF-010,PERF-013,PERF-014,PERF-015,PERF-017,
TRAIN-021,MODEL-021,MODEL-022,MODEL-023,RNG-001,RNG-002,RNG-003,
VALID-002,VALID-020,VALID-021,VALID-022,CKPT-001,ART-001,SRC-003

The gate benchmarks the actual teacher-driven collection kernel, proves that
candidate Torch thread counts preserve the complete sample stream, proves that
the memory-safe strong-only collector preserves the canonical unique-strong
stream, and tests exact process-parallel execution of the four independent
multiseed streams.  Policy/specialist fitting remains on the canonical 8-thread
implementation; its fixed cost is measured separately and included in the
projection.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import statistics
import sys
import threading
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_fresh_semantic_strategy_independent500_10105 as confirm
import audit_3h_semantic_fold_logit_calibrated_specialist_10115 as cal
import audit_3h_semantic_stratified_specialist_multiseed_10315 as multiseed
import build_3h_semantic_postlong_policy_10315 as policy
import postlong_10315_collection as compact
from spincore.solver import SolverLibrary

SCHEMA="SPINCORE_3H_SEMANTIC_POSTLONG_10315_PERFORMANCE_GATE_V1"
BENCH_EPISODES=2000
THREAD_PROFILES=(1,2,4,8)
FULL_SERIAL_EPISODES=58000       # 8k base + 50k evaluation
COMPACT_SERIAL_EPISODES=480000   # 180k augmentation + 300k specialist
MULTISEED_EPISODES_PER_SEED=120000
MIN_MULTI_SPEEDUP=1.20
MIN_PROJECTED_SPEEDUP=1.20
MIN_MEM_AVAILABLE_GIB=8.0
MAX_SWAP_USED_GIB=1.0


def _set_threads(threads:int)->None:
    threads=int(threads)
    os.environ["OMP_NUM_THREADS"]=str(threads)
    os.environ["MKL_NUM_THREADS"]=str(threads)
    os.environ["OPENBLAS_NUM_THREADS"]=str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _load_collection_context(adv_path:str,solver_path:str):
    # Collection depends only on the frozen 10315 semantic-Advantage teacher
    # and solver.  Do NOT deserialize the ~multi-GB 10105 training checkpoint
    # in collection workers: the parent validates that checkpoint once and the
    # collection path never consumes its payload.  This is especially important
    # for process-parallel multiseed validation, where redundant torch.load()
    # copies create a large transient memory fan-out without changing semantics.
    adv,_adv_payload=policy.load_adv(Path(adv_path))
    solver=SolverLibrary(Path(solver_path).resolve(strict=True))
    return adv,solver


def _full_worker(task):
    adv_path,solver_path,seed,episodes,threads=task
    _set_threads(int(threads))
    adv,solver=_load_collection_context(adv_path,solver_path)
    distill.MASTER_SEED=int(seed)
    started=time.perf_counter()
    a,b,stats=distill.collect_fresh(solver,adv,int(episodes))
    wall=float(time.perf_counter()-started)
    all_samples=list(a)+list(b)
    unique=cal.unique_strong(all_samples)
    return {
        "mode":"FULL_CANONICAL",
        "threads":int(threads),
        "seed":int(seed),
        "episodes":int(episodes),
        "wall_seconds":wall,
        "collection":{k:int(v) for k,v in stats.items()},
        "full_digest":compact.sample_digest(all_samples),
        "unique_strong_digest":compact.sample_digest(unique),
        "unique_strong_count":len(unique),
        "maxrss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "pid":int(os.getpid()),
    }


def _compact_worker(task):
    adv_path,solver_path,seed,episodes,threads=task
    _set_threads(int(threads))
    adv,solver=_load_collection_context(adv_path,solver_path)
    started=time.perf_counter()
    a,b,stats=compact.collect_strong_split(
        solver,
        adv,
        int(episodes),
        master_seed=int(seed),
    )
    unique=compact.unique_strong_from_split(a,b)
    wall=float(time.perf_counter()-started)
    return {
        "mode":"COMPACT_STRONG_ONLY",
        "threads":int(threads),
        "seed":int(seed),
        "episodes":int(episodes),
        "wall_seconds":wall,
        "collection":compact.canonical_collection_stats(stats),
        "unique_strong_digest":compact.sample_digest(unique),
        "unique_strong_count":len(unique),
        "maxrss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "pid":int(os.getpid()),
    }


def _run_isolated(fn,task):
    ctx=mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1,mp_context=ctx) as pool:
        return list(pool.map(fn,[task]))[0]


def _serial_four_worker(task):
    adv_path,solver_path,seeds,episodes,threads=task
    _set_threads(int(threads))
    adv,solver=_load_collection_context(adv_path,solver_path)
    rows=[]
    started=time.perf_counter()
    for seed in seeds:
        t0=time.perf_counter()
        a,b,stats=compact.collect_strong_split(
            solver,adv,int(episodes),master_seed=int(seed)
        )
        unique=compact.unique_strong_from_split(a,b)
        rows.append({
            "seed":int(seed),
            "wall_seconds":float(time.perf_counter()-t0),
            "collection":compact.canonical_collection_stats(stats),
            "unique_strong_digest":compact.sample_digest(unique),
            "unique_strong_count":len(unique),
        })
    return {
        "rows":rows,
        "wall_seconds":float(time.perf_counter()-started),
        "maxrss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "pid":int(os.getpid()),
    }


def _fit_worker(task):
    checkpoint,adv_path,solver_path,episodes=task
    _set_threads(8)
    cp=Path(checkpoint).resolve(strict=True)
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    adv,solver=_load_collection_context(adv_path,solver_path)
    d3=(payload.get("domains") or {}).get(distill.DOMAIN) or {}
    cfg=dict(payload.get("config") or {})
    distill.MASTER_SEED=int(policy.BASE_TRAIN_SEED)
    train,hold,_stats=distill.collect_fresh(solver,adv,int(episodes))
    if not train:
        raise RuntimeError("fit benchmark train stream empty")

    _step0_v1,_o0,_step0_sem,_so0=confirm.initial_policy_pair(d3,cfg)
    v1,v1opt,tail,tailopt=confirm.initial_policy_pair(d3,cfg)
    t0=time.perf_counter()
    confirm.train_500(v1,v1opt,tail,tailopt,train,cfg)
    tail_seconds=float(time.perf_counter()-t0)

    _v1b,_v1ob,fullpool,fullopt=confirm.initial_policy_pair(d3,cfg)
    t1=time.perf_counter()
    policy.train_semantic_only(fullpool,fullopt,list(train),cfg)
    fullpool_seconds=float(time.perf_counter()-t1)
    return {
        "episodes":int(episodes),
        "train_samples":len(train),
        "holdout_samples":len(hold),
        "tail_train500_seconds":tail_seconds,
        "fullpool_train500_seconds":fullpool_seconds,
        "fixed_fit_seconds":tail_seconds+fullpool_seconds,
        "maxrss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _meminfo():
    vals={}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key,rest=line.split(":",1)
            vals[key]=int(rest.strip().split()[0])
    except Exception:
        return None
    return {
        "mem_available_kib":vals.get("MemAvailable",0),
        "swap_total_kib":vals.get("SwapTotal",0),
        "swap_free_kib":vals.get("SwapFree",0),
    }


class Monitor:
    def __init__(self):
        self.stop=threading.Event()
        self.mem=[]
        self.swap=[]
        self.thread=None
    def _run(self):
        while not self.stop.wait(0.5):
            m=_meminfo()
            if m:
                self.mem.append(m["mem_available_kib"])
                self.swap.append(m["swap_total_kib"]-m["swap_free_kib"])
    def __enter__(self):
        self._run_once()
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.thread.start()
        return self
    def _run_once(self):
        m=_meminfo()
        if m:
            self.mem.append(m["mem_available_kib"])
            self.swap.append(m["swap_total_kib"]-m["swap_free_kib"])
    def __exit__(self,*_):
        self.stop.set()
        if self.thread:self.thread.join(timeout=3)
        self._run_once()
    def report(self):
        return {
            "min_mem_available_gib":min(self.mem)/(1024**2) if self.mem else None,
            "max_swap_used_gib":max(self.swap)/(1024**2) if self.swap else None,
        }


def atomic_json(payload,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--long-report",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    args=ap.parse_args()

    cp=args.checkpoint.resolve(strict=True)
    adv=args.semantic_advantage.resolve(strict=True)
    solver=args.solver.resolve(strict=True)
    long_report=args.long_report.resolve(strict=True)
    report_path=args.report.resolve()
    progress_path=report_path.with_name("postlong_10315_performance_gate_progress.json")

    def progress(stage:str, **extra):
        payload={
            "schema":"SPINCORE_3H_SEMANTIC_POSTLONG_10315_PERFORMANCE_PROGRESS_V1",
            "stage":str(stage),
            "pid":int(os.getpid()),
            "unix_time":float(time.time()),
            "meminfo":_meminfo(),
        }
        payload.update(extra)
        atomic_json(payload,progress_path)
        print(
            "POSTLONG_PERF_PHASE "
            +json.dumps(
                {
                    "stage":str(stage),
                    "meminfo":payload["meminfo"],
                    **{
                        k:v for k,v in extra.items()
                        if k in (
                            "threads","workers","speedup","exact",
                            "worker_maxrss_kib","checkpoint_size_gib","adv_size_gib",
                        )
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if distill.sha256(cp)!=distill.EXPECTED_SHA:
        raise RuntimeError("source checkpoint SHA mismatch")
    progress(
        "START",
        checkpoint_size_gib=float(cp.stat().st_size)/(1024**3),
        adv_size_gib=float(adv.stat().st_size)/(1024**3),
    )

    r=json.loads(long_report.read_text(encoding="utf-8"))
    if r.get("schema")!="SPINCORE_3H_SEMANTIC_LONG_CONTINUATION_10115_10315_V1":
        raise RuntimeError("wrong long report schema")
    if r.get("status")!="PASS" or int(r.get("completed_iteration",-1))!=10315:
        raise RuntimeError("long report is not completed PASS")
    if not r.get("final_safety_guard_pass"):
        raise RuntimeError("long report safety guard failed")
    if str(r.get("final_ensemble_sha256"))!=distill.sha256(adv):
        raise RuntimeError("teacher ensemble hash/report mismatch")

    task_base=(str(adv),str(solver),int(policy.AUGMENT_SEED),BENCH_EPISODES)
    with Monitor() as monitor:
        progress("REFERENCE_BEGIN",threads=8)
        reference=_run_isolated(_full_worker,task_base+(8,))
        progress("REFERENCE_PASS",threads=8)
        profiles=[]
        for threads in THREAD_PROFILES:
            progress("THREAD_PROFILE_BEGIN",threads=int(threads))
            full=_run_isolated(_full_worker,task_base+(int(threads),))
            compact_row=_run_isolated(_compact_worker,task_base+(int(threads),))
            full_exact=(
                full["full_digest"]==reference["full_digest"]
                and full["collection"]==reference["collection"]
            )
            compact_exact=(
                compact_row["unique_strong_digest"]==full["unique_strong_digest"]
                and compact_row["unique_strong_count"]==full["unique_strong_count"]
                and compact_row["collection"]==full["collection"]
            )
            projected_serial_component=(
                FULL_SERIAL_EPISODES*(full["wall_seconds"]/BENCH_EPISODES)
                +COMPACT_SERIAL_EPISODES*(compact_row["wall_seconds"]/BENCH_EPISODES)
            )
            profiles.append({
                "threads":int(threads),
                "full":full,
                "compact":compact_row,
                "full_stream_exact_to_canonical_8t":bool(full_exact),
                "compact_unique_strong_exact_to_full":bool(compact_exact),
                "projected_non_multiseed_collection_seconds":projected_serial_component,
            })
            progress(
                "THREAD_PROFILE_PASS",
                threads=int(threads),
                full_exact=bool(full_exact),
                compact_exact=bool(compact_exact),
                projected_non_multiseed_seconds=float(projected_serial_component),
                completed_profiles=[
                    {
                        "threads":int(x["threads"]),
                        "full_exact":bool(x["full_stream_exact_to_canonical_8t"]),
                        "compact_exact":bool(x["compact_unique_strong_exact_to_full"]),
                        "projected_non_multiseed_seconds":float(x["projected_non_multiseed_collection_seconds"]),
                    }
                    for x in profiles
                ],
            )
            print(
                "POSTLONG_PERF_THREAD_PROFILE "
                +json.dumps({
                    "threads":threads,
                    "full_seconds":full["wall_seconds"],
                    "compact_seconds":compact_row["wall_seconds"],
                    "full_exact":full_exact,
                    "compact_exact":compact_exact,
                    "projected_non_multiseed_seconds":projected_serial_component,
                },sort_keys=True),
                flush=True,
            )

        eligible=[
            row for row in profiles
            if row["full_stream_exact_to_canonical_8t"]
            and row["compact_unique_strong_exact_to_full"]
        ]
        if not eligible:
            raise RuntimeError("no exact collection-thread profile")
        selected=min(
            eligible,
            key=lambda row:row["projected_non_multiseed_collection_seconds"],
        )
        selected_threads=int(selected["threads"])

        progress("MULTISEED_SERIAL_BEGIN",threads=selected_threads,workers=1)
        serial4=_run_isolated(
            _serial_four_worker,
            (
                str(adv),str(solver),
                tuple(int(x) for x in multiseed.SEEDS),
                BENCH_EPISODES,selected_threads,
            ),
        )

        progress(
            "MULTISEED_SERIAL_PASS",
            threads=selected_threads,
            workers=1,
            serial_wall_seconds=float(serial4["wall_seconds"]),
            worker_maxrss_kib=int(serial4["maxrss_kib"]),
        )

        ctx=mp.get_context("spawn")
        parallel_tasks=[
            (
                str(adv),str(solver),int(seed),
                BENCH_EPISODES,selected_threads,
            )
            for seed in multiseed.SEEDS
        ]
        progress("MULTISEED_PARALLEL_BEGIN",threads=selected_threads,workers=4)
        t0=time.perf_counter()
        with ProcessPoolExecutor(max_workers=4,mp_context=ctx) as pool:
            parallel_rows=list(pool.map(_compact_worker,parallel_tasks,chunksize=1))
        parallel_wall=float(time.perf_counter()-t0)
        parallel_rows.sort(key=lambda x:int(x["seed"]))

        serial_by={int(x["seed"]):x for x in serial4["rows"]}
        multi_exact=True
        for row in parallel_rows:
            src=serial_by[int(row["seed"])]
            if (
                row["unique_strong_digest"]!=src["unique_strong_digest"]
                or row["unique_strong_count"]!=src["unique_strong_count"]
                or row["collection"]!=src["collection"]
            ):
                multi_exact=False
                break
        multi_speedup=serial4["wall_seconds"]/parallel_wall if parallel_wall>0 else 0.0
        progress(
            "MULTISEED_PARALLEL_PASS",
            threads=selected_threads,
            workers=4,
            speedup=float(multi_speedup),
            exact=bool(multi_exact),
            parallel_wall_seconds=float(parallel_wall),
        )

        progress("FIT_BENCHMARK_BEGIN",threads=8)
        fit=_run_isolated(
            _fit_worker,
            (str(cp),str(adv),str(solver),BENCH_EPISODES),
        )
        progress(
            "FIT_BENCHMARK_PASS",
            threads=8,
            fixed_fit_seconds=float(fit["fixed_fit_seconds"]),
            fit_maxrss_kib=int(fit["maxrss_kib"]),
        )
    resources=monitor.report()

    baseline_collection_seconds=(
        (FULL_SERIAL_EPISODES+COMPACT_SERIAL_EPISODES+4*MULTISEED_EPISODES_PER_SEED)
        *(reference["wall_seconds"]/BENCH_EPISODES)
    )
    candidate_nonmulti=float(selected["projected_non_multiseed_collection_seconds"])
    candidate_multi=parallel_wall*(MULTISEED_EPISODES_PER_SEED/BENCH_EPISODES)
    fixed_fit=float(fit["fixed_fit_seconds"])
    # Specialist 25-step fit and final metric/report serialization are unchanged.
    # Add a conservative 25% fixed-overhead reserve so the speedup claim is not
    # driven by omitting downstream noncollection work.
    fixed_overhead_conservative=1.25*fixed_fit
    projected_baseline=baseline_collection_seconds+fixed_overhead_conservative
    projected_candidate=candidate_nonmulti+candidate_multi+fixed_overhead_conservative
    projected_speedup=projected_baseline/projected_candidate if projected_candidate>0 else 0.0

    criteria={
        "selected_collection_profile_exact":bool(
            selected["full_stream_exact_to_canonical_8t"]
            and selected["compact_unique_strong_exact_to_full"]
        ),
        "parallel_four_seed_exact":bool(multi_exact),
        "parallel_four_seed_speedup_at_least_1_20x":bool(multi_speedup>=MIN_MULTI_SPEEDUP),
        "projected_end_to_end_speedup_at_least_1_20x":bool(projected_speedup>=MIN_PROJECTED_SPEEDUP),
        "memory_headroom_at_least_8gib":bool(
            resources["min_mem_available_gib"] is not None
            and resources["min_mem_available_gib"]>=MIN_MEM_AVAILABLE_GIB
        ),
        "swap_used_no_more_than_1gib":bool(
            resources["max_swap_used_gib"] is not None
            and resources["max_swap_used_gib"]<=MAX_SWAP_USED_GIB
        ),
    }
    passed=all(criteria.values())

    payload={
        "schema":SCHEMA,
        "status":"PASS" if passed else "FAIL",
        "long_report_sha256":distill.sha256(long_report),
        "teacher_ensemble_sha256":distill.sha256(adv),
        "benchmark_episodes":BENCH_EPISODES,
        "canonical_reference_threads":8,
        "thread_profiles":profiles,
        "selected_collection_threads":selected_threads,
        "multiseed":{
            "workers":4,
            "threads_per_worker":selected_threads,
            "declared_threads":4*selected_threads,
            "serial_wall_seconds":serial4["wall_seconds"],
            "parallel_wall_seconds":parallel_wall,
            "speedup":multi_speedup,
            "exact_parity":multi_exact,
            "serial_rows":serial4["rows"],
            "parallel_rows":parallel_rows,
        },
        "fixed_fit_benchmark":fit,
        "projection":{
            "current_serial_full_collector_seconds":baseline_collection_seconds,
            "candidate_non_multiseed_collection_seconds":candidate_nonmulti,
            "candidate_multiseed_collection_seconds":candidate_multi,
            "conservative_fixed_noncollection_seconds":fixed_overhead_conservative,
            "projected_current_total_seconds":projected_baseline,
            "projected_candidate_total_seconds":projected_candidate,
            "projected_end_to_end_speedup":projected_speedup,
        },
        "resources":resources,
        "criteria":criteria,
        "interpretation":(
            "PASS authorizes promoting the exact postlong-10315 stage manifest "
            "to READY with the selected collection thread count, canonical "
            "8-thread fitting, memory-safe strong-only augmentation/specialist "
            "collection, and four-process independent multiseed validation. "
            "It does not authorize any policy promotion by itself."
        ),
    }
    atomic_json(payload,report_path)
    progress(
        "COMPLETE_PASS" if passed else "COMPLETE_FAIL",
        selected_collection_threads=int(selected_threads),
        multiseed_speedup=float(multi_speedup),
        projected_end_to_end_speedup=float(projected_speedup),
        criteria=criteria,
        resources=resources,
        report=str(report_path),
    )
    print(
        "POSTLONG_10315_PERFORMANCE_GATE_PASS"
        if passed else
        "POSTLONG_10315_PERFORMANCE_GATE_FAIL",
        flush=True,
    )
    print("selected_collection_threads="+str(selected_threads))
    print("multiseed_speedup="+str(multi_speedup))
    print("projected_end_to_end_speedup="+str(projected_speedup))
    print("report="+str(report_path))
    return 0 if passed else 5


if __name__=="__main__":
    raise SystemExit(main())
