#!/usr/bin/env python3
from __future__ import annotations

"""Exact-parity process-parallel four-seed specialist confirmation @10315.

PROJECT_CONTRACT_IDS:
PERF-001,PERF-002,PERF-010,PERF-013,PERF-014,PERF-015,PERF-017,
TRAIN-021,MODEL-021,MODEL-022,MODEL-023,RNG-001,RNG-002,RNG-003,
VALID-020,VALID-021,VALID-022,CKPT-001,ART-001

Each seed stream is scientifically independent and already has a frozen seed.
The candidate executes those four streams concurrently, preserving the exact
per-seed sampler/action RNG semantics.  Workers retain only strong samples via
postlong_10315_collection, then the parent concatenates results in canonical
seed order and applies the unchanged pooled metrics/criteria from the serial
reference implementation.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_semantic_stratified_specialist_multiseed_10315 as serial
import postlong_10315_collection as compact
from spincore.solver import SolverLibrary


def _worker(task):
    (
        seed_index,seed,episodes,threads,
        semantic_advantage,specialist,solver_path,
    )=task
    os.environ["OMP_NUM_THREADS"]=str(int(threads))
    os.environ["MKL_NUM_THREADS"]=str(int(threads))
    os.environ["OPENBLAS_NUM_THREADS"]=str(int(threads))
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    adv=serial.load_adv(Path(semantic_advantage))
    base,spec_model,source=serial.load_specialist(Path(specialist))
    solver=SolverLibrary(Path(solver_path).resolve(strict=True))

    started=time.perf_counter()
    train,strong_hold,collection=compact.collect_strong_split(
        solver,
        adv,
        int(episodes),
        master_seed=int(seed),
        progress_prefix=f"MULTISEED_PARALLEL_SEED_{int(seed_index)}",
    )
    samples=compact.unique_strong_from_split(train,strong_hold)
    if int(episodes)==int(serial.EPISODES_PER_SEED) and len(samples)<600:
        raise RuntimeError(
            f"seed {seed_index} unique strong sample too small: {len(samples)}"
        )
    base_m,spec_m=serial.metrics_for(base,spec_model,samples)
    row={
        "seed_index":int(seed_index),
        "seed":int(seed),
        "episodes":int(episodes),
        "collection":{
            "episodes":int(collection["episodes"]),
            "train_episodes":int(collection["train_episodes"]),
            "holdout_episodes":int(collection["holdout_episodes"]),
            "decisions":int(collection["decisions"]),
            "train_samples":int(collection["train_samples"]),
            "holdout_samples":int(collection["holdout_samples"]),
        },
        "unique_strong_samples":len(samples),
        "base":base_m,
        "specialist":spec_m,
        "effect_sizes":serial.effect(base_m,spec_m),
    }
    return {
        "row":row,
        "samples":samples,
        "strong_digest":compact.sample_digest(samples),
        "wall_seconds":float(time.perf_counter()-started),
        "pid":int(os.getpid()),
        "maxrss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "source_selected_mode":source.get("selected_training_mode"),
        "source_selected_steps":int(source["selected_steps"]),
    }


def run_parallel(
    *,
    checkpoint:Path,
    semantic_advantage:Path,
    specialist:Path,
    solver:Path,
    episodes_per_seed:int,
    workers:int,
    threads_per_worker:int,
):
    if int(workers)<=0 or int(threads_per_worker)<=0:
        raise ValueError("positive workers/threads required")
    if int(workers)*int(threads_per_worker)>32:
        raise ValueError("postlong multiseed refuses >32 declared threads")
    tasks=[
        (
            index,int(seed),int(episodes_per_seed),int(threads_per_worker),
            str(semantic_advantage),str(specialist),str(solver),
        )
        for index,seed in enumerate(serial.SEEDS,1)
    ]
    ctx=mp.get_context("spawn")
    started=time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=ctx,
    ) as pool:
        results=list(pool.map(_worker,tasks,chunksize=1))
    wall=float(time.perf_counter()-started)
    results.sort(key=lambda x:int(x["row"]["seed_index"]))
    return results,wall


def build_report(results, *, episodes_per_seed:int, execution:dict):
    seed_rows=[x["row"] for x in results]
    pooled=[]
    for x in results:
        pooled.extend(x["samples"])

    if int(episodes_per_seed)!=int(serial.EPISODES_PER_SEED):
        return {
            "benchmark_only":True,
            "seed_rows":seed_rows,
            "strong_digests":[x["strong_digest"] for x in results],
            "pooled_unique_strong_observations":len(pooled),
            "execution":execution,
        }

    base_pool,spec_pool=serial.metrics_for(
        # base/specialist are not returned from workers; load once in parent
        # before calling this function in production main.
        execution["_base_model"],
        execution["_specialist_model"],
        pooled,
    )
    eff=serial.effect(base_pool,spec_pool)

    per_seed_sample_ok=all(
        r["unique_strong_samples"]>=600
        and int(r["specialist"]["high_target_count"])>=20
        for r in seed_rows
    )
    per_seed_no_catastrophic_high_bias=all(
        float(r["effect_sizes"]["high_target_bias_delta"])<=0.05
        for r in seed_rows
    )
    per_seed_no_ce_regression=all(
        float(r["effect_sizes"]["ce_ratio"])<=1.0
        for r in seed_rows
    )
    criteria={
        "four_new_seed_streams_completed":len(seed_rows)==4,
        "each_seed_has_at_least_600_unique_strong_and_20_high_target":
            per_seed_sample_ok,
        "pooled_has_at_least_2400_strong_states":len(pooled)>=2400,
        "pooled_has_at_least_100_high_target_states":
            int(spec_pool["high_target_count"])>=100,
        "pooled_ce_at_least_10pct_better_than_fullpool":
            float(eff["ce_ratio"])<=0.90,
        "pooled_tv_at_least_10pct_better_than_fullpool":
            float(eff["tv_ratio"])<=0.90,
        "pooled_fold_bias_at_least_30pct_better":
            float(eff["fold_bias_ratio"])<=0.70,
        "pooled_low_target_mean_at_least_30pct_better":
            float(eff["low_mean_ratio"])<=0.70,
        "pooled_low_target_p95_at_least_30pct_better":
            float(eff["low_p95_ratio"])<=0.70,
        "pooled_straight_flush_bias_at_least_30pct_better":
            float(eff["straight_flush_bias_ratio"])<=0.70,
        "pooled_river_bias_at_least_30pct_better":
            float(eff["river_bias_ratio"])<=0.70,
        "pooled_high_target_bias_not_worse_by_over_1pp":
            float(eff["high_target_bias_delta"])<=0.01,
        "no_single_seed_high_target_bias_regression_over_5pp":
            per_seed_no_catastrophic_high_bias,
        "no_single_seed_ce_regression":
            per_seed_no_ce_regression,
    }
    passed=all(criteria.values())

    src_mode=results[0]["source_selected_mode"]
    src_steps=int(results[0]["source_selected_steps"])
    if any(x["source_selected_mode"]!=src_mode for x in results):
        raise RuntimeError("worker specialist mode disagreement")
    if any(int(x["source_selected_steps"])!=src_steps for x in results):
        raise RuntimeError("worker specialist step disagreement")

    clean_execution={k:v for k,v in execution.items() if not k.startswith("_")}
    return {
        "schema":"SPINCORE_3H_SEMANTIC_STRATIFIED_SPECIALIST_MULTISEED_CONFIRMATION_10315_V1",
        "scope":"RESEARCH_ONLY_REBUILT_10315_MODEL_FOUR_NEW_SEEDS_NO_POSTHOC_SELECTION",
        "source_checkpoint_sha256":serial.EXPECTED_SHA,
        "semantic_completed_iteration":serial.FINAL_ITERATION,
        "specialist_schema":serial.SPECIALIST_SCHEMA,
        "specialist_selected_training_mode":src_mode,
        "specialist_selected_steps":src_steps,
        "episodes_per_seed":int(episodes_per_seed),
        "seeds":[int(x) for x in serial.SEEDS],
        "seed_rows":seed_rows,
        "pooled_unique_strong_observations":len(pooled),
        "pooled_fullpool":base_pool,
        "pooled_specialist":spec_pool,
        "pooled_effect_sizes":eff,
        "precommitted_criteria":criteria,
        "semantic_stratified_specialist_multiseed_pass":passed,
        "execution":clean_execution,
        "interpretation":(
            "PASS means the rebuilt 10315 target-stratified specialist reproduces "
            "its broad gains across four entirely new deterministic streams and "
            "the rare legitimate high-Fold issue does not reproduce materially "
            "in pooled evidence. Parallel execution changes only independent "
            "seed scheduling; per-seed streams and canonical pooled ordering are "
            "preserved. PASS authorizes the precommitted fresh-seed DC1 5k "
            "comparison only; no production/DC2/canonical-strength claim."
        ),
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage",type=Path,required=True)
    ap.add_argument("--stratified-specialist",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--threads-per-worker",type=int,default=8)
    ap.add_argument("--episodes-per-seed",type=int,default=serial.EPISODES_PER_SEED)
    args=ap.parse_args()

    if int(args.episodes_per_seed)!=int(serial.EPISODES_PER_SEED):
        raise SystemExit(
            "Production audit requires the frozen 120000 episodes/seed; "
            "reduced budgets are reserved for the dedicated performance gate."
        )

    cp=args.checkpoint.resolve(strict=True)
    adv=args.semantic_advantage.resolve(strict=True)
    specialist=args.stratified_specialist.resolve(strict=True)
    solver=args.solver.resolve(strict=True)

    if serial.distill.sha256(cp)!=serial.EXPECTED_SHA:
        raise RuntimeError("source checkpoint SHA mismatch")
    # Integrity is established by the frozen checkpoint SHA.  The multiseed
    # evaluator does not consume the training-checkpoint payload, so do not
    # deserialize this multi-GB file in the parent or any spawned worker.
    base,spec_model,source=serial.load_specialist(specialist)

    results,wall=run_parallel(
        checkpoint=cp,
        semantic_advantage=adv,
        specialist=specialist,
        solver=solver,
        episodes_per_seed=int(args.episodes_per_seed),
        workers=int(args.workers),
        threads_per_worker=int(args.threads_per_worker),
    )
    execution={
        "mode":"PROCESS_PARALLEL_INDEPENDENT_SEEDS",
        "workers":int(args.workers),
        "threads_per_worker":int(args.threads_per_worker),
        "declared_threads":int(args.workers)*int(args.threads_per_worker),
        "wall_seconds":wall,
        "worker_maxrss_kib":[int(x["maxrss_kib"]) for x in results],
        "worker_pids":[int(x["pid"]) for x in results],
        "strong_digests":[x["strong_digest"] for x in results],
        "_base_model":base,
        "_specialist_model":spec_model,
    }
    report=build_report(
        results,
        episodes_per_seed=int(args.episodes_per_seed),
        execution=execution,
    )
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )

    for row in report["seed_rows"]:
        print(
            "MULTISEED_SPECIALIST "
            +json.dumps({
                "seed_index":row["seed_index"],
                "strong":row["unique_strong_samples"],
                "high":row["specialist"]["high_target_count"],
                "ce_ratio":row["effect_sizes"]["ce_ratio"],
                "tv_ratio":row["effect_sizes"]["tv_ratio"],
                "high_bias_delta":row["effect_sizes"]["high_target_bias_delta"],
            },sort_keys=True),
            flush=True,
        )
    print("pooled_fullpool="+json.dumps(report["pooled_fullpool"],sort_keys=True))
    print("pooled_specialist="+json.dumps(report["pooled_specialist"],sort_keys=True))
    print("pooled_effect_sizes="+json.dumps(report["pooled_effect_sizes"],sort_keys=True))
    print("criteria="+json.dumps(report["precommitted_criteria"],sort_keys=True))
    print(
        "semantic_stratified_specialist_multiseed_pass="
        +str(report["semantic_stratified_specialist_multiseed_pass"])
    )
    print(f"report={args.report.resolve()}")
    print("SEMANTIC_STRATIFIED_SPECIALIST_MULTISEED_CONFIRMATION_COMPLETE")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
