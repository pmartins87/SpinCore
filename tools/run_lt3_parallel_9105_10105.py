#!/usr/bin/env python3
from __future__ import annotations

"""LT3 exact-parity 4x8 research continuation from finalized 9105 to 10105.

This block keeps the frozen 9105 checkpoint+ENS8 sidecar read-only, continues
the already validated learning process for exactly 1000 additional iterations,
preserves a raw 9600 milestone, checkpoints every 50 iterations, and finalizes
AveragePolicy only at the final 10105 endpoint. It is research-only and touches
no sealed holdout.
"""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import torch

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

from spincore.lean_functional_training import (
    compact_report,
    finalize,
    load_checkpoint,
    save_checkpoint,
)
from spincore.lean_parallel import ParallelRootExecutor
from spincore.solver import SolverLibrary

import lt3_hu_ens8_parallel_fit as par
import lt3_parallel_continuation_core as core

DOMAIN_HU="TRUE_HEADS_UP"
DOMAIN_3H="THREE_HANDED"
SOURCE_ITERATION=9105
TARGET_ITERATION=10105
MILESTONE_ITERATION=9600
CHECKPOINT_EVERY=50
ENSEMBLE_SIZE=8
MEMBER_STEPS=400
SCHEMA="SPINCORE_LT3_PARALLEL_CONTINUATION_9105_10105_V1"
ENSEMBLE_SCHEMA="SPINCORE_LT3_HU_ENS8_CURRENT_STATE_V1"


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,required=True)
    p.add_argument("--source-checkpoint",type=Path,required=True)
    p.add_argument("--source-ensemble",type=Path,required=True)
    p.add_argument("--output-checkpoint",type=Path,required=True)
    p.add_argument("--output-ensemble",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--work-dir",type=Path,required=True)
    p.add_argument("--milestone-dir",type=Path,required=True)
    p.add_argument("--workers",type=int,default=31)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--parallel-concurrency",type=int,default=4)
    p.add_argument("--parallel-member-threads",type=int,default=8)
    return p.parse_args()


def _atomic_json(path:Path,payload:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def _save_ensemble(
    path:Path,
    *,
    completed_iteration:int,
    source_checkpoint:Path,
    source_ensemble:Path,
    states,
    member_meta,
)->None:
    payload={
        "schema":ENSEMBLE_SCHEMA,
        "research_lane":"LT3_PARALLEL_9105_10105",
        "source_checkpoint":str(source_checkpoint.resolve()),
        "source_ensemble_state":str(source_ensemble.resolve()),
        "completed_iteration":int(completed_iteration),
        "ensemble_size":ENSEMBLE_SIZE,
        "member_steps":MEMBER_STEPS,
        "member_meta":member_meta,
        "members":states,
        "semantics":"mean raw Advantage outputs then unchanged lean_regret_matching_policy",
        "fit_seed_contract":"fixed ENS8_A init/batch seeds reused every iteration",
        "policy_rng_contract":"ensemble fit isolated from authoritative policy RNG",
        "fit_execution":"4 concurrent processes x 8 Torch threads, exact-parity gate passed",
        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
    }
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(payload,tmp)
    os.replace(tmp,path)


def _preserve_milestone(src:Path,dst:Path)->None:
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src,dst)
    except OSError:
        shutil.copy2(src,dst)


def _validate_source(config,completed):
    if int(completed)!=SOURCE_ITERATION:
        raise RuntimeError(f"expected source iteration 9105, got {completed}")
    if int(config.roots_per_iteration)!=600:
        raise RuntimeError("expected 600 roots/iteration")
    if int(config.advantage_steps_for_domain(DOMAIN_3H))!=100:
        raise RuntimeError("expected 3H fresh100")
    if int(config.hu_advantage_steps or 0)!=400:
        raise RuntimeError("expected HU fresh400")
    if int(config.hu_preflop_board_average_k)!=1:
        raise RuntimeError("expected K4 off")


def main()->int:
    args=parse_args()
    torch.set_num_threads(int(args.threads))

    solver_path=args.solver.resolve(strict=True)
    source_checkpoint=args.source_checkpoint.resolve(strict=True)
    source_ensemble=args.source_ensemble.resolve(strict=True)
    output_checkpoint=args.output_checkpoint.resolve()
    output_ensemble=args.output_ensemble.resolve()
    report_path=args.report.resolve()
    work_dir=args.work_dir.resolve()
    milestone_dir=args.milestone_dir.resolve()
    work_dir.mkdir(parents=True,exist_ok=True)

    source_hashes_before={
        "checkpoint":core.sha256_file(source_checkpoint),
        "ensemble":core.sha256_file(source_ensemble),
    }

    solver=SolverLibrary(solver_path)
    seed,source_config,completed,sampler,runtimes,history,source_finalized=load_checkpoint(
        source_checkpoint,solver=solver
    )
    _validate_source(source_config,completed)
    if not bool(source_finalized):
        raise RuntimeError("expected finalized source checkpoint at iteration 9105")

    config=replace(
        source_config,
        iterations=TARGET_ITERATION,
        hu_advantage_steps=MEMBER_STEPS,
        hu_preflop_board_average_k=1,
    )
    for runtime in runtimes.values():
        runtime.session.batch_mode="vectorized"

    hu=runtimes[DOMAIN_HU]
    hu_states,source_member_meta,source_ensemble_schema=core.load_ensemble_behavior(
        hu,source_ensemble,SOURCE_ITERATION
    )

    packed_dir=work_dir/"packed_hu_adv"
    mirror,packed_meta=par.PackedAdvantageReservoir.build(
        hu.bundle.adv_mem,packed_dir
    )
    mirror.bind_authoritative(hu.bundle.adv_mem)

    fit_contract=par.make_fit_contract(hu,config)
    fitter=par.ParallelEnsembleFitter(
        manifest_path=mirror.manifest_path,
        contract=fit_contract,
        concurrency=int(args.parallel_concurrency),
        threads_per_member=int(args.parallel_member_threads),
    )
    root_executor=ParallelRootExecutor(solver_path,int(args.workers))

    checkpoint_rows=[]
    timing_rows=[]
    recent_history=[]
    milestone_files={}
    overall_started=time.perf_counter()

    try:
        for iteration in range(SOURCE_ITERATION+1,TARGET_ITERATION+1):
            report,hu_states,hu_member_meta=core.run_one_iteration(
                seed=int(seed),
                iteration=int(iteration),
                config=config,
                sampler=sampler,
                runtimes=runtimes,
                root_executor=root_executor,
                hu_states=hu_states,
                fit_mode="parallel",
                parallel_fitter=fitter,
            )
            history.append(report)
            recent_history.append(report)
            recent_history=recent_history[-10:]
            completed=int(iteration)

            timing_rows.append({
                "iteration":completed,
                "wall_seconds":float(report["wall_seconds"]),
                "hu_fit_seconds":float(
                    report["domains"][DOMAIN_HU]["advantage_fit_seconds"]
                ),
                "three_handed_tree_seconds":float(
                    report["domains"][DOMAIN_3H]["tree_seconds"]
                ),
                "hu_tree_seconds":float(
                    report["domains"][DOMAIN_HU]["tree_seconds"]
                ),
            })

            checkpoint_due=(
                completed%CHECKPOINT_EVERY==0
                or completed==TARGET_ITERATION
            )
            if checkpoint_due:
                save_started=time.perf_counter()
                mirror.flush()
                save_checkpoint(
                    output_checkpoint,
                    seed=int(seed),
                    config=config,
                    completed_iteration=completed,
                    sampler=sampler,
                    runtimes=runtimes,
                    history=history,
                    finalized=False,
                )
                _save_ensemble(
                    output_ensemble,
                    completed_iteration=completed,
                    source_checkpoint=source_checkpoint,
                    source_ensemble=source_ensemble,
                    states=hu_states,
                    member_meta=hu_member_meta,
                )
                checkpoint_rows.append({
                    "iteration":completed,
                    "seconds":float(time.perf_counter()-save_started),
                    "checkpoint_bytes":int(output_checkpoint.stat().st_size),
                    "ensemble_bytes":int(output_ensemble.stat().st_size),
                })

                if completed==MILESTONE_ITERATION:
                    milestone_cp=milestone_dir/f"checkpoint_{MILESTONE_ITERATION}.pt"
                    milestone_ens=milestone_dir/f"hu_ensemble_state_{MILESTONE_ITERATION}.pt"
                    _preserve_milestone(output_checkpoint,milestone_cp)
                    _preserve_milestone(output_ensemble,milestone_ens)
                    milestone_files={
                        "iteration":MILESTONE_ITERATION,
                        "checkpoint":str(milestone_cp.resolve()),
                        "checkpoint_sha256":core.sha256_file(milestone_cp),
                        "ensemble":str(milestone_ens.resolve()),
                        "ensemble_sha256":core.sha256_file(milestone_ens),
                        "finalized":False,
                    }
                    print(
                        "LT3_PARALLEL_MILESTONE_9600_PRESERVED "
                        +json.dumps(milestone_files,sort_keys=True),
                        flush=True,
                    )

            _atomic_json(report_path,{
                "schema":SCHEMA,
                "status":"RUNNING",
                "research_lane":"LT3_PARALLEL_9105_10105",
                "source_iteration":SOURCE_ITERATION,
                "target_iteration":TARGET_ITERATION,
                "completed_iteration":completed,
                "parallel_fit_layout":"4x8",
                "checkpoint_every":CHECKPOINT_EVERY,
                "milestone_iteration":MILESTONE_ITERATION,
                "milestone_files":milestone_files,
                "checkpoint_rows":checkpoint_rows,
                "timing_rows":timing_rows,
                "recent_history":recent_history,
                "source_unchanged_so_far":True,
                "holdout_touched":False,
                "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
            })

            print(
                "LT3_PARALLEL_ITERATION "
                +json.dumps({
                    "iteration":completed,
                    "target":TARGET_ITERATION,
                    "hu_adv_seen":int(hu.bundle.adv_mem.seen),
                    "hu_strategy_seen":int(hu.bundle.pol_mem.seen),
                    "hu_fit_seconds":float(
                        report["domains"][DOMAIN_HU]["advantage_fit_seconds"]
                    ),
                    "hu_member_loss_last":[
                        round(float(x["loss_last"]),6)
                        for x in hu_member_meta
                    ],
                    "wall_seconds":float(report["wall_seconds"]),
                    "mirror_updates":int(mirror.write_updates),
                },sort_keys=True),
                flush=True,
            )
    finally:
        root_executor.close()
        fitter.close()
        mirror.flush()

    print("LT3_PARALLEL_FINALIZE average-policy",flush=True)
    finalize_started=time.perf_counter()
    final=finalize(config=config,runtimes=runtimes)
    finalize_seconds=float(time.perf_counter()-finalize_started)

    final_save_started=time.perf_counter()
    save_checkpoint(
        output_checkpoint,
        seed=int(seed),
        config=config,
        completed_iteration=int(completed),
        sampler=sampler,
        runtimes=runtimes,
        history=history,
        finalized=True,
    )
    _save_ensemble(
        output_ensemble,
        completed_iteration=int(completed),
        source_checkpoint=source_checkpoint,
        source_ensemble=source_ensemble,
        states=hu_states,
        member_meta=hu_member_meta,
    )
    final_save_seconds=float(time.perf_counter()-final_save_started)

    source_hashes_after={
        "checkpoint":core.sha256_file(source_checkpoint),
        "ensemble":core.sha256_file(source_ensemble),
    }
    if source_hashes_after!=source_hashes_before:
        raise RuntimeError("source 9105 artifacts changed during continuation")

    compact=compact_report(
        seed=int(seed),
        config=config,
        history=history,
        final=final,
    )
    payload={
        "schema":SCHEMA,
        "status":"PASS",
        "research_lane":"LT3_PARALLEL_9105_10105",
        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
        "source_checkpoint":str(source_checkpoint),
        "source_ensemble":str(source_ensemble),
        "source_ensemble_schema":source_ensemble_schema,
        "source_iteration":SOURCE_ITERATION,
        "target_iteration":TARGET_ITERATION,
        "completed_iteration":int(completed),
        "source_finalized":bool(source_finalized),
        "source_hashes_before":source_hashes_before,
        "source_hashes_after":source_hashes_after,
        "source_unchanged":True,
        "intervention":{
            "poker_or_training_algorithm_change":False,
            "execution_only":"HU ENS8 independent fresh400 fits executed 4x8 process-parallel",
            "parallel_fit_layout":"4x8",
            "exact_parity_gate":"SPINCORE_LT3_PARALLEL_8200_GATE_V1 PASS",
            "three_handed":"unchanged fresh100",
            "true_heads_up":"unchanged ENS8 raw-Advantage average of eight fresh400 fits",
            "ensemble_size":ENSEMBLE_SIZE,
            "member_steps":MEMBER_STEPS,
            "hu_preflop_board_average_k":1,
            "roots_per_iteration":int(config.roots_per_iteration),
        },
        "method":{
            "workers":int(args.workers),
            "torch_threads_parent":int(torch.get_num_threads()),
            "parallel_concurrency":int(args.parallel_concurrency),
            "parallel_member_threads":int(args.parallel_member_threads),
            "additional_iterations":TARGET_ITERATION-SOURCE_ITERATION,
            "new_training_roots":(
                TARGET_ITERATION-SOURCE_ITERATION
            )*int(config.roots_per_iteration),
            "checkpoint_every":CHECKPOINT_EVERY,
            "mmap_persistent_mirror":True,
            "packed_meta":packed_meta,
            "fit_pool_startup_seconds":float(fitter.startup_seconds),
            "lt2_final_holdout_touched":False,
            "lt3_holdout_touched":False,
        },
        "source_member_meta_count":len(source_member_meta),
        "milestone_files":milestone_files,
        "checkpoint_rows":checkpoint_rows,
        "timing_rows":timing_rows,
        "recent_history":recent_history,
        "finalize_seconds":finalize_seconds,
        "final_save_seconds":final_save_seconds,
        "final":final,
        "compact_final":compact.get("final"),
        "output_checkpoint":str(output_checkpoint.resolve()),
        "output_checkpoint_sha256":core.sha256_file(output_checkpoint),
        "output_ensemble":str(output_ensemble.resolve()),
        "output_ensemble_sha256":core.sha256_file(output_ensemble),
        "ordinary_checkpoint_warning":"HU current behavior requires output_ensemble sidecar",
        "wall_seconds":float(time.perf_counter()-overall_started),
        "holdout_touched":False,
    }
    _atomic_json(report_path,payload)
    mirror.close()

    print("LT3_PARALLEL_9105_10105_TRAINING_PASS")
    print(f"report={report_path}")
    print(f"checkpoint={output_checkpoint}")
    print(f"ensemble={output_ensemble}")
    if milestone_files:
        print(f"milestone_9600_checkpoint={milestone_files['checkpoint']}")
        print(f"milestone_9600_ensemble={milestone_files['ensemble']}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
