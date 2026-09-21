#!/usr/bin/env python3
from __future__ import annotations

"""One-iteration semantic parity + three-iteration throughput gate from LT3@8200.

Run each arm in a separate process so the 2.8-GB checkpoint and Python
reservoir graph are fully released between arms.
"""

import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

from spincore.lean_functional_training import DOMAINS,load_checkpoint
from spincore.lean_parallel import ParallelRootExecutor
from spincore.solver import SolverLibrary

import lt3_hu_ens8_parallel_fit as par
import lt3_parallel_continuation_core as core

DOMAIN_HU="TRUE_HEADS_UP"
DOMAIN_3H="THREE_HANDED"
SOURCE_ITERATION=8200
SCHEMA_ARM="SPINCORE_LT3_PARALLEL_8200_ARM_V1"
SCHEMA_GATE="SPINCORE_LT3_PARALLEL_8200_GATE_V1"


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)

    for name in ("sequential","parallel"):
        a=sub.add_parser(name)
        a.add_argument("--solver",type=Path,required=True)
        a.add_argument("--source-checkpoint",type=Path,required=True)
        a.add_argument("--source-ensemble",type=Path,required=True)
        a.add_argument("--output",type=Path,required=True)
        a.add_argument("--work-dir",type=Path,required=True)
        a.add_argument("--workers",type=int,default=31)
        a.add_argument("--threads",type=int,default=8)
        a.add_argument("--parallel-concurrency",type=int,default=4)
        a.add_argument("--parallel-member-threads",type=int,default=8)

    c=sub.add_parser("compare")
    c.add_argument("--sequential-report",type=Path,required=True)
    c.add_argument("--parallel-report",type=Path,required=True)
    c.add_argument("--source-progress-report",type=Path)
    c.add_argument("--lt2-8100-report",type=Path)
    c.add_argument("--output",type=Path,required=True)
    return p.parse_args()


def _mem_available_gib()->float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1])/1024.0/1024.0
    return float("nan")


def _validate_source(config,completed):
    if int(completed)!=SOURCE_ITERATION:
        raise RuntimeError(f"expected source iteration 8200, got {completed}")
    if int(config.roots_per_iteration)!=600:
        raise RuntimeError("expected 600 roots/iteration")
    if int(config.advantage_steps_for_domain(DOMAIN_3H))!=100:
        raise RuntimeError("expected 3H fresh100")
    if int(config.hu_advantage_steps or 0)!=400:
        raise RuntimeError("expected HU fresh400")
    if int(config.hu_preflop_board_average_k)!=1:
        raise RuntimeError("expected K4 off")


def run_arm(args,fit_mode:str)->int:
    torch.set_num_threads(int(args.threads))
    source=args.source_checkpoint.resolve(strict=True)
    ensemble=args.source_ensemble.resolve(strict=True)
    solver_path=args.solver.resolve(strict=True)
    out=args.output.resolve()
    work=args.work_dir.resolve()
    work.mkdir(parents=True,exist_ok=True)
    out.parent.mkdir(parents=True,exist_ok=True)

    sha_before={
        "checkpoint":core.sha256_file(source),
        "ensemble":core.sha256_file(ensemble),
    }
    mem_before=_mem_available_gib()

    solver=SolverLibrary(solver_path)
    seed,source_config,completed,sampler,runtimes,history,finalized=load_checkpoint(
        source,solver=solver
    )
    _validate_source(source_config,completed)
    config=replace(
        source_config,
        iterations=SOURCE_ITERATION+(3 if fit_mode=="parallel" else 1),
        hu_advantage_steps=400,
        hu_preflop_board_average_k=1,
    )
    for runtime in runtimes.values():
        runtime.session.batch_mode="vectorized"

    hu=runtimes[DOMAIN_HU]
    hu_states,source_member_meta,source_schema=core.load_ensemble_behavior(
        hu,ensemble,SOURCE_ITERATION
    )

    mirror=None
    packed_meta=None
    fitter=None
    if fit_mode=="parallel":
        packed_dir=work/"packed_hu_adv"
        if packed_dir.exists():
            import shutil
            shutil.rmtree(packed_dir)
        mirror,packed_meta=par.PackedAdvantageReservoir.build(
            hu.bundle.adv_mem,packed_dir
        )
        audits=core.attach_write_audits(runtimes,hu_mirror=mirror)
        contract=par.make_fit_contract(hu,config)
        fitter=par.ParallelEnsembleFitter(
            manifest_path=mirror.manifest_path,
            contract=contract,
            concurrency=int(args.parallel_concurrency),
            threads_per_member=int(args.parallel_member_threads),
        )
        fit_pool_startup=float(fitter.startup_seconds)
    else:
        audits=core.attach_write_audits(runtimes)
        fit_pool_startup=0.0

    root_executor=ParallelRootExecutor(solver_path,int(args.workers))
    rows=[]
    first_fingerprint=None
    try:
        count=3 if fit_mode=="parallel" else 1
        for offset in range(1,count+1):
            iteration=SOURCE_ITERATION+offset
            report,hu_states,hu_meta=core.run_one_iteration(
                seed=int(seed),
                iteration=int(iteration),
                config=config,
                sampler=sampler,
                runtimes=runtimes,
                root_executor=root_executor,
                hu_states=hu_states,
                fit_mode=fit_mode,
                parallel_fitter=fitter,
            )
            rows.append(report)
            if offset==1:
                first_fingerprint=core.semantic_fingerprint(
                    sampler=sampler,
                    runtimes=runtimes,
                    hu_states=hu_states,
                    audits=audits,
                )
            print(
                f"LT3_8200_GATE_{fit_mode.upper()} iteration={iteration} "
                f"wall={report['wall_seconds']:.3f}s "
                f"hu_fit={report['domains'][DOMAIN_HU]['advantage_fit_seconds']:.3f}s",
                flush=True,
            )
    finally:
        root_executor.close()
        if fitter is not None:
            fitter.close()
        if mirror is not None:
            mirror.close()

    sha_after={
        "checkpoint":core.sha256_file(source),
        "ensemble":core.sha256_file(ensemble),
    }
    result={
        "schema":SCHEMA_ARM,
        "fit_mode":fit_mode,
        "source_iteration":SOURCE_ITERATION,
        "source_checkpoint":str(source),
        "source_ensemble":str(ensemble),
        "source_ensemble_schema":source_schema,
        "source_hashes_before":sha_before,
        "source_hashes_after":sha_after,
        "source_unchanged":sha_before==sha_after,
        "source_finalized":bool(finalized),
        "iterations":rows,
        "first_iteration_fingerprint":first_fingerprint,
        "fit_pool_startup_seconds":fit_pool_startup,
        "packed_meta":packed_meta,
        "mirror_write_updates":(
            int(mirror.write_updates) if mirror is not None else None
        ),
        "mem_available_gib_before":mem_before,
        "mem_available_gib_after":_mem_available_gib(),
        "holdout_touched":False,
        "new_roots":int(len(rows))*int(config.roots_per_iteration),
        "source_member_meta_count":len(source_member_meta),
    }
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"LT3_8200_GATE_{fit_mode.upper()}_ARM_PASS")
    return 0


def _checkpoint_seconds(path:Path|None):
    if path is None or not path.is_file():
        return None
    try:
        d=json.loads(path.read_text(encoding="utf-8"))
        rows=d.get("checkpoint_rows") or []
        if not rows:
            return None
        return float(rows[-1]["seconds"])
    except Exception:
        return None


def _historical_final_overhead(path:Path|None):
    if path is None or not path.is_file():
        return None
    try:
        d=json.loads(path.read_text(encoding="utf-8"))
        total=float(d["wall_seconds"])
        iterations=sum(float(r["wall_seconds"]) for r in d.get("pilot_history") or [])
        checkpoints=sum(float(r["seconds"]) for r in d.get("checkpoint_rows") or [])
        # This residual includes executor setup/teardown, final AveragePolicy fit
        # and the final checkpoint save.  It is a conservative end-block
        # overhead estimate rather than a pure finalization timing.
        residual=total-iterations-checkpoints
        return max(0.0,float(residual))
    except Exception:
        return None


def compare(args)->int:
    seq=json.loads(args.sequential_report.read_text(encoding="utf-8"))
    parrep=json.loads(args.parallel_report.read_text(encoding="utf-8"))
    if seq.get("schema")!=SCHEMA_ARM or parrep.get("schema")!=SCHEMA_ARM:
        raise RuntimeError("wrong arm schema")
    if seq.get("fit_mode")!="sequential" or parrep.get("fit_mode")!="parallel":
        raise RuntimeError("arm identity drift")

    exact=seq["first_iteration_fingerprint"]==parrep["first_iteration_fingerprint"]
    seq_wall=float(seq["iterations"][0]["wall_seconds"])
    par_first=float(parrep["iterations"][0]["wall_seconds"])
    parallel_walls=[float(x["wall_seconds"]) for x in parrep["iterations"]]
    median_parallel=float(statistics.median(parallel_walls))
    whole_speedup=seq_wall/par_first if par_first>0 else 0.0
    checkpoint_seconds=_checkpoint_seconds(args.source_progress_report)
    final_overhead=_historical_final_overhead(args.lt2_8100_report)

    amortized=median_parallel
    if checkpoint_seconds is not None:
        amortized+=checkpoint_seconds/50.0

    # Give finalization a one-time allowance when estimating the block target.
    budget=24.0*3600.0
    if final_overhead is not None:
        budget=max(0.0,budget-final_overhead)
    intervals=max(1,int(round((budget/amortized)/50.0))*50)
    target=SOURCE_ITERATION+intervals
    projected=intervals*amortized+(final_overhead or 0.0)

    source_same=bool(seq["source_unchanged"] and parrep["source_unchanged"])
    verdict=(
        "PASS"
        if exact and source_same and whole_speedup>=1.15
        else "FAIL"
    )
    out={
        "schema":SCHEMA_GATE,
        "verdict":verdict,
        "source_iteration":SOURCE_ITERATION,
        "semantic_parity_exact":bool(exact),
        "source_unchanged":bool(source_same),
        "sequential_iteration_8201_wall_seconds":seq_wall,
        "parallel_iteration_8201_wall_seconds":par_first,
        "parallel_iteration_wall_seconds":parallel_walls,
        "parallel_median_wall_seconds":median_parallel,
        "whole_iteration_speedup_same_iteration":float(whole_speedup),
        "checkpoint_seconds_historical":checkpoint_seconds,
        "checkpoint_amortized_seconds_per_iteration":(
            checkpoint_seconds/50.0 if checkpoint_seconds is not None else None
        ),
        "historical_end_block_overhead_seconds":final_overhead,
        "planning_seconds_per_iteration":float(amortized),
        "recommended_24h":{
            "start_iteration":SOURCE_ITERATION,
            "additional_iterations":int(intervals),
            "target_iteration":int(target),
            "projected_total_seconds":float(projected),
            "projected_total_hours":float(projected/3600.0),
            "rounding":"nearest 50 iterations",
            "planning_only_until_reviewed":True,
        },
        "parallel_fit_layout":"4x8",
        "holdout_touched":False,
        "gate_new_roots":{
            "sequential_disposable":int(seq["new_roots"]),
            "parallel_disposable":int(parrep["new_roots"]),
            "source_mutation":0,
        },
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== LT3 8200 PARALLEL END-TO-END GATE ===")
    print(f"semantic_parity_exact={exact}")
    print(f"sequential_8201={seq_wall:.3f}s")
    print(f"parallel_8201={par_first:.3f}s")
    print(f"whole_iteration_speedup={whole_speedup:.3f}x")
    print(f"parallel_median_3={median_parallel:.3f}s")
    print(
        f"recommended_24h_target={target} "
        f"additional={intervals} projected_hours={projected/3600.0:.2f}"
    )
    print(f"VERDICT={verdict}")
    print("LT3_8200_PARALLEL_END_TO_END_GATE_COMPLETE")
    return 0 if verdict=="PASS" else 2


def main()->int:
    args=parse_args()
    if args.mode=="sequential":
        return run_arm(args,"sequential")
    if args.mode=="parallel":
        return run_arm(args,"parallel")
    return compare(args)


if __name__=="__main__":
    raise SystemExit(main())
