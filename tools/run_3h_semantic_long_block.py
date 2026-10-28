#!/usr/bin/env python3
from __future__ import annotations

"""Generic resumable 3H Semantic ENS8 long-block continuation.

The training algorithm is the validated semantic-online THREE_HANDED path:
64 roots per iteration, eight fresh semantic Advantage fits x1600, canonical
lean regret matching, saturated 2M Advantage reservoir, fixed protected 50k
positions, and safety/integrity diagnostics every 500 block iterations.

This generic carrier exists so successive long-horizon blocks can reuse the same
implementation without cloning a new Python trainer for every 2,000 iterations.

PROJECT_CONTRACT_IDS:
TRAIN-020,TRAIN-022,TRAIN-024,MODEL-020,RNG-001,RNG-002,RNG-003,
SAFE-002,VALID-034,VALID-035,CKPT-004,ART-015
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import run_3h_semantic_online_pilot_10105 as pilot
import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import run_3h_semantic_long_15315_17315 as reference_long

from spincore.lean_functional_training import load_checkpoint
from spincore_nn.reservoir import UniformReservoir
from spincore.solver import SolverLibrary

SCHEMA="SPINCORE_3H_SEMANTIC_LONG_BLOCK_RESULT_V1"
RESUME_SCHEMA="SPINCORE_3H_SEMANTIC_LONG_BLOCK_RESUME_V1"
ENSEMBLE_SCHEMA="SPINCORE_3H_SEMANTIC_LONG_BLOCK_ENSEMBLE_V1"

SOURCE_CHECKPOINT_ITERATION=10105
ROOTS_PER_ITERATION=pilot.ROOTS_PER_ITERATION
MEMBER_STEPS=pilot.MEMBER_STEPS
ENSEMBLE_SIZE=8
PROTECTED_COUNT=50000
MILESTONE_INTERVAL=500


def atomic_torch(payload,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(payload,tmp)
    os.replace(tmp,path)


def atomic_json(payload,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def state_dicts_equal(a,b):
    if set(a)!=set(b):
        return False
    return all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a)


def ensemble_states_equal(left,right):
    if len(left)!=len(right):
        return False
    return all(state_dicts_equal(a,b) for a,b in zip(left,right))


def states_finite(states):
    for state in states:
        for value in state.values():
            if not torch.isfinite(value.detach()).all().item():
                return False
    return True


def load_source_report(path:Path):
    report=json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if report.get("status")!="PASS":
        raise RuntimeError("source 3H semantic report is not PASS")
    schema=str(report.get("schema") or "")
    if schema==SCHEMA:
        completed=int(report.get("completed_iteration",-1))
    elif schema.startswith("SPINCORE_3H_SEMANTIC_LONG_CONTINUATION_"):
        completed=int(report.get("completed_iteration",-1))
    else:
        raise RuntimeError(f"unsupported source 3H semantic report schema: {schema!r}")
    if report.get("final_safety_guard_pass") is not True:
        raise RuntimeError("source 3H semantic report lacks passing safety guard")
    return report,completed


def load_source_endpoint(*,source_resume:Path,source_ensemble:Path,source_report:Path,expected_start:int):
    report,report_completed=load_source_report(source_report)
    if report_completed!=int(expected_start):
        raise RuntimeError(
            f"source 3H report endpoint {report_completed} != expected {expected_start}"
        )
    resume_sha=pilot.sha256(source_resume)
    ensemble_sha=pilot.sha256(source_ensemble)
    report_sha=pilot.sha256(source_report)
    if str(report.get("resume_state_sha256"))!=resume_sha:
        raise RuntimeError("source 3H resume hash does not match PASS report")
    if str(report.get("final_ensemble_sha256"))!=ensemble_sha:
        raise RuntimeError("source 3H ensemble hash does not match PASS report")

    rp=torch.load(source_resume.resolve(strict=True),map_location="cpu",weights_only=False)
    ep=torch.load(source_ensemble.resolve(strict=True),map_location="cpu",weights_only=False)

    rschema=str(rp.get("schema") or "")
    if rschema==RESUME_SCHEMA or rschema.startswith("SPINCORE_3H_SEMANTIC_LONG_RESUME_"):
        completed=int(rp.get("completed_iteration",-1))
        states=list(rp.get("semantic_members") or [])
        meta=list(rp.get("member_seed_contract") or [])
    else:
        raise RuntimeError(f"unsupported source 3H semantic resume schema: {rschema!r}")

    eschema=str(ep.get("schema") or "")
    if eschema in (ENSEMBLE_SCHEMA, reference_long.ENSEMBLE_SCHEMA):
        ensemble_completed=int(ep.get("completed_iteration",-1))
        ensemble_states=list(ep.get("members") or [])
    else:
        raise RuntimeError(f"unsupported source 3H semantic ensemble schema: {eschema!r}")

    if completed!=int(expected_start) or ensemble_completed!=int(expected_start):
        raise RuntimeError("source 3H resume/ensemble endpoint mismatch")
    if len(states)!=ENSEMBLE_SIZE or len(ensemble_states)!=ENSEMBLE_SIZE:
        raise RuntimeError("source 3H semantic ensemble size drift")
    if len(meta)!=ENSEMBLE_SIZE:
        raise RuntimeError("source 3H semantic member metadata drift")
    if not states_finite(states) or not states_finite(ensemble_states):
        raise RuntimeError("source 3H semantic endpoint contains non-finite tensors")
    if not ensemble_states_equal(states,ensemble_states):
        raise RuntimeError("source 3H resume/final-ensemble state mismatch")

    return {
        "report":report,
        "report_sha256":report_sha,
        "resume":rp,
        "resume_sha256":resume_sha,
        "ensemble":ep,
        "ensemble_sha256":ensemble_sha,
        "states":states,
        "member_meta":meta,
        "completed":completed,
    }


def restore_source_runtime(source,*,sampler,r3):
    rp=source["resume"]
    r3.bundle.adv_mem=UniformReservoir.from_state_dict(rp["adv_mem"])
    r3.session.bundle=r3.bundle
    r3.session.collector.advantage_memory=r3.bundle.adv_mem
    sampler.rng.bit_generator.state=rp["sampler_rng_state"]


def save_ensemble(path:Path,*,source,start_iteration:int,completed_iteration:int,states,member_meta):
    atomic_torch({
        "schema":ENSEMBLE_SCHEMA,
        "source_checkpoint_sha256":pilot.EXPECTED_SHA,
        "source_endpoint_resume_sha256":source["resume_sha256"],
        "source_endpoint_ensemble_sha256":source["ensemble_sha256"],
        "source_endpoint_report_sha256":source["report_sha256"],
        "source_semantic_iteration":int(start_iteration),
        "completed_iteration":int(completed_iteration),
        "ensemble_size":ENSEMBLE_SIZE,
        "member_steps":MEMBER_STEPS,
        "members":states,
        "member_seed_contract":list(member_meta),
        "behavior_contract":"CANONICAL_LEAN_REGRET_MATCHING_SOFTMAX_FALLBACK",
        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
    },path)


def save_resume(
    path:Path,
    *,
    source,
    sampler,
    memory,
    states,
    member_meta,
    start_iteration:int,
    target_iteration:int,
    completed_iteration:int,
    recent_rows,
    milestones,
):
    atomic_torch({
        "schema":RESUME_SCHEMA,
        "source_checkpoint_sha256":pilot.EXPECTED_SHA,
        "source_endpoint_resume_sha256":source["resume_sha256"],
        "source_endpoint_ensemble_sha256":source["ensemble_sha256"],
        "source_endpoint_report_sha256":source["report_sha256"],
        "source_semantic_iteration":int(start_iteration),
        "target_iteration":int(target_iteration),
        "completed_iteration":int(completed_iteration),
        "sampler_rng_state":sampler.rng.bit_generator.state,
        "adv_mem":memory.state_dict(),
        "semantic_members":states,
        "member_seed_contract":list(member_meta),
        "recent_rows":list(recent_rows[-20:]),
        "milestones":dict(milestones),
        "behavior_contract":"CANONICAL_LEAN_REGRET_MATCHING_SOFTMAX_FALLBACK",
        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
    },path)


def restore_local_resume(path:Path,*,source,sampler,r3,start_iteration:int,target_iteration:int):
    p=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=RESUME_SCHEMA:
        raise RuntimeError("wrong local generic 3H long-resume schema")
    if str(p.get("source_endpoint_resume_sha256"))!=source["resume_sha256"]:
        raise RuntimeError("local 3H resume source-resume binding mismatch")
    if str(p.get("source_endpoint_ensemble_sha256"))!=source["ensemble_sha256"]:
        raise RuntimeError("local 3H resume source-ensemble binding mismatch")
    if str(p.get("source_endpoint_report_sha256"))!=source["report_sha256"]:
        raise RuntimeError("local 3H resume source-report binding mismatch")
    if int(p.get("source_semantic_iteration",-1))!=int(start_iteration):
        raise RuntimeError("local 3H resume start drift")
    if int(p.get("target_iteration",-1))!=int(target_iteration):
        raise RuntimeError("local 3H resume target drift")
    completed=int(p.get("completed_iteration",-1))
    if not int(start_iteration)<=completed<=int(target_iteration):
        raise RuntimeError("local 3H resume completed iteration out of range")

    r3.bundle.adv_mem=UniformReservoir.from_state_dict(p["adv_mem"])
    r3.session.bundle=r3.bundle
    r3.session.collector.advantage_memory=r3.bundle.adv_mem
    sampler.rng.bit_generator.state=p["sampler_rng_state"]
    states=list(p.get("semantic_members") or [])
    meta=list(p.get("member_seed_contract") or [])
    if len(states)!=ENSEMBLE_SIZE or len(meta)!=ENSEMBLE_SIZE or not states_finite(states):
        raise RuntimeError("local 3H resume ensemble/meta drift")
    models=[distill.load_semantic_advantage(s) for s in states]
    return completed,models,states,meta,list(p.get("recent_rows") or []),dict(p.get("milestones") or {})


def parse_args():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage-models",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--spin-bundle",type=Path,required=True)
    ap.add_argument("--source-resume",type=Path,required=True)
    ap.add_argument("--source-final-ensemble",type=Path,required=True)
    ap.add_argument("--source-report",type=Path,required=True)
    ap.add_argument("--start-iteration",type=int,required=True)
    ap.add_argument("--target-iteration",type=int,required=True)
    ap.add_argument("--run-dir",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    return ap.parse_args()


def main()->int:
    args=parse_args()
    start_iteration=int(args.start_iteration)
    target_iteration=int(args.target_iteration)
    if start_iteration<=SOURCE_CHECKPOINT_ITERATION or target_iteration<=start_iteration:
        raise RuntimeError("invalid generic 3H semantic interval")
    if (target_iteration-start_iteration)%MILESTONE_INTERVAL!=0:
        raise RuntimeError("generic 3H target must align to 500-iteration safety milestone")

    torch.set_num_threads(int(args.threads))
    run_dir=args.run_dir.resolve()
    run_dir.mkdir(parents=True,exist_ok=True)
    report_path=run_dir/"3h_semantic_long_block.json"
    resume_path=run_dir/"resume_state.pt"

    checkpoint=args.checkpoint.resolve(strict=True)
    if pilot.sha256(checkpoint)!=pilot.EXPECTED_SHA:
        raise RuntimeError("source checkpoint SHA mismatch")

    source_resume=args.source_resume.resolve(strict=True)
    source_final=args.source_final_ensemble.resolve(strict=True)
    source_report=args.source_report.resolve(strict=True)
    source=load_source_endpoint(
        source_resume=source_resume,
        source_ensemble=source_final,
        source_report=source_report,
        expected_start=start_iteration,
    )
    source_hashes_before={
        "checkpoint":pilot.sha256(checkpoint),
        "resume":pilot.sha256(source_resume),
        "ensemble":pilot.sha256(source_final),
        "report":pilot.sha256(source_report),
    }

    solver=SolverLibrary(args.solver.resolve(strict=True))
    seed,source_config,completed_source,sampler,runtimes,_history,_finalized=load_checkpoint(
        checkpoint,solver=solver
    )
    if int(completed_source)!=SOURCE_CHECKPOINT_ITERATION:
        raise RuntimeError("source checkpoint iteration drift")
    r3=runtimes[pilot.DOMAIN]
    restore_source_runtime(source,sampler=sampler,r3=r3)
    if len(r3.bundle.adv_mem.items)!=2_000_000:
        raise RuntimeError("expected saturated 2M 3H Advantage reservoir")

    _initial_payload,_initial_models,artifact_member_meta=pilot.load_semantic_artifact(
        args.semantic_advantage_models
    )
    source_member_meta=list(source["member_meta"])
    if [
        (int(x["init_seed"]),int(x["batch_seed"])) for x in source_member_meta
    ] != [
        (int(x["init_seed"]),int(x["batch_seed"])) for x in artifact_member_meta
    ]:
        raise RuntimeError("source 3H member seed contract drift")

    probe=torch.load(args.probe.resolve(strict=True),map_location="cpu",weights_only=False)
    if probe.get("schema")!=pilot.PROBE_SCHEMA:
        raise RuntimeError("controlled split probe schema mismatch")
    if str(probe.get("source_checkpoint_sha256"))!=pilot.EXPECTED_SHA:
        raise RuntimeError("controlled split probe source mismatch")
    split_rng=random.Random(int(probe["holdout_seed"]))
    protected=set(split_rng.sample(
        range(len(r3.bundle.adv_mem.items)),
        int(probe["holdout_size"]),
    ))
    if len(protected)!=PROTECTED_COUNT:
        raise RuntimeError("protected split size drift")
    train_pool=[i for i in range(len(r3.bundle.adv_mem.items)) if i not in protected]
    if len(train_pool)!=1_950_000:
        raise RuntimeError("controlled split train pool drift")

    source_models=[
        distill.load_semantic_advantage(s)
        for s in list(source["states"])
    ]
    models=list(source_models)
    states=list(source["states"])
    member_meta=source_member_meta
    completed=start_iteration
    recent_rows=[]
    milestone_rows={}

    if resume_path.is_file():
        completed,models,states,member_meta,recent_rows,milestone_rows=restore_local_resume(
            resume_path,
            source=source,
            sampler=sampler,
            r3=r3,
            start_iteration=start_iteration,
            target_iteration=target_iteration,
        )
        print(f"SEMANTIC_GENERIC_LONG_RESUME iteration={completed}",flush=True)

    print("SEMANTIC_GENERIC_LONG_PRECOMPUTE_BEGIN",flush=True)
    sem_rows,precompute_seconds=pilot.precompute_semantics(r3.bundle.adv_mem)
    print(
        f"SEMANTIC_GENERIC_LONG_PRECOMPUTE_PASS seconds={precompute_seconds:.3f}",
        flush=True,
    )

    traces,games=pilot.shadow.replay_states(
        solver,args.spin_bundle.resolve(strict=True),200
    )
    initial_diag=pilot.advantage_dc1_summary(source_models,traces)

    started=time.perf_counter()
    block_roots=(completed-start_iteration)*ROOTS_PER_ITERATION
    try:
        for iteration in range(completed+1,target_iteration+1):
            seen_before=int(r3.bundle.adv_mem.seen)
            tree=pilot.collect_roots(
                solver=solver,
                sampler=sampler,
                memory=r3.bundle.adv_mem,
                models=models,
                seed=int(seed),
                source_config=source_config,
                iteration=int(iteration),
            )
            if int(tree.get("roots",-1))!=ROOTS_PER_ITERATION:
                raise RuntimeError("generic 3H semantic root-count drift")
            models,states,fit_meta,fit_seconds=pilot.fit_ensemble(
                r3.bundle.adv_mem,
                sem_rows,
                train_pool,
                member_meta,
                lr=float(source_config.learning_rate),
                batch_size=int(source_config.batch_size),
                iteration=int(iteration),
            )
            if not math.isfinite(float(fit_seconds)) or not states_finite(states):
                raise RuntimeError("generic 3H semantic non-finite fit")
            block_roots+=int(tree["roots"])
            completed=int(iteration)
            row={
                "iteration":completed,
                "block_iteration":completed-start_iteration,
                "adv_seen_before":seen_before,
                "adv_seen_after":int(r3.bundle.adv_mem.seen),
                "tree":tree,
                "fit_seconds":float(fit_seconds),
                "member_meta":fit_meta,
            }
            recent_rows.append(row)
            recent_rows=recent_rows[-20:]

            if (completed-start_iteration)%MILESTONE_INTERVAL==0:
                current_diag=pilot.advantage_dc1_summary(models,traces)
                guard,guard_pass=reference_long.safety_guard(initial_diag,current_diag)
                milestone_rows[str(completed)]={
                    "dc1_advantage_diagnostic":current_diag,
                    "safety_guard":guard,
                    "safety_guard_pass":guard_pass,
                }
                save_ensemble(
                    run_dir/f"semantic_long_ensemble_{completed}.pt",
                    source=source,
                    start_iteration=start_iteration,
                    completed_iteration=completed,
                    states=states,
                    member_meta=member_meta,
                )
                print(
                    "SEMANTIC_GENERIC_LONG_MILESTONE "
                    +json.dumps({
                        "iteration":completed,
                        "pairwise_tv":current_diag["pairwise_tv_mean"],
                        "argmax_disagreement":current_diag["pairwise_argmax_disagreement_mean"],
                        "no_draw_raw_allin":current_diag["high_card_jam_no_immediate_draw"]["raw_ensemble_action_probability_mean"],
                        "guard_pass":guard_pass,
                    },sort_keys=True),
                    flush=True,
                )
                if not guard_pass:
                    save_resume(
                        resume_path,
                        source=source,
                        sampler=sampler,
                        memory=r3.bundle.adv_mem,
                        states=states,
                        member_meta=member_meta,
                        start_iteration=start_iteration,
                        target_iteration=target_iteration,
                        completed_iteration=completed,
                        recent_rows=recent_rows,
                        milestones=milestone_rows,
                    )
                    atomic_json({
                        "schema":SCHEMA,
                        "status":"STOPPED_SAFETY",
                        "source_checkpoint_sha256":pilot.EXPECTED_SHA,
                        "source_semantic_iteration":start_iteration,
                        "target_iteration":target_iteration,
                        "completed_iteration":completed,
                        "final_safety_guard_pass":False,
                        "milestones":milestone_rows,
                        "recent_rows":recent_rows,
                        "hu_training_performed":False,
                        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
                    },report_path)
                    print("SEMANTIC_GENERIC_LONG_SAFETY_STOP",flush=True)
                    return 5

            save_resume(
                resume_path,
                source=source,
                sampler=sampler,
                memory=r3.bundle.adv_mem,
                states=states,
                member_meta=member_meta,
                start_iteration=start_iteration,
                target_iteration=target_iteration,
                completed_iteration=completed,
                recent_rows=recent_rows,
                milestones=milestone_rows,
            )
            atomic_json({
                "schema":SCHEMA,
                "status":"RUNNING",
                "source_checkpoint_sha256":pilot.EXPECTED_SHA,
                "source_semantic_iteration":start_iteration,
                "target_iteration":target_iteration,
                "completed_iteration":completed,
                "additional_iterations":target_iteration-start_iteration,
                "roots_per_iteration":ROOTS_PER_ITERATION,
                "block_new_roots_so_far":(completed-start_iteration)*ROOTS_PER_ITERATION,
                "ensemble_size":ENSEMBLE_SIZE,
                "member_steps":MEMBER_STEPS,
                "protected_controlled_split_positions":len(protected),
                "initial_diagnostic":initial_diag,
                "milestones":milestone_rows,
                "recent_rows":recent_rows,
                "source_endpoint_resume_sha256":source["resume_sha256"],
                "source_endpoint_ensemble_sha256":source["ensemble_sha256"],
                "source_endpoint_report_sha256":source["report_sha256"],
                "hu_training_performed":False,
                "average_policy_training_performed":False,
                "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
            },report_path)
            print(
                "SEMANTIC_GENERIC_LONG_ITERATION "
                +json.dumps({
                    "iteration":completed,
                    "target_iteration":target_iteration,
                    "block_iteration":completed-start_iteration,
                    "adv_seen_after":int(r3.bundle.adv_mem.seen),
                    "tree_seconds":tree["seconds"],
                    "fit_seconds":fit_seconds,
                },sort_keys=True),
                flush=True,
            )
    finally:
        pass

    final_diag=pilot.advantage_dc1_summary(models,traces)
    final_guard,final_guard_pass=reference_long.safety_guard(initial_diag,final_diag)
    if not final_guard_pass:
        raise RuntimeError("final generic 3H safety guard drifted after milestone check")
    if completed!=target_iteration:
        raise RuntimeError("generic 3H semantic block ended before target")

    expected_roots=(target_iteration-start_iteration)*ROOTS_PER_ITERATION
    if block_roots!=expected_roots:
        raise RuntimeError(
            f"generic 3H root accounting drift: {block_roots} != {expected_roots}"
        )

    final_ensemble=run_dir/f"semantic_long_ensemble_final_{target_iteration}.pt"
    save_ensemble(
        final_ensemble,
        source=source,
        start_iteration=start_iteration,
        completed_iteration=target_iteration,
        states=states,
        member_meta=member_meta,
    )
    save_resume(
        resume_path,
        source=source,
        sampler=sampler,
        memory=r3.bundle.adv_mem,
        states=states,
        member_meta=member_meta,
        start_iteration=start_iteration,
        target_iteration=target_iteration,
        completed_iteration=target_iteration,
        recent_rows=recent_rows,
        milestones=milestone_rows,
    )

    source_hashes_after={
        "checkpoint":pilot.sha256(checkpoint),
        "resume":pilot.sha256(source_resume),
        "ensemble":pilot.sha256(source_final),
        "report":pilot.sha256(source_report),
    }
    if source_hashes_after!=source_hashes_before:
        raise RuntimeError("frozen 3H source endpoint changed during continuation")

    report={
        "schema":SCHEMA,
        "status":"PASS",
        "source_checkpoint_sha256":pilot.EXPECTED_SHA,
        "source_semantic_iteration":start_iteration,
        "target_iteration":target_iteration,
        "completed_iteration":target_iteration,
        "additional_iterations":target_iteration-start_iteration,
        "expected_new_roots":expected_roots,
        "new_roots_recorded":block_roots,
        "behavior_contract":"CANONICAL_LEAN_REGRET_MATCHING_SOFTMAX_FALLBACK",
        "roots_per_iteration":ROOTS_PER_ITERATION,
        "ensemble_size":ENSEMBLE_SIZE,
        "member_steps":MEMBER_STEPS,
        "protected_controlled_split_positions":len(protected),
        "fixed_diagnostic_games":games,
        "initial_diagnostic":initial_diag,
        "final_diagnostic":final_diag,
        "final_safety_guard":final_guard,
        "final_safety_guard_pass":final_guard_pass,
        "milestones":milestone_rows,
        "recent_rows":recent_rows,
        "source_endpoint_resume_sha256":source["resume_sha256"],
        "source_endpoint_ensemble_sha256":source["ensemble_sha256"],
        "source_endpoint_report_sha256":source["report_sha256"],
        "source_10105_mutated":False,
        "hu_training_performed":False,
        "average_policy_training_performed":False,
        "final_ensemble":str(final_ensemble.resolve()),
        "final_ensemble_sha256":pilot.sha256(final_ensemble),
        "resume_state":str(resume_path.resolve()),
        "resume_state_sha256":pilot.sha256(resume_path),
        "precompute_seconds":precompute_seconds,
        "wall_seconds_this_invocation":float(time.perf_counter()-started),
        "source_unchanged":True,
        "source_hashes_before":source_hashes_before,
        "source_hashes_after":source_hashes_after,
        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
        "interpretation":(
            "PASS means one frozen 3H Semantic ENS8 long block completed from the "
            "bound prior endpoint with the validated teacher architecture and "
            "precommitted pathology guards. It is not a strength/promotion result."
        ),
    }
    atomic_json(report,report_path)

    print("SEMANTIC_GENERIC_LONG_BLOCK_PASS")
    print(f"source_iteration={start_iteration}")
    print(f"target_iteration={target_iteration}")
    print(f"new_roots_recorded={block_roots}")
    print(f"final_ensemble_sha256={report['final_ensemble_sha256']}")
    print(f"resume_state_sha256={report['resume_state_sha256']}")
    print(f"report={report_path}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
