#!/usr/bin/env python3
from __future__ import annotations

"""Resumable long 3H semantic-Advantage macroblock from 10315 to 15315.

This is the next long-horizon compute macroblock. Short-horizon strength checks are intentionally omitted; only frozen safety/integrity milestones may stop the run.

The architecture is intentionally frozen:
- THREE_HANDED only;
- canonical lean regret-matching with softmax fallback;
- 64 roots per online iteration;
- eight semantic Advantage members;
- each member is fresh-fit for 1600 steps with the already-validated deterministic
  init/batch-seed contract;
- same 2M Advantage reservoir and the same protected 50k reservoir positions;
- source LT3@10105 checkpoint and HU@10105 sidecar remain read-only;
- no AveragePolicy or specialist is trained during this compute-heavy phase.

At the end we only freeze the new semantic Advantage teacher.  Fresh final
AveragePolicy distillation + strong-specialist rebuilding are separate,
post-training gates so historical target conflict is not reintroduced.

Milestone safety checks are pathology guards, not quality claims.  A safety
failure stops the long run but cannot promote an earlier milestone.
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
import run_3h_semantic_research_10105_10115 as research
import run_3h_semantic_long_10115_10315 as previous_long
import audit_3h_fresh_semantic_strategy_distill_10105 as distill

from spincore.lean_functional_training import load_checkpoint
from spincore_nn.reservoir import UniformReservoir
from spincore.solver import SolverLibrary

SCHEMA="SPINCORE_3H_SEMANTIC_LONG_CONTINUATION_10315_15315_V1"
RESUME_SCHEMA="SPINCORE_3H_SEMANTIC_LONG_RESUME_10315_15315_V1"
ENSEMBLE_SCHEMA="SPINCORE_3H_SEMANTIC_LONG_ENSEMBLE_V1"

SOURCE_CHECKPOINT_ITERATION=10105
START_ITERATION=10315
TARGET_ITERATION=15315
MILESTONES=(10815,11315,11815,12315,12815,13315,13815,14315,14815,15315)
EXPECTED_ADDITIONAL_ITERATIONS=5000
EXPECTED_NEW_ROOTS=EXPECTED_ADDITIONAL_ITERATIONS*pilot.ROOTS_PER_ITERATION

# Frozen pathology guards.  These are deliberately loose enough not to select
# for a particular poker style; they exist only to stop gross training drift.
MAX_PAIRWISE_TV_RATIO=1.35
MAX_ARGMAX_DISAGREEMENT_RATIO=1.35
MAX_NO_DRAW_ALLIN_DELTA=0.15


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


def clone_states(models):
    return [
        {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        for m in models
    ]


def state_dicts_equal(a,b):
    if set(a)!=set(b):
        return False
    return all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a)


def ensemble_states_equal(left,right):
    if len(left)!=len(right):
        return False
    return all(state_dicts_equal(a,b) for a,b in zip(left,right))


def save_ensemble(path:Path,states,member_meta,iteration:int,source_resume_sha:str):
    atomic_torch({
        "schema":ENSEMBLE_SCHEMA,
        "source_checkpoint_sha256":pilot.EXPECTED_SHA,
        "source_semantic_iteration":START_ITERATION,
        "source_resume_sha256":source_resume_sha,
        "completed_iteration":int(iteration),
        "ensemble_size":8,
        "member_steps":pilot.MEMBER_STEPS,
        "members":states,
        "member_seed_contract":member_meta,
        "behavior_contract":"CANONICAL_LEAN_REGRET_MATCHING_SOFTMAX_FALLBACK",
        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
    },path)


def save_long_resume(
    path:Path,
    *,
    sampler,
    memory,
    states,
    member_meta,
    completed_iteration:int,
    online_rows,
    milestones,
    source_resume_sha:str,
):
    atomic_torch({
        "schema":RESUME_SCHEMA,
        "source_checkpoint_sha256":pilot.EXPECTED_SHA,
        "source_checkpoint_iteration":SOURCE_CHECKPOINT_ITERATION,
        "source_semantic_iteration":START_ITERATION,
        "source_resume_sha256":source_resume_sha,
        "target_iteration":TARGET_ITERATION,
        "completed_iteration":int(completed_iteration),
        "sampler_rng_state":sampler.rng.bit_generator.state,
        "adv_mem":memory.state_dict(),
        "semantic_members":states,
        "member_seed_contract":member_meta,
        "online_rows":list(online_rows),
        "milestones":dict(milestones),
        "behavior_contract":"CANONICAL_LEAN_REGRET_MATCHING_SOFTMAX_FALLBACK",
        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
    },path)


def restore_long_resume(path:Path,sampler,r3,expected_member_meta,source_resume_sha:str):
    p=torch.load(path,map_location="cpu",weights_only=False)
    if p.get("schema")!=RESUME_SCHEMA:
        raise RuntimeError("wrong long-resume schema")
    if p.get("source_checkpoint_sha256")!=pilot.EXPECTED_SHA:
        raise RuntimeError("long-resume source checkpoint mismatch")
    if str(p.get("source_resume_sha256"))!=source_resume_sha:
        raise RuntimeError("long-resume 10115 source mismatch")
    if int(p.get("target_iteration",-1))!=TARGET_ITERATION:
        raise RuntimeError("long-resume target mismatch")
    completed=int(p.get("completed_iteration",-1))
    if not START_ITERATION < completed <= TARGET_ITERATION:
        raise RuntimeError("long-resume completed iteration out of range")

    meta=list(p.get("member_seed_contract") or [])
    if [
        (int(x["init_seed"]),int(x["batch_seed"])) for x in meta
    ] != [
        (int(x["init_seed"]),int(x["batch_seed"])) for x in expected_member_meta
    ]:
        raise RuntimeError("long-resume member seed contract drift")

    r3.bundle.adv_mem=UniformReservoir.from_state_dict(p["adv_mem"])
    r3.session.bundle=r3.bundle
    r3.session.collector.advantage_memory=r3.bundle.adv_mem
    sampler.rng.bit_generator.state=p["sampler_rng_state"]

    states=list(p["semantic_members"])
    if len(states)!=8:
        raise RuntimeError("long-resume ensemble size drift")
    models=[distill.load_semantic_advantage(s) for s in states]
    return (
        completed,
        models,
        states,
        list(p.get("online_rows") or []),
        dict(p.get("milestones") or {}),
    )


def finite_diag(diag):
    vals=[
        diag["pairwise_tv_mean"],
        diag["pairwise_argmax_disagreement_mean"],
        diag["high_card_jam_no_immediate_draw"]["raw_ensemble_action_probability_mean"],
        diag["high_card_jam_no_immediate_draw"]["policy_mixture_action_probability_mean"],
    ]
    return all(math.isfinite(float(x)) for x in vals)


def safety_guard(initial,current):
    ini_nd=initial["high_card_jam_no_immediate_draw"]
    cur_nd=current["high_card_jam_no_immediate_draw"]
    criteria={
        "diagnostics_finite":finite_diag(current),
        "pairwise_tv_not_over_35pct_worse":bool(
            current["pairwise_tv_mean"]
            <=MAX_PAIRWISE_TV_RATIO*initial["pairwise_tv_mean"]
        ),
        "argmax_disagreement_not_over_35pct_worse":bool(
            current["pairwise_argmax_disagreement_mean"]
            <=MAX_ARGMAX_DISAGREEMENT_RATIO*initial["pairwise_argmax_disagreement_mean"]
        ),
        "no_draw_raw_allin_not_up_over_15pp":bool(
            cur_nd["raw_ensemble_action_probability_mean"]
            <=ini_nd["raw_ensemble_action_probability_mean"]+MAX_NO_DRAW_ALLIN_DELTA
        ),
        "no_draw_member_mix_allin_not_up_over_15pp":bool(
            cur_nd["policy_mixture_action_probability_mean"]
            <=ini_nd["policy_mixture_action_probability_mean"]+MAX_NO_DRAW_ALLIN_DELTA
        ),
    }
    return criteria,all(criteria.values())


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage-models",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--spin-bundle",type=Path,required=True)
    ap.add_argument("--source-resume",type=Path,required=True)
    ap.add_argument("--source-final-ensemble",type=Path,required=True)
    ap.add_argument("--run-dir",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    args=ap.parse_args()

    torch.set_num_threads(int(args.threads))
    run_dir=args.run_dir.resolve()
    run_dir.mkdir(parents=True,exist_ok=True)
    report_path=run_dir/"3h_semantic_long_10315_15315.json"
    resume_path=run_dir/"resume_state.pt"

    source=args.checkpoint.resolve(strict=True)
    if pilot.sha256(source)!=pilot.EXPECTED_SHA:
        raise RuntimeError("source checkpoint SHA mismatch")

    source_resume=args.source_resume.resolve(strict=True)
    source_resume_sha=pilot.sha256(source_resume)
    source_final=args.source_final_ensemble.resolve(strict=True)

    solver=SolverLibrary(args.solver.resolve(strict=True))
    seed,source_config,completed_source,sampler,runtimes,_history,_finalized=load_checkpoint(
        source,solver=solver
    )
    if int(completed_source)!=SOURCE_CHECKPOINT_ITERATION:
        raise RuntimeError("source checkpoint iteration drift")
    r3=runtimes[pilot.DOMAIN]
    if len(r3.bundle.adv_mem.items)!=2_000_000:
        raise RuntimeError("expected saturated 2M 3H Advantage reservoir")

    _initial_payload,_initial_models,member_meta=pilot.load_semantic_artifact(
        args.semantic_advantage_models
    )

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
    if len(protected)!=50000:
        raise RuntimeError("protected split size drift")
    train_pool=[i for i in range(len(r3.bundle.adv_mem.items)) if i not in protected]
    if len(train_pool)!=1_950_000:
        raise RuntimeError("controlled split train pool drift")

    src_ens=torch.load(source_final,map_location="cpu",weights_only=False)
    if src_ens.get("schema")!=ENSEMBLE_SCHEMA:
        raise RuntimeError("source semantic ensemble schema mismatch")
    if int(src_ens.get("completed_iteration",-1))!=START_ITERATION:
        raise RuntimeError("source semantic ensemble iteration mismatch")
    source_models=[
        distill.load_semantic_advantage(s)
        for s in list(src_ens.get("members") or [])
    ]
    if len(source_models)!=8:
        raise RuntimeError("source semantic ensemble member-count drift")

    if resume_path.is_file():
        completed,models,states,online_rows,milestone_rows=restore_long_resume(
            resume_path,sampler,r3,member_meta,source_resume_sha
        )
        print(f"SEMANTIC_LONG_RESUME iteration={completed}",flush=True)
    else:
        previous_payload=torch.load(source_resume,map_location="cpu",weights_only=False)
        previous_source_sha=str(previous_payload.get("source_resume_sha256") or "")
        if not previous_source_sha:
            raise RuntimeError("10315 source resume lacks lineage hash")
        completed,models,states,_old_rows,_old_milestones=previous_long.restore_long_resume(
            source_resume,sampler,r3,member_meta,previous_source_sha
        )
        if completed!=START_ITERATION:
            raise RuntimeError("source long resume is not at 10315")
        if not ensemble_states_equal(states,list(src_ens["members"])):
            raise RuntimeError("10315 resume/final-ensemble state mismatch")
        online_rows=[]
        milestone_rows={}

    print("SEMANTIC_LONG_PRECOMPUTE_BEGIN",flush=True)
    sem_rows,precompute_seconds=pilot.precompute_semantics(r3.bundle.adv_mem)
    print(
        f"SEMANTIC_LONG_PRECOMPUTE_PASS seconds={precompute_seconds:.3f}",
        flush=True,
    )

    traces,games=pilot.shadow.replay_states(
        solver,args.spin_bundle.resolve(strict=True),200
    )
    initial_diag=pilot.advantage_dc1_summary(source_models,traces)
    current_diag=pilot.advantage_dc1_summary(models,traces)

    started=time.perf_counter()
    for iteration in range(completed+1,TARGET_ITERATION+1):
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
        models,states,fit_meta,fit_seconds=pilot.fit_ensemble(
            r3.bundle.adv_mem,
            sem_rows,
            train_pool,
            member_meta,
            lr=float(source_config.learning_rate),
            batch_size=int(source_config.batch_size),
            iteration=int(iteration),
        )
        row={
            "iteration":int(iteration),
            "adv_seen_before":seen_before,
            "adv_seen_after":int(r3.bundle.adv_mem.seen),
            "tree":tree,
            "fit_seconds":fit_seconds,
            "member_meta":fit_meta,
        }
        online_rows.append(row)
        completed=int(iteration)

        if completed in MILESTONES:
            current_diag=pilot.advantage_dc1_summary(models,traces)
            guard,guard_pass=safety_guard(initial_diag,current_diag)
            milestone_rows[str(completed)]={
                "dc1_advantage_diagnostic":current_diag,
                "safety_guard":guard,
                "safety_guard_pass":guard_pass,
            }
            save_ensemble(
                run_dir/f"semantic_long_ensemble_{completed}.pt",
                states,member_meta,completed,source_resume_sha
            )
            print(
                "SEMANTIC_LONG_MILESTONE "
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
                save_long_resume(
                    resume_path,
                    sampler=sampler,
                    memory=r3.bundle.adv_mem,
                    states=states,
                    member_meta=member_meta,
                    completed_iteration=completed,
                    online_rows=online_rows,
                    milestones=milestone_rows,
                    source_resume_sha=source_resume_sha,
                )
                report={
                    "schema":SCHEMA,
                    "status":"STOPPED_SAFETY",
                    "source_checkpoint_sha256":pilot.EXPECTED_SHA,
                    "source_semantic_iteration":START_ITERATION,
                    "target_iteration":TARGET_ITERATION,
                    "completed_iteration":completed,
                    "behavior_contract":"CANONICAL_LEAN_REGRET_MATCHING_SOFTMAX_FALLBACK",
                    "roots_per_iteration":pilot.ROOTS_PER_ITERATION,
                    "ensemble_size":8,
                    "member_steps":pilot.MEMBER_STEPS,
                    "protected_controlled_split_positions":len(protected),
                    "initial_diagnostic":initial_diag,
                    "milestones":milestone_rows,
                    "online_rows":online_rows,
                    "source_10105_mutated":False,
                    "hu_training_performed":False,
                    "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
                }
                atomic_json(report,report_path)
                print("SEMANTIC_LONG_SAFETY_STOP",flush=True)
                return 5

        save_long_resume(
            resume_path,
            sampler=sampler,
            memory=r3.bundle.adv_mem,
            states=states,
            member_meta=member_meta,
            completed_iteration=completed,
            online_rows=online_rows,
            milestones=milestone_rows,
            source_resume_sha=source_resume_sha,
        )

        partial={
            "schema":SCHEMA,
            "status":"RUNNING",
            "source_checkpoint_sha256":pilot.EXPECTED_SHA,
            "source_semantic_iteration":START_ITERATION,
            "target_iteration":TARGET_ITERATION,
            "completed_iteration":completed,
            "behavior_contract":"CANONICAL_LEAN_REGRET_MATCHING_SOFTMAX_FALLBACK",
            "roots_per_iteration":pilot.ROOTS_PER_ITERATION,
            "ensemble_size":8,
            "member_steps":pilot.MEMBER_STEPS,
            "protected_controlled_split_positions":len(protected),
            "initial_diagnostic":initial_diag,
            "milestones":milestone_rows,
            "online_rows":online_rows,
            "source_10105_mutated":False,
            "hu_training_performed":False,
            "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
        }
        atomic_json(partial,report_path)
        print(
            "SEMANTIC_LONG_ITERATION "
            +json.dumps({
                "iteration":completed,
                "adv_seen_after":int(r3.bundle.adv_mem.seen),
                "tree_seconds":tree["seconds"],
                "fit_seconds":fit_seconds,
            },sort_keys=True),
            flush=True,
        )

    final_diag=pilot.advantage_dc1_summary(models,traces)
    final_guard,final_guard_pass=safety_guard(initial_diag,final_diag)
    if not final_guard_pass:
        raise RuntimeError("final diagnostic guard drifted after milestone checks")

    final_ensemble=run_dir/"semantic_long_ensemble_final_15315.pt"
    save_ensemble(
        final_ensemble,states,member_meta,TARGET_ITERATION,source_resume_sha
    )
    save_long_resume(
        resume_path,
        sampler=sampler,
        memory=r3.bundle.adv_mem,
        states=states,
        member_meta=member_meta,
        completed_iteration=TARGET_ITERATION,
        online_rows=online_rows,
        milestones=milestone_rows,
        source_resume_sha=source_resume_sha,
    )

    if pilot.sha256(source)!=pilot.EXPECTED_SHA:
        raise RuntimeError("source 10105 checkpoint mutated")

    actual_new_roots=sum(int(r["tree"]["roots"]) for r in online_rows)
    report={
        "schema":SCHEMA,
        "status":"PASS",
        "source_checkpoint_sha256":pilot.EXPECTED_SHA,
        "source_checkpoint_iteration":SOURCE_CHECKPOINT_ITERATION,
        "source_semantic_iteration":START_ITERATION,
        "source_resume_sha256":source_resume_sha,
        "target_iteration":TARGET_ITERATION,
        "completed_iteration":TARGET_ITERATION,
        "additional_iterations":TARGET_ITERATION-START_ITERATION,
        "expected_new_roots":EXPECTED_NEW_ROOTS,
        "new_roots_recorded":actual_new_roots,
        "behavior_contract":"CANONICAL_LEAN_REGRET_MATCHING_SOFTMAX_FALLBACK",
        "roots_per_iteration":pilot.ROOTS_PER_ITERATION,
        "ensemble_size":8,
        "member_steps":pilot.MEMBER_STEPS,
        "protected_controlled_split_positions":len(protected),
        "fixed_diagnostic_games":games,
        "initial_diagnostic":initial_diag,
        "final_diagnostic":final_diag,
        "final_safety_guard":final_guard,
        "final_safety_guard_pass":final_guard_pass,
        "milestones":milestone_rows,
        "online_rows":online_rows,
        "source_10105_mutated":False,
        "hu_training_performed":False,
        "final_ensemble":str(final_ensemble.resolve()),
        "final_ensemble_sha256":pilot.sha256(final_ensemble),
        "resume_state":str(resume_path.resolve()),
        "resume_state_sha256":pilot.sha256(resume_path),
        "precompute_seconds":precompute_seconds,
        "wall_seconds_this_invocation":float(time.perf_counter()-started),
        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
        "interpretation":(
            "PASS means the frozen semantic-Advantage architecture completed a "
            f"5000-iteration long 3H continuation without triggering precommitted "
            "pathology guards. It does not itself produce or promote a deployable "
            "AveragePolicy. The next gate must freshly distill the final teacher, "
            "rebuild the strong specialist, and independently validate that policy."
        ),
    }
    atomic_json(report,report_path)

    print("=== 3H semantic long continuation 10315->15315 ===")
    print("final_guard="+json.dumps(final_guard,sort_keys=True))
    print(f"final_ensemble_sha256={report['final_ensemble_sha256']}")
    print(f"resume_state_sha256={report['resume_state_sha256']}")
    print(f"wall_hours={report['wall_seconds_this_invocation']/3600.0}")
    print(f"report={report_path}")
    print("SEMANTIC_LONG_10315_15315_TRAINING_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
