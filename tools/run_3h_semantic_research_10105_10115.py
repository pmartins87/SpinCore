#!/usr/bin/env python3
from __future__ import annotations

"""Resumable 3H V1+semantic research continuation from frozen 10105 to 10115.

This lane follows the bounded online pilot that passed at 10107.  It replays the
same 10106/10107 contract from the read-only 10105 source, verifies the 10107
milestone against the admitted pilot, then continues through 10115.

Research-only:
- THREE_HANDED only;
- 64 roots per online iteration;
- eight semantic Advantage members, each fresh-fit for 1600 steps;
- exact validated member init/batch seed contract;
- original controlled-split 50k reservoir positions excluded from minibatches;
- source 10105 checkpoint and HU sidecar remain read-only;
- rolling resume state is saved after every completed online iteration;
- compact ensemble milestone snapshots are saved at 10107, 10110 and 10115;
- final semantic behavior generates fresh strategy targets;
- frozen 500-step semantic AveragePolicy distillation is evaluated on a new
  independent stream.

No production promotion and no DC2 occur here.
"""

import argparse
import json
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
import audit_3h_fresh_semantic_strategy_independent500_10105 as confirm

from spincore.lean_functional_training import load_checkpoint
from spincore_nn.reservoir import UniformReservoir
from spincore.solver import SolverLibrary

SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_CONTINUATION_10105_10115_V1"
RESUME_SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_RESUME_V1"
SOURCE_ITERATION=10105
TARGET_ITERATION=10115
MILESTONES=(10107,10110,10115)
EXPECTED_PILOT_10107={
    "pairwise_tv_mean":0.4156071527337772,
    "pairwise_argmax_disagreement_mean":0.42253876502431414,
    "no_draw_raw_allin":0.1870395436080371,
    "no_draw_member_mix_allin":0.18710446272712586,
}
TAIL_TRAIN_SEED=20260927 ^ 0x10115A
TAIL_EVAL_SEED=20260927 ^ 0x10115E


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


def save_ensemble(path:Path,states,member_meta,iteration:int):
    atomic_torch({
        "schema":"SPINCORE_3H_SEMANTIC_RESEARCH_ENSEMBLE_V1",
        "source_checkpoint_sha256":pilot.EXPECTED_SHA,
        "source_iteration":SOURCE_ITERATION,
        "completed_iteration":int(iteration),
        "ensemble_size":8,
        "member_steps":1600,
        "members":states,
        "member_seed_contract":member_meta,
        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
    },path)


def save_resume(
    path:Path,
    *,
    sampler,
    memory,
    states,
    member_meta,
    completed_iteration:int,
    online_rows,
    milestones,
):
    atomic_torch({
        "schema":RESUME_SCHEMA,
        "source_checkpoint_sha256":pilot.EXPECTED_SHA,
        "source_iteration":SOURCE_ITERATION,
        "target_iteration":TARGET_ITERATION,
        "completed_iteration":int(completed_iteration),
        "sampler_rng_state":sampler.rng.bit_generator.state,
        "adv_mem":memory.state_dict(),
        "semantic_members":states,
        "member_seed_contract":member_meta,
        "online_rows":list(online_rows),
        "milestones":dict(milestones),
        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
    },path)


def restore_resume(path:Path,sampler,r3,expected_member_meta):
    p=torch.load(path,map_location="cpu",weights_only=False)
    if p.get("schema")!=RESUME_SCHEMA:
        raise RuntimeError("wrong research resume schema")
    if p.get("source_checkpoint_sha256")!=pilot.EXPECTED_SHA:
        raise RuntimeError("research resume source mismatch")
    completed=int(p.get("completed_iteration",-1))
    if not SOURCE_ITERATION < completed <= TARGET_ITERATION:
        raise RuntimeError("research resume iteration out of range")
    r3.bundle.adv_mem=UniformReservoir.from_state_dict(p["adv_mem"])
    r3.session.bundle=r3.bundle
    r3.session.collector.advantage_memory=r3.bundle.adv_mem
    sampler.rng.bit_generator.state=p["sampler_rng_state"]
    states=list(p["semantic_members"])
    if len(states)!=8:
        raise RuntimeError("research resume ensemble-size drift")
    meta=list(p.get("member_seed_contract") or [])
    if [
        (int(x["init_seed"]),int(x["batch_seed"])) for x in meta
    ] != [
        (int(x["init_seed"]),int(x["batch_seed"])) for x in expected_member_meta
    ]:
        raise RuntimeError("research resume member seed-contract drift")
    models=[distill.load_semantic_advantage(s) for s in states]
    return (
        completed,models,states,list(p.get("online_rows") or []),
        dict(p.get("milestones") or {})
    )


def pilot_reproduction_check(summary):
    nd=summary["high_card_jam_no_immediate_draw"]
    actual={
        "pairwise_tv_mean":float(summary["pairwise_tv_mean"]),
        "pairwise_argmax_disagreement_mean":float(summary["pairwise_argmax_disagreement_mean"]),
        "no_draw_raw_allin":float(nd["raw_ensemble_action_probability_mean"]),
        "no_draw_member_mix_allin":float(nd["policy_mixture_action_probability_mean"]),
    }
    diffs={k:abs(actual[k]-v) for k,v in EXPECTED_PILOT_10107.items()}
    passed=all(x<=1e-6 for x in diffs.values())
    return {"expected":EXPECTED_PILOT_10107,"actual":actual,"abs_diff":diffs,"pass":passed}


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage-models",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--spin-bundle",type=Path,required=True)
    ap.add_argument("--run-dir",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--canonical-replay",action="store_true")
    args=ap.parse_args()

    torch.set_num_threads(int(args.threads))
    run_dir=args.run_dir.resolve()
    run_dir.mkdir(parents=True,exist_ok=True)
    report_path=run_dir/"3h_semantic_research_10105_10115.json"
    resume_path=run_dir/"resume_state.pt"

    source=args.checkpoint.resolve(strict=True)
    actual=pilot.sha256(source)
    if actual!=pilot.EXPECTED_SHA:
        raise SystemExit(f"checkpoint SHA mismatch: {actual}")
    source_sha_before=actual

    solver=SolverLibrary(args.solver.resolve(strict=True))
    seed,source_config,completed_source,sampler,runtimes,_history,_finalized=load_checkpoint(
        source,solver=solver
    )
    if int(completed_source)!=SOURCE_ITERATION:
        raise RuntimeError("source iteration drift")
    r3=runtimes[pilot.DOMAIN]
    if len(r3.bundle.adv_mem.items)!=2_000_000:
        raise RuntimeError("expected saturated source 3H Advantage reservoir")

    _sem_payload,initial_models,member_meta=pilot.load_semantic_artifact(
        args.semantic_advantage_models
    )

    probe=torch.load(args.probe.resolve(strict=True),map_location="cpu",weights_only=False)
    if probe.get("schema")!=pilot.PROBE_SCHEMA:
        raise RuntimeError("controlled split probe schema mismatch")
    split_rng=random.Random(int(probe["holdout_seed"]))
    protected=set(split_rng.sample(
        range(len(r3.bundle.adv_mem.items)),
        int(probe["holdout_size"]),
    ))
    if len(protected)!=50000:
        raise RuntimeError("protected split size drift")
    train_pool=[i for i in range(len(r3.bundle.adv_mem.items)) if i not in protected]

    if resume_path.is_file():
        completed,models,states,online_rows,milestone_rows=restore_resume(
            resume_path,sampler,r3,member_meta
        )
        print(f"SEMANTIC_RESEARCH_RESUME iteration={completed}",flush=True)
    else:
        completed=SOURCE_ITERATION
        models=initial_models
        states=clone_states(models)
        online_rows=[]
        milestone_rows={}

    print("SEMANTIC_RESEARCH_PRECOMPUTE_BEGIN",flush=True)
    sem_rows,precompute_seconds=pilot.precompute_semantics(r3.bundle.adv_mem)
    print(f"SEMANTIC_RESEARCH_PRECOMPUTE_PASS seconds={precompute_seconds:.3f}",flush=True)

    traces,games=pilot.shadow.replay_states(
        solver,args.spin_bundle.resolve(strict=True),200
    )
    initial_dc1=pilot.advantage_dc1_summary(initial_models,traces)

    overall_started=time.perf_counter()
    for iteration in range(completed+1,TARGET_ITERATION+1):
        before_seen=int(r3.bundle.adv_mem.seen)
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
        summary=pilot.advantage_dc1_summary(models,traces)
        row={
            "iteration":int(iteration),
            "adv_seen_before":before_seen,
            "adv_seen_after":int(r3.bundle.adv_mem.seen),
            "tree":tree,
            "fit_seconds":fit_seconds,
            "dc1":summary,
            "member_meta":fit_meta,
        }
        online_rows.append(row)
        completed=int(iteration)

        if completed==10107:
            reproduction=pilot_reproduction_check(summary)
            row["bounded_pilot_reproduction"]=reproduction
            row["bounded_pilot_reproduction_role"]=(
                "DESCRIPTIVE_HISTORICAL_COMPARISON_ONLY"
                if args.canonical_replay
                else "HARD_REPRODUCTION_GUARD"
            )
            if (not args.canonical_replay) and (not reproduction["pass"]):
                save_resume(
                    resume_path,sampler=sampler,memory=r3.bundle.adv_mem,
                    states=states,member_meta=member_meta,
                    completed_iteration=completed,online_rows=online_rows,
                    milestones=milestone_rows,
                )
                raise RuntimeError(
                    "10107 deterministic replay does not reproduce admitted bounded pilot"
                )

        if completed in MILESTONES:
            milestone_rows[str(completed)]=summary
            save_ensemble(
                run_dir/f"semantic_ensemble_{completed}.pt",
                states,member_meta,completed
            )

        save_resume(
            resume_path,
            sampler=sampler,
            memory=r3.bundle.adv_mem,
            states=states,
            member_meta=member_meta,
            completed_iteration=completed,
            online_rows=online_rows,
            milestones=milestone_rows,
        )
        partial={
            "schema":SCHEMA,
            "status":"RUNNING",
            "source_iteration":SOURCE_ITERATION,
            "target_iteration":TARGET_ITERATION,
            "completed_iteration":completed,
            "online_rows":online_rows,
            "milestones":milestone_rows,
            "protected_controlled_split_positions":len(protected),
            "source_checkpoint_mutated":False,
            "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
        }
        atomic_json(partial,report_path)
        print("SEMANTIC_RESEARCH_ITERATION "+json.dumps({
            "iteration":completed,
            "adv_seen_after":int(r3.bundle.adv_mem.seen),
            "tree_seconds":tree["seconds"],
            "fit_seconds":fit_seconds,
            "dc1_pairwise_tv":summary["pairwise_tv_mean"],
            "dc1_no_draw_raw_allin":summary["high_card_jam_no_immediate_draw"]["raw_ensemble_action_probability_mean"],
        },sort_keys=True),flush=True)

    final_dc1=pilot.advantage_dc1_summary(models,traces)

    # Fresh strategy targets from the final online semantic ensemble.
    distill.MASTER_SEED=TAIL_TRAIN_SEED
    policy_train,discarded_holdout,train_collection=distill.collect_fresh(
        solver,models,pilot.POLICY_TRAIN_EPISODES
    )
    if len(policy_train)<20000:
        raise RuntimeError("research tail policy train stream too small")

    mini,cfg=pilot.policy_pair_from_source(r3.bundle,source_config)
    v1,v1opt,sem,semopt=confirm.initial_policy_pair(mini,cfg)
    confirm.train_500(v1,v1opt,sem,semopt,policy_train,cfg)

    step0_v1,_,step0_sem,_=confirm.initial_policy_pair(mini,cfg)
    distill.MASTER_SEED=TAIL_EVAL_SEED
    eval_a,eval_b,eval_collection=distill.collect_fresh(
        solver,models,pilot.POLICY_EVAL_EPISODES
    )
    independent=list(eval_a)+list(eval_b)
    if len(independent)<5000:
        raise RuntimeError("research tail independent eval too small")
    step0=distill.metrics(step0_v1,step0_sem,independent)
    final_policy=distill.metrics(v1,sem,independent)
    hc=final_policy["high_card_no_draw_allin"]
    if int(hc["count"])<150:
        raise RuntimeError("research tail high-card subgroup too small")

    policy_dc1=distill.dc1(
        solver,args.spin_bundle.resolve(strict=True),v1,sem,200
    )

    ini_nd=initial_dc1["high_card_jam_no_immediate_draw"]
    fin_nd=final_dc1["high_card_jam_no_immediate_draw"]
    criteria={
        "final_pairwise_tv_not_worse_by_over_15pct":bool(
            final_dc1["pairwise_tv_mean"]<=1.15*initial_dc1["pairwise_tv_mean"]
        ),
        "final_argmax_disagreement_not_worse_by_over_15pct":bool(
            final_dc1["pairwise_argmax_disagreement_mean"]
            <=1.15*initial_dc1["pairwise_argmax_disagreement_mean"]
        ),
        "final_no_draw_raw_allin_not_up_by_over_10pp":bool(
            fin_nd["raw_ensemble_action_probability_mean"]
            <=ini_nd["raw_ensemble_action_probability_mean"]+0.10
        ),
        "final_no_draw_member_mix_allin_not_up_by_over_10pp":bool(
            fin_nd["policy_mixture_action_probability_mean"]
            <=ini_nd["policy_mixture_action_probability_mean"]+0.10
        ),
        "tail_semantic_ce_at_least_10pct_better_than_v1":bool(
            final_policy["semantic_weighted_ce"]<=0.90*final_policy["v1_weighted_ce"]
        ),
        "tail_semantic_tv_at_least_15pct_better_than_v1":bool(
            final_policy["semantic_weighted_tv"]<=0.85*final_policy["v1_weighted_tv"]
        ),
        "tail_semantic_high_card_bias_at_least_50pct_better":bool(
            float(hc["v1_abs_bias"])>0
            and float(hc["semantic_abs_bias"])<=0.50*float(hc["v1_abs_bias"])
        ),
        "tail_semantic_ce_at_least_10pct_better_than_step0":bool(
            final_policy["semantic_weighted_ce"]<=0.90*step0["semantic_weighted_ce"]
        ),
    }
    passed=all(criteria.values())

    save_ensemble(
        run_dir/"semantic_ensemble_final_10115.pt",
        states,member_meta,TARGET_ITERATION
    )
    torch.save({
        "schema":"SPINCORE_3H_SEMANTIC_RESEARCH_TAIL_POLICY_V1",
        "source_checkpoint_sha256":actual,
        "completed_iteration":TARGET_ITERATION,
        "selected_steps":500,
        "model_state":{k:v.detach().cpu() for k,v in sem.state_dict().items()},
        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
    },run_dir/"semantic_tail_policy_10115.pt")

    source_sha_after=pilot.sha256(source)
    if source_sha_after!=source_sha_before:
        raise RuntimeError("source 10105 checkpoint changed during research continuation")

    report={
        "schema":SCHEMA,
        "status":"PASS" if passed else "FAIL",
        "source_checkpoint_sha256":actual,
        "source_iteration":SOURCE_ITERATION,
        "target_iteration":TARGET_ITERATION,
        "completed_iteration":completed,
        "source_checkpoint_mutated":False,
        "source_checkpoint_sha_after":source_sha_after,
        "hu_training_performed":False,
        "behavior_contract":(
            "CANONICAL_LEAN_REGRET_MATCHING_SOFTMAX_FALLBACK"
            if args.canonical_replay
            else "HISTORICAL_ADMITTED_SEMANTIC_SIGMA_CONTRACT"
        ),
        "canonical_replay":bool(args.canonical_replay),
        "online_iterations":TARGET_ITERATION-SOURCE_ITERATION,
        "roots_per_iteration":pilot.ROOTS_PER_ITERATION,
        "ensemble_size":8,
        "member_steps":1600,
        "protected_controlled_split_positions":len(protected),
        "online_rows":online_rows,
        "milestones":milestone_rows,
        "advantage_dc1":{
            "balanced_games":games,
            "trace_decisions":len(traces),
            "initial":initial_dc1,
            "final":final_dc1,
        },
        "tail_policy":{
            "train_collection":train_collection,
            "discarded_selection_holdout_samples":len(discarded_holdout),
            "selected_steps":500,
            "independent_eval_collection":eval_collection,
            "independent_eval_samples":len(independent),
            "step0_independent_eval":step0,
            "step500_independent_eval":final_policy,
            "dc1":policy_dc1,
        },
        "precommitted_criteria":criteria,
        "semantic_research_continuation_pass":passed,
        "wall_seconds_this_invocation":float(time.perf_counter()-overall_started),
        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
        "interpretation":(
            "PASS means the semantic 3H lane survived ten bounded online feedback "
            "iterations and still distills into a semantic AveragePolicy better than V1 "
            "on a new final fresh-target stream. It remains research-only; PASS is a gate "
            "to a larger development benchmark, not production or DC2 authorization."
        ),
    }
    atomic_json(report,report_path)
    print("=== 3H semantic research continuation 10105->10115 ===")
    print("criteria="+json.dumps(criteria,sort_keys=True))
    print(f"semantic_research_continuation_pass={passed}")
    print(f"report={report_path}")
    print("3H_SEMANTIC_RESEARCH_10105_10115_RUN_COMPLETE")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
