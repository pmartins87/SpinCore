#!/usr/bin/env python3
from __future__ import annotations

"""LT3 Heavy H1: isolated continuation of the frozen LT2 ENS8@8100 research state.

This is a RESEARCH-ONLY continuation. It never mutates or resumes in-place from
the LT2 production artifacts.

Source:
- LT2 ordinary checkpoint @8100 (reservoirs / AveragePolicy / counters);
- mandatory LT2 HU ENS8 sidecar @8100 (the actual current HU behavior).

Intervention:
- preserve the exact LT2 online-feedback mechanics;
- THREE_HANDED: one fresh 100-step Advantage fit per iteration;
- TRUE_HEADS_UP: eight fresh 400-step Advantage fits per iteration;
- same fixed ENS8_A init/batch seed contract;
- raw HU Advantage outputs averaged before unchanged lean regret matching;
- K4 remains off (K=1);
- 31 root workers, parent Torch threads=8.

H1 adds 500 iterations by default (8101..8600), i.e. 300k new roots.
No LT2 holdout seed or LT3 holdout seed is touched here.
"""

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

from spincore.lean_action_policy import LeanEnsembleActionAdvantagePolicy
from spincore.lean_concurrent_iteration import _collect_policy_from_episodes, _presample_plan
from spincore.lean_functional_training import (
    DOMAINS,
    _advantage_reset_seed,
    compact_report,
    finalize,
    load_checkpoint,
    save_checkpoint,
)
from spincore.lean_parallel import ParallelRootExecutor
from spincore.solver import SolverLibrary

import run_lt2_hu_ens8_online_pilot as lt2ens

DOMAIN_HU="TRUE_HEADS_UP"
DOMAIN_3H="THREE_HANDED"
SCHEMA="SPINCORE_LT3_HEAVY_ENS8_H1_V1"
ENSEMBLE_SCHEMA="SPINCORE_LT3_HU_ENS8_CURRENT_STATE_V1"
SOURCE_ENSEMBLE_SCHEMA="SPINCORE_LT2_HU_ENS8_CURRENT_STATE_V1"
SOURCE_ITERATION=8100
ENSEMBLE_SIZE=8
MEMBER_STEPS=400


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--source-checkpoint",type=Path,required=True)
    p.add_argument("--source-ensemble-state",type=Path,required=True)
    p.add_argument("--output-checkpoint",type=Path,required=True)
    p.add_argument("--output-ensemble-state",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--additional-iterations",type=int,default=500)
    p.add_argument("--checkpoint-every",type=int,default=50)
    p.add_argument("--workers",type=int,default=31)
    p.add_argument("--threads",type=int,default=8)
    return p.parse_args()


def _clone_state(model)->dict[str,torch.Tensor]:
    return {
        key:value.detach().cpu().clone()
        for key,value in model.state_dict().items()
    }


def _save_ensemble(
    path:Path,
    *,
    completed_iteration:int,
    source_checkpoint:Path,
    source_ensemble:Path,
    states:tuple[dict[str,torch.Tensor],...],
    member_meta:list[dict[str,Any]],
)->None:
    payload={
        "schema":ENSEMBLE_SCHEMA,
        "research_lane":"LT3_HEAVY_H1",
        "source_checkpoint":str(source_checkpoint.resolve()),
        "source_ensemble_state":str(source_ensemble.resolve()),
        "completed_iteration":int(completed_iteration),
        "ensemble_size":ENSEMBLE_SIZE,
        "member_steps":MEMBER_STEPS,
        "member_meta":member_meta,
        "members":states,
        "semantics":"mean raw Advantage outputs then unchanged lean_regret_matching_policy",
        "fit_seed_contract":"same fixed ENS8_A init/batch seeds as LT2; reused every iteration",
        "policy_rng_contract":"ensemble minibatch RNG isolated; authoritative policy RNG restored before sampled-policy collection",
        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
    }
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(payload,tmp)
    os.replace(tmp,path)


def _save_progress(path:Path,payload:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def _load_source_ensemble(
    runtime,
    source_ensemble:Path,
)->tuple[tuple[dict[str,torch.Tensor],...],list[dict[str,Any]]]:
    payload=torch.load(source_ensemble,map_location="cpu",weights_only=False)
    if payload.get("schema")!=SOURCE_ENSEMBLE_SCHEMA:
        raise RuntimeError(f"wrong source ensemble schema: {payload.get('schema')!r}")
    if int(payload.get("completed_iteration",-1))!=SOURCE_ITERATION:
        raise RuntimeError("source ensemble is not iteration 8100")
    if int(payload.get("ensemble_size",-1))!=ENSEMBLE_SIZE:
        raise RuntimeError("source ensemble size drift")
    if int(payload.get("member_steps",-1))!=MEMBER_STEPS:
        raise RuntimeError("source ensemble member-step drift")
    members=payload.get("members")
    if not isinstance(members,(list,tuple)) or len(members)!=ENSEMBLE_SIZE:
        raise RuntimeError("source ensemble member payload drift")

    models=[]
    states=[]
    for member,state in enumerate(members):
        model=copy.deepcopy(runtime.bundle.advantage)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        states.append({
            key:value.detach().cpu().clone()
            for key,value in state.items()
        })

    behavior=LeanEnsembleActionAdvantagePolicy(
        models,
        selected_representation="C0_V1_FROZEN_CONTROL",
        device="cpu",
        ready=True,
    )
    runtime.session.behavior=behavior
    runtime.session.collector.policy=behavior
    runtime.bundle.counters["advantage_ready"]=1

    member_meta=list(payload.get("member_meta") or [])
    return tuple(states),member_meta


def main()->int:
    args=parse_args()
    if args.additional_iterations<=0:
        raise SystemExit("--additional-iterations must be positive")
    if args.checkpoint_every<=0 or args.workers<=1 or args.threads<=0:
        raise SystemExit("invalid checkpoint/workers/threads")

    solver_path=args.solver.resolve(strict=True)
    source_checkpoint=args.source_checkpoint.resolve(strict=True)
    source_ensemble=args.source_ensemble_state.resolve(strict=True)

    torch.set_num_threads(int(args.threads))
    solver=SolverLibrary(solver_path)

    seed,source_config,completed,sampler,runtimes,history,source_finalized=load_checkpoint(
        source_checkpoint,solver=solver
    )
    if int(completed)!=SOURCE_ITERATION:
        raise RuntimeError(f"expected source iteration {SOURCE_ITERATION}, got {completed}")
    if int(source_config.advantage_steps_for_domain(DOMAIN_3H))!=100:
        raise RuntimeError("LT3 H1 requires source 3H fresh100 contract")
    if int(source_config.hu_advantage_steps or 0)!=MEMBER_STEPS:
        raise RuntimeError("LT3 H1 requires source HU400 contract")
    if int(source_config.hu_preflop_board_average_k)!=1:
        raise RuntimeError("LT3 H1 requires K4/off canonical K=1")

    target=int(completed)+int(args.additional_iterations)
    config=replace(
        source_config,
        iterations=target,
        hu_advantage_steps=MEMBER_STEPS,
        hu_preflop_board_average_k=1,
    )
    if int(config.roots_per_iteration)!=600:
        raise RuntimeError(
            f"unexpected source roots_per_iteration={config.roots_per_iteration}; "
            "H1 preregistration requires 600"
        )

    for runtime in runtimes.values():
        runtime.session.batch_mode="vectorized"

    hu_runtime=runtimes[DOMAIN_HU]
    hu_states,source_member_meta=_load_source_ensemble(hu_runtime,source_ensemble)

    recent_history=[]
    checkpoint_rows=[]
    overall_started=time.perf_counter()
    executor=ParallelRootExecutor(solver_path,int(args.workers))
    try:
        for iteration in range(int(completed)+1,target+1):
            iter_started=time.perf_counter()
            plans=_presample_plan(
                seed=int(seed),
                iteration=int(iteration),
                config=config,
                sampler=sampler,
            )
            report={"iteration":int(iteration),"domains":{},"hu_ensemble_size":ENSEMBLE_SIZE}
            base={}

            # Roots are collected under the previous complete HU ensemble.
            for domain in DOMAINS:
                runtime=runtimes[domain]
                plan=plans[domain]
                base[domain]={
                    "nodes_before":int(runtime.bundle.counters["nodes"]),
                    "adv_before":int(runtime.bundle.adv_mem.seen),
                    "pol_before":int(runtime.bundle.pol_mem.seen),
                }
                stats=executor.collect(
                    domain=domain,
                    bundle=runtime.bundle,
                    iteration=int(iteration),
                    exact_opponent_levels=int(config.exact_opponent_levels),
                    jobs=plan["jobs"],
                    hu_preflop_board_average_k=1,
                    ensemble_model_states=(hu_states if domain==DOMAIN_HU else None),
                )
                base[domain]["tree_seconds"]=float(stats["seconds"])
                base[domain]["execution_mode"]=f"parallel_{executor.workers}x1"
                base[domain]["root_behavior_ensemble_size"]=int(stats.get("ensemble_size",1))

            # 3H remains the exact LT2 fresh100 path.
            r3=runtimes[DOMAIN_3H]
            fit3_started=time.perf_counter()
            r3.session.reset_advantage_network(
                init_seed=_advantage_reset_seed(seed,DOMAIN_3H,iteration),
                lr=float(config.learning_rate),
            )
            losses3=r3.session.train_advantage(
                steps=int(config.advantage_steps_for_domain(DOMAIN_3H)),
                batch_size=int(config.batch_size),
            )
            fit3_seconds=float(time.perf_counter()-fit3_started)

            # HU remains exact ENS8_A fresh400-per-member mechanics.
            _models,hu_states,hu_member_meta,hu_fit_seconds=lt2ens._fit_hu_ensemble(
                hu_runtime,config,count_optimizer_steps=True
            )

            for domain in DOMAINS:
                runtime=runtimes[domain]
                plan=plans[domain]
                policy_report=_collect_policy_from_episodes(
                    seed=int(seed),
                    iteration=int(iteration),
                    domain=domain,
                    episodes=plan["policy_episodes"],
                    runtime=runtime,
                )
                before=base[domain]
                if domain==DOMAIN_3H:
                    fit_seconds=fit3_seconds
                    optimizer_steps=int(config.advantage_steps_for_domain(domain))
                    loss_last=float(losses3[-1])
                    extra={"ensemble_size":1}
                else:
                    fit_seconds=hu_fit_seconds
                    optimizer_steps=ENSEMBLE_SIZE*MEMBER_STEPS
                    loss_last=float(
                        sum(x["loss_last"] for x in hu_member_meta)/len(hu_member_meta)
                    )
                    extra={
                        "ensemble_size":ENSEMBLE_SIZE,
                        "member_steps":MEMBER_STEPS,
                        "member_loss_last":[float(x["loss_last"]) for x in hu_member_meta],
                        "member_fit_seconds":[float(x["fit_seconds"]) for x in hu_member_meta],
                    }

                report["domains"][domain]={
                    "roots":int(plan["roots"]),
                    "nodes":int(runtime.bundle.counters["nodes"])-int(before["nodes_before"]),
                    "advantage_samples":int(runtime.bundle.adv_mem.seen)-int(before["adv_before"]),
                    "strategy_samples":int(runtime.bundle.pol_mem.seen)-int(before["pol_before"]),
                    "tree_seconds":float(before["tree_seconds"]),
                    "seconds_per_root":float(before["tree_seconds"]/plan["roots"]),
                    "execution_mode":before["execution_mode"],
                    "root_behavior_ensemble_size":int(before["root_behavior_ensemble_size"]),
                    "advantage_fit_seconds":float(fit_seconds),
                    "advantage_optimizer_steps_total_this_iteration":int(optimizer_steps),
                    "advantage_loss_last":loss_last,
                    "sampled_policy":policy_report,
                    **extra,
                }

            report["wall_seconds"]=float(time.perf_counter()-iter_started)
            history.append(report)
            recent_history.append(report)
            completed=iteration

            if completed%int(args.checkpoint_every)==0 or completed==target:
                save_started=time.perf_counter()
                save_checkpoint(
                    args.output_checkpoint,
                    seed=int(seed),
                    config=config,
                    completed_iteration=int(completed),
                    sampler=sampler,
                    runtimes=runtimes,
                    history=history,
                    finalized=False,
                )
                _save_ensemble(
                    args.output_ensemble_state,
                    completed_iteration=int(completed),
                    source_checkpoint=source_checkpoint,
                    source_ensemble=source_ensemble,
                    states=hu_states,
                    member_meta=hu_member_meta,
                )
                checkpoint_rows.append({
                    "iteration":int(completed),
                    "seconds":float(time.perf_counter()-save_started),
                    "checkpoint_bytes":int(args.output_checkpoint.stat().st_size),
                    "ensemble_state_bytes":int(args.output_ensemble_state.stat().st_size),
                })

            _save_progress(args.report,{
                "schema":SCHEMA,
                "status":"RUNNING",
                "research_lane":"LT3_HEAVY_H1",
                "production_lt2_untouched":True,
                "source_iteration":SOURCE_ITERATION,
                "target_iteration":int(target),
                "completed_iteration":int(completed),
                "ensemble_size":ENSEMBLE_SIZE,
                "member_steps":MEMBER_STEPS,
                "checkpoint_rows":checkpoint_rows,
                "recent_history":recent_history,
                "holdout_touched":False,
            })

            print(
                "LT3_H1_ITERATION "
                +json.dumps({
                    "iteration":int(iteration),
                    "target":int(target),
                    "hu_adv_seen":int(hu_runtime.bundle.adv_mem.seen),
                    "hu_strategy_seen":int(hu_runtime.bundle.pol_mem.seen),
                    "hu_fit_seconds":float(hu_fit_seconds),
                    "hu_member_loss_last":[
                        round(float(x["loss_last"]),6) for x in hu_member_meta
                    ],
                    "wall_seconds":float(report["wall_seconds"]),
                },sort_keys=True),
                flush=True,
            )
    finally:
        executor.close()

    print("LT3_H1_FINALIZE average-policy",flush=True)
    final=finalize(config=config,runtimes=runtimes)

    save_checkpoint(
        args.output_checkpoint,
        seed=int(seed),
        config=config,
        completed_iteration=int(completed),
        sampler=sampler,
        runtimes=runtimes,
        history=history,
        finalized=True,
    )
    _save_ensemble(
        args.output_ensemble_state,
        completed_iteration=int(completed),
        source_checkpoint=source_checkpoint,
        source_ensemble=source_ensemble,
        states=hu_states,
        member_meta=hu_member_meta,
    )

    compact=compact_report(seed=int(seed),config=config,history=history,final=final)
    payload={
        "schema":SCHEMA,
        "status":"PASS",
        "research_lane":"LT3_HEAVY_H1",
        "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
        "source_checkpoint":str(source_checkpoint),
        "source_ensemble_state":str(source_ensemble),
        "source_iteration":SOURCE_ITERATION,
        "target_iteration":int(target),
        "completed_iteration":int(completed),
        "source_finalized":bool(source_finalized),
        "intervention":{
            "three_handed":"unchanged single fresh100",
            "true_heads_up":"ENS8 raw-Advantage average of eight fresh400 fits",
            "ensemble_size":ENSEMBLE_SIZE,
            "member_steps":MEMBER_STEPS,
            "member_seed_contract":"same fixed ENS8_A init/batch seeds as LT2",
            "policy_rng_separated_from_fit_rng":True,
            "hu_preflop_board_average_k":1,
            "roots_per_iteration":int(config.roots_per_iteration),
        },
        "method":{
            "source_checkpoint_read_only":True,
            "source_ensemble_read_only":True,
            "lt2_production_artifacts_mutated":False,
            "lt2_final_holdout_touched":False,
            "lt3_holdout_touched":False,
            "workers":int(args.workers),
            "torch_threads_parent":int(torch.get_num_threads()),
            "additional_iterations":int(args.additional_iterations),
            "new_training_roots":int(args.additional_iterations)*int(config.roots_per_iteration),
            "new_hu_optimizer_steps":int(args.additional_iterations)*ENSEMBLE_SIZE*MEMBER_STEPS,
            "new_3h_optimizer_steps":int(args.additional_iterations)*int(config.advantage_steps_for_domain(DOMAIN_3H)),
        },
        "source_member_meta":source_member_meta,
        "checkpoint_rows":checkpoint_rows,
        "recent_history":recent_history,
        "final":final,
        "compact_final":compact.get("final"),
        "output_checkpoint":str(args.output_checkpoint.resolve()),
        "output_ensemble_state":str(args.output_ensemble_state.resolve()),
        "ordinary_checkpoint_warning":"HU current behavior requires output_ensemble_state sidecar; ordinary checkpoint alone is not the LT3 HU current policy",
        "wall_seconds":float(time.perf_counter()-overall_started),
    }
    _save_progress(args.report,payload)

    print("LT3_HEAVY_ENS8_H1_TRAINING_PASS")
    print(f"report={args.report.resolve()}")
    print(f"checkpoint={args.output_checkpoint.resolve()}")
    print(f"ensemble_state={args.output_ensemble_state.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
