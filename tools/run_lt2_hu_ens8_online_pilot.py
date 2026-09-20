#!/usr/bin/env python3
from __future__ import annotations

"""Isolated LT2 HU ENS8 online-feedback pilot from the frozen iteration-8000 checkpoint.

Intervention:
- THREE_HANDED remains the canonical single fresh Advantage fit at 100 steps.
- TRUE_HEADS_UP uses eight independent fresh 400-step Advantage fits on the
  same mature HU reservoir.
- The eight raw Advantage outputs are averaged before the unchanged
  lean_regret_matching_policy.
- The exact same eight predeclared fit seeds are reused every iteration. This
  removes iteration-to-iteration fit-seed churn and isolates reservoir
  evolution.
- Ensemble-fit minibatch RNG is separated from the authoritative policy
  sampling RNG, so adding ensemble members does not consume extra policy RNG.

The source checkpoint is never mutated. This tool writes an isolated ordinary
checkpoint plus a mandatory ensemble sidecar. The ordinary checkpoint stores
the last HU member only and MUST NOT be interpreted as the full current HU
behavior without the sidecar.
"""

import argparse
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

import audit_lt2_stage_a_b_first_divergence as fd
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

DOMAIN_HU="TRUE_HEADS_UP"
DOMAIN_3H="THREE_HANDED"
SCHEMA="SPINCORE_LT2_HU_ENS8_ONLINE_PILOT_V1"
ENSEMBLE_SCHEMA="SPINCORE_LT2_HU_ENS8_CURRENT_STATE_V1"
ENSEMBLE_SIZE=8
MEMBER_STEPS=400


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--source-checkpoint",type=Path,required=True)
    p.add_argument("--output-checkpoint",type=Path,required=True)
    p.add_argument("--ensemble-state",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--additional-iterations",type=int,default=100)
    p.add_argument("--checkpoint-every",type=int,default=25)
    p.add_argument("--workers",type=int,default=31)
    p.add_argument("--threads",type=int,default=8)
    return p.parse_args()


def _member_seeds(member:int)->tuple[int,int]:
    if not 0<=int(member)<ENSEMBLE_SIZE:
        raise ValueError("ensemble member out of range")
    # Exactly the predeclared ENS8_A seeds used by the mature replicated gate.
    init_seed=fd._mix64(20260920,int(member),0xE115E)&0x7FFFFFFF
    batch_seed=fd._mix64(20260920,int(member),0xEBA7C4)&0x7FFFFFFF
    return int(init_seed),int(batch_seed)


def _clone_state(model)->dict[str,torch.Tensor]:
    return {
        key:value.detach().cpu().clone()
        for key,value in model.state_dict().items()
    }


def _fit_hu_ensemble(runtime,config,*,count_optimizer_steps:bool)->tuple[list[Any],tuple[dict,...],list[dict[str,Any]],float]:
    """Fit all eight members without consuming authoritative policy RNG."""
    policy_rng_state=runtime.bundle.batch_rng.getstate()
    counters_before=dict(runtime.bundle.counters)
    models=[]
    states=[]
    meta=[]
    started=time.perf_counter()
    try:
        for member in range(ENSEMBLE_SIZE):
            init_seed,batch_seed=_member_seeds(member)
            runtime.session.reset_advantage_network(
                init_seed=int(init_seed),
                lr=float(config.learning_rate),
            )
            runtime.bundle.batch_rng.seed(int(batch_seed))
            fit_started=time.perf_counter()
            losses=runtime.session.train_advantage(
                steps=MEMBER_STEPS,
                batch_size=int(config.batch_size),
            )
            elapsed=float(time.perf_counter()-fit_started)
            if len(losses)!=MEMBER_STEPS:
                raise RuntimeError("ENS8 member step-count drift")
            if not all(torch.isfinite(torch.tensor(float(x))) for x in losses):
                raise RuntimeError("nonfinite ENS8 fit loss")
            model=runtime.bundle.advantage
            models.append(model)
            states.append(_clone_state(model))
            meta.append({
                "member":int(member),
                "init_seed":int(init_seed),
                "batch_seed":int(batch_seed),
                "steps":MEMBER_STEPS,
                "fit_seconds":elapsed,
                "loss_last":float(losses[-1]),
            })
    finally:
        runtime.bundle.batch_rng.setstate(policy_rng_state)

    if not count_optimizer_steps:
        # Bootstrap defines the source-8000 ensemble policy but is not an
        # iteration. Preserve all authoritative training counters exactly.
        runtime.bundle.counters.clear()
        runtime.bundle.counters.update(counters_before)

    behavior=LeanEnsembleActionAdvantagePolicy(
        models,
        selected_representation="C0_V1_FROZEN_CONTROL",
        device="cpu",
        ready=True,
    )
    runtime.session.behavior=behavior
    runtime.session.collector.policy=behavior
    runtime.bundle.counters["advantage_ready"]=1
    return models,tuple(states),meta,float(time.perf_counter()-started)


def _save_ensemble(path:Path,*,completed_iteration:int,source_checkpoint:Path,states:tuple[dict,...],member_meta:list[dict[str,Any]])->None:
    payload={
        "schema":ENSEMBLE_SCHEMA,
        "source_checkpoint":str(source_checkpoint.resolve()),
        "completed_iteration":int(completed_iteration),
        "ensemble_size":ENSEMBLE_SIZE,
        "member_steps":MEMBER_STEPS,
        "member_meta":member_meta,
        "members":states,
        "semantics":"mean raw Advantage outputs then unchanged lean_regret_matching_policy",
        "fit_seed_contract":"same predeclared ENS8_A init/batch seeds reused every iteration",
        "policy_rng_contract":"ensemble minibatch RNG isolated; authoritative policy RNG restored before sampled-policy collection",
    }
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(payload,tmp)
    os.replace(tmp,path)


def _save_progress_report(path:Path,payload:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def main():
    args=parse_args()
    if args.additional_iterations<=0 or args.checkpoint_every<=0 or args.workers<=1 or args.threads<=0:
        raise SystemExit("invalid positive pilot arguments")

    solver_path=args.solver.resolve(strict=True)
    source=args.source_checkpoint.resolve(strict=True)
    torch.set_num_threads(int(args.threads))
    solver=SolverLibrary(solver_path)

    seed,source_config,completed,sampler,runtimes,history,finalized=load_checkpoint(
        source,solver=solver
    )
    if int(completed)!=8000:
        raise RuntimeError(f"expected source iteration 8000, got {completed}")
    if int(source_config.advantage_steps_for_domain(DOMAIN_3H))!=100:
        raise RuntimeError("expected frozen 3H advantage budget 100")
    if int(source_config.hu_preflop_board_average_k)!=1:
        raise RuntimeError("ENS8 pilot requires K4/off canonical K=1")

    target=int(completed)+int(args.additional_iterations)
    config=replace(
        source_config,
        iterations=target,
        hu_advantage_steps=MEMBER_STEPS,
        hu_preflop_board_average_k=1,
    )

    for runtime in runtimes.values():
        runtime.session.batch_mode="vectorized"

    hu_runtime=runtimes[DOMAIN_HU]
    # Bootstrap the exact already-validated ENS8_A composition from the frozen
    # source reservoir. These fits define iteration-8000 current ensemble
    # behavior but are not charged to iteration optimizer counters.
    _models,hu_states,bootstrap_meta,bootstrap_seconds=_fit_hu_ensemble(
        hu_runtime,config,count_optimizer_steps=False
    )

    pilot_history=[]
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

            # Root collection under the pre-fit policies. HU sees the previous
            # iteration's complete ensemble, never an arbitrary member.
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

            # 3H remains unchanged: one fresh 100-step fit.
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

            # HU intervention: eight fresh400 estimators from the same newly
            # updated reservoir. Raw outputs are averaged for policy collection
            # and for the next iteration's root traversal.
            _models,hu_states,hu_member_meta,hu_fit_seconds=_fit_hu_ensemble(
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
                    advantage_steps=int(config.advantage_steps_for_domain(domain))
                    loss_last=float(losses3[-1])
                    extra={"ensemble_size":1}
                else:
                    fit_seconds=hu_fit_seconds
                    advantage_steps=ENSEMBLE_SIZE*MEMBER_STEPS
                    loss_last=float(sum(x["loss_last"] for x in hu_member_meta)/len(hu_member_meta))
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
                    "advantage_optimizer_steps_total_this_iteration":int(advantage_steps),
                    "advantage_loss_last":loss_last,
                    "sampled_policy":policy_report,
                    **extra,
                }

            report["wall_seconds"]=float(time.perf_counter()-iter_started)
            history.append(report)
            pilot_history.append(report)
            completed=iteration

            if (
                completed%int(args.checkpoint_every)==0
                or completed==target
            ):
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
                    args.ensemble_state,
                    completed_iteration=int(completed),
                    source_checkpoint=source,
                    states=hu_states,
                    member_meta=hu_member_meta,
                )
                checkpoint_rows.append({
                    "iteration":int(completed),
                    "seconds":float(time.perf_counter()-save_started),
                    "checkpoint_bytes":int(args.output_checkpoint.stat().st_size),
                    "ensemble_state_bytes":int(args.ensemble_state.stat().st_size),
                })

            print(
                "ENS8_ITERATION "
                +json.dumps({
                    "iteration":int(iteration),
                    "hu_adv_seen":int(hu_runtime.bundle.adv_mem.seen),
                    "hu_strategy_seen":int(hu_runtime.bundle.pol_mem.seen),
                    "hu_fit_seconds":float(hu_fit_seconds),
                    "hu_member_loss_last":[round(float(x["loss_last"]),6) for x in hu_member_meta],
                    "hu_policy_actions":policy_report["action_counts"] if domain==DOMAIN_HU else None,
                    "wall_seconds":float(report["wall_seconds"]),
                },sort_keys=True),
                flush=True,
            )

            partial={
                "schema":SCHEMA,
                "status":"RUNNING",
                "source_checkpoint":str(source),
                "source_iteration":8000,
                "target_iteration":int(target),
                "completed_iteration":int(completed),
                "ensemble_size":ENSEMBLE_SIZE,
                "member_steps":MEMBER_STEPS,
                "bootstrap_seconds":float(bootstrap_seconds),
                "bootstrap_member_meta":bootstrap_meta,
                "checkpoint_rows":checkpoint_rows,
                "pilot_history":pilot_history,
            }
            _save_progress_report(args.report,partial)
    finally:
        executor.close()

    print("FINALIZE average-policy",flush=True)
    final=finalize(config=config,runtimes=runtimes)

    # Final ordinary checkpoint + mandatory ensemble sidecar.
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
        args.ensemble_state,
        completed_iteration=int(completed),
        source_checkpoint=source,
        states=hu_states,
        member_meta=hu_member_meta,
    )

    compact=compact_report(seed=int(seed),config=config,history=history,final=final)
    payload={
        "schema":SCHEMA,
        "status":"PASS",
        "source_checkpoint":str(source),
        "source_iteration":8000,
        "target_iteration":int(target),
        "completed_iteration":int(completed),
        "source_finalized":bool(finalized),
        "intervention":{
            "three_handed":"unchanged single fresh100",
            "true_heads_up":"ENS8 raw-Advantage average of eight fresh400 fits",
            "ensemble_size":ENSEMBLE_SIZE,
            "member_steps":MEMBER_STEPS,
            "member_seed_contract":"exact predeclared ENS8_A seeds reused every iteration",
            "policy_rng_separated_from_fit_rng":True,
            "hu_preflop_board_average_k":1,
        },
        "method":{
            "source_checkpoint_read_only":True,
            "holdout_touched":False,
            "workers":int(args.workers),
            "torch_threads_parent":int(torch.get_num_threads()),
            "additional_iterations":int(args.additional_iterations),
            "new_training_roots":int(args.additional_iterations)*int(config.roots_per_iteration),
        },
        "bootstrap_seconds":float(bootstrap_seconds),
        "bootstrap_member_meta":bootstrap_meta,
        "checkpoint_rows":checkpoint_rows,
        "pilot_history":pilot_history,
        "final":final,
        "compact_final":compact.get("final"),
        "output_checkpoint":str(args.output_checkpoint.resolve()),
        "ensemble_state":str(args.ensemble_state.resolve()),
        "ordinary_checkpoint_warning":"HU current behavior requires ensemble_state sidecar; ordinary checkpoint stores only the final single member in bundle.advantage",
        "wall_seconds":float(time.perf_counter()-overall_started),
    }
    _save_progress_report(args.report,payload)
    print("LT2_HU_ENS8_ONLINE_PILOT_PASS")
    print(f"report={args.report.resolve()}")
    print(f"checkpoint={args.output_checkpoint.resolve()}")
    print(f"ensemble_state={args.ensemble_state.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
