#!/usr/bin/env python3
from __future__ import annotations

"""Broad EV gate for two disjoint size-4 mature HU Advantage ensembles.

Recreates the same eight deterministic 400-step fits used by the mature
root-stability gate, then evaluates two disjoint raw-Advantage ensembles over
complete paired HU hands against UNIFORM_LEGAL, PASSIVE_CALLER and JAMMER.

Context policies are production current behavior at iterations 7500, 7600 and
8000.  Common scenarios, deals, seats and hero RNG are used across all hero
policies.  Source checkpoints are read only.  No CFR roots or holdout seeds.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import torch

import audit_lt2_hu_policy_chain as chain
import audit_lt2_hu_preflop_conditional_resampling as cond
import audit_lt2_jammer_facing_allin_target_overlay as overlay
import audit_lt2_stage_a_b_first_divergence as fd
import audit_lt2_hu_b400_broad_generalization as broadgen
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_functional_training import load_checkpoint
from spincore.r7_5_action_cfr import legal_mask
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.solver import Episode, SolverLibrary

BASELINES=("UNIFORM_LEGAL","PASSIVE_CALLER","JAMMER")
POLICIES=("PROD_7500","PROD_7600","PROD_8000","ENS4_LEFT","ENS4_RIGHT")
FORENSIC_SEEDS=(20260920,20260921,20260922,20260923,20260924,20260925)
DOMAIN="TRUE_HEADS_UP"

_SOLVER=None
_STAGES=None


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--stage-7500",type=Path,required=True)
    p.add_argument("--stage-7600",type=Path,required=True)
    p.add_argument("--stage-8000",type=Path,required=True)
    p.add_argument("--budget",type=int,default=400)
    p.add_argument("--scenarios-per-seed",type=int,default=5000)
    p.add_argument("--workers",type=int,default=31)
    p.add_argument("--threads-fit",type=int,default=8)
    p.add_argument("--report",type=Path,required=True)
    return p.parse_args()


def _mean_ci(values):
    xs=[float(x) for x in values]
    n=len(xs)
    mean=float(statistics.fmean(xs))
    sem=0.0 if n<=1 else float(statistics.stdev(xs)/math.sqrt(n))
    h=1.96*sem
    return {"n":n,"mean":mean,"sem":sem,"ci95_low":mean-h,"ci95_high":mean+h}


def _build_candidates(*,solver,stage8000,out_dir,budget,threads):
    torch.set_num_threads(int(threads))
    _seed,config,iteration,_sampler,runtimes,_history,_finalized=load_checkpoint(
        stage8000,solver=solver
    )
    if int(iteration)!=8000:
        raise RuntimeError(f"unexpected source iteration {iteration}")
    runtime=runtimes[DOMAIN]
    runtime.session.batch_mode="vectorized"
    paths=[]
    meta=[]
    for rep in range(8):
        init_seed=fd._mix64(20260920,rep,0xE115E)&0x7fffffff
        batch_seed=fd._mix64(20260920,rep,0xEBA7C4)&0x7fffffff
        runtime.session.reset_advantage_network(
            init_seed=int(init_seed),lr=float(config.learning_rate)
        )
        runtime.bundle.batch_rng.seed(int(batch_seed))
        losses=runtime.session.train_advantage(
            steps=int(budget),batch_size=int(config.batch_size)
        )
        path=out_dir/f"fresh400_r{rep}.pt"
        broadgen._save_candidate_snapshot(
            runtime=runtime,
            source_checkpoint=stage8000,
            iteration=iteration,
            path=path,
        )
        paths.append(path)
        meta.append({
            "replicate":rep,
            "init_seed":int(init_seed),
            "batch_seed":int(batch_seed),
            "steps":int(budget),
            "loss_last":float(losses[-1]),
            "snapshot":str(path.resolve()),
        })
    return paths,meta


def _init_worker(solver_path,names,snapshots):
    global _SOLVER,_STAGES
    import os
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    os.environ["SPINCORE_TORCH_THREADS"]="1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _SOLVER=SolverLibrary(solver_path)
    _STAGES={
        str(name):overlay.StageModels(Path(path),_SOLVER)
        for name,path in zip(names,snapshots)
    }


def _hero_distribution(policy,state):
    active_mask,legal=chain._legal_context(state)
    observation=state.neural_bytes()
    if policy in ("PROD_7500","PROD_7600","PROD_8000"):
        stage=_STAGES[policy]
        probs=stage.runtime.session.behavior(state,observation,legal)
        out=tuple(float(x) for x in probs)
    else:
        members=range(0,4) if policy=="ENS4_LEFT" else range(4,8)
        mask=legal_mask(legal)
        raws=[
            cond._model_raw(_STAGES[f"R{i}"].runtime,observation,mask)
            for i in members
        ]
        raw=torch.stack(raws,dim=0).mean(dim=0)
        out=lean_regret_matching_policy(raw.tolist(),legal)
    mass=sum(out[a] for a in legal)
    if not (0.999<=mass<=1.001):
        raise RuntimeError(f"behavior probability mass drift: {mass}")
    return int(active_mask),legal,out


def _play_one(
    episode:Episode,*,
    deal_seed:int,
    hero_seat:int,
    hero_policy:str,
    opponent_policy:str,
    scenario_index:int,
    master_seed:int,
):
    state=_SOLVER.create(episode,int(deal_seed))
    seat_rng={
        seat:random.Random(
            fd._mix64(master_seed,scenario_index,seat,100+BASELINES.index(opponent_policy))
        )
        for seat in range(3)
    }
    seat_rng[int(hero_seat)]=random.Random(
        fd._mix64(master_seed,scenario_index,hero_seat,777)
    )
    decisions=0
    try:
        while not state.terminal:
            actor=int(state.actor)
            if actor==int(hero_seat):
                active_mask,legal,probs=_hero_distribution(hero_policy,state)
                slot=chain._sample_probs(legal,probs,seat_rng[actor])
            else:
                active_mask,slot=chain._baseline_action(
                    opponent_policy,state,seat_rng[actor]
                )
            chain.apply_lean(state,active_mask,slot)
            decisions+=1
            if decisions>200:
                raise RuntimeError("ensemble EV hand exceeded 200 decisions")
        delta=state.terminal_chip_delta()
        if sum(delta)!=0:
            raise RuntimeError(f"nonzero-sum delta: {delta}")
        return int(delta[int(hero_seat)])
    finally:
        state.close()


def _worker(task):
    seed,scenario_index,episode,deal_seed=task
    if not episode.game_is_hu:
        return []
    live=[s for s,stack in enumerate(episode.stacks) if int(stack)>0]
    if len(live)!=2:
        raise RuntimeError("HU episode without exactly two live seats")
    rows=[]
    for baseline in BASELINES:
        for hero in live:
            values={
                policy:_play_one(
                    episode,
                    deal_seed=int(deal_seed),
                    hero_seat=int(hero),
                    hero_policy=policy,
                    opponent_policy=baseline,
                    scenario_index=int(scenario_index),
                    master_seed=int(seed),
                )
                for policy in POLICIES
            }
            row={
                "seed":int(seed),
                "scenario":int(scenario_index),
                "hero_seat":int(hero),
                "baseline":baseline,
                **{p.lower():int(v) for p,v in values.items()},
            }
            for ens in ("ENS4_LEFT","ENS4_RIGHT"):
                k=ens.lower()
                row[f"{k}_minus_prod_7500"]=int(values[ens]-values["PROD_7500"])
                row[f"{k}_minus_prod_7600"]=int(values[ens]-values["PROD_7600"])
                row[f"{k}_minus_prod_8000"]=int(values[ens]-values["PROD_8000"])
            row["ens4_right_minus_left"]=int(values["ENS4_RIGHT"]-values["ENS4_LEFT"])
            rows.append(row)
    return rows


def _cluster(rows,baseline,key):
    by={}
    for row in rows:
        if row["baseline"]!=baseline:
            continue
        k=(int(row["seed"]),int(row["scenario"]))
        by.setdefault(k,[]).append(float(row[key]))
    return [float(statistics.fmean(v)) for v in by.values()]


def _summary(rows):
    out={}
    for baseline in BASELINES:
        block={
            "clusters":len({(r["seed"],r["scenario"]) for r in rows if r["baseline"]==baseline}),
            "absolute":{},
            "ensembles":{},
        }
        for policy in POLICIES:
            block["absolute"][policy]=_mean_ci(
                _cluster(rows,baseline,policy.lower())
            )
        for ens in ("ENS4_LEFT","ENS4_RIGHT"):
            k=ens.lower()
            block["ensembles"][ens]={
                "minus_prod_7500":_mean_ci(_cluster(rows,baseline,f"{k}_minus_prod_7500")),
                "minus_prod_7600":_mean_ci(_cluster(rows,baseline,f"{k}_minus_prod_7600")),
                "minus_prod_8000":_mean_ci(_cluster(rows,baseline,f"{k}_minus_prod_8000")),
            }
        block["ens4_right_minus_left"]=_mean_ci(
            _cluster(rows,baseline,"ens4_right_minus_left")
        )
        out[baseline]=block
    return out


def main():
    args=parse_args()
    if args.budget<=0 or args.scenarios_per_seed<=0 or args.workers<=0:
        raise SystemExit("positive args required")
    solver_path=args.solver.resolve(strict=True)
    p7500=args.stage_7500.resolve(strict=True)
    p7600=args.stage_7600.resolve(strict=True)
    p8000=args.stage_8000.resolve(strict=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)

    local_solver=SolverLibrary(solver_path)
    prod7500=args.report.parent/"prod7500.pt"
    prod7600=args.report.parent/"prod7600.pt"
    prod8000=args.report.parent/"prod8000.pt"
    m7500=overlay._extract_stage_snapshot(p7500,prod7500)
    m7600=overlay._extract_stage_snapshot(p7600,prod7600)
    m8000=overlay._extract_stage_snapshot(p8000,prod8000)
    if (int(m7500["completed_iteration"]),int(m7600["completed_iteration"]),int(m8000["completed_iteration"]))!=(7500,7600,8000):
        raise RuntimeError("production iteration mismatch")

    candidate_paths,candidate_meta=_build_candidates(
        solver=local_solver,
        stage8000=p8000,
        out_dir=args.report.parent,
        budget=int(args.budget),
        threads=int(args.threads_fit),
    )
    del local_solver
    gc.collect()

    names=["PROD_7500","PROD_7600","PROD_8000"]+[f"R{i}" for i in range(8)]
    snapshots=[prod7500,prod7600,prod8000,*candidate_paths]

    tasks=[]
    hu_counts={}
    for seed in FORENSIC_SEEDS:
        sampler=LegacyScenarioSampler(seed=int(seed)^0x5CE0A710,config=LegacyScenarioConfig())
        hu=0
        for idx in range(int(args.scenarios_per_seed)):
            ep=sampler.sample_episode()
            if ep.game_is_hu:
                hu+=1
            deal_seed=fd._mix64(int(seed),int(idx),0xD34A1)
            tasks.append((int(seed),int(idx),ep,int(deal_seed)))
        hu_counts[str(seed)]=hu

    rows=[]
    ctx=mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers),len(tasks)),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(solver_path),names,[str(p.resolve()) for p in snapshots]),
    ) as pool:
        for chunk in pool.map(_worker,tasks,chunksize=1):
            rows.extend(chunk)

    summary=_summary(rows)
    out={
        "schema":"SPINCORE_LT2_HU_MATURE_ENS4_BROAD_EV_V1",
        "sources":{
            "prod_7500":str(p7500),
            "prod_7600":str(p7600),
            "prod_8000":str(p8000),
        },
        "candidate_meta":candidate_meta,
        "ensembles":{
            "ENS4_LEFT":[0,1,2,3],
            "ENS4_RIGHT":[4,5,6,7],
            "semantics":"mean raw Advantage outputs then unchanged lean_regret_matching_policy at every hero decision",
        },
        "method":{
            "read_only_source_checkpoints":True,
            "new_training_roots":0,
            "source_training_memory_writes":0,
            "candidate_optimizer_steps":8*int(args.budget),
            "future_holdout_seeds_touched":False,
            "forensic_seeds":list(FORENSIC_SEEDS),
            "scenarios_per_seed":int(args.scenarios_per_seed),
            "hu_scenarios_by_seed":hu_counts,
            "baselines":list(BASELINES),
            "policies":list(POLICIES),
            "pairing":"same HU scenario, deal, hero seat and hero RNG across all policies",
        },
        "summary":summary,
        "rows_count":len(rows),
    }
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== MATURE HU SIZE-4 ENSEMBLE BROAD EV ===")
    for baseline in BASELINES:
        b=summary[baseline]
        print(baseline)
        for ens in ("ENS4_LEFT","ENS4_RIGHT"):
            x=b["ensembles"][ens]
            a=x["minus_prod_7500"]; c=x["minus_prod_7600"]
            print(
                f"  {ens}: vs7500={a['mean']:+.3f} "
                f"CI=[{a['ci95_low']:+.3f},{a['ci95_high']:+.3f}] "
                f"vs7600={c['mean']:+.3f} "
                f"CI=[{c['ci95_low']:+.3f},{c['ci95_high']:+.3f}]"
            )
        d=b["ens4_right_minus_left"]
        print(
            f"  RIGHT-LEFT={d['mean']:+.3f} "
            f"CI=[{d['ci95_low']:+.3f},{d['ci95_high']:+.3f}]"
        )
    print("LT2_HU_MATURE_ENS4_BROAD_EV_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
