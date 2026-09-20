#!/usr/bin/env python3
from __future__ import annotations

"""Replicated mature HU size-8 ensemble broad-EV gate.

Build two independent deterministic groups of eight fresh 400-step Advantage
fits from the frozen iteration-8000 HU reservoir. Evaluate each group in a
separate worker pool to keep memory close to the successful ENS4 gate. Both
groups see identical forensic scenarios, deals, seats and hero RNG.

Read-only source checkpoints; no CFR roots; no holdout.
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
PROD_POLICIES=("PROD_7500","PROD_7600","PROD_8000")
FORENSIC_SEEDS=(20260920,20260921,20260922,20260923,20260924,20260925)
DOMAIN="TRUE_HEADS_UP"

_SOLVER=None
_STAGES=None
_ENSEMBLE_NAME=None


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


def _fit_group(*,solver,stage8000,out_dir,budget,threads,start_rep):
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
    for rep in range(int(start_rep),int(start_rep)+8):
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
            "replicate":int(rep),
            "init_seed":int(init_seed),
            "batch_seed":int(batch_seed),
            "steps":int(budget),
            "loss_last":float(losses[-1]),
            "snapshot":str(path.resolve()),
        })
    return paths,meta


def _init_worker(solver_path,names,snapshots,ensemble_name):
    global _SOLVER,_STAGES,_ENSEMBLE_NAME
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
    _ENSEMBLE_NAME=str(ensemble_name)


def _hero_distribution(policy,state):
    active_mask,legal=chain._legal_context(state)
    observation=state.neural_bytes()
    if policy in PROD_POLICIES:
        stage=_STAGES[policy]
        probs=stage.runtime.session.behavior(state,observation,legal)
        out=tuple(float(x) for x in probs)
    elif policy==_ENSEMBLE_NAME:
        mask=legal_mask(legal)
        raws=[
            cond._model_raw(_STAGES[f"R{i}"].runtime,observation,mask)
            for i in range(8)
        ]
        raw=torch.stack(raws,dim=0).mean(dim=0)
        out=lean_regret_matching_policy(raw.tolist(),legal)
    else:
        raise ValueError(policy)
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
                raise RuntimeError("ENS8 EV hand exceeded 200 decisions")
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
    policies=(*PROD_POLICIES,_ENSEMBLE_NAME)
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
                for policy in policies
            }
            rows.append({
                "seed":int(seed),
                "scenario":int(scenario_index),
                "hero_seat":int(hero),
                "baseline":baseline,
                **{p.lower():int(v) for p,v in values.items()},
            })
    return rows


def _cluster(rows,baseline,key):
    by={}
    for row in rows:
        if row["baseline"]!=baseline:
            continue
        ck=(int(row["seed"]),int(row["scenario"]))
        by.setdefault(ck,[]).append(float(row[key]))
    return [float(statistics.fmean(v)) for v in by.values()]


def _evaluate_group(*,solver_path,prod_paths,candidate_paths,tasks,workers,ensemble_name):
    names=[*PROD_POLICIES]+[f"R{i}" for i in range(8)]
    snapshots=[*prod_paths,*candidate_paths]
    rows=[]
    ctx=mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(workers),len(tasks)),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(
            str(solver_path),
            names,
            [str(p.resolve()) for p in snapshots],
            str(ensemble_name),
        ),
    ) as pool:
        for chunk in pool.map(_worker,tasks,chunksize=1):
            rows.extend(chunk)
    return rows


def _summarize_group(rows,ensemble_name):
    out={}
    ekey=ensemble_name.lower()
    for baseline in BASELINES:
        block={
            "clusters":len({(r["seed"],r["scenario"]) for r in rows if r["baseline"]==baseline}),
            "absolute":{},
            "deltas":{},
        }
        for policy in (*PROD_POLICIES,ensemble_name):
            block["absolute"][policy]=_mean_ci(
                _cluster(rows,baseline,policy.lower())
            )
        for prod in PROD_POLICIES:
            vals=[
                e-p for e,p in zip(
                    _cluster(rows,baseline,ekey),
                    _cluster(rows,baseline,prod.lower()),
                )
            ]
            block["deltas"][f"{ensemble_name}_minus_{prod}"]=_mean_ci(vals)
        out[baseline]=block
    return out


def _paired_difference(rows_a,rows_b):
    index_b={
        (r["seed"],r["scenario"],r["hero_seat"],r["baseline"]):r
        for r in rows_b
    }
    out={}
    for baseline in BASELINES:
        by={}
        for ra in rows_a:
            if ra["baseline"]!=baseline:
                continue
            k=(ra["seed"],ra["scenario"],ra["hero_seat"],ra["baseline"])
            rb=index_b[k]
            ck=(int(ra["seed"]),int(ra["scenario"]))
            by.setdefault(ck,[]).append(
                float(rb["ens8_b"])-float(ra["ens8_a"])
            )
        out[baseline]=_mean_ci([
            float(statistics.fmean(v)) for v in by.values()
        ])
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

    group_a_paths,meta_a=_fit_group(
        solver=local_solver,stage8000=p8000,out_dir=args.report.parent,
        budget=int(args.budget),threads=int(args.threads_fit),start_rep=0,
    )
    group_b_paths,meta_b=_fit_group(
        solver=local_solver,stage8000=p8000,out_dir=args.report.parent,
        budget=int(args.budget),threads=int(args.threads_fit),start_rep=8,
    )
    del local_solver
    gc.collect()

    tasks=[]
    hu_counts={}
    for seed in FORENSIC_SEEDS:
        sampler=LegacyScenarioSampler(
            seed=int(seed)^0x5CE0A710,config=LegacyScenarioConfig()
        )
        hu=0
        for idx in range(int(args.scenarios_per_seed)):
            ep=sampler.sample_episode()
            if ep.game_is_hu:
                hu+=1
            deal_seed=fd._mix64(int(seed),int(idx),0xD34A1)
            tasks.append((int(seed),int(idx),ep,int(deal_seed)))
        hu_counts[str(seed)]=hu

    prod_paths=(prod7500,prod7600,prod8000)
    rows_a=_evaluate_group(
        solver_path=solver_path,prod_paths=prod_paths,
        candidate_paths=group_a_paths,tasks=tasks,workers=args.workers,
        ensemble_name="ENS8_A",
    )
    gc.collect()
    rows_b=_evaluate_group(
        solver_path=solver_path,prod_paths=prod_paths,
        candidate_paths=group_b_paths,tasks=tasks,workers=args.workers,
        ensemble_name="ENS8_B",
    )

    summary_a=_summarize_group(rows_a,"ENS8_A")
    summary_b=_summarize_group(rows_b,"ENS8_B")
    pair=_paired_difference(rows_a,rows_b)

    out={
        "schema":"SPINCORE_LT2_HU_MATURE_ENS8_REPLICATED_BROAD_EV_V1",
        "sources":{
            "prod_7500":str(p7500),
            "prod_7600":str(p7600),
            "prod_8000":str(p8000),
        },
        "candidate_meta":{"ENS8_A":meta_a,"ENS8_B":meta_b},
        "ensembles":{
            "ENS8_A":list(range(0,8)),
            "ENS8_B":list(range(8,16)),
            "semantics":"mean raw Advantage outputs then unchanged lean_regret_matching_policy at every hero decision",
        },
        "method":{
            "read_only_source_checkpoints":True,
            "new_training_roots":0,
            "source_training_memory_writes":0,
            "candidate_optimizer_steps":16*int(args.budget),
            "future_holdout_seeds_touched":False,
            "forensic_seeds":list(FORENSIC_SEEDS),
            "scenarios_per_seed":int(args.scenarios_per_seed),
            "hu_scenarios_by_seed":hu_counts,
            "baselines":list(BASELINES),
            "pairing":"identical HU scenario, deal, hero seat and hero RNG across ENS8_A/ENS8_B and productions",
            "group_evaluation":"separate worker pools to bound memory; deterministic identical task/RNG schedule",
        },
        "summary":{"ENS8_A":summary_a,"ENS8_B":summary_b},
        "ens8_b_minus_a":pair,
        "rows_count_each_group":len(rows_a),
    }
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== MATURE HU REPLICATED SIZE-8 ENSEMBLE BROAD EV ===")
    for baseline in BASELINES:
        a=summary_a[baseline]["absolute"]["ENS8_A"]
        b=summary_b[baseline]["absolute"]["ENS8_B"]
        d=pair[baseline]
        print(
            f"{baseline}: A={a['mean']:+.3f} "
            f"CI=[{a['ci95_low']:+.3f},{a['ci95_high']:+.3f}] "
            f"B={b['mean']:+.3f} "
            f"CI=[{b['ci95_low']:+.3f},{b['ci95_high']:+.3f}] "
            f"B-A={d['mean']:+.3f} "
            f"CI=[{d['ci95_low']:+.3f},{d['ci95_high']:+.3f}]"
        )
    print("LT2_HU_MATURE_ENS8_REPLICATED_BROAD_EV_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
