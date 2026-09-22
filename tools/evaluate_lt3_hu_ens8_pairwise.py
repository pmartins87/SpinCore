#!/usr/bin/env python3
from __future__ import annotations

"""Paired HU development comparison for two frozen ENS8 sidecars.

This is a development diagnostic, not a sealed holdout and not an exploitability
proof. It compares current HU behavior with exact seat rotation on shared deals,
plus paired deltas against transparent weak baselines.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import torch

import audit_lt2_hu_policy_chain as chain
import audit_lt2_stage_a_b_first_divergence as fd
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import legal_mask
from spincore.solver import Episode, SolverLibrary
from spincore_nn.action_models import collate_action_observations, make_advantage_action_model

REPRESENTATION="C0_V1_FROZEN_CONTROL"
ALLOWED_SCHEMAS={
    "SPINCORE_LT2_HU_ENS8_CURRENT_STATE_V1",
    "SPINCORE_LT3_HU_ENS8_CURRENT_STATE_V1",
}
BASELINES=("UNIFORM_LEGAL","PASSIVE_CALLER","JAMMER")

_SOLVER=None
_BEFORE=None
_AFTER=None


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--before-ensemble",type=Path,required=True)
    p.add_argument("--after-ensemble",type=Path,required=True)
    p.add_argument("--scenarios",type=int,default=3000)
    p.add_argument("--workers",type=int,default=31)
    p.add_argument("--seed",type=int,default=20260922)
    p.add_argument("--report",type=Path,required=True)
    return p.parse_args()


def _load(path:Path):
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("schema") not in ALLOWED_SCHEMAS:
        raise RuntimeError(f"unsupported ENS8 schema {payload.get('schema')!r}: {path}")
    members=payload.get("members")
    if int(payload.get("ensemble_size",-1))!=8 or not isinstance(members,(list,tuple)) or len(members)!=8:
        raise RuntimeError(f"bad ENS8 member payload: {path}")
    models=[]
    for idx,state in enumerate(members):
        _,model=make_advantage_action_model(REPRESENTATION,device="cpu",seed=idx)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models,int(payload.get("completed_iteration",-1)),str(payload.get("schema"))


def _dist(models,state):
    active_mask,legal=chain._legal_context(state)
    batch=collate_action_observations(
        REPRESENTATION,
        [state.neural_bytes()],
        [legal_mask(legal)],
        device="cpu",
    )
    with torch.no_grad():
        raw=torch.stack([m(batch)[0] for m in models],dim=0).mean(dim=0).detach().cpu().tolist()
    return int(active_mask),legal,lean_regret_matching_policy(raw,legal)


def _sample(models,state,rng):
    active_mask,legal,probs=_dist(models,state)
    return int(active_mask),chain._sample_probs(legal,probs,rng)


def _init_worker(solver_path,before_path,after_path):
    global _SOLVER,_BEFORE,_AFTER
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
    _BEFORE,_,_=_load(Path(before_path))
    _AFTER,_,_=_load(Path(after_path))


def _play_direct(ep:Episode,deal_seed:int,after_seat:int,scenario:int,seed:int)->int:
    live=[i for i,s in enumerate(ep.stacks) if int(s)>0]
    opp=live[1] if after_seat==live[0] else live[0]
    state=_SOLVER.create(ep,int(deal_seed))
    rngs={
        int(seat):random.Random(fd._mix64(seed,scenario,9000,int(seat)))
        for seat in live
    }
    try:
        decisions=0
        while not state.terminal:
            actor=int(state.actor)
            models=_AFTER if actor==int(after_seat) else _BEFORE
            mask,slot=_sample(models,state,rngs[actor])
            chain.apply_lean(state,mask,slot)
            decisions+=1
            if decisions>200:
                raise RuntimeError("HU direct hand exceeded 200 decisions")
        delta=tuple(int(x) for x in state.terminal_chip_delta())
        if sum(delta)!=0:
            raise RuntimeError(f"non-zero-sum terminal delta {delta}")
        return int(delta[int(after_seat)])
    finally:
        state.close()


def _play_weak(ep:Episode,deal_seed:int,hero:int,models,baseline:str,scenario:int,seed:int)->int:
    state=_SOLVER.create(ep,int(deal_seed))
    rngs={
        seat:random.Random(fd._mix64(seed,scenario,seat,10000+BASELINES.index(baseline)))
        for seat in range(3)
    }
    rngs[int(hero)]=random.Random(fd._mix64(seed,scenario,hero,10777))
    try:
        decisions=0
        while not state.terminal:
            actor=int(state.actor)
            if actor==int(hero):
                mask,slot=_sample(models,state,rngs[actor])
            else:
                mask,slot=chain._baseline_action(baseline,state,rngs[actor])
            chain.apply_lean(state,mask,slot)
            decisions+=1
            if decisions>200:
                raise RuntimeError("HU weak-baseline hand exceeded 200 decisions")
        delta=tuple(int(x) for x in state.terminal_chip_delta())
        if sum(delta)!=0:
            raise RuntimeError(f"non-zero-sum terminal delta {delta}")
        return int(delta[int(hero)])
    finally:
        state.close()


def _worker(task):
    scenario,ep,deal_seed,seed=task
    if not ep.game_is_hu:
        return []
    live=[i for i,s in enumerate(ep.stacks) if int(s)>0]
    if len(live)!=2:
        raise RuntimeError("HU scenario without exactly two live seats")
    rows=[]
    for after_seat in live:
        rows.append({
            "mode":"DIRECT",
            "scenario":int(scenario),
            "after_seat":int(after_seat),
            "after_chip_delta":_play_direct(ep,deal_seed,after_seat,scenario,seed),
        })
    for baseline in BASELINES:
        for hero in live:
            before=_play_weak(ep,deal_seed,hero,_BEFORE,baseline,scenario,seed)
            after=_play_weak(ep,deal_seed,hero,_AFTER,baseline,scenario,seed)
            rows.append({
                "mode":"WEAK",
                "scenario":int(scenario),
                "baseline":baseline,
                "hero_seat":int(hero),
                "before":int(before),
                "after":int(after),
                "delta":int(after-before),
            })
    return rows


def _mean_ci(values):
    xs=[float(x) for x in values]
    if not xs:
        return {"scenario_clusters":0,"mean":float("nan"),"sem":float("nan"),"ci95_low":float("nan"),"ci95_high":float("nan")}
    mean=float(statistics.fmean(xs))
    sem=0.0 if len(xs)==1 else float(statistics.stdev(xs)/math.sqrt(len(xs)))
    h=1.96*sem
    return {"scenario_clusters":len(xs),"mean":mean,"sem":sem,"ci95_low":mean-h,"ci95_high":mean+h}


def _cluster(rows,key,mode,baseline=None):
    by={}
    for r in rows:
        if r["mode"]!=mode:
            continue
        if baseline is not None and r.get("baseline")!=baseline:
            continue
        by.setdefault(int(r["scenario"]),[]).append(float(r[key]))
    return _mean_ci([statistics.fmean(v) for v in by.values()])


def main()->int:
    args=parse_args()
    solver=args.solver.resolve(strict=True)
    before=args.before_ensemble.resolve(strict=True)
    after=args.after_ensemble.resolve(strict=True)
    _,before_iter,before_schema=_load(before)
    _,after_iter,after_schema=_load(after)
    if before_iter<0 or after_iter<0:
        raise RuntimeError("missing ensemble iteration metadata")

    sampler=LegacyScenarioSampler(seed=int(args.seed)^0x5CE0A710,config=LegacyScenarioConfig())
    tasks=[]
    hu_scenarios=0
    for i in range(int(args.scenarios)):
        ep=sampler.sample_episode()
        if ep.game_is_hu:
            hu_scenarios+=1
        tasks.append((i,ep,fd._mix64(int(args.seed),i,0xD34A1),int(args.seed)))

    rows=[]
    ctx=mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers),max(1,len(tasks))),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(solver),str(before),str(after)),
    ) as pool:
        for chunk in pool.map(_worker,tasks,chunksize=1):
            rows.extend(chunk)

    direct=_cluster(rows,"after_chip_delta","DIRECT")
    weak={}
    for baseline in BASELINES:
        weak[baseline]={
            "before":_cluster(rows,"before","WEAK",baseline),
            "after":_cluster(rows,"after","WEAK",baseline),
            "delta_after_minus_before":_cluster(rows,"delta","WEAK",baseline),
        }

    report={
        "schema":"SPINCORE_LT3_HU_ENS8_PAIRWISE_DEV_V1",
        "before_iteration":before_iter,
        "after_iteration":after_iter,
        "before_ensemble":str(before),
        "after_ensemble":str(after),
        "before_schema":before_schema,
        "after_schema":after_schema,
        "seed":int(args.seed),
        "scenarios_requested":int(args.scenarios),
        "hu_scenarios":int(hu_scenarios),
        "workers":int(min(int(args.workers),max(1,len(tasks)))),
        "direct_after_vs_before":direct,
        "weak_baselines":weak,
        "method":{
            "sampler":"legacy empirical full sampler; non-HU scenarios skipped",
            "direct":"after ENS8 versus before ENS8; after rotated through both HU seats on same deal",
            "weak_pairing":"same scenario, deal, hero seat and RNG stream for before/after",
            "ci":"normal 95% CI over scenario-cluster means",
            "holdout_touched":False,
            "warning":"development diagnostic only; not sealed holdout or exploitability proof",
        },
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("=== LT3 HU ENS8 pairwise development comparison ===")
    print(f"before={before_iter} after={after_iter} HU_scenarios={hu_scenarios}")
    print(f"DIRECT after_vs_before={direct['mean']:+.3f} CI95=[{direct['ci95_low']:+.3f},{direct['ci95_high']:+.3f}]")
    for baseline,x in weak.items():
        d=x["delta_after_minus_before"]
        print(f"WEAK {baseline} delta={d['mean']:+.3f} CI95=[{d['ci95_low']:+.3f},{d['ci95_high']:+.3f}]")
    print("LT3_HU_ENS8_PAIRWISE_DEV_PASS")
    print(f"report={args.report.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
