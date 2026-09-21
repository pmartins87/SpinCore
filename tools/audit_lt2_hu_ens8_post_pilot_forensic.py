#!/usr/bin/env python3
from __future__ import annotations

"""Post-pilot forensic adjudication for the LT2 HU ENS8 online pilot.

Read-only gate:
- reconstruct exact source-8000 ENS8_A from frozen reservoir;
- load final iteration-8100 ENS8 sidecar;
- deterministic root drift on every forensic HU root;
- broad weak-baseline EV for current ENS8 8000/8100, current 7600 and
  AveragePolicy 8000/8100.

No CFR roots, source-memory writes or holdout seeds.
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
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import torch

import audit_lt2_hu_behavior_first_divergence as behfd
import audit_lt2_hu_policy_chain as chain
import audit_lt2_hu_preflop_conditional_resampling as cond
import audit_lt2_jammer_facing_allin_target_overlay as overlay
import audit_lt2_stage_a_b_first_divergence as fd
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_functional_training import load_checkpoint
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import legal_mask
from spincore.solver import Episode, SolverLibrary
from spincore_nn.action_models import collate_action_observations, make_advantage_action_model

DOMAIN="TRUE_HEADS_UP"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
FORENSIC_SEEDS=(20260920,20260921,20260922,20260923,20260924,20260925)
BASELINES=("UNIFORM_LEGAL","PASSIVE_CALLER","JAMMER")
POLICIES=("AVG_8000","AVG_8100","BEH_7600","ENS8_8000","ENS8_8100")
ENSEMBLE_SCHEMA="SPINCORE_LT2_HU_ENS8_CURRENT_STATE_V1"
ENSEMBLE_SIZE=8
MEMBER_STEPS=400

_SOLVER=None
_STAGE7600=None
_STAGE8000=None
_STAGE8100=None
_ENS8000=None
_ENS8100=None


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--stage-7600",type=Path,required=True)
    p.add_argument("--stage-8000",type=Path,required=True)
    p.add_argument("--stage-8100",type=Path,required=True)
    p.add_argument("--ensemble-8100",type=Path,required=True)
    p.add_argument("--scenarios-per-seed",type=int,default=5000)
    p.add_argument("--workers",type=int,default=31)
    p.add_argument("--threads-fit",type=int,default=8)
    p.add_argument("--eval-batch-size",type=int,default=1024)
    p.add_argument("--report",type=Path,required=True)
    return p.parse_args()


def _mean_ci(values):
    xs=[float(x) for x in values]
    n=len(xs)
    if n==0:
        return {"n":0,"mean":float("nan"),"sem":float("nan"),"ci95_low":float("nan"),"ci95_high":float("nan")}
    mean=float(statistics.fmean(xs))
    sem=0.0 if n==1 else float(statistics.stdev(xs)/math.sqrt(n))
    h=1.96*sem
    return {"n":n,"mean":mean,"sem":sem,"ci95_low":mean-h,"ci95_high":mean+h}


def _member_seeds(member:int):
    return (
        int(fd._mix64(20260920,int(member),0xE115E)&0x7fffffff),
        int(fd._mix64(20260920,int(member),0xEBA7C4)&0x7fffffff),
    )


def _clone_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def _build_source_ensemble(*,solver,checkpoint,out_path,threads):
    torch.set_num_threads(int(threads))
    _seed,config,iteration,_sampler,runtimes,_history,_finalized=load_checkpoint(
        checkpoint,solver=solver
    )
    if int(iteration)!=8000:
        raise RuntimeError(f"expected source iteration 8000, got {iteration}")
    runtime=runtimes[DOMAIN]
    runtime.session.batch_mode="vectorized"
    members=[]
    meta=[]
    for member in range(ENSEMBLE_SIZE):
        init_seed,batch_seed=_member_seeds(member)
        runtime.session.reset_advantage_network(
            init_seed=int(init_seed),lr=float(config.learning_rate)
        )
        runtime.bundle.batch_rng.seed(int(batch_seed))
        losses=runtime.session.train_advantage(
            steps=MEMBER_STEPS,batch_size=int(config.batch_size)
        )
        if len(losses)!=MEMBER_STEPS:
            raise RuntimeError("source ENS8 step-count drift")
        members.append(_clone_state(runtime.bundle.advantage))
        meta.append({
            "member":member,
            "init_seed":init_seed,
            "batch_seed":batch_seed,
            "steps":MEMBER_STEPS,
            "loss_last":float(losses[-1]),
        })
    payload={
        "schema":ENSEMBLE_SCHEMA,
        "completed_iteration":8000,
        "ensemble_size":ENSEMBLE_SIZE,
        "member_steps":MEMBER_STEPS,
        "member_meta":meta,
        "members":members,
        "semantics":"mean raw Advantage outputs then unchanged lean_regret_matching_policy",
        "source_reconstruction":True,
    }
    torch.save(payload,out_path)
    return meta


def _load_ensemble(path):
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("schema")!=ENSEMBLE_SCHEMA:
        raise RuntimeError(f"bad ensemble sidecar schema: {path}")
    if int(payload.get("ensemble_size",-1))!=ENSEMBLE_SIZE or len(payload.get("members") or [])!=ENSEMBLE_SIZE:
        raise RuntimeError(f"bad ensemble size: {path}")
    models=[]
    for index,state in enumerate(payload["members"]):
        _,model=make_advantage_action_model(
            REPRESENTATION,device="cpu",seed=index
        )
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models,int(payload["completed_iteration"])


def _collect_root_corpus(solver,scenarios):
    rows=[]
    counts={}
    for seed in FORENSIC_SEEDS:
        sampler=LegacyScenarioSampler(seed=int(seed)^0x5CE0A710,config=LegacyScenarioConfig())
        hu=0
        for idx in range(int(scenarios)):
            ep=sampler.sample_episode()
            if not ep.game_is_hu:
                continue
            hu+=1
            deal_seed=fd._mix64(int(seed),int(idx),0xD34A1)
            state=solver.create(ep,int(deal_seed))
            try:
                _mask,legal=behfd._legal_context(state)
                live=[int(x) for x in ep.stacks if int(x)>0]
                effective=min(live)/float(ep.big_blind)
                rows.append({
                    "seed":int(seed),
                    "scenario":int(idx),
                    "blind":f"{ep.small_blind}/{ep.big_blind}",
                    "effective_stack_bb":float(effective),
                    "observation":state.neural_bytes(),
                    "legal":tuple(int(x) for x in legal),
                    "legal_mask":legal_mask(legal),
                })
            finally:
                state.close()
        counts[str(seed)]=hu
    return rows,counts


def _predict_policy(models,corpus,batch_size):
    chunks=[]
    with torch.no_grad():
        for start in range(0,len(corpus),int(batch_size)):
            block=corpus[start:start+int(batch_size)]
            batch=collate_action_observations(
                REPRESENTATION,
                [r["observation"] for r in block],
                [r["legal_mask"] for r in block],
                device="cpu",
            )
            raw=torch.stack([m(batch) for m in models],dim=0).mean(dim=0).detach().cpu()
            probs=[
                lean_regret_matching_policy(raw[i].tolist(),r["legal"])
                for i,r in enumerate(block)
            ]
            chunks.append(torch.tensor(probs,dtype=torch.float32))
    return torch.cat(chunks,dim=0)


def _seed_cluster_ci(corpus,values):
    by={}
    for row,value in zip(corpus,values):
        by.setdefault(int(row["seed"]),[]).append(float(value))
    return _mean_ci([statistics.fmean(v) for v in by.values()])


def _root_summary(corpus,a,b):
    tv=0.5*torch.abs(a-b).sum(dim=1)
    arg=(torch.argmax(a,dim=1)!=torch.argmax(b,dim=1)).float()
    out={
        "root_count":len(corpus),
        "mean_tv":_seed_cluster_ci(corpus,tv.tolist()),
        "argmax_disagreement":_seed_cluster_ci(corpus,arg.tolist()),
        "action_mass":{
            "ENS8_8000":{},
            "ENS8_8100":{},
            "delta_8100_minus_8000":{},
        },
    }
    for action,name in ((0,"FOLD"),(1,"CHECK_CALL"),(3,"POT_33"),(9,"ALL_IN")):
        av=a[:,action].tolist(); bv=b[:,action].tolist()
        dv=[y-x for x,y in zip(av,bv)]
        out["action_mass"]["ENS8_8000"][name]=_seed_cluster_ci(corpus,av)
        out["action_mass"]["ENS8_8100"][name]=_seed_cluster_ci(corpus,bv)
        out["action_mass"]["delta_8100_minus_8000"][name]=_seed_cluster_ci(corpus,dv)

    buckets=[
        ("<=3bb",lambda x:x<=3),
        (">3-5bb",lambda x:3<x<=5),
        (">5-8bb",lambda x:5<x<=8),
        (">8-12bb",lambda x:8<x<=12),
        (">12bb",lambda x:x>12),
    ]
    out["by_stack"]={}
    for label,pred in buckets:
        idx=[i for i,r in enumerate(corpus) if pred(float(r["effective_stack_bb"]))]
        if not idx:
            continue
        ta=a[idx]; tb=b[idx]
        ttv=0.5*torch.abs(ta-tb).sum(dim=1)
        out["by_stack"][label]={
            "n":len(idx),
            "mean_tv":float(ttv.mean().item()),
            "all_in_8000":float(ta[:,9].mean().item()),
            "all_in_8100":float(tb[:,9].mean().item()),
            "all_in_delta":float((tb[:,9]-ta[:,9]).mean().item()),
            "pot33_delta":float((tb[:,3]-ta[:,3]).mean().item()),
        }
    return out


def _init_worker(solver_path,snap7600,snap8000,snap8100,ens8000,ens8100):
    global _SOLVER,_STAGE7600,_STAGE8000,_STAGE8100,_ENS8000,_ENS8100
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
    _STAGE7600=overlay.StageModels(Path(snap7600),_SOLVER)
    _STAGE8000=overlay.StageModels(Path(snap8000),_SOLVER)
    _STAGE8100=overlay.StageModels(Path(snap8100),_SOLVER)
    _ENS8000,_=_load_ensemble(Path(ens8000))
    _ENS8100,_=_load_ensemble(Path(ens8100))


def _ensemble_distribution(models,state):
    active_mask,legal=chain._legal_context(state)
    obs=state.neural_bytes()
    mask=legal_mask(legal)
    batch=collate_action_observations(
        REPRESENTATION,[obs],[mask],device="cpu"
    )
    with torch.no_grad():
        raw=torch.stack([m(batch)[0] for m in models],dim=0).mean(dim=0).detach().cpu().tolist()
    out=lean_regret_matching_policy(raw,legal)
    return int(active_mask),legal,out


def _hero_distribution(policy,state):
    if policy=="AVG_8000":
        return _STAGE8000.average_policy(state)
    if policy=="AVG_8100":
        return _STAGE8100.average_policy(state)
    if policy=="BEH_7600":
        active_mask,legal=chain._legal_context(state)
        obs=state.neural_bytes()
        probs=_STAGE7600.runtime.session.behavior(state,obs,legal)
        return int(active_mask),legal,tuple(float(x) for x in probs)
    if policy=="ENS8_8000":
        return _ensemble_distribution(_ENS8000,state)
    if policy=="ENS8_8100":
        return _ensemble_distribution(_ENS8100,state)
    raise ValueError(policy)


def _play_one(episode:Episode,*,deal_seed,hero_seat,hero_policy,baseline,scenario_index,master_seed):
    state=_SOLVER.create(episode,int(deal_seed))
    rngs={
        seat:random.Random(fd._mix64(master_seed,scenario_index,seat,100+BASELINES.index(baseline)))
        for seat in range(3)
    }
    rngs[int(hero_seat)]=random.Random(fd._mix64(master_seed,scenario_index,hero_seat,777))
    decisions=0
    try:
        while not state.terminal:
            actor=int(state.actor)
            if actor==int(hero_seat):
                active_mask,legal,probs=_hero_distribution(hero_policy,state)
                slot=chain._sample_probs(legal,probs,rngs[actor])
            else:
                active_mask,slot=chain._baseline_action(baseline,state,rngs[actor])
            chain.apply_lean(state,active_mask,slot)
            decisions+=1
            if decisions>200:
                raise RuntimeError("post-pilot hand exceeded 200 decisions")
        delta=state.terminal_chip_delta()
        if sum(delta)!=0:
            raise RuntimeError(f"nonzero-sum delta {delta}")
        return int(delta[int(hero_seat)])
    finally:
        state.close()


def _worker(task):
    seed,idx,ep,deal_seed=task
    if not ep.game_is_hu:
        return []
    live=[s for s,stack in enumerate(ep.stacks) if int(stack)>0]
    rows=[]
    for baseline in BASELINES:
        for hero in live:
            vals={
                p:_play_one(
                    ep,deal_seed=deal_seed,hero_seat=hero,hero_policy=p,
                    baseline=baseline,scenario_index=idx,master_seed=seed
                )
                for p in POLICIES
            }
            rows.append({
                "seed":seed,"scenario":idx,"hero_seat":hero,"baseline":baseline,
                **{p.lower():int(v) for p,v in vals.items()},
                "ens8_8100_minus_8000":int(vals["ENS8_8100"]-vals["ENS8_8000"]),
                "ens8_8100_minus_beh7600":int(vals["ENS8_8100"]-vals["BEH_7600"]),
                "avg_8100_minus_8000":int(vals["AVG_8100"]-vals["AVG_8000"]),
                "chain_gap_8000":int(vals["AVG_8000"]-vals["ENS8_8000"]),
                "chain_gap_8100":int(vals["AVG_8100"]-vals["ENS8_8100"]),
                "chain_gap_delta":int(
                    (vals["AVG_8100"]-vals["ENS8_8100"])
                    -(vals["AVG_8000"]-vals["ENS8_8000"])
                ),
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


def _ev_summary(rows):
    out={}
    for baseline in BASELINES:
        block={
            "clusters":len({(r["seed"],r["scenario"]) for r in rows if r["baseline"]==baseline}),
            "absolute":{},
            "deltas":{},
        }
        for p in POLICIES:
            block["absolute"][p]=_mean_ci(_cluster(rows,baseline,p.lower()))
        for key in (
            "ens8_8100_minus_8000",
            "ens8_8100_minus_beh7600",
            "avg_8100_minus_8000",
            "chain_gap_8000",
            "chain_gap_8100",
            "chain_gap_delta",
        ):
            block["deltas"][key]=_mean_ci(_cluster(rows,baseline,key))
        out[baseline]=block
    return out


def main():
    args=parse_args()
    if args.scenarios_per_seed<=0 or args.workers<=0 or args.threads_fit<=0:
        raise SystemExit("positive arguments required")

    solver_path=args.solver.resolve(strict=True)
    p7600=args.stage_7600.resolve(strict=True)
    p8000=args.stage_8000.resolve(strict=True)
    p8100=args.stage_8100.resolve(strict=True)
    ens8100=args.ensemble_8100.resolve(strict=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)

    solver=SolverLibrary(solver_path)
    ens8000=args.report.parent/"hu_ensemble_8000_reconstructed.pt"
    source_meta=_build_source_ensemble(
        solver=solver,checkpoint=p8000,out_path=ens8000,threads=args.threads_fit
    )

    e8100=torch.load(ens8100,map_location="cpu",weights_only=False)
    if e8100.get("schema")!=ENSEMBLE_SCHEMA or int(e8100.get("completed_iteration",-1))!=8100:
        raise RuntimeError("iteration-8100 ensemble sidecar mismatch")

    # deterministic root drift
    corpus,hu_counts=_collect_root_corpus(solver,args.scenarios_per_seed)
    m8000,_=_load_ensemble(ens8000)
    m8100,_=_load_ensemble(ens8100)
    pred8000=_predict_policy(m8000,corpus,args.eval_batch_size)
    pred8100=_predict_policy(m8100,corpus,args.eval_batch_size)
    root=_root_summary(corpus,pred8000,pred8100)
    del m8000,m8100,solver,pred8000,pred8100
    gc.collect()

    # lightweight snapshots for full-hand worker pool
    local_solver=SolverLibrary(solver_path)
    s7600=args.report.parent/"stage7600.pt"
    s8000=args.report.parent/"stage8000.pt"
    s8100=args.report.parent/"stage8100.pt"
    meta7600=overlay._extract_stage_snapshot(p7600,s7600)
    meta8000=overlay._extract_stage_snapshot(p8000,s8000)
    meta8100=overlay._extract_stage_snapshot(p8100,s8100)
    if (int(meta7600["completed_iteration"]),int(meta8000["completed_iteration"]),int(meta8100["completed_iteration"]))!=(7600,8000,8100):
        raise RuntimeError("checkpoint iteration mismatch")
    del local_solver
    gc.collect()

    tasks=[]
    for seed in FORENSIC_SEEDS:
        sampler=LegacyScenarioSampler(seed=int(seed)^0x5CE0A710,config=LegacyScenarioConfig())
        for idx in range(int(args.scenarios_per_seed)):
            ep=sampler.sample_episode()
            deal_seed=fd._mix64(int(seed),int(idx),0xD34A1)
            tasks.append((int(seed),int(idx),ep,int(deal_seed)))

    rows=[]
    ctx=mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers),len(tasks)),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(
            str(solver_path),str(s7600.resolve()),str(s8000.resolve()),str(s8100.resolve()),
            str(ens8000.resolve()),str(ens8100.resolve()),
        ),
    ) as pool:
        for chunk in pool.map(_worker,tasks,chunksize=1):
            rows.extend(chunk)

    ev=_ev_summary(rows)

    out={
        "schema":"SPINCORE_LT2_HU_ENS8_POST_PILOT_FORENSIC_V1",
        "sources":{
            "stage_7600":str(p7600),
            "stage_8000":str(p8000),
            "stage_8100":str(p8100),
            "ensemble_8100":str(ens8100),
            "ensemble_8000_reconstructed":str(ens8000.resolve()),
        },
        "source_8000_ensemble_meta":source_meta,
        "method":{
            "read_only_source_artifacts":True,
            "new_training_roots":0,
            "source_training_memory_writes":0,
            "future_holdout_seeds_touched":False,
            "forensic_seeds":list(FORENSIC_SEEDS),
            "scenarios_per_seed":int(args.scenarios_per_seed),
            "hu_scenarios_by_seed":hu_counts,
            "baselines":list(BASELINES),
            "policies":list(POLICIES),
            "pairing":"same HU scenario, deal, hero seat and hero RNG across policies",
            "warning":"design-set weak-baseline/trajectory diagnostic; not exploitability/GTO proof",
        },
        "root_drift_8100_minus_8000":root,
        "ev":ev,
        "rows_count":len(rows),
    }
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== LT2 HU ENS8 POST-PILOT FORENSIC ===")
    rd=root["action_mass"]["delta_8100_minus_8000"]
    print(
        f"ROOT: TV={root['mean_tv']['mean']:.4f} "
        f"argmax={root['argmax_disagreement']['mean']:.4f} "
        f"ALL_IN_delta={rd['ALL_IN']['mean']:+.4f} "
        f"POT33_delta={rd['POT_33']['mean']:+.4f}"
    )
    for baseline in BASELINES:
        b=ev[baseline]
        d1=b["deltas"]["ens8_8100_minus_8000"]
        d2=b["deltas"]["ens8_8100_minus_beh7600"]
        da=b["deltas"]["avg_8100_minus_8000"]
        print(
            f"{baseline}: ENS8100-ENS8000={d1['mean']:+.3f} "
            f"CI=[{d1['ci95_low']:+.3f},{d1['ci95_high']:+.3f}] "
            f"ENS8100-BEH7600={d2['mean']:+.3f} "
            f"CI=[{d2['ci95_low']:+.3f},{d2['ci95_high']:+.3f}] "
            f"AVG8100-AVG8000={da['mean']:+.3f} "
            f"CI=[{da['ci95_low']:+.3f},{da['ci95_high']:+.3f}]"
        )
    print("LT2_HU_ENS8_POST_PILOT_FORENSIC_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
