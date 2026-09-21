#!/usr/bin/env python3
from __future__ import annotations

"""Final sealed-holdout validation for frozen LT2 HU ENS8@8100 candidate.

IMPORTANT: decision rules are frozen in docs/LT2_HU_ENS8_FINAL_HOLDOUT_PROTOCOL_20260921.md
before these seeds are evaluated. Do not alter criteria after outcomes.

No CFR roots, no optimizer steps on source artifacts, no training-memory writes.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
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
import audit_lt2_jammer_facing_allin_target_overlay as overlay
import audit_lt2_stage_a_b_first_divergence as fd
import audit_lt2_hu_ens8_post_pilot_forensic as post
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import legal_mask
from spincore.solver import Episode, SolverLibrary
from spincore_nn.action_models import collate_action_observations

DOMAIN="TRUE_HEADS_UP"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
HOLDOUT_SEEDS=(20261001,20261002,20261003,20261004,20261005,20261006)
BASELINES=("UNIFORM_LEGAL","PASSIVE_CALLER","JAMMER")
ECOSYSTEM=("AVG_7600","AVG_8000","BEH_7600","ENS8_8000")
MARGIN=-3.0

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
    p.add_argument("--scenarios-per-seed",type=int,default=6000)
    p.add_argument("--workers",type=int,default=31)
    p.add_argument("--threads-fit",type=int,default=8)
    p.add_argument("--git-commit",type=str,required=True)
    p.add_argument("--report",type=Path,required=True)
    return p.parse_args()


def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _mean_ci(values):
    xs=[float(x) for x in values]
    n=len(xs)
    if n==0:
        return {"n":0,"mean":float("nan"),"sem":float("nan"),"ci95_low":float("nan"),"ci95_high":float("nan")}
    mean=float(statistics.fmean(xs))
    sem=0.0 if n==1 else float(statistics.stdev(xs)/math.sqrt(n))
    h=1.96*sem
    return {"n":n,"mean":mean,"sem":sem,"ci95_low":mean-h,"ci95_high":mean+h}


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
    _ENS8000,_=post._load_ensemble(Path(ens8000))
    _ENS8100,_=post._load_ensemble(Path(ens8100))


def _ensemble_distribution(models,state):
    active_mask,legal=chain._legal_context(state)
    obs=state.neural_bytes()
    batch=collate_action_observations(
        REPRESENTATION,[obs],[legal_mask(legal)],device="cpu"
    )
    with torch.no_grad():
        raw=torch.stack([m(batch)[0] for m in models],dim=0).mean(dim=0).detach().cpu().tolist()
    return int(active_mask),legal,lean_regret_matching_policy(raw,legal)


def _distribution(policy,state):
    if policy=="AVG_7600":
        return _STAGE7600.average_policy(state)
    if policy=="AVG_8000":
        return _STAGE8000.average_policy(state)
    if policy=="AVG_8100":
        return _STAGE8100.average_policy(state)
    if policy=="BEH_7600":
        active_mask,legal=chain._legal_context(state)
        probs=_STAGE7600.runtime.session.behavior(state,state.neural_bytes(),legal)
        return int(active_mask),legal,tuple(float(x) for x in probs)
    if policy=="ENS8_8000":
        return _ensemble_distribution(_ENS8000,state)
    if policy=="ENS8_8100":
        return _ensemble_distribution(_ENS8100,state)
    raise ValueError(policy)


def _sample_policy(policy,state,rng):
    active_mask,legal,probs=_distribution(policy,state)
    return int(active_mask),chain._sample_probs(legal,probs,rng)


def _play_learned(episode:Episode,*,deal_seed,labels,scenario,seed,stream_tag):
    state=_SOLVER.create(episode,int(deal_seed))
    rngs={int(seat):random.Random(fd._mix64(seed,scenario,stream_tag,int(seat))) for seat in labels}
    decisions=0
    try:
        while not state.terminal:
            actor=int(state.actor)
            active_mask,slot=_sample_policy(labels[actor],state,rngs[actor])
            chain.apply_lean(state,active_mask,slot)
            decisions+=1
            if decisions>200:
                raise RuntimeError("holdout learned hand exceeded 200 decisions")
        delta=tuple(int(x) for x in state.terminal_chip_delta())
        if sum(delta)!=0:
            raise RuntimeError(f"nonzero-sum {delta}")
        return delta
    finally:
        state.close()


def _play_weak(episode:Episode,*,deal_seed,hero_seat,hero_policy,baseline,scenario,seed):
    state=_SOLVER.create(episode,int(deal_seed))
    rngs={
        seat:random.Random(fd._mix64(seed,scenario,seat,5000+BASELINES.index(baseline)))
        for seat in range(3)
    }
    rngs[int(hero_seat)]=random.Random(fd._mix64(seed,scenario,hero_seat,5777))
    decisions=0
    try:
        while not state.terminal:
            actor=int(state.actor)
            if actor==int(hero_seat):
                active_mask,slot=_sample_policy(hero_policy,state,rngs[actor])
            else:
                active_mask,slot=chain._baseline_action(baseline,state,rngs[actor])
            chain.apply_lean(state,active_mask,slot)
            decisions+=1
            if decisions>200:
                raise RuntimeError("holdout weak hand exceeded 200 decisions")
        delta=tuple(int(x) for x in state.terminal_chip_delta())
        if sum(delta)!=0:
            raise RuntimeError(f"nonzero-sum {delta}")
        return int(delta[int(hero_seat)])
    finally:
        state.close()


def _eco_opponent(seed,scenario,hero,opp):
    return ECOSYSTEM[int(fd._mix64(seed,scenario,hero,opp,0xEC05)%len(ECOSYSTEM))]


def _worker(task):
    seed,scenario,episode,deal_seed=task
    if not episode.game_is_hu:
        return []
    live=[s for s,stack in enumerate(episode.stacks) if int(stack)>0]
    if len(live)!=2:
        raise RuntimeError("HU episode without two live seats")
    rows=[]

    # Common learned ecosystem.
    for hero in live:
        opp=live[1] if hero==live[0] else live[0]
        opp_policy=_eco_opponent(seed,scenario,hero,opp)
        vals={}
        for policy in ("ENS8_8000","ENS8_8100","AVG_8100"):
            delta=_play_learned(
                episode,deal_seed=deal_seed,
                labels={int(hero):policy,int(opp):opp_policy},
                scenario=scenario,seed=seed,stream_tag=6000+int(hero),
            )
            vals[policy]=int(delta[int(hero)])
        rows.append({
            "mode":"ECOSYSTEM","seed":seed,"scenario":scenario,"hero_seat":hero,
            "opponent_policy":opp_policy,
            "ens8_8000":vals["ENS8_8000"],
            "ens8_8100":vals["ENS8_8100"],
            "avg_8100":vals["AVG_8100"],
            "ens8_8100_minus_8000":vals["ENS8_8100"]-vals["ENS8_8000"],
            "ens8_8100_minus_avg8100":vals["ENS8_8100"]-vals["AVG_8100"],
        })

    # Direct seat-balanced.
    for candidate,reference,label,tag in (
        ("ENS8_8100","ENS8_8000","ENS8_8100_vs_ENS8_8000",7000),
        ("ENS8_8100","AVG_8100","ENS8_8100_vs_AVG_8100",7100),
    ):
        for cand_seat in live:
            opp=live[1] if cand_seat==live[0] else live[0]
            delta=_play_learned(
                episode,deal_seed=deal_seed,
                labels={int(cand_seat):candidate,int(opp):reference},
                scenario=scenario,seed=seed,stream_tag=tag+int(cand_seat),
            )
            rows.append({
                "mode":"DIRECT","pair":label,"seed":seed,"scenario":scenario,
                "candidate_seat":cand_seat,
                "candidate_chip_delta":int(delta[int(cand_seat)]),
            })

    # Transparent weak baselines.
    for baseline in BASELINES:
        for hero in live:
            v8000=_play_weak(
                episode,deal_seed=deal_seed,hero_seat=hero,hero_policy="ENS8_8000",
                baseline=baseline,scenario=scenario,seed=seed
            )
            v8100=_play_weak(
                episode,deal_seed=deal_seed,hero_seat=hero,hero_policy="ENS8_8100",
                baseline=baseline,scenario=scenario,seed=seed
            )
            rows.append({
                "mode":"WEAK","baseline":baseline,"seed":seed,"scenario":scenario,
                "hero_seat":hero,"ens8_8000":v8000,"ens8_8100":v8100,
                "ens8_8100_minus_8000":v8100-v8000,
            })
    return rows


def _cluster(rows,mode,key,*,pair=None,baseline=None):
    by={}
    for row in rows:
        if row["mode"]!=mode:
            continue
        if pair is not None and row.get("pair")!=pair:
            continue
        if baseline is not None and row.get("baseline")!=baseline:
            continue
        ck=(int(row["seed"]),int(row["scenario"]))
        by.setdefault(ck,[]).append(float(row[key]))
    return [float(statistics.fmean(v)) for v in by.values()]


def main():
    args=parse_args()
    solver_path=args.solver.resolve(strict=True)
    p7600=args.stage_7600.resolve(strict=True)
    p8000=args.stage_8000.resolve(strict=True)
    p8100=args.stage_8100.resolve(strict=True)
    e8100=args.ensemble_8100.resolve(strict=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)

    artifact_hashes={
        "stage_7600":_sha256(p7600),
        "stage_8000":_sha256(p8000),
        "stage_8100":_sha256(p8100),
        "ensemble_8100":_sha256(e8100),
    }

    solver=SolverLibrary(solver_path)
    e8000=args.report.parent/"hu_ensemble_8000_reconstructed.pt"
    source_meta=post._build_source_ensemble(
        solver=solver,checkpoint=p8000,out_path=e8000,threads=args.threads_fit
    )
    artifact_hashes["ensemble_8000_reconstructed"]=_sha256(e8000)

    s7600=args.report.parent/"stage7600.pt"
    s8000=args.report.parent/"stage8000.pt"
    s8100=args.report.parent/"stage8100.pt"
    overlay._extract_stage_snapshot(p7600,s7600)
    overlay._extract_stage_snapshot(p8000,s8000)
    overlay._extract_stage_snapshot(p8100,s8100)
    del solver
    gc.collect()

    tasks=[]
    hu_counts={}
    for seed in HOLDOUT_SEEDS:
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
        initargs=(
            str(solver_path),str(s7600),str(s8000),str(s8100),str(e8000),str(e8100)
        ),
    ) as pool:
        for chunk in pool.map(_worker,tasks,chunksize=1):
            rows.extend(chunk)

    eco=[r for r in rows if r["mode"]=="ECOSYSTEM"]
    ecosystem={
        "clusters":len({(r["seed"],r["scenario"]) for r in eco}),
        "seat_runs":len(eco),
        "absolute":{
            "ENS8_8000":_mean_ci(_cluster(eco,"ECOSYSTEM","ens8_8000")),
            "ENS8_8100":_mean_ci(_cluster(eco,"ECOSYSTEM","ens8_8100")),
            "AVG_8100":_mean_ci(_cluster(eco,"ECOSYSTEM","avg_8100")),
        },
        "deltas":{
            "ENS8_8100_MINUS_8000":_mean_ci(_cluster(eco,"ECOSYSTEM","ens8_8100_minus_8000")),
            "ENS8_8100_MINUS_AVG8100":_mean_ci(_cluster(eco,"ECOSYSTEM","ens8_8100_minus_avg8100")),
        },
        "opponent_counts":{p:sum(1 for r in eco if r["opponent_policy"]==p) for p in ECOSYSTEM},
    }

    direct={}
    for label in ("ENS8_8100_vs_ENS8_8000","ENS8_8100_vs_AVG_8100"):
        direct[label]=_mean_ci(_cluster(rows,"DIRECT","candidate_chip_delta",pair=label))

    weak={}
    for baseline in BASELINES:
        weak[baseline]={
            "ENS8_8000":_mean_ci(_cluster(rows,"WEAK","ens8_8000",baseline=baseline)),
            "ENS8_8100":_mean_ci(_cluster(rows,"WEAK","ens8_8100",baseline=baseline)),
            "ENS8_8100_MINUS_8000":_mean_ci(_cluster(rows,"WEAK","ens8_8100_minus_8000",baseline=baseline)),
        }

    criteria={}
    def crit(name,actual,threshold,relation):
        passed=(actual>threshold) if relation==">" else (actual>=threshold)
        criteria[name]={
            "actual_ci95_low":float(actual),
            "threshold":float(threshold),
            "relation":relation,
            "pass":bool(passed),
        }

    crit("ecosystem_abs_ens8_8100",
         ecosystem["absolute"]["ENS8_8100"]["ci95_low"],0.0,">")
    crit("ecosystem_ens8_8100_minus_8000_noninferiority",
         ecosystem["deltas"]["ENS8_8100_MINUS_8000"]["ci95_low"],MARGIN,">")
    crit("ecosystem_ens8_8100_minus_avg8100",
         ecosystem["deltas"]["ENS8_8100_MINUS_AVG8100"]["ci95_low"],0.0,">")
    crit("direct_ens8_8100_vs_8000_noninferiority",
         direct["ENS8_8100_vs_ENS8_8000"]["ci95_low"],MARGIN,">")
    crit("direct_ens8_8100_vs_avg8100",
         direct["ENS8_8100_vs_AVG_8100"]["ci95_low"],0.0,">")
    for baseline in BASELINES:
        crit(f"weak_{baseline}_abs_ens8_8100",
             weak[baseline]["ENS8_8100"]["ci95_low"],0.0,">")
        crit(f"weak_{baseline}_8100_minus_8000_noninferiority",
             weak[baseline]["ENS8_8100_MINUS_8000"]["ci95_low"],MARGIN,">")

    verdict="PASS" if all(x["pass"] for x in criteria.values()) else "FAIL"

    out={
        "schema":"SPINCORE_LT2_HU_ENS8_FINAL_HOLDOUT_V1",
        "verdict":verdict,
        "candidate":"TRUE_HEADS_UP current ENS8 iteration 8100",
        "git_commit":args.git_commit,
        "artifact_hashes":artifact_hashes,
        "source_8000_ensemble_meta":source_meta,
        "method":{
            "holdout_seeds":list(HOLDOUT_SEEDS),
            "scenarios_per_seed":int(args.scenarios_per_seed),
            "hu_scenarios_by_seed":hu_counts,
            "read_only_source_artifacts":True,
            "new_training_roots":0,
            "source_training_memory_writes":0,
            "optimizer_steps_for_source_artifacts":0,
            "pre_registered_margin_chips":MARGIN,
            "decision_protocol":"docs/LT2_HU_ENS8_FINAL_HOLDOUT_PROTOCOL_20260921.md",
            "no_posthoc_tuning":True,
            "warning":"finite predefined validation battery; not mathematical GTO/exploitability proof",
        },
        "ecosystem":ecosystem,
        "direct":direct,
        "weak_baselines":weak,
        "criteria":criteria,
        "rows_count":len(rows),
    }
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== LT2 HU ENS8 FINAL HOLDOUT ===")
    print(f"VERDICT={verdict}")
    for name,x in criteria.items():
        print(
            f"{'PASS' if x['pass'] else 'FAIL'} {name}: "
            f"CI95_low={x['actual_ci95_low']:+.3f} threshold {x['relation']} {x['threshold']:+.3f}"
        )
    print("LT2_HU_ENS8_FINAL_HOLDOUT_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
