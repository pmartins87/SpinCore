#!/usr/bin/env python3
from __future__ import annotations

"""Independent design-set learned-ecosystem crossplay for the LT2 HU ENS8 candidate.

Seeds 20260926..20260930 are preregistered here before seeing outcomes. They
were not referenced elsewhere in the repository when the gate was created and
are distinct from sealed holdout seeds 20261001..20261006.

Historical opponent ecosystem excludes iteration 8100:
AVG7600, AVG8000, BEH7600, ENS8_8000.

Primary paired hero comparisons use identical scenario/deal/seat/opponent
assignment/RNG:
ENS8_8100 - ENS8_8000
AVG_8100  - AVG_8000
ENS8_8100 - AVG_8100

Additional seat-balanced direct crossplay evaluates the same three pairings.
No CFR roots, optimizer steps on source artifacts, or holdout.
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

import audit_lt2_hu_policy_chain as chain
import audit_lt2_hu_preflop_conditional_resampling as cond
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
DESIGN_SEEDS=(20260926,20260927,20260928,20260929,20260930)

HERO_POLICIES=("AVG_8000","AVG_8100","ENS8_8000","ENS8_8100")
ECOSYSTEM=("AVG_7600","AVG_8000","BEH_7600","ENS8_8000")
DIRECT_PAIRS=(
    ("ENS8_8100","ENS8_8000","ENS8_8100_vs_ENS8_8000"),
    ("AVG_8100","AVG_8000","AVG_8100_vs_AVG_8000"),
    ("ENS8_8100","AVG_8100","ENS8_8100_vs_AVG_8100"),
)

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
    probs=lean_regret_matching_policy(raw,legal)
    return int(active_mask),legal,probs


def _distribution(policy,state):
    if policy=="AVG_7600":
        return _STAGE7600.average_policy(state)
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


def _sample(policy,state,rng):
    active_mask,legal,probs=_distribution(policy,state)
    return int(active_mask),chain._sample_probs(legal,probs,rng)


def _play(episode:Episode,*,deal_seed,labels,scenario,seed,stream_tag):
    state=_SOLVER.create(episode,int(deal_seed))
    rngs={
        int(seat):random.Random(fd._mix64(int(seed),int(scenario),int(stream_tag),int(seat)))
        for seat in labels
    }
    decisions=0
    try:
        while not state.terminal:
            actor=int(state.actor)
            policy=labels[actor]
            active_mask,slot=_sample(policy,state,rngs[actor])
            chain.apply_lean(state,active_mask,slot)
            decisions+=1
            if decisions>200:
                raise RuntimeError("learned-ecosystem hand exceeded 200 decisions")
        delta=tuple(int(x) for x in state.terminal_chip_delta())
        if sum(delta)!=0:
            raise RuntimeError(f"nonzero-sum delta {delta}")
        return delta
    finally:
        state.close()


def _ecosystem_opponent(seed,scenario,hero,opp):
    idx=int(fd._mix64(int(seed),int(scenario),int(hero),int(opp),0xEC05)%len(ECOSYSTEM))
    return ECOSYSTEM[idx]


def _worker(task):
    seed,scenario,episode,deal_seed=task
    if not episode.game_is_hu:
        return []
    live=[s for s,stack in enumerate(episode.stacks) if int(stack)>0]
    if len(live)!=2:
        raise RuntimeError("HU episode without two live seats")
    rows=[]

    # Paired historical-ecosystem hero comparison.
    for hero in live:
        opp=live[1] if hero==live[0] else live[0]
        opp_policy=_ecosystem_opponent(seed,scenario,hero,opp)
        values={}
        stream_tag=1000+int(hero)
        for policy in HERO_POLICIES:
            labels={int(hero):policy,int(opp):opp_policy}
            delta=_play(
                episode,deal_seed=deal_seed,labels=labels,
                scenario=scenario,seed=seed,stream_tag=stream_tag
            )
            values[policy]=int(delta[int(hero)])
        rows.append({
            "mode":"ECOSYSTEM",
            "seed":int(seed),"scenario":int(scenario),"hero_seat":int(hero),
            "opponent_policy":opp_policy,
            **{p.lower():int(v) for p,v in values.items()},
            "ens8_8100_minus_8000":int(values["ENS8_8100"]-values["ENS8_8000"]),
            "avg_8100_minus_8000":int(values["AVG_8100"]-values["AVG_8000"]),
            "ens8_8100_minus_avg8100":int(values["ENS8_8100"]-values["AVG_8100"]),
        })

    # Seat-balanced direct candidate-vs-reference play.
    for candidate,reference,label in DIRECT_PAIRS:
        for cand_seat in live:
            opp=live[1] if cand_seat==live[0] else live[0]
            labels={int(cand_seat):candidate,int(opp):reference}
            delta=_play(
                episode,deal_seed=deal_seed,labels=labels,
                scenario=scenario,seed=seed,
                stream_tag=2000+DIRECT_PAIRS.index((candidate,reference,label))*10+int(cand_seat),
            )
            rows.append({
                "mode":"DIRECT",
                "pair":label,
                "seed":int(seed),"scenario":int(scenario),
                "candidate_seat":int(cand_seat),
                "candidate_chip_delta":int(delta[int(cand_seat)]),
            })
    return rows


def _cluster(rows,mode,key,*,pair=None):
    by={}
    for row in rows:
        if row["mode"]!=mode:
            continue
        if pair is not None and row.get("pair")!=pair:
            continue
        ck=(int(row["seed"]),int(row["scenario"]))
        by.setdefault(ck,[]).append(float(row[key]))
    return [float(statistics.fmean(v)) for v in by.values()]


def _per_seed(rows,key):
    out={}
    for seed in DESIGN_SEEDS:
        subset=[r for r in rows if r["mode"]=="ECOSYSTEM" and int(r["seed"])==int(seed)]
        vals=_cluster(subset,"ECOSYSTEM",key)
        out[str(seed)]=float(statistics.fmean(vals)) if vals else float("nan")
    return out


def main():
    args=parse_args()
    if args.scenarios_per_seed<=0 or args.workers<=0 or args.threads_fit<=0:
        raise SystemExit("positive arguments required")

    solver_path=args.solver.resolve(strict=True)
    p7600=args.stage_7600.resolve(strict=True)
    p8000=args.stage_8000.resolve(strict=True)
    p8100=args.stage_8100.resolve(strict=True)
    e8100=args.ensemble_8100.resolve(strict=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)

    solver=SolverLibrary(solver_path)
    e8000=args.report.parent/"hu_ensemble_8000_reconstructed.pt"
    source_meta=post._build_source_ensemble(
        solver=solver,checkpoint=p8000,out_path=e8000,threads=args.threads_fit
    )

    s7600=args.report.parent/"stage7600.pt"
    s8000=args.report.parent/"stage8000.pt"
    s8100=args.report.parent/"stage8100.pt"
    m7600=overlay._extract_stage_snapshot(p7600,s7600)
    m8000=overlay._extract_stage_snapshot(p8000,s8000)
    m8100=overlay._extract_stage_snapshot(p8100,s8100)
    if (int(m7600["completed_iteration"]),int(m8000["completed_iteration"]),int(m8100["completed_iteration"]))!=(7600,8000,8100):
        raise RuntimeError("iteration mismatch")
    del solver
    gc.collect()

    tasks=[]
    hu_counts={}
    for seed in DESIGN_SEEDS:
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
            str(solver_path),str(s7600.resolve()),str(s8000.resolve()),str(s8100.resolve()),
            str(e8000.resolve()),str(e8100.resolve()),
        ),
    ) as pool:
        for chunk in pool.map(_worker,tasks,chunksize=1):
            rows.extend(chunk)

    eco=[r for r in rows if r["mode"]=="ECOSYSTEM"]
    opponent_counts={p:sum(1 for r in eco if r["opponent_policy"]==p) for p in ECOSYSTEM}
    ecosystem={
        "clusters":len({(r["seed"],r["scenario"]) for r in eco}),
        "seat_runs":len(eco),
        "opponent_counts":opponent_counts,
        "absolute":{
            p:_mean_ci(_cluster(eco,"ECOSYSTEM",p.lower()))
            for p in HERO_POLICIES
        },
        "deltas":{
            key:_mean_ci(_cluster(eco,"ECOSYSTEM",key))
            for key in (
                "ens8_8100_minus_8000",
                "avg_8100_minus_8000",
                "ens8_8100_minus_avg8100",
            )
        },
        "per_seed":{
            key:_per_seed(eco,key)
            for key in (
                "ens8_8100_minus_8000",
                "avg_8100_minus_8000",
                "ens8_8100_minus_avg8100",
            )
        },
    }
    direct={}
    for _cand,_ref,label in DIRECT_PAIRS:
        direct[label]=_mean_ci(
            _cluster(rows,"DIRECT","candidate_chip_delta",pair=label)
        )

    out={
        "schema":"SPINCORE_LT2_HU_ENS8_LEARNED_ECOSYSTEM_CROSSPLAY_V1",
        "sources":{
            "stage_7600":str(p7600),
            "stage_8000":str(p8000),
            "stage_8100":str(p8100),
            "ensemble_8100":str(e8100),
            "ensemble_8000_reconstructed":str(e8000.resolve()),
        },
        "source_8000_ensemble_meta":source_meta,
        "method":{
            "read_only_source_artifacts":True,
            "new_training_roots":0,
            "source_training_memory_writes":0,
            "optimizer_steps_for_source_artifacts":0,
            "future_holdout_seeds_touched":False,
            "design_seeds":list(DESIGN_SEEDS),
            "design_seed_note":"pre-registered unused repository literals before outcome observation; separate from holdout 20261001..20261006",
            "scenarios_per_seed":int(args.scenarios_per_seed),
            "hu_scenarios_by_seed":hu_counts,
            "historical_opponent_ecosystem":list(ECOSYSTEM),
            "hero_policies":list(HERO_POLICIES),
            "pairing":"same scenario, deal, hero seat, historical opponent assignment and per-seat RNG across ecosystem hero policies",
            "warning":"learned-policy relative-strength design diagnostic; not exploitability/GTO proof",
        },
        "ecosystem":ecosystem,
        "direct":direct,
        "rows_count":len(rows),
    }
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== LT2 HU ENS8 LEARNED-ECOSYSTEM CROSSPLAY ===")
    for key,x in ecosystem["deltas"].items():
        print(
            f"ECOSYSTEM {key}: {x['mean']:+.3f} "
            f"CI=[{x['ci95_low']:+.3f},{x['ci95_high']:+.3f}]"
        )
    for label,x in direct.items():
        print(
            f"DIRECT {label}: candidateEV={x['mean']:+.3f} "
            f"CI=[{x['ci95_low']:+.3f},{x['ci95_high']:+.3f}]"
        )
    print("LT2_HU_ENS8_LEARNED_ECOSYSTEM_CROSSPLAY_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
