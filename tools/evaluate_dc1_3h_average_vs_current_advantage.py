#!/usr/bin/env python3
from __future__ import annotations

"""Replay the frozen 200-scenario DC1 smoke while tracing 3H AveragePolicy vs current Advantage.

Played actions remain sampled from the actual 10105 hybrid research policy:
THREE_HANDED AveragePolicy + TRUE_HEADS_UP ENS8.  The 3H current Advantage
distribution is observational metadata only and consumes no benchmark RNG.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import statistics
import sys
from typing import Any

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))

from spincore.deepcrusher_benchmark import (
    DEEPC_RUSHER_POLICY_ID,
    SPINCORE_POLICY_ID,
    ExternalExactAction,
    OfflineHeadToHeadEngine,
)
from spincore.decision_sanity import sanity_flags
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_solver_actions import resolve_lean_exact
from spincore_nn.action_models import (
    collate_action_observations,
    make_advantage_action_model,
)

REPRESENTATION="C0_V1_FROZEN_CONTROL"
PROBE_SCHEMA="SPINCORE_3H_CURRENT_ADVANTAGE_PROBE_V1"

_SOLVER=None
_SPIN=None
_DC=None
_ENGINE_SEED=None


def mix64(*values:int)->int:
    x=0x9E3779B97F4A7C15
    mask=(1<<64)-1
    for value in values:
        y=int(value)&mask
        x^=(y+0x9E3779B97F4A7C15+((x<<6)&mask)+(x>>2))&mask
        x&=mask
    return x


class HybridWith3HCurrentProbe:
    policy_id=SPINCORE_POLICY_ID
    def __init__(self,agent,current_advantage):
        self.agent=agent
        self.current_advantage=current_advantage.eval()
        self._last={}

    def choose_exact(self,state,*,seat:int,rng:random.Random)->ExternalExactAction:
        active_mask,legal,probs=self.agent.distribution(state)
        domain=self.agent.domain_for_state(state)

        current=None
        current_raw=None
        if domain=="THREE_HANDED":
            obs=state.neural_bytes()
            batch=collate_action_observations(
                REPRESENTATION,[obs],[legal_mask(legal)],device="cpu"
            )
            with torch.no_grad():
                raw=self.current_advantage(batch)[0].detach().cpu().tolist()
            cur=lean_regret_matching_policy(raw,legal)
            current=[
                {
                    "slot":int(a),
                    "name":str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")),
                    "probability":float(cur[a]),
                }
                for a in legal
            ]
            current_raw=[
                {
                    "slot":int(a),
                    "name":str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")),
                    "raw":float(raw[a]),
                }
                for a in legal
            ]

        x=rng.random()
        cumulative=0.0
        slot=int(legal[-1])
        for candidate in legal:
            cumulative+=float(probs[candidate])
            if x<cumulative:
                slot=int(candidate)
                break
        action_type,amount_to=resolve_lean_exact(state,active_mask,slot)

        detail={
            "oracle":"HybridWith3HCurrentProbe",
            "selection":"SAMPLED_ACTUAL_HYBRID_POLICY",
            "domain":domain,
            "active_mask":int(active_mask),
            "sample_u":float(x),
            "legal_actions":[
                {
                    "slot":int(a),
                    "name":str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")),
                    "probability":float(probs[a]),
                }
                for a in legal
            ],
            "selected_slot":int(slot),
            "selected_slot_name":str(NAME_BY_SLOT.get(int(slot),f"SLOT_{int(slot)}")),
            "selected_probability":float(probs[slot]),
            "resolved_action_type":int(action_type),
            "resolved_amount_to":int(amount_to),
        }
        if current is not None:
            detail["current_3h_advantage_policy"]=current
            detail["current_3h_advantage_raw"]=current_raw
        self._last[int(seat)]=detail
        return ExternalExactAction(int(action_type),int(amount_to))

    def decision_metadata(self,*,seat:int):
        row=self._last.get(int(seat))
        return None if row is None else dict(row)


def init_worker(solver_path:str,bundle_path:str,probe_path:str,seed:int):
    global _SOLVER,_SPIN,_DC,_ENGINE_SEED
    import os
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    os.environ["SPINCORE_TORCH_THREADS"]="1"
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass

    from spincore.solver import SolverLibrary
    from spincore.deepcrusher_policy import DeepCrusherR8Policy
    from spincore.lean_hybrid_deployment_agent import LeanHybridDeploymentAgent

    _SOLVER=SolverLibrary(solver_path)
    agent=LeanHybridDeploymentAgent.from_bundle(bundle_path,device="cpu",seed=0)
    p=torch.load(probe_path,map_location="cpu",weights_only=False)
    if p.get("schema")!=PROBE_SCHEMA or p.get("representation")!=REPRESENTATION:
        raise RuntimeError("wrong 3H current-Advantage probe artifact")
    _,adv=make_advantage_action_model(REPRESENTATION,device="cpu",seed=0)
    adv.load_state_dict(p["advantage"])
    _SPIN=HybridWith3HCurrentProbe(agent,adv)
    _DC=DeepCrusherR8Policy.from_repository(ROOT)
    _ENGINE_SEED=int(seed)


def worker(task):
    index,episode,deal_seed,max_decisions=task
    traces=[]
    engine=OfflineHeadToHeadEngine(
        _SOLVER,
        spincore_policy=_SPIN,
        deepcrusher_policy=_DC,
        master_seed=int(_ENGINE_SEED),
        max_decisions=int(max_decisions),
        decision_sink=traces.append,
    )
    obs=engine.play_balanced_block(
        episode,deal_seed=int(deal_seed),scenario_index=int(index)
    )
    return {
        "traces":[asdict(t) for t in traces],
        "games":len(obs),
    }


def pct(values,p):
    if not values:return None
    s=sorted(values)
    return float(s[min(len(s)-1,max(0,math.ceil(p*len(s))-1))])


def probs(detail,key):
    return {str(x["name"]):float(x["probability"]) for x in (detail.get(key) or [])}


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--spin-bundle",type=Path,required=True)
    ap.add_argument("--current-3h-probe",type=Path,required=True)
    ap.add_argument("--scenarios",type=int,default=200)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--seed",type=int,default=20260923)
    ap.add_argument("--max-decisions",type=int,default=200)
    ap.add_argument("--report",type=Path,required=True)
    args=ap.parse_args()

    sampler=LegacyScenarioSampler(
        seed=int(args.seed)^0x5CE0A710,
        config=LegacyScenarioConfig(),
    )
    tasks=[]
    for index in range(int(args.scenarios)):
        episode=sampler.sample_episode()
        tasks.append((
            int(index),episode,int(mix64(args.seed,index,0xD34A1)),int(args.max_decisions)
        ))

    all_traces=[]
    games=0
    ctx=mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers),int(args.scenarios)),
        mp_context=ctx,
        initializer=init_worker,
        initargs=(
            str(args.solver.resolve()),
            str(args.spin_bundle.resolve()),
            str(args.current_3h_probe.resolve()),
            int(args.seed),
        ),
    ) as pool:
        for chunk in pool.map(worker,tasks,chunksize=1):
            all_traces.extend(chunk["traces"])
            games+=int(chunk["games"])

    # Rehydrate only fields needed by sanity_flags.
    from spincore.deepcrusher_benchmark import DecisionTrace
    traces=[DecisionTrace(**row) for row in all_traces]

    tvs=[]
    argdiff=[]
    flagged=[]
    for t in traces:
        if t.policy_id!=SPINCORE_POLICY_ID or t.domain!="THREE_HANDED":
            continue
        detail=dict(t.policy_detail or {})
        apol=probs(detail,"legal_actions")
        cur=probs(detail,"current_3h_advantage_policy")
        if cur:
            names=set(apol)|set(cur)
            tvs.append(0.5*sum(abs(apol.get(k,0.0)-cur.get(k,0.0)) for k in names))
            argdiff.append(float(max(apol,key=apol.get)!=max(cur,key=cur.get)))
        for flag in sanity_flags(t):
            if flag.code not in ("POSTFLOP_DEEP_HIGH_CARD_JAM","POSTFLOP_TRIPS_PLUS_FOLD"):
                continue
            row=asdict(flag)
            row["average_policy"]=apol
            row["current_3h_advantage_policy"]=cur
            action=str(t.action_name)
            row["played_action_average_policy_probability"]=apol.get(action)
            row["played_action_current_advantage_probability"]=cur.get(action)
            row["average_policy_argmax"]=max(apol,key=apol.get) if apol else None
            row["current_advantage_argmax"]=max(cur,key=cur.get) if cur else None
            flagged.append(row)

    high=[x for x in flagged if x["code"]=="POSTFLOP_DEEP_HIGH_CARD_JAM"]
    trips=[x for x in flagged if x["code"]=="POSTFLOP_TRIPS_PLUS_FOLD"]

    def flag_summary(rows,action):
        if not rows:
            return {"count":0}
        a=[float(x["average_policy"].get(action,0.0)) for x in rows]
        c=[float(x["current_3h_advantage_policy"].get(action,0.0)) for x in rows]
        return {
            "count":len(rows),
            "average_policy_action_probability_mean":statistics.fmean(a),
            "current_advantage_action_probability_mean":statistics.fmean(c),
            "current_advantage_action_probability_median":statistics.median(c),
            "current_advantage_argmax_is_played_action_count":sum(
                x["current_advantage_argmax"]==action for x in rows
            ),
            "current_advantage_probability_lower_than_average_policy_count":sum(
                cc<aa for aa,cc in zip(a,c)
            ),
            "current_advantage_probability_zero_count":sum(cc==0.0 for cc in c),
        }

    report={
        "schema":"SPINCORE_DC1_3H_AVERAGE_VS_CURRENT_ADVANTAGE_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_PLAYED_TRAJECTORY_UNCHANGED",
        "seed":int(args.seed),
        "scenarios":int(args.scenarios),
        "balanced_games":games,
        "trace_decisions":len(traces),
        "three_handed_policy_drift":{
            "decision_count":len(tvs),
            "tv_mean":statistics.fmean(tvs) if tvs else None,
            "tv_median":statistics.median(tvs) if tvs else None,
            "tv_p95":pct(tvs,0.95),
            "argmax_disagreement_rate":statistics.fmean(argdiff) if argdiff else None,
        },
        "three_handed_high_card_jam":flag_summary(high,"ALL_IN"),
        "three_handed_trips_plus_fold":flag_summary(trips,"FOLD"),
        "flags":flagged,
        "interpretation":(
            "Played trajectories use the unchanged actual hybrid policy. Current 3H Advantage is "
            "counterfactual diagnostic metadata only. If weird AveragePolicy actions lose most "
            "probability under current Advantage, the anomaly is concentrated in time-averaging/"
            "distillation/generalization rather than the latest 3H Advantage signal. If current "
            "Advantage supports the same actions, replacing AveragePolicy would not solve them."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("=== DC1 3H AveragePolicy vs current Advantage ===")
    print("policy_drift="+json.dumps(report["three_handed_policy_drift"],sort_keys=True))
    print("high_card_jam="+json.dumps(report["three_handed_high_card_jam"],sort_keys=True))
    print("trips_fold="+json.dumps(report["three_handed_trips_plus_fold"],sort_keys=True))
    print(f"report={args.report.resolve()}")
    print("DC1_3H_AVERAGE_VS_CURRENT_ADVANTAGE_AUDIT_PASS")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
