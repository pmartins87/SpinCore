#!/usr/bin/env python3
from __future__ import annotations

"""Replay the frozen DC1 smoke and compare 3H AveragePolicy/current/ENS8 refits.

The benchmark trajectory is still sampled from the unchanged actual hybrid:
THREE_HANDED finalized AveragePolicy + TRUE_HEADS_UP validated ENS8.

Current 3H Advantage and the eight independent 3H refits are observational only.
The eight refits are combined by averaging raw Advantage outputs before the
unchanged lean regret-matching transform, matching the validated HU ensemble
semantics.
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

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))

from spincore.deepcrusher_benchmark import (
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
PROBE_SCHEMA="SPINCORE_3H_ENS8_DIAGNOSTIC_PROBE_V1"
REPLICAS=8

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


def named_policy(values,legal):
    return [
        {
            "slot":int(a),
            "name":str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")),
            "probability":float(values[a]),
        }
        for a in legal
    ]


def named_raw(values,legal):
    return [
        {
            "slot":int(a),
            "name":str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")),
            "raw":float(values[a]),
        }
        for a in legal
    ]


class HybridWith3HEns8Probe:
    policy_id=SPINCORE_POLICY_ID
    def __init__(self,agent,current_model,replica_models):
        self.agent=agent
        self.current=current_model.eval()
        self.replicas=[m.eval() for m in replica_models]
        self._last={}

    def choose_exact(self,state,*,seat:int,rng:random.Random)->ExternalExactAction:
        active_mask,legal,avg=self.agent.distribution(state)
        domain=self.agent.domain_for_state(state)

        detail={
            "oracle":"HybridWith3HEns8Probe",
            "selection":"SAMPLED_ACTUAL_HYBRID_POLICY",
            "domain":domain,
            "active_mask":int(active_mask),
            "legal_actions":named_policy(avg,legal),
        }

        if domain=="THREE_HANDED":
            obs=state.neural_bytes()
            batch=collate_action_observations(
                REPRESENTATION,[obs],[legal_mask(legal)],device="cpu"
            )
            with torch.no_grad():
                current_raw=self.current(batch)[0].detach().cpu()
                replica_raw=[
                    model(batch)[0].detach().cpu()
                    for model in self.replicas
                ]
            current_policy=lean_regret_matching_policy(current_raw.tolist(),legal)
            ens_raw=torch.stack(replica_raw,dim=0).mean(dim=0)
            ens_policy=lean_regret_matching_policy(ens_raw.tolist(),legal)
            member_policies=[
                lean_regret_matching_policy(raw.tolist(),legal)
                for raw in replica_raw
            ]
            detail["current_3h_advantage_policy"]=named_policy(current_policy,legal)
            detail["current_3h_advantage_raw"]=named_raw(current_raw.tolist(),legal)
            detail["ens8_3h_advantage_policy"]=named_policy(ens_policy,legal)
            detail["ens8_3h_advantage_raw"]=named_raw(ens_raw.tolist(),legal)
            detail["ens8_member_policies"]=[
                named_policy(policy,legal) for policy in member_policies
            ]

        x=rng.random()
        cumulative=0.0
        slot=int(legal[-1])
        for candidate in legal:
            cumulative+=float(avg[candidate])
            if x<cumulative:
                slot=int(candidate)
                break
        action_type,amount_to=resolve_lean_exact(state,active_mask,slot)
        detail.update({
            "sample_u":float(x),
            "selected_slot":int(slot),
            "selected_slot_name":str(NAME_BY_SLOT.get(int(slot),f"SLOT_{int(slot)}")),
            "selected_probability":float(avg[slot]),
            "resolved_action_type":int(action_type),
            "resolved_amount_to":int(amount_to),
        })
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
        raise RuntimeError("wrong 3H ENS8 diagnostic probe")
    members=list(p.get("members") or [])
    if len(members)!=REPLICAS:
        raise RuntimeError("3H ENS8 probe member count drift")

    def make(state):
        _,model=make_advantage_action_model(REPRESENTATION,device="cpu",seed=0)
        model.load_state_dict(state)
        return model

    current=make(p["current_checkpoint_member"])
    replicas=[make(state) for state in members]
    _SPIN=HybridWith3HEns8Probe(agent,current,replicas)
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
    return {"traces":[asdict(t) for t in traces],"games":len(obs)}


def probs(detail,key):
    return {str(x["name"]):float(x["probability"]) for x in (detail.get(key) or [])}


def tv(a,b):
    names=set(a)|set(b)
    return 0.5*sum(abs(a.get(k,0.0)-b.get(k,0.0)) for k in names)


def argmax_name(p):
    return max(p,key=p.get) if p else None


def pct(values,p):
    if not values:return None
    s=sorted(float(x) for x in values)
    return float(s[min(len(s)-1,max(0,math.ceil(p*len(s))-1))])


def drift_summary(a,b):
    values=[tv(x,y) for x,y in zip(a,b)]
    arg=[float(argmax_name(x)!=argmax_name(y)) for x,y in zip(a,b)]
    return {
        "decision_count":len(values),
        "tv_mean":statistics.fmean(values) if values else None,
        "tv_median":statistics.median(values) if values else None,
        "tv_p95":pct(values,0.95),
        "argmax_disagreement_rate":statistics.fmean(arg) if arg else None,
    }


def action_summary(rows,action):
    if not rows:
        return {"count":0}
    def vals(key):
        return [float(r[key].get(action,0.0)) for r in rows]
    av=vals("average_policy")
    cur=vals("current_policy")
    ens=vals("ens8_policy")
    member_matrix=[
        [float(member.get(action,0.0)) for member in r["member_policies"]]
        for r in rows
    ]
    member_stds=[statistics.pstdev(x) for x in member_matrix]
    return {
        "count":len(rows),
        "average_probability_mean":statistics.fmean(av),
        "current_probability_mean":statistics.fmean(cur),
        "ens8_probability_mean":statistics.fmean(ens),
        "ens8_probability_median":statistics.median(ens),
        "ens8_argmax_is_action_count":sum(argmax_name(r["ens8_policy"])==action for r in rows),
        "ens8_probability_zero_count":sum(x==0.0 for x in ens),
        "ens8_probability_lower_than_current_count":sum(e<c for e,c in zip(ens,cur)),
        "ens8_probability_lower_than_average_count":sum(e<a for e,a in zip(ens,av)),
        "member_action_probability_pstdev_mean":statistics.fmean(member_stds),
        "all_8_members_argmax_action_count":sum(
            all(argmax_name(member)==action for member in r["member_policies"])
            for r in rows
        ),
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--spin-bundle",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
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
            str(args.probe.resolve()),
            int(args.seed),
        ),
    ) as pool:
        for chunk in pool.map(worker,tasks,chunksize=1):
            all_traces.extend(chunk["traces"])
            games+=int(chunk["games"])

    from spincore.deepcrusher_benchmark import DecisionTrace
    traces=[DecisionTrace(**row) for row in all_traces]

    avgs=[]; currents=[]; ensembles=[]
    rows=[]
    for t in traces:
        if t.policy_id!=SPINCORE_POLICY_ID or t.domain!="THREE_HANDED":
            continue
        detail=dict(t.policy_detail or {})
        avg=probs(detail,"legal_actions")
        cur=probs(detail,"current_3h_advantage_policy")
        ens=probs(detail,"ens8_3h_advantage_policy")
        members=[
            {str(x["name"]):float(x["probability"]) for x in member}
            for member in (detail.get("ens8_member_policies") or [])
        ]
        if not (avg and cur and ens and len(members)==REPLICAS):
            raise RuntimeError("missing 3H diagnostic policy metadata")
        avgs.append(avg); currents.append(cur); ensembles.append(ens)

        for flag in sanity_flags(t):
            if flag.code not in ("POSTFLOP_DEEP_HIGH_CARD_JAM","POSTFLOP_TRIPS_PLUS_FOLD"):
                continue
            rows.append({
                "code":flag.code,
                "severity":flag.severity,
                "reason":flag.reason,
                "context":flag.context,
                "average_policy":avg,
                "current_policy":cur,
                "ens8_policy":ens,
                "member_policies":members,
                "average_argmax":argmax_name(avg),
                "current_argmax":argmax_name(cur),
                "ens8_argmax":argmax_name(ens),
            })

    high=[r for r in rows if r["code"]=="POSTFLOP_DEEP_HIGH_CARD_JAM"]
    high_no_draw=[
        r for r in high
        if not bool((r.get("context") or {}).get("has_immediate_straight_or_flush_draw"))
    ]
    high_draw=[r for r in high if r not in high_no_draw]
    trips=[r for r in rows if r["code"]=="POSTFLOP_TRIPS_PLUS_FOLD"]

    report={
        "schema":"SPINCORE_DC1_3H_ENS8_UNCERTAINTY_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_PLAYED_TRAJECTORY_UNCHANGED",
        "seed":int(args.seed),
        "scenarios":int(args.scenarios),
        "balanced_games":games,
        "trace_decisions":len(traces),
        "three_handed_decisions":len(avgs),
        "drift":{
            "average_vs_current":drift_summary(avgs,currents),
            "average_vs_ens8":drift_summary(avgs,ensembles),
            "current_vs_ens8":drift_summary(currents,ensembles),
        },
        "high_card_jam_all":action_summary(high,"ALL_IN"),
        "high_card_jam_no_immediate_straight_or_flush_draw":action_summary(high_no_draw,"ALL_IN"),
        "high_card_jam_with_immediate_straight_or_flush_draw":action_summary(high_draw,"ALL_IN"),
        "trips_plus_fold":action_summary(trips,"FOLD"),
        "flags":rows,
        "interpretation":(
            "If independent same-budget 3H refits disagree strongly and ENS8 materially changes the "
            "weird-action probabilities, the final single current Advantage is high-variance. If ENS8 "
            "retains the high-card aggression across independent learners, the signal is in the shared "
            "Advantage reservoir/learning target rather than one unlucky final fit. AveragePolicy remains "
            "the actually played 3H policy in this replay."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("=== DC1 3H ENS8 uncertainty ===")
    print("drift="+json.dumps(report["drift"],sort_keys=True))
    print("high_all="+json.dumps(report["high_card_jam_all"],sort_keys=True))
    print("high_no_draw="+json.dumps(report["high_card_jam_no_immediate_straight_or_flush_draw"],sort_keys=True))
    print("trips="+json.dumps(report["trips_plus_fold"],sort_keys=True))
    print(f"report={args.report.resolve()}")
    print("DC1_3H_ENS8_UNCERTAINTY_AUDIT_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
