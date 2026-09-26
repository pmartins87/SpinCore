#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate controlled-split 3H Advantage budget snapshots on fixed DC1 states."""

import argparse
from dataclasses import asdict
import json
import math
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
from spincore_nn.action_models import collate_action_observations, make_advantage_action_model

REPRESENTATION="C0_V1_FROZEN_CONTROL"
PROBE_SCHEMA="SPINCORE_3H_ADVANTAGE_CONTROLLED_SPLIT_PROBE_V1"
BUDGETS=(100,200,400,800,1600)
REPLICAS=8


def mix64(*values:int)->int:
    x=0x9E3779B97F4A7C15
    mask=(1<<64)-1
    for value in values:
        y=int(value)&mask
        x^=(y+0x9E3779B97F4A7C15+((x<<6)&mask)+(x>>2))&mask
        x&=mask
    return x


def named(values,legal):
    return {str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")):float(values[a]) for a in legal}


def argmax_name(p):
    return max(p,key=p.get) if p else None


def tv(a,b):
    keys=set(a)|set(b)
    return 0.5*sum(abs(a.get(k,0.0)-b.get(k,0.0)) for k in keys)


def pct(vals,p):
    if not vals:return None
    s=sorted(float(x) for x in vals)
    return float(s[min(len(s)-1,max(0,math.ceil(p*len(s))-1))])


def load_model(state):
    _,m=make_advantage_action_model(REPRESENTATION,device="cpu",seed=0)
    m.load_state_dict(state);m.eval()
    return m


class CapturePolicy:
    policy_id=SPINCORE_POLICY_ID
    def __init__(self,agent):
        self.agent=agent
        self._last={}

    def choose_exact(self,state,*,seat:int,rng:random.Random)->ExternalExactAction:
        active_mask,legal,probs=self.agent.distribution(state)
        domain=self.agent.domain_for_state(state)
        x=rng.random()
        cumulative=0.0
        slot=int(legal[-1])
        for a in legal:
            cumulative+=float(probs[a])
            if x<cumulative:
                slot=int(a);break
        action_type,amount_to=resolve_lean_exact(state,active_mask,slot)
        self._last[int(seat)]={
            "oracle":"ControlledBudgetCapture",
            "selection":"SAMPLED_ACTUAL_HYBRID_POLICY",
            "domain":domain,
            "active_mask":int(active_mask),
            "sample_u":float(x),
            "selected_slot":int(slot),
            "selected_slot_name":str(NAME_BY_SLOT.get(slot,f"SLOT_{slot}")),
            "selected_probability":float(probs[slot]),
            "legal_actions":[
                {"slot":int(a),"name":str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")),"probability":float(probs[a])}
                for a in legal
            ],
            "observation_hex":state.neural_bytes().hex() if domain=="THREE_HANDED" else None,
            "legal_slots":[int(a) for a in legal] if domain=="THREE_HANDED" else None,
        }
        return ExternalExactAction(int(action_type),int(amount_to))

    def decision_metadata(self,*,seat:int):
        x=self._last.get(int(seat))
        return None if x is None else dict(x)


def evaluate_one(models,obs_hex,legal):
    obs=bytes.fromhex(obs_hex)
    batch=collate_action_observations(
        REPRESENTATION,[obs],[legal_mask(tuple(legal))],device="cpu"
    )
    raws=[]
    with torch.no_grad():
        for m in models:
            raws.append(m(batch)[0].detach().cpu().tolist())
    members=[
        named(lean_regret_matching_policy(raw,tuple(legal)),tuple(legal))
        for raw in raws
    ]
    raw_mean=[statistics.fmean([raw[a] for raw in raws]) for a in range(10)]
    raw_ens=named(lean_regret_matching_policy(raw_mean,tuple(legal)),tuple(legal))
    mix={
        k:statistics.fmean([p.get(k,0.0) for p in members])
        for k in set().union(*(p.keys() for p in members))
    }
    pair_tvs=[];pair_arg=[]
    for i in range(REPLICAS):
        for j in range(i+1,REPLICAS):
            pair_tvs.append(tv(members[i],members[j]))
            pair_arg.append(float(argmax_name(members[i])!=argmax_name(members[j])))
    votes={}
    for p in members:
        a=argmax_name(p);votes[a]=votes.get(a,0)+1
    return {
        "raw_ensemble_policy":raw_ens,
        "policy_mixture":mix,
        "member_policies":members,
        "pairwise_tv_mean":statistics.fmean(pair_tvs),
        "pairwise_argmax_disagreement":statistics.fmean(pair_arg),
        "max_argmax_votes":max(votes.values()),
        "unanimous_argmax":max(votes.values())==REPLICAS,
        "member_argmax":[argmax_name(p) for p in members],
        "all_in_positive_raw_count":sum(raw[9]>0.0 for raw in raws),
        "fold_positive_raw_count":sum(raw[0]>0.0 for raw in raws),
    }


def summarize(rows,budget):
    vals=[r["budget"][str(budget)] for r in rows]
    return {
        "decision_count":len(vals),
        "member_pairwise_tv_mean":statistics.fmean([x["pairwise_tv_mean"] for x in vals]),
        "member_pairwise_tv_median":statistics.median([x["pairwise_tv_mean"] for x in vals]),
        "member_pairwise_tv_p95":pct([x["pairwise_tv_mean"] for x in vals],0.95),
        "member_pairwise_argmax_disagreement_mean":statistics.fmean([x["pairwise_argmax_disagreement"] for x in vals]),
        "mean_max_argmax_vote_share":statistics.fmean([x["max_argmax_votes"]/REPLICAS for x in vals]),
        "unanimous_argmax_rate":statistics.fmean([float(x["unanimous_argmax"]) for x in vals]),
        "raw_ensemble_vs_policy_mixture_tv_mean":statistics.fmean([
            tv(x["raw_ensemble_policy"],x["policy_mixture"]) for x in vals
        ]),
    }


def weird(rows,budget,action):
    if not rows:return {"count":0}
    vals=[r["budget"][str(budget)] for r in rows]
    raw=[x["raw_ensemble_policy"].get(action,0.0) for x in vals]
    mix=[x["policy_mixture"].get(action,0.0) for x in vals]
    poskey="all_in_positive_raw_count" if action=="ALL_IN" else "fold_positive_raw_count"
    return {
        "count":len(rows),
        "raw_ensemble_action_probability_mean":statistics.fmean(raw),
        "policy_mixture_action_probability_mean":statistics.fmean(mix),
        "raw_ensemble_argmax_action_count":sum(argmax_name(x["raw_ensemble_policy"])==action for x in vals),
        "majority_member_argmax_action_count":sum(
            sum(a==action for a in x["member_argmax"])>=5 for x in vals
        ),
        "unanimous_member_argmax_action_count":sum(
            all(a==action for a in x["member_argmax"]) for x in vals
        ),
        "mean_positive_raw_member_count":statistics.fmean([x[poskey] for x in vals]),
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--spin-bundle",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--scenarios",type=int,default=200)
    ap.add_argument("--seed",type=int,default=20260923)
    args=ap.parse_args()

    from spincore.solver import SolverLibrary
    from spincore.deepcrusher_policy import DeepCrusherR8Policy
    from spincore.lean_hybrid_deployment_agent import LeanHybridDeploymentAgent

    solver=SolverLibrary(args.solver.resolve(strict=True))
    agent=LeanHybridDeploymentAgent.from_bundle(args.spin_bundle.resolve(strict=True),device="cpu",seed=0)
    spin=CapturePolicy(agent)
    deep=DeepCrusherR8Policy.from_repository(ROOT)

    sampler=LegacyScenarioSampler(
        seed=int(args.seed)^0x5CE0A710,
        config=LegacyScenarioConfig(),
    )
    traces=[]
    games=0
    for index in range(int(args.scenarios)):
        episode=sampler.sample_episode()
        engine=OfflineHeadToHeadEngine(
            solver,
            spincore_policy=spin,
            deepcrusher_policy=deep,
            master_seed=int(args.seed),
            max_decisions=200,
            decision_sink=traces.append,
        )
        obs=engine.play_balanced_block(
            episode,
            deal_seed=int(mix64(args.seed,index,0xD34A1)),
            scenario_index=int(index),
        )
        games+=len(obs)

    probe=torch.load(args.probe.resolve(strict=True),map_location="cpu",weights_only=False)
    if probe.get("schema")!=PROBE_SCHEMA or tuple(int(x) for x in probe.get("budgets") or [])!=BUDGETS:
        raise SystemExit("wrong controlled-split probe")

    snapshots=probe["snapshots"]
    models={b:[load_model(s) for s in snapshots[str(b)]] for b in BUDGETS}

    decisions=[]
    flagged=[]
    for t in traces:
        if t.policy_id!=SPINCORE_POLICY_ID or t.domain!="THREE_HANDED":
            continue
        d=dict(t.policy_detail or {})
        obs_hex=d.get("observation_hex")
        legal=d.get("legal_slots")
        if not obs_hex or not legal:
            raise RuntimeError("missing captured 3H observation")
        row={"budget":{}}
        for b in BUDGETS:
            row["budget"][str(b)]=evaluate_one(models[b],obs_hex,legal)
        decisions.append(row)
        for flag in sanity_flags(t):
            if flag.code in ("POSTFLOP_DEEP_HIGH_CARD_JAM","POSTFLOP_TRIPS_PLUS_FOLD"):
                flagged.append({
                    "code":flag.code,
                    "context":flag.context,
                    "budget":row["budget"],
                    "average_policy":{
                        str(x["name"]):float(x["probability"]) for x in d["legal_actions"]
                    },
                })

    high=[r for r in flagged if r["code"]=="POSTFLOP_DEEP_HIGH_CARD_JAM"]
    nodraw=[r for r in high if not bool((r["context"] or {}).get("has_immediate_straight_or_flush_draw"))]
    draw=[r for r in high if r not in nodraw]
    trips=[r for r in flagged if r["code"]=="POSTFLOP_TRIPS_PLUS_FOLD"]

    out={
        "schema":"SPINCORE_3H_ADVANTAGE_CONTROLLED_BUDGET_CURVE_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_PLAYED_TRAJECTORY_UNCHANGED",
        "scenarios":int(args.scenarios),
        "balanced_games":games,
        "trace_decisions":len(traces),
        "three_handed_decisions":len(decisions),
        "holdout_validation":probe["validation"],
        "budgets":{},
        "interpretation":(
            "All replicas use the same fixed 50k reservoir holdout excluded from training. "
            "This makes validation MSE directly comparable from 100 through 1600 steps. "
            "The DC1 section separately tests whether greater target-regression fit stabilizes "
            "policy behavior or changes the suspicious high-card/trips actions."
        ),
    }
    for b in BUDGETS:
        out["budgets"][str(b)]={
            "global":summarize(decisions,b),
            "high_card_jam_all":weird(high,b,"ALL_IN"),
            "high_card_jam_no_immediate_draw":weird(nodraw,b,"ALL_IN"),
            "high_card_jam_with_immediate_draw":weird(draw,b,"ALL_IN"),
            "trips_plus_fold":weird(trips,b,"FOLD"),
        }

    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    for b in BUDGETS:
        print(
            f"CONTROLLED_BUDGET_{b} "
            + json.dumps({
                "holdout":out["holdout_validation"][str(b)],
                "global":out["budgets"][str(b)]["global"],
                "high_no_draw":out["budgets"][str(b)]["high_card_jam_no_immediate_draw"],
                "trips":out["budgets"][str(b)]["trips_plus_fold"],
            },sort_keys=True),
            flush=True,
        )
    print(f"report={args.report.resolve()}")
    print("3H_ADVANTAGE_CONTROLLED_BUDGET_CURVE_AUDIT_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
