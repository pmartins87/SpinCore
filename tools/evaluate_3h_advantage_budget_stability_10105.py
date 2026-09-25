#!/usr/bin/env python3
from __future__ import annotations

"""Replay fixed DC1 and measure 3H Advantage stability at 100/200/400 fit steps."""

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
from spincore_nn.action_models import collate_action_observations, make_advantage_action_model

REPRESENTATION="C0_V1_FROZEN_CONTROL"
PROBE_SCHEMA="SPINCORE_3H_ADVANTAGE_BUDGET_PROBE_V1"
BUDGETS=(100,200,400)
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
    return {str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")):float(values[a]) for a in legal}


def named_raw(values,legal):
    return {str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")):float(values[a]) for a in legal}


def argmax_name(p):
    return max(p,key=p.get) if p else None


def tv(a,b):
    keys=set(a)|set(b)
    return 0.5*sum(abs(a.get(k,0.0)-b.get(k,0.0)) for k in keys)


def mix_policies(member_policies):
    keys=set().union(*(p.keys() for p in member_policies))
    return {k:statistics.fmean([p.get(k,0.0) for p in member_policies]) for k in keys}


def pair_metrics(member_policies):
    tvs=[]
    args=[]
    for i in range(len(member_policies)):
        for j in range(i+1,len(member_policies)):
            tvs.append(tv(member_policies[i],member_policies[j]))
            args.append(float(argmax_name(member_policies[i])!=argmax_name(member_policies[j])))
    votes={}
    for p in member_policies:
        a=argmax_name(p)
        votes[a]=votes.get(a,0)+1
    return {
        "pairwise_tv_mean":statistics.fmean(tvs),
        "pairwise_argmax_disagreement":statistics.fmean(args),
        "max_argmax_votes":max(votes.values()),
        "unanimous_argmax":max(votes.values())==len(member_policies),
    }


class BudgetProbePolicy:
    policy_id=SPINCORE_POLICY_ID
    def __init__(self,agent,current,budget_models):
        self.agent=agent
        self.current=current.eval()
        self.models={int(b):[m.eval() for m in models] for b,models in budget_models.items()}
        self._last={}

    def choose_exact(self,state,*,seat:int,rng:random.Random)->ExternalExactAction:
        active_mask,legal,avg_values=self.agent.distribution(state)
        domain=self.agent.domain_for_state(state)
        detail={
            "oracle":"BudgetProbePolicy",
            "selection":"SAMPLED_ACTUAL_HYBRID_POLICY",
            "domain":domain,
            "active_mask":int(active_mask),
            "average_policy":named_policy(avg_values,legal),
        }

        if domain=="THREE_HANDED":
            obs=state.neural_bytes()
            batch=collate_action_observations(
                REPRESENTATION,[obs],[legal_mask(legal)],device="cpu"
            )
            with torch.no_grad():
                current_raw=self.current(batch)[0].detach().cpu().tolist()
            current_policy=lean_regret_matching_policy(current_raw,legal)
            detail["current_policy"]=named_policy(current_policy,legal)
            detail["budgets"]={}
            with torch.no_grad():
                for budget,models in self.models.items():
                    raws=[m(batch)[0].detach().cpu().tolist() for m in models]
                    member_policies=[
                        named_policy(lean_regret_matching_policy(raw,legal),legal)
                        for raw in raws
                    ]
                    raw_mean=[
                        statistics.fmean([raw[a] for raw in raws])
                        for a in range(10)
                    ]
                    raw_ensemble=named_policy(
                        lean_regret_matching_policy(raw_mean,legal),legal
                    )
                    policy_mix=mix_policies(member_policies)
                    pm=pair_metrics(member_policies)
                    detail["budgets"][str(budget)]={
                        "raw_ensemble_policy":raw_ensemble,
                        "policy_mixture":policy_mix,
                        "pairwise_tv_mean":pm["pairwise_tv_mean"],
                        "pairwise_argmax_disagreement":pm["pairwise_argmax_disagreement"],
                        "max_argmax_votes":pm["max_argmax_votes"],
                        "unanimous_argmax":pm["unanimous_argmax"],
                        "member_argmax":[argmax_name(p) for p in member_policies],
                        "member_all_in_probability":[p.get("ALL_IN",0.0) for p in member_policies],
                        "member_fold_probability":[p.get("FOLD",0.0) for p in member_policies],
                        "member_all_in_raw":[float(raw[9]) for raw in raws],
                        "member_fold_raw":[float(raw[0]) for raw in raws],
                        "all_in_positive_raw_count":sum(float(raw[9])>0.0 for raw in raws),
                        "fold_positive_raw_count":sum(float(raw[0])>0.0 for raw in raws),
                    }

        x=rng.random()
        cumulative=0.0
        slot=int(legal[-1])
        for candidate in legal:
            cumulative+=float(avg_values[candidate])
            if x<cumulative:
                slot=int(candidate); break
        action_type,amount_to=resolve_lean_exact(state,active_mask,slot)
        detail.update({
            "sample_u":float(x),
            "selected_slot":int(slot),
            "selected_slot_name":str(NAME_BY_SLOT.get(int(slot),f"SLOT_{int(slot)}")),
            "selected_probability":float(avg_values[slot]),
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
    if p.get("schema")!=PROBE_SCHEMA or tuple(p.get("budgets") or [])!=BUDGETS:
        raise RuntimeError("wrong 3H budget probe")

    def make(state):
        _,m=make_advantage_action_model(REPRESENTATION,device="cpu",seed=0)
        m.load_state_dict(state)
        return m

    current=make(p["current_checkpoint_member"])
    budget_models={
        int(b):[make(state) for state in p["snapshots"][str(b)]]
        for b in BUDGETS
    }
    _SPIN=BudgetProbePolicy(agent,current,budget_models)
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


def pct(vals,p):
    if not vals:return None
    s=sorted(float(x) for x in vals)
    return float(s[min(len(s)-1,max(0,math.ceil(p*len(s))-1))])


def summarize_decisions(rows,budget):
    vals=[r["budget"][str(budget)] for r in rows]
    ptv=[x["pairwise_tv_mean"] for x in vals]
    parg=[x["pairwise_argmax_disagreement"] for x in vals]
    rawmix=[tv(x["raw_ensemble_policy"],x["policy_mixture"]) for x in vals]
    return {
        "decision_count":len(vals),
        "member_pairwise_tv_mean":statistics.fmean(ptv),
        "member_pairwise_tv_median":statistics.median(ptv),
        "member_pairwise_tv_p95":pct(ptv,0.95),
        "member_pairwise_argmax_disagreement_mean":statistics.fmean(parg),
        "unanimous_argmax_rate":statistics.fmean([float(x["unanimous_argmax"]) for x in vals]),
        "mean_max_argmax_vote_share":statistics.fmean([x["max_argmax_votes"]/REPLICAS for x in vals]),
        "raw_ensemble_vs_policy_mixture_tv_mean":statistics.fmean(rawmix),
        "raw_ensemble_vs_policy_mixture_tv_p95":pct(rawmix,0.95),
    }


def weird_summary(rows,budget,action):
    if not rows:return {"count":0}
    bs=[r["budget"][str(budget)] for r in rows]
    raw=[x["raw_ensemble_policy"].get(action,0.0) for x in bs]
    mix=[x["policy_mixture"].get(action,0.0) for x in bs]
    positive_key="all_in_positive_raw_count" if action=="ALL_IN" else "fold_positive_raw_count"
    member_prob_key="member_all_in_probability" if action=="ALL_IN" else "member_fold_probability"
    return {
        "count":len(rows),
        "raw_ensemble_action_probability_mean":statistics.fmean(raw),
        "policy_mixture_action_probability_mean":statistics.fmean(mix),
        "raw_minus_mixture_mean":statistics.fmean([a-b for a,b in zip(raw,mix)]),
        "raw_ensemble_argmax_action_count":sum(argmax_name(x["raw_ensemble_policy"])==action for x in bs),
        "majority_member_argmax_action_count":sum(
            sum(a==action for a in x["member_argmax"])>=5 for x in bs
        ),
        "unanimous_member_argmax_action_count":sum(
            all(a==action for a in x["member_argmax"]) for x in bs
        ),
        "mean_positive_raw_member_count":statistics.fmean([x[positive_key] for x in bs]),
        "mean_member_action_probability_pstdev":statistics.fmean([
            statistics.pstdev(x[member_prob_key]) for x in bs
        ]),
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--spin-bundle",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--scenarios",type=int,default=200)
    ap.add_argument("--workers",type=int,default=4)
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
        ep=sampler.sample_episode()
        tasks.append((index,ep,int(mix64(args.seed,index,0xD34A1)),int(args.max_decisions)))

    all_traces=[]; games=0
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
    decisions=[]
    flags=[]
    for t in traces:
        if t.policy_id!=SPINCORE_POLICY_ID or t.domain!="THREE_HANDED":
            continue
        detail=dict(t.policy_detail or {})
        b=detail.get("budgets") or {}
        if any(str(x) not in b for x in BUDGETS):
            raise RuntimeError("missing budget metadata")
        row={"budget":b}
        decisions.append(row)
        for flag in sanity_flags(t):
            if flag.code not in ("POSTFLOP_DEEP_HIGH_CARD_JAM","POSTFLOP_TRIPS_PLUS_FOLD"):
                continue
            flags.append({
                "code":flag.code,
                "context":flag.context,
                "budget":b,
                "average_policy":detail.get("average_policy"),
                "current_policy":detail.get("current_policy"),
            })

    high=[r for r in flags if r["code"]=="POSTFLOP_DEEP_HIGH_CARD_JAM"]
    nodraw=[
        r for r in high
        if not bool((r["context"] or {}).get("has_immediate_straight_or_flush_draw"))
    ]
    draw=[r for r in high if r not in nodraw]
    trips=[r for r in flags if r["code"]=="POSTFLOP_TRIPS_PLUS_FOLD"]

    out={
        "schema":"SPINCORE_3H_ADVANTAGE_BUDGET_STABILITY_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_PLAYED_TRAJECTORY_UNCHANGED",
        "seed":int(args.seed),
        "scenarios":int(args.scenarios),
        "balanced_games":games,
        "trace_decisions":len(traces),
        "three_handed_decisions":len(decisions),
        "budgets":{},
        "interpretation":(
            "The 100-step contract is the historical 3H current-Advantage budget. "
            "If pairwise independent-fit instability falls materially by 200/400 steps, "
            "fresh100 is underfitting the frozen reservoir. If it remains high, the dominant "
            "issue is not optimizer budget alone. Policy-mixture is reported separately from "
            "raw-Advantage ensembling because regret matching is nonlinear around zero."
        ),
    }
    for budget in BUDGETS:
        out["budgets"][str(budget)]={
            "global":summarize_decisions(decisions,budget),
            "high_card_jam_all":weird_summary(high,budget,"ALL_IN"),
            "high_card_jam_no_immediate_draw":weird_summary(nodraw,budget,"ALL_IN"),
            "high_card_jam_with_immediate_draw":weird_summary(draw,budget,"ALL_IN"),
            "trips_plus_fold":weird_summary(trips,budget,"FOLD"),
        }

    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    for budget in BUDGETS:
        print(f"BUDGET_{budget} "+json.dumps(out["budgets"][str(budget)],sort_keys=True))
    print(f"report={args.report.resolve()}")
    print("3H_ADVANTAGE_BUDGET_STABILITY_AUDIT_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
