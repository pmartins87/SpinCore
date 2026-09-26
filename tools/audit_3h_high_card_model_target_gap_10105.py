#!/usr/bin/env python3
from __future__ import annotations

"""Compare the 1600-step 3H Advantage ensemble to local stored targets.

This diagnostic consumes:
- the corrected V2 high-card target-neighborhood report;
- the controlled-split 100..1600 probe artifact;
- the unchanged fixed DC1 200-scenario trajectory.

It does not train, generate CFR roots, mutate the checkpoint, or alter played
benchmark semantics.  Its purpose is to determine whether extreme ALL_IN mass
at the exact flagged states is supported by local stored Advantage targets or
is mainly a model/generalization + regret-matching effect.
"""

import argparse
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
from spincore.deepcrusher_policy import DeepCrusherR8Policy
from spincore.decision_sanity import sanity_flags
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_solver_actions import resolve_lean_exact
from spincore.lean_hybrid_deployment_agent import LeanHybridDeploymentAgent
from spincore.solver import SolverLibrary
from spincore_nn.action_models import collate_action_observations, make_advantage_action_model

REPRESENTATION="C0_V1_FROZEN_CONTROL"
TARGET_SCHEMA="SPINCORE_3H_HIGH_CARD_ADVANTAGE_TARGET_AUDIT_V2"
PROBE_SCHEMA="SPINCORE_3H_ADVANTAGE_CONTROLLED_SPLIT_PROBE_V1"
BUDGET=1600
REPLICAS=8
SEED=20260923
ALL_IN=9


def mix64(*values:int)->int:
    x=0x9E3779B97F4A7C15
    mask=(1<<64)-1
    for value in values:
        y=int(value)&mask
        x^=(y+0x9E3779B97F4A7C15+((x<<6)&mask)+(x>>2))&mask
        x&=mask
    return x


def rank(card:int)->int:
    return 2+int(card)//4


def card_label(card:int)->str:
    return "23456789TJQKA"[rank(card)-2] + "cdhs"[int(card)%4]


def named(values,legal):
    return {
        str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")):float(values[a])
        for a in legal
    }


def argmax_name(p):
    return max(p,key=p.get) if p else None


def tv(a,b):
    keys=set(a)|set(b)
    return 0.5*sum(abs(a.get(k,0.0)-b.get(k,0.0)) for k in keys)


def pearson(xs,ys):
    if len(xs)<2:return None
    mx=statistics.fmean(xs);my=statistics.fmean(ys)
    vx=sum((x-mx)**2 for x in xs);vy=sum((y-my)**2 for y in ys)
    if vx<=0 or vy<=0:return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(vx*vy)


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
            "domain":domain,
            "legal_slots":[int(a) for a in legal],
            "observation_hex":state.neural_bytes().hex() if domain=="THREE_HANDED" else None,
        }
        return ExternalExactAction(int(action_type),int(amount_to))

    def decision_metadata(self,*,seat:int):
        row=self._last.get(int(seat))
        return None if row is None else dict(row)


def load_model(state):
    _,m=make_advantage_action_model(REPRESENTATION,device="cpu",seed=0)
    m.load_state_dict(state)
    m.eval()
    return m


def replay_flagged(solver,bundle_path,scenarios):
    agent=LeanHybridDeploymentAgent.from_bundle(bundle_path,device="cpu",seed=0)
    spin=CapturePolicy(agent)
    deep=DeepCrusherR8Policy.from_repository(ROOT)
    sampler=LegacyScenarioSampler(seed=SEED^0x5CE0A710,config=LegacyScenarioConfig())
    traces=[]
    for index in range(int(scenarios)):
        episode=sampler.sample_episode()
        engine=OfflineHeadToHeadEngine(
            solver,
            spincore_policy=spin,
            deepcrusher_policy=deep,
            master_seed=SEED,
            max_decisions=200,
            decision_sink=traces.append,
        )
        engine.play_balanced_block(
            episode,
            deal_seed=int(mix64(SEED,index,0xD34A1)),
            scenario_index=int(index),
        )

    out={}
    for t in traces:
        if t.policy_id!=SPINCORE_POLICY_ID or t.domain!="THREE_HANDED":
            continue
        flags=[f for f in sanity_flags(t) if f.code=="POSTFLOP_DEEP_HIGH_CARD_JAM"]
        if not flags:
            continue
        if bool(flags[0].context.get("has_immediate_straight_or_flush_draw")):
            continue
        d=dict(t.policy_detail or {})
        obs=d.get("observation_hex")
        legal=tuple(int(x) for x in d.get("legal_slots") or ())
        if not obs or ALL_IN not in legal:
            raise RuntimeError("missing exact captured 3H state")
        key=(int(t.scenario_index),int(t.lineup_index),int(t.decision_index))
        out[key]={
            "observation":bytes.fromhex(obs),
            "legal":legal,
            "hole":[int(x) for x in (t.hole_cards or ())],
            "board":[int(x) for x in t.board],
        }
    return out,len(traces)


def evaluate_exact(models,observation,legal):
    batch=collate_action_observations(
        REPRESENTATION,[observation],[legal_mask(tuple(legal))],device="cpu"
    )
    raws=[]
    with torch.no_grad():
        for m in models:
            raws.append(m(batch)[0].detach().cpu().tolist())
    member_policies=[
        named(lean_regret_matching_policy(raw,tuple(legal)),tuple(legal))
        for raw in raws
    ]
    raw_mean=[
        statistics.fmean([raw[a] for raw in raws])
        for a in range(10)
    ]
    raw_ens=named(
        lean_regret_matching_policy(raw_mean,tuple(legal)),tuple(legal)
    )
    mix={
        k:statistics.fmean([p.get(k,0.0) for p in member_policies])
        for k in set().union(*(p.keys() for p in member_policies))
    }
    votes=[argmax_name(p) for p in member_policies]
    return {
        "raw_ensemble_advantage":named(raw_mean,tuple(legal)),
        "raw_ensemble_policy":raw_ens,
        "member_policy_mixture":mix,
        "member_all_in_raw":[float(raw[ALL_IN]) for raw in raws],
        "member_all_in_policy":[float(p.get("ALL_IN",0.0)) for p in member_policies],
        "member_argmax":votes,
        "all_in_positive_raw_members":sum(raw[ALL_IN]>0.0 for raw in raws),
        "all_in_argmax_members":sum(v=="ALL_IN" for v in votes),
        "raw_ensemble_vs_policy_mixture_tv":tv(raw_ens,mix),
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--spin-bundle",type=Path,required=True)
    ap.add_argument("--target-report",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--scenarios",type=int,default=200)
    args=ap.parse_args()

    target_report=json.loads(args.target_report.resolve(strict=True).read_text(encoding="utf-8"))
    if target_report.get("schema")!=TARGET_SCHEMA:
        raise SystemExit(f"expected {TARGET_SCHEMA}")
    if int(target_report.get("no_immediate_draw_flags",-1))!=17:
        raise SystemExit("expected 17 no-draw flags")

    probe=torch.load(args.probe.resolve(strict=True),map_location="cpu",weights_only=False)
    if probe.get("schema")!=PROBE_SCHEMA:
        raise SystemExit(f"expected {PROBE_SCHEMA}")
    states=list(probe["snapshots"][str(BUDGET)])
    if len(states)!=REPLICAS:
        raise SystemExit("expected eight 1600-step replicas")
    models=[load_model(s) for s in states]

    solver=SolverLibrary(args.solver.resolve(strict=True))
    replay,trace_count=replay_flagged(
        solver,args.spin_bundle.resolve(strict=True),args.scenarios
    )
    if len(replay)!=17:
        raise RuntimeError(f"expected 17 replayed no-draw flags, got {len(replay)}")

    rows=[]
    for item in target_report["targets"]:
        t=item["target"]
        key=(int(t["scenario_index"]),int(t["lineup_index"]),int(t["decision_index"]))
        if key not in replay:
            raise RuntimeError(f"V2 target missing from exact replay: {key}")
        r=replay[key]
        pred=evaluate_exact(models,r["observation"],r["legal"])
        e=item["subsets"]["E_near_geometry_history_suffix2"]
        f=item["subsets"]["F_same_hole_ranks"]
        recent=e["iteration_cohorts"]["9106_10105"]
        ai_raw=float(pred["raw_ensemble_advantage"].get("ALL_IN",0.0))
        ai_policy=float(pred["raw_ensemble_policy"].get("ALL_IN",0.0))
        ai_mix=float(pred["member_policy_mixture"].get("ALL_IN",0.0))
        ew=e["all_in_target_iteration_weighted_mean"]
        em=e["all_in_target_mean"]
        er=recent["all_in_target_mean"]
        row={
            "key":{"scenario_index":key[0],"lineup_index":key[1],"decision_index":key[2]},
            "hole_labels":[card_label(x) for x in r["hole"]],
            "board_labels":[card_label(x) for x in r["board"]],
            "street":int(t["street"]),
            "facing_class":str(t["facing_class"]),
            "contesting_effective_stack_bb":float(t["contesting_effective_stack_bb"]),
            "local_E":{
                "count":int(e["count"]),
                "all_in_target_mean":em,
                "all_in_target_iteration_weighted_mean":ew,
                "all_in_target_positive_fraction":e["all_in_target_positive_fraction"],
                "all_in_target_argmax_fraction":e["all_in_target_argmax_fraction"],
                "recent_count":int(recent["count"]),
                "recent_all_in_target_mean":er,
                "recent_all_in_target_positive_fraction":recent["all_in_target_positive_fraction"],
            },
            "same_hole_rank_F":{
                "count":int(f["count"]),
                "all_in_target_mean":f["all_in_target_mean"],
                "all_in_target_iteration_weighted_mean":f["all_in_target_iteration_weighted_mean"],
                "recent_count":int(f["iteration_cohorts"]["9106_10105"]["count"]),
                "recent_all_in_target_mean":f["iteration_cohorts"]["9106_10105"]["all_in_target_mean"],
            },
            "model_1600":pred,
            "comparison":{
                "raw_all_in_advantage":ai_raw,
                "raw_all_in_positive":bool(ai_raw>0.0),
                "raw_ensemble_all_in_policy":ai_policy,
                "member_policy_mixture_all_in":ai_mix,
                "member_all_in_argmax_count":int(pred["all_in_argmax_members"]),
                "member_all_in_positive_raw_count":int(pred["all_in_positive_raw_members"]),
                "sign_agrees_E_weighted":(
                    None if ew is None else
                    ((ai_raw>0.0 and ew>0.0) or (ai_raw<0.0 and ew<0.0) or (ai_raw==0.0 and ew==0.0))
                ),
                "model_positive_E_weighted_negative":bool(
                    ew is not None and ai_raw>0.0 and ew<0.0
                ),
                "model_positive_recent_negative":bool(
                    er is not None and ai_raw>0.0 and er<0.0
                ),
            },
        }
        rows.append(row)

    e_weighted=[r["local_E"]["all_in_target_iteration_weighted_mean"] for r in rows]
    e_mean=[r["local_E"]["all_in_target_mean"] for r in rows]
    recent=[r["local_E"]["recent_all_in_target_mean"] for r in rows]
    raws=[r["comparison"]["raw_all_in_advantage"] for r in rows]
    mix=[r["comparison"]["member_policy_mixture_all_in"] for r in rows]
    rawp=[r["comparison"]["raw_ensemble_all_in_policy"] for r in rows]

    strong_recent=[
        r for r in rows if r["local_E"]["recent_count"]>=30
    ]
    false_e=[
        r for r in rows if r["comparison"]["model_positive_E_weighted_negative"]
    ]
    high_jam_negative=[
        r for r in rows
        if r["comparison"]["raw_ensemble_all_in_policy"]>=0.50
        and r["local_E"]["all_in_target_iteration_weighted_mean"] is not None
        and r["local_E"]["all_in_target_iteration_weighted_mean"]<0.0
    ]

    aggregate={
        "states":len(rows),
        "E_count_min":min(r["local_E"]["count"] for r in rows),
        "E_count_median":statistics.median(r["local_E"]["count"] for r in rows),
        "E_count_max":max(r["local_E"]["count"] for r in rows),
        "E_weighted_target_positive_states":sum(x>0 for x in e_weighted if x is not None),
        "E_weighted_target_negative_states":sum(x<0 for x in e_weighted if x is not None),
        "E_unweighted_target_positive_states":sum(x>0 for x in e_mean if x is not None),
        "E_unweighted_target_negative_states":sum(x<0 for x in e_mean if x is not None),
        "recent_target_positive_states":sum(x>0 for x in recent if x is not None),
        "recent_target_negative_states":sum(x<0 for x in recent if x is not None),
        "model_raw_all_in_positive_states":sum(x>0 for x in raws),
        "model_raw_all_in_negative_states":sum(x<0 for x in raws),
        "model_positive_E_weighted_negative_states":len(false_e),
        "raw_ensemble_all_in_ge_50_with_E_weighted_negative_states":len(high_jam_negative),
        "mean_raw_ensemble_all_in_policy":statistics.fmean(rawp),
        "mean_member_policy_mixture_all_in":statistics.fmean(mix),
        "pearson_model_raw_all_in_vs_E_weighted_mean":pearson(
            raws,[float(x) for x in e_weighted]
        ),
        "pearson_model_raw_all_in_vs_E_unweighted_mean":pearson(
            raws,[float(x) for x in e_mean]
        ),
        "strong_recent_E_count_ge_30_states":len(strong_recent),
        "strong_recent_negative_mean_states":sum(
            r["local_E"]["recent_all_in_target_mean"]<0 for r in strong_recent
        ),
        "strong_recent_positive_mean_states":sum(
            r["local_E"]["recent_all_in_target_mean"]>0 for r in strong_recent
        ),
        "F_same_hole_rank_zero_count_states":sum(
            r["same_hole_rank_F"]["count"]==0 for r in rows
        ),
        "F_same_hole_rank_median_count":statistics.median(
            r["same_hole_rank_F"]["count"] for r in rows
        ),
        "false_positive_examples":[
            {
                "key":r["key"],
                "hole_labels":r["hole_labels"],
                "board_labels":r["board_labels"],
                "raw_all_in_advantage":r["comparison"]["raw_all_in_advantage"],
                "raw_ensemble_all_in_policy":r["comparison"]["raw_ensemble_all_in_policy"],
                "member_policy_mixture_all_in":r["comparison"]["member_policy_mixture_all_in"],
                "E_count":r["local_E"]["count"],
                "E_weighted_mean":r["local_E"]["all_in_target_iteration_weighted_mean"],
                "E_recent_count":r["local_E"]["recent_count"],
                "E_recent_mean":r["local_E"]["recent_all_in_target_mean"],
                "F_count":r["same_hole_rank_F"]["count"],
                "F_mean":r["same_hole_rank_F"]["all_in_target_mean"],
            }
            for r in false_e
        ],
    }

    out={
        "schema":"SPINCORE_3H_HIGH_CARD_MODEL_TARGET_GAP_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_NO_TRAINING_NO_CFR_ROOTS",
        "budget":BUDGET,
        "replicas":REPLICAS,
        "target_report_schema":TARGET_SCHEMA,
        "probe_schema":PROBE_SCHEMA,
        "trace_decisions":trace_count,
        "aggregate":aggregate,
        "states":rows,
        "interpretation":(
            "A positive exact-state model ALL_IN raw Advantage together with a negative "
            "iteration-weighted local-E stored target mean is evidence of model/generalization "
            "sign mismatch at that diagnostic neighborhood. It is not a proof that the poker "
            "action is strategically wrong because E remains an approximate neighborhood and "
            "same-hole-rank F coverage is often sparse. If the exact model raw values are small "
            "while regret matching produces large ALL_IN mass, nonlinear sign-threshold "
            "amplification is also implicated."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== 3H high-card model-target gap @1600 ===")
    print(json.dumps(aggregate,indent=2,sort_keys=True))
    print(f"report={args.report.resolve()}")
    print("3H_HIGH_CARD_MODEL_TARGET_GAP_AUDIT_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
