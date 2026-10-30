#!/usr/bin/env python3
from __future__ import annotations

"""Paired direct-play/common-chance gate for teacher @17315 aggregation A vs B.

PROJECT_CONTRACT_IDS:
MODEL-020,MODEL-025,VALID-025,VALID-031,VALID-034,VALID-035,SAFE-001,SAFE-010,RNG-001,RNG-002,RNG-003

Research-only. Uses the exact same frozen Semantic ENS8 teacher members and
changes only the ensemble behavior map:
A = mean raw Advantages -> lean regret matching.
B = lean regret matching per member -> mean member policies.
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

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_teacher_sanity_17315 as sanity
import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import evaluate_deepcrusher_dc1 as dcbase

from spincore.deepcrusher_benchmark import OfflineHeadToHeadEngine,SPINCORE_POLICY_ID
from spincore.deepcrusher_policy import DeepCrusherR8Policy
from spincore.legacy_scenario import LegacyScenarioConfig,LegacyScenarioSampler
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import lean_legal_actions,resolve_lean_exact
from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore.solver import SolverLibrary
from spincore_nn.action_models import collate_action_observations
from spincore_nn.semantic_v1 import semantic_vector_from_observation

SCHEMA="SPINCORE_TEACHER_MEMBER_POLICY_MEAN_DIRECT_PLAY_17315_V1"
ENSEMBLE_SCHEMA="SPINCORE_3H_SEMANTIC_LONG_ENSEMBLE_V1"
ITERATION=17315
DOMAIN="THREE_HANDED"

_SOLVER=None
_POLICY=None
_DC=None
_ENGINE_SEED=None


def mean_ci(values:list[float])->dict[str,float|int]:
    n=len(values)
    if n==0:
        return {"n":0,"mean":float("nan"),"sem":float("nan"),"ci95_low":float("nan"),"ci95_high":float("nan")}
    mean=float(statistics.fmean(values))
    sem=0.0 if n==1 else float(statistics.stdev(values)/math.sqrt(n))
    half=1.96*sem
    return {"n":n,"mean":mean,"sem":sem,"ci95_low":mean-half,"ci95_high":mean+half}


class MemberPolicyMeanTeacherPolicy(sanity.SemanticAdvantageTeacherPolicy):
    def distribution(self,state):
        street=sanity.street_from_state(state)
        active=FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal=tuple(int(x) for x in lean_legal_actions(state,active))
        if not legal:
            raise RuntimeError("nonterminal state has no legal action")
        obs=state.neural_bytes()
        batch=collate_action_observations(
            distill.REPRESENTATION,
            [obs],
            [legal_mask(legal)],
            device="cpu",
        )
        sb=dict(batch)
        sb["semantic"]=torch.tensor(
            np.asarray([semantic_vector_from_observation(obs)],dtype=np.float32),
            dtype=torch.float32,
        )
        policies=[]
        with torch.no_grad():
            for model in self.models:
                model.eval()
                raw=model(sb)[0].detach().cpu().tolist()
                policies.append(
                    lean_regret_matching_policy(raw,legal)
                )
        out=[0.0]*10
        for action in legal:
            out[action]=sum(float(p[action]) for p in policies)/float(len(policies))
        mass=sum(out[a] for a in legal)
        if abs(mass-1.0)>1e-8:
            raise RuntimeError(f"member-policy mean mass drift: {mass}")
        return int(active),legal,tuple(out)


def init_worker(solver_path:str,ensemble_path:str,root_path:str,seed:int,mode:str)->None:
    global _SOLVER,_POLICY,_DC,_ENGINE_SEED
    import os
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    os.environ["OPENBLAS_NUM_THREADS"]="1"
    os.environ["SPINCORE_TORCH_THREADS"]="1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _SOLVER=SolverLibrary(solver_path)
    if not _SOLVER.explicit_deal_available:
        raise RuntimeError("explicit-deal solver ABI required")
    payload=torch.load(ensemble_path,map_location="cpu",weights_only=False)
    if payload.get("schema")!=ENSEMBLE_SCHEMA:
        raise RuntimeError("wrong ensemble schema")
    if int(payload.get("completed_iteration",-1))!=ITERATION:
        raise RuntimeError("teacher iteration mismatch")
    states=list(payload.get("members") or [])
    if len(states)!=8:
        raise RuntimeError("expected eight semantic members")
    models=[distill.load_semantic_advantage(state) for state in states]
    if mode=="A_CURRENT_RAW_MEAN_RM":
        _POLICY=sanity.SemanticAdvantageTeacherPolicy(models,"17315_A_RAW_MEAN")
    elif mode=="B_MEMBER_POLICY_MEAN":
        _POLICY=MemberPolicyMeanTeacherPolicy(models,"17315_B_MEMBER_POLICY_MEAN")
    else:
        raise RuntimeError(f"unknown mode: {mode}")
    _DC=DeepCrusherR8Policy.from_repository(root_path)
    _ENGINE_SEED=int(seed)


def worker(task):
    index,episode,deal_seed,max_decisions=task
    traces=[]
    engine=OfflineHeadToHeadEngine(
        _SOLVER,
        spincore_policy=_POLICY,
        deepcrusher_policy=_DC,
        master_seed=int(_ENGINE_SEED),
        max_decisions=int(max_decisions),
        decision_sink=traces.append,
    )
    observations=engine.play_balanced_block(
        episode,
        deal_seed=int(deal_seed),
        scenario_index=int(index),
    )
    summary=dcbase._scenario_summary(observations)
    summary.update({
        "scenario":int(index),
        "domain":DOMAIN,
        "blind":f"{int(episode.small_blind)}/{int(episode.big_blind)}",
        "dealer_id":int(episode.dealer_id),
        "deal_seed":int(deal_seed),
    })

    flag_counts={}
    opportunities={}
    opportunity_policy={}
    spin_decisions=0
    for trace in traces:
        if trace.policy_id!=SPINCORE_POLICY_ID:
            continue
        spin_decisions+=1
        for code in sanity._opportunity_codes(trace):
            opportunities[code]=opportunities.get(code,0)+1
            action_name=sanity._target_action_for_opportunity(code)
            probability=sanity._policy_probability(trace,action_name)
            stats=opportunity_policy.setdefault(code,{
                "opportunities":0,
                "probability_sum":0.0,
                "probability_max":0.0,
                "ge_50pct":0,
                "ge_80pct":0,
                "ge_95pct":0,
                "top_probability_rows":[],
            })
            stats["opportunities"]+=1
            stats["probability_sum"]+=float(probability)
            stats["probability_max"]=max(float(stats["probability_max"]),float(probability))
            stats["ge_50pct"]+=int(probability>=0.50)
            stats["ge_80pct"]+=int(probability>=0.80)
            stats["ge_95pct"]+=int(probability>=0.95)
            stats["top_probability_rows"].append(
                sanity._opportunity_policy_row(trace,code,probability)
            )
            stats["top_probability_rows"].sort(
                key=lambda x:float(x["target_probability"]),reverse=True
            )
            stats["top_probability_rows"]=stats["top_probability_rows"][:5]
        for flag in sanity.sanity_flags(trace):
            code=str(flag.code)
            flag_counts[code]=flag_counts.get(code,0)+1

    return {
        "scenario":int(index),
        "summary":summary,
        "spin_decisions":spin_decisions,
        "flag_counts":flag_counts,
        "opportunities":opportunities,
        "opportunity_policy":opportunity_policy,
    }


def merge_counts(rows,key):
    out={}
    for row in rows:
        for code,value in row[key].items():
            out[str(code)]=out.get(str(code),0)+int(value)
    return dict(sorted(out.items()))


def summarize_candidate(rows):
    rows=sorted(rows,key=lambda x:int(x["scenario"]))
    summaries=[r["summary"] for r in rows]
    by_domain,_=dcbase._summaries(summaries)
    flags=merge_counts(rows,"flag_counts")
    opportunities=merge_counts(rows,"opportunities")
    policy=sanity._merge_opportunity_policy(rows)
    spin_decisions=sum(int(r["spin_decisions"]) for r in rows)
    rates={}
    for code in sorted(set(flags)|set(opportunities)):
        n=int(opportunities.get(code,0))
        f=int(flags.get(code,0))
        rates[code]={
            "flags":f,
            "opportunities":n,
            "flag_rate_per_opportunity":None if n==0 else f/float(n),
        }
    return {
        "rows":rows,
        "summary":by_domain[DOMAIN],
        "flag_counts":flags,
        "opportunity_counts":opportunities,
        "opportunity_policy":policy,
        "rates":rates,
        "spin_decisions":spin_decisions,
    }


def run_candidate(args,tasks,mode):
    rows=[]
    ctx=mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers),len(tasks)),
        mp_context=ctx,
        initializer=init_worker,
        initargs=(
            str(args.solver.resolve()),
            str(args.teacher.resolve()),
            str(ROOT.resolve()),
            int(args.seed),
            str(mode),
        ),
    ) as pool:
        for row in pool.map(worker,tasks,chunksize=1):
            rows.append(row)
    return summarize_candidate(rows)


def surface_mean(summary,code):
    row=(summary.get("opportunity_policy") or {}).get(code)
    return None if row is None else row.get("mean_target_probability")


def adjudicate(A,B,paired):
    checks={
        "ev_noninferiority":float(paired["ci95_low"])>=-0.50,
        "zero_aa_fold":int(B["flag_counts"].get("PREFLOP_AA_FOLD",0))==0,
        "zero_monster_fold":int(B["flag_counts"].get("POSTFLOP_MONSTER_FOLD",0))==0,
    }
    ah=(A["opportunity_policy"].get("POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW") or {})
    bh=(B["opportunity_policy"].get("POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW") or {})
    aopp=int(ah.get("opportunities",0))
    bopp=int(bh.get("opportunities",0))
    afrac=0.0 if aopp==0 else int(ah.get("ge_80pct_count",0))/float(aopp)
    bfrac=0.0 if bopp==0 else int(bh.get("ge_80pct_count",0))/float(bopp)
    checks["high_card_no_draw_ge80"]=bfrac<=afrac+1e-15 and bfrac<=0.0025

    a72=surface_mean(A,"PREFLOP_DEEP_72O_JAM")
    b72=surface_mean(B,"PREFLOP_DEEP_72O_JAM")
    checks["deep_72o_mean_allin"]=(
        a72 is not None and b72 is not None and float(b72)<=float(a72)+0.02
    )
    at=surface_mean(A,"POSTFLOP_TRIPS_PLUS_FOLD")
    bt=surface_mean(B,"POSTFLOP_TRIPS_PLUS_FOLD")
    checks["trips_plus_fold_mean"]=(
        at is not None and bt is not None and float(bt)<=float(at)+0.02
    )
    passed=all(checks.values())
    return {
        "status":"PASS" if passed else "FAIL",
        "checks":checks,
        "preferred_aggregation":"B_MEMBER_POLICY_MEAN" if passed else None,
        "production_change_authorized":False,
        "long_training_unblocked":False,
        "next_gate":"FALLBACK_MODEL_SIGN_ERROR_GATE" if passed else "AGGREGATION_RESEARCH_REOPEN",
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--teacher",type=Path,required=True)
    ap.add_argument("--training-report",type=Path,required=True)
    ap.add_argument("--prereg",type=Path,required=True)
    ap.add_argument("--scenarios",type=int,default=10000)
    ap.add_argument("--workers",type=int,default=16)
    ap.add_argument("--seed",type=int,default=2026103017)
    ap.add_argument("--max-decisions",type=int,default=300)
    ap.add_argument("--report",type=Path,required=True)
    args=ap.parse_args()

    prereg=json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("schema")!="SPINCORE_TEACHER_MEMBER_POLICY_MEAN_DIRECT_PLAY_PREREG_V1":
        raise RuntimeError("wrong prereg schema")
    if prereg.get("status")!="FROZEN_BEFORE_RESULTS":
        raise RuntimeError("prereg is not frozen")
    frozen=prereg["design"]
    for key,actual in (
        ("scenarios",int(args.scenarios)),
        ("seed",int(args.seed)),
        ("workers",int(args.workers)),
        ("max_decisions",int(args.max_decisions)),
    ):
        if int(frozen[key])!=actual:
            raise RuntimeError(f"runtime {key} differs from prereg")

    training=json.loads(args.training_report.read_text(encoding="utf-8"))
    if training.get("status")!="PASS" or int(training.get("completed_iteration",-1))!=ITERATION:
        raise RuntimeError("training report is not PASS @17315")
    if training.get("final_ensemble_sha256")!=dcbase._sha256(args.teacher):
        raise RuntimeError("teacher SHA does not match PASS report")

    sampler=LegacyScenarioSampler(
        seed=int(args.seed)^0x5CE0A710,
        config=LegacyScenarioConfig(),
    )
    tasks=[]
    for index in range(int(args.scenarios)):
        episode=sampler.sample_episode(force_domain=DOMAIN)
        tasks.append((
            int(index),
            episode,
            int(dcbase._mix64(args.seed,index,0xD34A1)),
            int(args.max_decisions),
        ))

    print("DIRECT_PLAY_A_BEGIN",flush=True)
    A=run_candidate(args,tasks,"A_CURRENT_RAW_MEAN_RM")
    print("DIRECT_PLAY_A_PASS",flush=True)
    print("DIRECT_PLAY_B_BEGIN",flush=True)
    B=run_candidate(args,tasks,"B_MEMBER_POLICY_MEAN")
    print("DIRECT_PLAY_B_PASS",flush=True)

    amap={int(r["scenario"]):r for r in A["rows"]}
    bmap={int(r["scenario"]):r for r in B["rows"]}
    if set(amap)!=set(bmap) or len(amap)!=int(args.scenarios):
        raise RuntimeError("A/B scenario pairing drift")
    delta=[]
    for i in sorted(amap):
        av=float(amap[i]["summary"]["spincore_chips_per_exposure"])
        bv=float(bmap[i]["summary"]["spincore_chips_per_exposure"])
        delta.append(bv-av)
    paired=mean_ci(delta)
    decision=adjudicate(A,B,paired)

    def compact(x):
        return {k:v for k,v in x.items() if k!="rows"}
    report={
        "schema":SCHEMA,
        "status":"PASS",
        "teacher_iteration":ITERATION,
        "teacher_sha256":dcbase._sha256(args.teacher),
        "prereg_sha256":dcbase._sha256(args.prereg),
        "seed":int(args.seed),
        "scenarios":int(args.scenarios),
        "workers":int(args.workers),
        "candidate_A":compact(A),
        "candidate_B":compact(B),
        "paired_B_minus_A_spincore_chips_per_policy_seat_hand":paired,
        "decision":decision,
        "production_change_authorized":False,
        "long_training_unblocked":False,
        "interpretation":{
            "pairing":"A and B share episode/deal schedules. Their trajectories may diverge after different sampled actions, so the paired EV delta is a common-chance direct-play comparison, not identical-state replay.",
            "scope":"Development research gate against the frozen DeepCrusher opponent. No production promotion or long-training authorization follows automatically."
        },
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== Member-policy-mean direct-play gate ===")
    print("decision="+decision["status"])
    print("checks="+json.dumps(decision["checks"],sort_keys=True))
    print("paired_B_minus_A="+json.dumps(paired,sort_keys=True))
    print("A_flags="+json.dumps(A["flag_counts"],sort_keys=True))
    print("B_flags="+json.dumps(B["flag_counts"],sort_keys=True))
    print("A_high_card_no_draw="+json.dumps(A["opportunity_policy"].get("POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW"),sort_keys=True))
    print("B_high_card_no_draw="+json.dumps(B["opportunity_policy"].get("POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW"),sort_keys=True))
    print("A_72o="+json.dumps(A["opportunity_policy"].get("PREFLOP_DEEP_72O_JAM"),sort_keys=True))
    print("B_72o="+json.dumps(B["opportunity_policy"].get("PREFLOP_DEEP_72O_JAM"),sort_keys=True))
    print(f"report={args.report.resolve()}")
    print("SPINCORE_TEACHER_MEMBER_POLICY_MEAN_DIRECT_PLAY_17315_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
