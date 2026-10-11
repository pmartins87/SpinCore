#!/usr/bin/env python3
from __future__ import annotations

"""Shadow poker-learning diagnostic: 3H Advantage teacher 10315 vs 12815.

PROJECT_CONTRACT_IDS:
DC-002,SAFE-001,TRAIN-022,VALID-001,VALID-003,VALID-025,VALID-031,VALID-034,RNG-001,RNG-002,RNG-003

This is explicitly NON-DECISION diagnostic evidence. It may not stop, extend,
select, promote, tune, or otherwise alter the active 10315->15315 macroblock.
It evaluates the raw semantic-Advantage teachers directly, avoiding
AveragePolicy/specialist rebuild variance.
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
sys.path.insert(0,str(ROOT/"tools"))

import evaluate_deepcrusher_dc1 as base
import audit_3h_fresh_semantic_strategy_distill_10105 as distill

from spincore.deepcrusher_benchmark import SPINCORE_POLICY_ID,ExternalExactAction,OfflineHeadToHeadEngine
from spincore.deepcrusher_policy import DeepCrusherR8Policy
from spincore.decision_sanity import sanity_flags
from spincore.legacy_scenario import LegacyScenarioConfig,LegacyScenarioSampler
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import lean_legal_actions,resolve_lean_exact
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore.solver import SolverLibrary

ENSEMBLE_SCHEMA="SPINCORE_3H_SEMANTIC_LONG_ENSEMBLE_V1"
DOMAIN="THREE_HANDED"

_SOLVER=None
_POLICY=None
_DC=None
_ENGINE_SEED=None

def street_from_state(state):
    payload=state.neural_bytes_v2()
    if len(payload)!=830 or not payload.startswith(b"SPNNIV2\x00"):
        raise RuntimeError("invalid SPNNIV2 state")
    street=int(payload[112])
    if street not in (0,1,2,3):
        raise RuntimeError("invalid street")
    return street

class SemanticAdvantageTeacherPolicy:
    policy_id=SPINCORE_POLICY_ID
    def __init__(self,models,label):
        self.models=[m.eval() for m in models]
        self.label=str(label)
        self._last={}

    def distribution(self,state):
        street=street_from_state(state)
        active=FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal=tuple(int(x) for x in lean_legal_actions(state,active))
        if not legal:
            raise RuntimeError("nonterminal state has no legal action")
        obs=state.neural_bytes()
        sigma=distill.semantic_sigma(self.models,obs,legal)
        return int(active),legal,tuple(float(x) for x in sigma)

    def choose_exact(self,state,*,seat:int,rng:random.Random):
        active,legal,probs=self.distribution(state)
        x=rng.random(); cumulative=0.0; slot=int(legal[-1])
        for a in legal:
            cumulative+=float(probs[a])
            if x<cumulative:
                slot=int(a); break
        action_type,amount_to=resolve_lean_exact(state,active,slot)
        self._last[int(seat)]={
            "oracle":"SemanticAdvantageTeacherPolicy",
            "teacher":self.label,
            "sample_u":float(x),
            "selected_slot":int(slot),
            "selected_slot_name":str(NAME_BY_SLOT.get(int(slot),f"SLOT_{int(slot)}")),
            "selected_probability":float(probs[slot]),
            "legal_actions":[
                {"slot":int(a),"name":str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")),
                 "probability":float(probs[a])}
                for a in legal
            ],
        }
        return ExternalExactAction(int(action_type),int(amount_to))

    def decision_metadata(self,*,seat:int):
        x=self._last.get(int(seat))
        return None if x is None else dict(x)

def init_worker(solver_path,ensemble_path,expected_iteration,root_path,seed):
    global _SOLVER,_POLICY,_DC,_ENGINE_SEED
    import os
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    os.environ["OPENBLAS_NUM_THREADS"]="1"
    os.environ["SPINCORE_TORCH_THREADS"]="1"
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass

    _SOLVER=SolverLibrary(solver_path)
    if not _SOLVER.explicit_deal_available:
        raise RuntimeError("explicit-deal solver ABI required")
    p=torch.load(ensemble_path,map_location="cpu",weights_only=False)
    if p.get("schema")!=ENSEMBLE_SCHEMA:
        raise RuntimeError(f"wrong ensemble schema: {p.get('schema')!r}")
    if int(p.get("completed_iteration",-1))!=int(expected_iteration):
        raise RuntimeError("teacher iteration mismatch")
    states=list(p.get("members") or [])
    if len(states)!=8:
        raise RuntimeError("expected 8 semantic teacher members")
    models=[distill.load_semantic_advantage(s) for s in states]
    _POLICY=SemanticAdvantageTeacherPolicy(models,str(expected_iteration))
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
    obs=engine.play_balanced_block(
        episode,deal_seed=int(deal_seed),scenario_index=int(index)
    )
    summary=base._scenario_summary(obs)
    summary.update({
        "scenario":int(index),
        "blind":f"{int(episode.small_blind)}/{int(episode.big_blind)}",
        "dealer_id":int(episode.dealer_id),
        "deal_seed":int(deal_seed),
    })
    flags={}
    spin_decisions=0
    for tr in traces:
        if tr.policy_id!=SPINCORE_POLICY_ID:
            continue
        spin_decisions+=1
        for flag in sanity_flags(tr):
            flags[flag.code]=flags.get(flag.code,0)+1
    summary["spin_decisions"]=spin_decisions
    summary["sanity_flags"]=flags
    return summary

def mean_ci(values):
    n=len(values)
    if not n:
        return {"n":0,"mean":None,"sem":None,"ci95_low":None,"ci95_high":None}
    mean=float(statistics.fmean(values))
    sem=0.0 if n==1 else float(statistics.stdev(values)/math.sqrt(n))
    h=1.96*sem
    return {"n":n,"mean":mean,"sem":sem,"ci95_low":mean-h,"ci95_high":mean+h}

def summarize_arm(rows):
    sc=[float(r["spincore_chips_per_exposure"]) for r in rows]
    margin=[float(r["paired_spincore_minus_deepcrusher_chips_per_exposure"]) for r in rows]
    flags={}
    for r in rows:
        for k,v in (r.get("sanity_flags") or {}).items():
            flags[k]=flags.get(k,0)+int(v)
    return {
        "scenario_clusters":len(rows),
        "balanced_games":sum(int(r["games"]) for r in rows),
        "spin_decisions":sum(int(r["spin_decisions"]) for r in rows),
        "spincore_chips_per_policy_seat_hand":mean_ci(sc),
        "paired_spincore_minus_deepcrusher":mean_ci(margin),
        "sanity_flag_counts":dict(sorted(flags.items())),
    }

def run_arm(*,solver,ensemble,iteration,tasks,workers,seed):
    rows=[]
    ctx=mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=ctx,
        initializer=init_worker,
        initargs=(str(solver),str(ensemble),int(iteration),str(ROOT.resolve()),int(seed)),
    ) as pool:
        for row in pool.map(worker,tasks,chunksize=1):
            rows.append(row)
    rows.sort(key=lambda x:int(x["scenario"]))
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--teacher-10315",type=Path,required=True)
    ap.add_argument("--teacher-12815",type=Path,required=True)
    ap.add_argument("--scenarios",type=int,default=1000)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--seed",type=int,default=20261011)
    ap.add_argument("--max-decisions",type=int,default=300)
    ap.add_argument("--report",type=Path,required=True)
    args=ap.parse_args()

    for p in (args.solver,args.teacher_10315,args.teacher_12815,base.DC_SOURCE,base.LIB1,base.LIB2):
        if not p.is_file():
            raise SystemExit(f"missing input: {p}")

    sampler=LegacyScenarioSampler(
        seed=int(args.seed)^0x5CE0A710,
        config=LegacyScenarioConfig(),
    )
    tasks=[]
    for i in range(int(args.scenarios)):
        ep=sampler.sample_episode(force_domain=DOMAIN)
        tasks.append((int(i),ep,int(base._mix64(args.seed,i,0xD34A1)),int(args.max_decisions)))

    old=run_arm(
        solver=args.solver.resolve(),ensemble=args.teacher_10315.resolve(),
        iteration=10315,tasks=tasks,workers=min(args.workers,args.scenarios),seed=args.seed
    )
    new=run_arm(
        solver=args.solver.resolve(),ensemble=args.teacher_12815.resolve(),
        iteration=12815,tasks=tasks,workers=min(args.workers,args.scenarios),seed=args.seed
    )
    if len(old)!=len(new):
        raise RuntimeError("arm scenario-count mismatch")
    for a,b in zip(old,new):
        if (a["scenario"],a["deal_seed"],a["blind"],a["dealer_id"]) != (
            b["scenario"],b["deal_seed"],b["blind"],b["dealer_id"]
        ):
            raise RuntimeError("paired scenario identity drift")

    delta=[
        float(b["spincore_chips_per_exposure"])-float(a["spincore_chips_per_exposure"])
        for a,b in zip(old,new)
    ]
    report={
        "schema":"SPINCORE_3H_TEACHER_SHADOW_LEARNING_10315_12815_V1",
        "scope":"CURIOSITY_DIAGNOSTIC_ONLY_NEVER_FOR_TRAINING_SELECTION_PROMOTION_OR_EARLY_STOP",
        "canonical_quality_claim_authorized":False,
        "dc0_status":"PENDING_REAL_OPENHOLDEM_PARITY",
        "seed":int(args.seed),
        "scenarios":int(args.scenarios),
        "workers":int(min(args.workers,args.scenarios)),
        "domain":"THREE_HANDED_ONLY",
        "pairing":"same empirical scenario, exact deal seed, balanced AAB/ABB lineups, same master RNG seed in both teacher arms",
        "teacher_10315":summarize_arm(old),
        "teacher_12815":summarize_arm(new),
        "paired_12815_minus_10315_spincore_chip_ev":mean_ci(delta),
        "interpretation_contract":{
            "positive_resolved":"CI95 low > 0: fresh shadow evidence that teacher 12815 outperforms teacher 10315 on this diagnostic sample",
            "negative_resolved":"CI95 high < 0: fresh shadow evidence in the opposite direction on this diagnostic sample",
            "otherwise":"INCONCLUSIVE",
            "governance":"Result is permanently excluded from macroblock stop/extend decisions, rebuild selection, promotion thresholds, and final 15315 strength evidence."
        }
    }
    d=report["paired_12815_minus_10315_spincore_chip_ev"]
    if d["ci95_low"] is not None and float(d["ci95_low"])>0:
        report["shadow_learning_status"]="POSITIVE_RESOLVED"
    elif d["ci95_high"] is not None and float(d["ci95_high"])<0:
        report["shadow_learning_status"]="NEGATIVE_RESOLVED"
    else:
        report["shadow_learning_status"]="INCONCLUSIVE"

    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("=== 3H teacher shadow poker-learning diagnostic ===")
    print("teacher_10315="+json.dumps(report["teacher_10315"],sort_keys=True))
    print("teacher_12815="+json.dumps(report["teacher_12815"],sort_keys=True))
    print("paired_12815_minus_10315="+json.dumps(d,sort_keys=True))
    print("shadow_learning_status="+report["shadow_learning_status"])
    print(f"report={args.report.resolve()}")
    print("SPINCORE_3H_TEACHER_SHADOW_LEARNING_10315_12815_COMPLETE")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
