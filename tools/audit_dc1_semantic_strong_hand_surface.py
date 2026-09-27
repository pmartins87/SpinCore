#!/usr/bin/env python3
from __future__ import annotations

"""Diagnose the strong-made-hand Fold surface of semantic-10115 on DC1 states.

This replays only the semantic-10115 1k development arm on the exact frozen
scenario/deal seed sequence.  At every THREE_HANDED postflop decision where the
hero has trips-or-better and FOLD is legal, it records three conditional
policies on the same state:

1. finalized 10105 V1 AveragePolicy;
2. semantic 10115 tail AveragePolicy actually used by the benchmark;
3. semantic 10115 Advantage ENS8 regret-matched behavior that generated the
   fresh strategy targets.

The purpose is attribution, not strength measurement: determine whether the
new POSTFLOP_TRIPS_PLUS_FOLD flags come from the online Advantage behavior or
from AveragePolicy distillation/generalization.
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

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import evaluate_deepcrusher_dc1 as base
from evaluate_deepcrusher_dc1_semantic_candidate import (
    EXPECTED_SOURCE_SHA,
    REPRESENTATION,
    SEMANTIC_COMPLETED_ITERATION,
    SEMANTIC_POLICY_SCHEMA,
    SemanticHybridBenchmarkPolicy,
)

from spincore.deepcrusher_benchmark import OfflineHeadToHeadEngine
from spincore.deepcrusher_card_symbols import evaluate_cards
from spincore.deepcrusher_policy import DeepCrusherR8Policy
from spincore.deepcrusher_state import state_view
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.lean_hybrid_deployment_agent import LeanHybridDeploymentAgent
from spincore.r7_5_action_contract import NAME_BY_SLOT

ADV_SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_ENSEMBLE_V1"
CATEGORY_NAMES={
    0:"HIGH_CARD",1:"PAIR",2:"TWO_PAIR",3:"TRIPS",4:"STRAIGHT",
    5:"FLUSH",6:"FULL_HOUSE",7:"QUADS",8:"STRAIGHT_FLUSH",
}
FOLD_SLOT=0

_SOLVER=None
_SPIN=None
_DC=None
_SEED=None


def made_category(state):
    view=state_view(state)
    cards=tuple(
        (int(rank),int(suit))
        for rank,suit in zip(view.ranks,view.openholdem_suits)
        if int(rank)>0
    )
    value=evaluate_cards(cards)
    return int(value.category),tuple(int(x) for x in value.tie),view


class StrongHandAuditPolicy(SemanticHybridBenchmarkPolicy):
    def __init__(self,base_agent,semantic_policy,adv_models):
        super().__init__(base_agent,semantic_policy)
        self.adv_models=list(adv_models)

    def choose_exact(self,state,*,seat:int,rng:random.Random):
        active,legal,probs=self.distribution(state)
        domain=self.domain_for_state(state)

        x=rng.random()
        cumulative=0.0
        slot=int(legal[-1])
        for a in legal:
            cumulative+=float(probs[a])
            if x<cumulative:
                slot=int(a)
                break

        from spincore.lean_solver_actions import resolve_lean_exact
        action_type,amount_to=resolve_lean_exact(state,active,slot)

        detail={
            "oracle":"StrongHandAuditPolicy",
            "selection":"SAMPLED_POLICY",
            "domain":domain,
            "active_mask":int(active),
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

        if domain=="THREE_HANDED":
            category,tie,view=made_category(state)
            if int(view.street)>0 and category>=3 and FOLD_SLOT in legal:
                b_active,b_legal,b_probs=self.base_agent.distribution(state)
                if int(b_active)!=int(active) or tuple(int(a) for a in b_legal)!=tuple(int(a) for a in legal):
                    raise RuntimeError("baseline/semantic legal-action contract drift")
                adv=distill.semantic_sigma(
                    self.adv_models,state.neural_bytes(),tuple(int(a) for a in legal)
                )
                detail["strong_hand_probe"]={
                    "category":CATEGORY_NAMES[category],
                    "category_id":category,
                    "tiebreak":list(tie),
                    "street":int(view.street),
                    "pot_bb":float(view.pot_bb),
                    "to_call_bb":float(view.to_call_bb),
                    "to_call_over_pot":(
                        float(view.to_call_bb/view.pot_bb) if view.pot_bb>0 else None
                    ),
                    "effective_stack_bb":float(view.effective_stack_bb),
                    "baseline_10105_fold":float(b_probs[FOLD_SLOT]),
                    "semantic_tail_10115_fold":float(probs[FOLD_SLOT]),
                    "semantic_advantage_10115_fold":float(adv[FOLD_SLOT]),
                    "tail_minus_advantage_fold":float(
                        probs[FOLD_SLOT]-adv[FOLD_SLOT]
                    ),
                }

        self._last[int(seat)]=detail
        from spincore.deepcrusher_benchmark import ExternalExactAction
        return ExternalExactAction(int(action_type),int(amount_to))


def init_worker(solver_path,bundle_path,semantic_policy_path,adv_path,root_path,seed):
    global _SOLVER,_SPIN,_DC,_SEED
    import os
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    os.environ["SPINCORE_TORCH_THREADS"]="1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    from spincore.solver import SolverLibrary
    from audit_3h_average_policy_semantic_continuation_10105 import V1SemanticPolicyNet

    _SOLVER=SolverLibrary(solver_path)
    if not _SOLVER.explicit_deal_available:
        raise RuntimeError("strong-hand audit requires explicit-deal solver ABI")

    base_agent=LeanHybridDeploymentAgent.from_bundle(bundle_path,device="cpu",seed=0)

    pp=torch.load(semantic_policy_path,map_location="cpu",weights_only=False)
    if pp.get("schema")!=SEMANTIC_POLICY_SCHEMA:
        raise RuntimeError("wrong semantic tail-policy schema")
    if pp.get("source_checkpoint_sha256")!=EXPECTED_SOURCE_SHA:
        raise RuntimeError("semantic tail-policy source mismatch")
    if int(pp.get("completed_iteration",-1))!=SEMANTIC_COMPLETED_ITERATION:
        raise RuntimeError("semantic tail-policy iteration mismatch")
    model=V1SemanticPolicyNet()
    model.load_state_dict(pp["model_state"])
    model.eval()

    ap=torch.load(adv_path,map_location="cpu",weights_only=False)
    if ap.get("schema")!=ADV_SCHEMA:
        raise RuntimeError("wrong semantic Advantage ensemble schema")
    if ap.get("source_checkpoint_sha256")!=EXPECTED_SOURCE_SHA:
        raise RuntimeError("semantic Advantage source mismatch")
    if int(ap.get("completed_iteration",-1))!=SEMANTIC_COMPLETED_ITERATION:
        raise RuntimeError("semantic Advantage iteration mismatch")
    states=list(ap.get("members") or [])
    if len(states)!=8:
        raise RuntimeError("semantic Advantage ensemble-size drift")
    adv_models=[distill.load_semantic_advantage(s) for s in states]

    _SPIN=StrongHandAuditPolicy(base_agent,model,adv_models)
    _DC=DeepCrusherR8Policy.from_repository(root_path)
    _SEED=int(seed)


def worker(task):
    index,episode,deal_seed,max_decisions=task
    traces=[]
    engine=OfflineHeadToHeadEngine(
        _SOLVER,
        spincore_policy=_SPIN,
        deepcrusher_policy=_DC,
        master_seed=int(_SEED),
        max_decisions=int(max_decisions),
        decision_sink=traces.append,
    )
    observations=engine.play_balanced_block(
        episode,deal_seed=int(deal_seed),scenario_index=int(index)
    )
    return {
        "summary":base._scenario_summary(observations),
        "traces":[asdict(t) for t in traces],
    }


def distribution_stats(values):
    if not values:
        return {"count":0}
    vals=[float(x) for x in values]
    return {
        "count":len(vals),
        "mean":statistics.fmean(vals),
        "median":statistics.median(vals),
        "p90":float(np.quantile(np.asarray(vals),0.90)),
        "p95":float(np.quantile(np.asarray(vals),0.95)),
        "max":max(vals),
        "sum_expected_folds":sum(vals),
        "count_ge_0_10":sum(x>=0.10 for x in vals),
        "count_ge_0_25":sum(x>=0.25 for x in vals),
        "count_ge_0_50":sum(x>=0.50 for x in vals),
        "count_ge_0_90":sum(x>=0.90 for x in vals),
    }


def summarize_rows(rows):
    return {
        "baseline_10105_fold":distribution_stats(
            [r["baseline_10105_fold"] for r in rows]
        ),
        "semantic_tail_10115_fold":distribution_stats(
            [r["semantic_tail_10115_fold"] for r in rows]
        ),
        "semantic_advantage_10115_fold":distribution_stats(
            [r["semantic_advantage_10115_fold"] for r in rows]
        ),
        "tail_minus_advantage_fold":distribution_stats(
            [r["tail_minus_advantage_fold"] for r in rows]
        ),
        "actual_sampled_folds":sum(bool(r["actual_fold"]) for r in rows),
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--base-spin-bundle",type=Path,required=True)
    ap.add_argument("--semantic-policy",type=Path,required=True)
    ap.add_argument("--semantic-advantage",type=Path,required=True)
    ap.add_argument("--scenarios",type=int,default=1000)
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
        ep=sampler.sample_episode()
        tasks.append((
            int(index),ep,int(base._mix64(args.seed,index,0xD34A1)),
            int(args.max_decisions),
        ))

    probes=[]
    scenario_rows=[]
    ctx=mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers),int(args.scenarios)),
        mp_context=ctx,
        initializer=init_worker,
        initargs=(
            str(args.solver.resolve()),
            str(args.base_spin_bundle.resolve()),
            str(args.semantic_policy.resolve()),
            str(args.semantic_advantage.resolve()),
            str(ROOT.resolve()),
            int(args.seed),
        ),
    ) as pool:
        for chunk in pool.map(worker,tasks,chunksize=1):
            scenario_rows.append(chunk["summary"])
            for tr in chunk["traces"]:
                if tr["policy_id"]!="SPINCORE" or tr["domain"]!="THREE_HANDED":
                    continue
                detail=dict(tr.get("policy_detail") or {})
                probe=detail.get("strong_hand_probe")
                if not probe:
                    continue
                row=dict(probe)
                row.update({
                    "scenario_index":int(tr["scenario_index"]),
                    "lineup_index":int(tr["lineup_index"]),
                    "decision_index":int(tr["decision_index"]),
                    "actor":int(tr["actor"]),
                    "blind":str(tr["blind"]),
                    "actual_action":str(
                        {0:"FOLD",1:"CHECK",2:"CALL",3:"BET_TO",4:"RAISE_TO",5:"ALL_IN"}
                        [int(tr["action_type"])]
                    ),
                    "actual_fold":int(tr["action_type"])==0,
                })
                probes.append(row)

    actual_folds=sum(bool(r["actual_fold"]) for r in probes)
    if actual_folds!=6:
        raise RuntimeError(
            f"semantic 1k strong-hand trajectory drift: expected 6 sampled folds, got {actual_folds}"
        )

    unique_fold_states={
        (r["scenario_index"],r["decision_index"],r["actor"])
        for r in probes if r["actual_fold"]
    }

    by_category={}
    for category in sorted({r["category"] for r in probes}):
        rows=[r for r in probes if r["category"]==category]
        by_category[category]=summarize_rows(rows)

    by_street={}
    for street in sorted({int(r["street"]) for r in probes}):
        rows=[r for r in probes if int(r["street"])==street]
        by_street[str(street)]=summarize_rows(rows)

    flagged=sorted(
        [r for r in probes if r["actual_fold"]],
        key=lambda r:(
            int(r["scenario_index"]),int(r["lineup_index"]),int(r["decision_index"])
        ),
    )
    top_tail=sorted(
        probes,key=lambda r:float(r["semantic_tail_10115_fold"]),reverse=True
    )[:30]

    report={
        "schema":"SPINCORE_DC1_SEMANTIC_STRONG_HAND_ATTRIBUTION_V1",
        "scope":"DEVELOPMENT_ONLY_ATTRIBUTION_NO_STRENGTH_CLAIM",
        "seed":int(args.seed),
        "scenarios":int(args.scenarios),
        "strong_hand_opportunities":len(probes),
        "actual_sampled_folds":actual_folds,
        "unique_actual_fold_states":len(unique_fold_states),
        "overall":summarize_rows(probes),
        "by_category":by_category,
        "by_street":by_street,
        "actual_fold_rows":flagged,
        "top_30_semantic_tail_fold_probability":top_tail,
        "interpretation_contract":{
            "advantage_high_tail_low":(
                "If semantic Advantage fold mass itself is high on the problem "
                "states, investigate online CFR/Advantage targets."
            ),
            "advantage_low_tail_high":(
                "If semantic Advantage fold mass is low while the semantic tail "
                "AveragePolicy fold mass is high, the regression is a strategy "
                "distillation/generalization problem."
            ),
            "both_low_sampled_flags":(
                "If both policy probabilities are low, excess observed folds may "
                "be sampling noise rather than a broad policy surface."
            ),
        },
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )

    print("=== semantic 10115 strong-hand attribution ===")
    print("overall="+json.dumps(report["overall"],sort_keys=True))
    print(f"actual_sampled_folds={actual_folds}")
    print(f"unique_actual_fold_states={len(unique_fold_states)}")
    print(f"report={args.report.resolve()}")
    print("DC1_SEMANTIC_STRONG_HAND_ATTRIBUTION_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
