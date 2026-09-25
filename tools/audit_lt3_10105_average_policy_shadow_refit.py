#!/usr/bin/env python3
from __future__ import annotations

"""Shadow-continue the frozen 10105 THREE_HANDED AveragePolicy fit.

Purpose: distinguish AveragePolicy underfitting from irreducible/nonstationary
strategy-target variation.  The frozen checkpoint is read-only.  A copy of the
final policy + Adam state is trained for at most 4000 additional strategy-only
steps on the current 3H strategy reservoir, excluding a fixed diagnostic
holdout.  Metrics are evaluated at fixed milestones.

This does NOT authorize deployment of the shadow model.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

from spincore.deepcrusher_benchmark import (
    DEEPC_RUSHER_POLICY_ID,
    SPINCORE_POLICY_ID,
    ExternalExactAction,
    OfflineHeadToHeadEngine,
    balanced_three_handed_lineups,
)
from spincore.deepcrusher_policy import DeepCrusherR8Policy
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_hybrid_deployment_agent import _street_from_state
from spincore.lean_solver_actions import lean_legal_actions, resolve_lean_exact
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import legal_mask
from spincore.solver import SolverLibrary
from spincore_nn.action_models import (
    collate_action_observations,
    make_policy_action_model,
)
from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.training import train_step
import audit_lt3_10105_trip_local_geometry as local

EXPECTED_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
DOMAIN="THREE_HANDED"
MASTER_SEED=20260923
SCENARIO_INDEX=86
LINEUP_INDEX=2
TARGET_HOLE=(40,26)
TARGET_BOARD=(24,27,11)
SHADOW_SEED=20260924
MILESTONES=(0,500,1000,2000,4000)


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def mix64(*values:int)->int:
    x=0x9E3779B97F4A7C15
    mask=(1<<64)-1
    for value in values:
        y=int(value)&mask
        x^=(y+0x9E3779B97F4A7C15+((x<<6)&mask)+(x>>2))&mask
        x&=mask
    return x


class CaptureOriginalPolicy:
    policy_id=SPINCORE_POLICY_ID
    def __init__(self,model):
        self.model=model.eval()
        self.target_observation=None
        self.target_legal=None
        self.target_active_mask=None
        self.target_sample_u=None
        self.target_selected_slot=None

    def choose_exact(self,state,*,seat:int,rng:random.Random)->ExternalExactAction:
        street=_street_from_state(state)
        active_mask=FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal=tuple(int(x) for x in lean_legal_actions(state,active_mask))
        obs=state.neural_bytes()
        batch=collate_action_observations(
            REPRESENTATION,[obs],[legal_mask(legal)],device="cpu"
        )
        with torch.no_grad():
            probs=self.model.probabilities(batch)[0].detach().cpu().tolist()

        x=rng.random()
        cumulative=0.0
        slot=int(legal[-1])
        for candidate in legal:
            cumulative+=float(probs[candidate])
            if x<cumulative:
                slot=int(candidate)
                break

        public=state.public_snapshot()
        deal=state.deal_snapshot() if state.owner.explicit_deal_available else None
        if (
            int(seat)==1
            and int(public.street)==1
            and int(public.pot)==120
            and int(public.to_call)==60
            and deal is not None
            and tuple(int(x) for x in deal.holes[1])==TARGET_HOLE
            and tuple(int(x) for x in deal.board[:3])==TARGET_BOARD
        ):
            self.target_observation=bytes(obs)
            self.target_legal=tuple(legal)
            self.target_active_mask=int(active_mask)
            self.target_sample_u=float(x)
            self.target_selected_slot=int(slot)

        action_type,amount_to=resolve_lean_exact(state,active_mask,slot)
        return ExternalExactAction(int(action_type),int(amount_to))


def capture_exact_target_observation(solver_path:Path,model):
    solver=SolverLibrary(solver_path.resolve(strict=True))
    spin=CaptureOriginalPolicy(model)
    deep=DeepCrusherR8Policy.from_repository(ROOT)
    sampler=LegacyScenarioSampler(
        seed=MASTER_SEED^0x5CE0A710,
        config=LegacyScenarioConfig(),
    )
    episode=None
    for _ in range(SCENARIO_INDEX+1):
        episode=sampler.sample_episode()
    assert episode is not None
    traces=[]
    engine=OfflineHeadToHeadEngine(
        solver,
        spincore_policy=spin,
        deepcrusher_policy=deep,
        master_seed=MASTER_SEED,
        max_decisions=200,
        decision_sink=traces.append,
    )
    lineup=balanced_three_handed_lineups()[LINEUP_INDEX]
    engine.play_hand(
        episode,
        deal_seed=mix64(MASTER_SEED,SCENARIO_INDEX,0xD34A1),
        scenario_index=SCENARIO_INDEX,
        lineup=lineup,
        lineup_index=LINEUP_INDEX,
    )
    if spin.target_observation is None:
        raise RuntimeError("failed to reproduce exact Q8/884 target state")
    if abs(float(spin.target_sample_u)-0.052933228485035455)>1e-15:
        raise RuntimeError("exact target replay RNG drift")
    if int(spin.target_selected_slot)!=0:
        raise RuntimeError("exact target replay no longer selects FOLD")
    return spin.target_observation,spin.target_legal


def subset_indices(items):
    groups={name:[] for name in (
        "A_flop_paired_board_trips_facing_action",
        "B_three_seat_topology_one_opponent_folded",
        "C_halfpot_price",
        "D_halfpot_stack_5_10bb",
        "E_near_target_pot_geometry",
        "F_target_dealer_rel",
    )}
    target_status=(0,1,0)
    def top(f):
        return f["topology_live_count"]==3 and f["statuses"]==target_status
    for idx,s in enumerate(items):
        f=local.features(s)
        if f is None or not f["fold_legal"]:
            continue
        groups["A_flop_paired_board_trips_facing_action"].append(idx)
        if not top(f): continue
        groups["B_three_seat_topology_one_opponent_folded"].append(idx)
        if f["to_call_over_pot"] is None or not (0.45<=f["to_call_over_pot"]<=0.55): continue
        groups["C_halfpot_price"].append(idx)
        if not (5.0<=f["hero_stack_bb"]<=10.0): continue
        groups["D_halfpot_stack_5_10bb"].append(idx)
        if not (3.5<=f["pot_bb"]<=4.5 and 1.75<=f["to_call_bb"]<=2.25 and 1.75<=f["current_bet_bb"]<=2.25): continue
        groups["E_near_target_pot_geometry"].append(idx)
        if f["dealer_rel"]!=1: continue
        groups["F_target_dealer_rel"].append(idx)
    return groups


def eval_samples(model,samples,batch_size=4096):
    if not samples:
        return {"n":0}
    ces=[]; tvs=[]; mismatch=[]; fold_target=[]; fold_pred=[]; weights=[]
    model.eval()
    for start in range(0,len(samples),batch_size):
        chunk=samples[start:start+batch_size]
        batch,target_t,weight_t=vectorized_batch(chunk,"cpu")
        with torch.no_grad():
            logits=model(batch).masked_fill(~batch["legal"],-1e9)
            logp=torch.log_softmax(logits,dim=-1).detach().cpu()
            pred=torch.softmax(logits,dim=-1).detach().cpu()
        target=target_t.detach().cpu()
        legal=batch["legal"].detach().cpu()
        for i,s in enumerate(chunk):
            la=[j for j,v in enumerate(legal[i].tolist()) if bool(v)]
            tgt=[float(x) for x in target[i].tolist()]
            pr=[float(x) for x in pred[i].tolist()]
            ce=-sum(tgt[a]*float(logp[i,a]) for a in la)
            tv=0.5*sum(abs(tgt[a]-pr[a]) for a in la)
            ces.append(ce);tvs.append(tv)
            mismatch.append(float(max(la,key=lambda a:tgt[a])!=max(la,key=lambda a:pr[a])))
            weights.append(float(s.weight))
            if 0 in la:
                fold_target.append(tgt[0]);fold_pred.append(pr[0])
    sw=sum(weights)
    return {
        "n":len(samples),
        "cross_entropy_mean":statistics.fmean(ces),
        "cross_entropy_iteration_weighted_mean":sum(c*w for c,w in zip(ces,weights))/sw,
        "tv_mean":statistics.fmean(tvs),
        "tv_median":statistics.median(tvs),
        "argmax_mismatch_rate":statistics.fmean(mismatch),
        "fold_target_mean":statistics.fmean(fold_target) if fold_target else None,
        "fold_policy_mean":statistics.fmean(fold_pred) if fold_pred else None,
        "fold_mae":statistics.fmean(abs(a-b) for a,b in zip(fold_target,fold_pred)) if fold_target else None,
    }


def exact_probs(model,obs,legal):
    batch=collate_action_observations(
        REPRESENTATION,[obs],[legal_mask(legal)],device="cpu"
    )
    with torch.no_grad():
        p=model.probabilities(batch)[0].detach().cpu().tolist()
    return {str(a):float(p[a]) for a in legal}


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--holdout-size",type=int,default=50000)
    args=ap.parse_args()

    cp=args.checkpoint.resolve(strict=True)
    actual=sha256(cp)
    if actual!=EXPECTED_SHA:
        raise SystemExit(f"checkpoint SHA mismatch: {actual}")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    d3=(payload.get("domains") or {}).get(DOMAIN) or {}
    cfg=dict(payload.get("config") or {})
    mem=d3.get("pol_mem") or {}
    items=list(mem.get("items") or [])
    if len(items)!=int(mem.get("capacity",len(items))):
        raise SystemExit("strategy reservoir must be saturated for this audit")

    _,original=make_policy_action_model(REPRESENTATION,device="cpu",seed=0)
    original.load_state_dict(d3["policy"])
    target_obs,target_legal=capture_exact_target_observation(args.solver,original)

    groups=subset_indices(items)
    protected=set()
    for indices in groups.values():
        protected.update(indices)

    rng=random.Random(SHADOW_SEED)
    candidates=[i for i in range(len(items)) if i not in protected]
    holdout_n=min(int(args.holdout_size),len(candidates))
    holdout_idx=set(rng.sample(candidates,holdout_n))
    protected.update(holdout_idx)
    train_idx=[i for i in range(len(items)) if i not in protected]
    if len(train_idx)<1024:
        raise SystemExit("insufficient shadow train pool")

    holdout=[items[i] for i in sorted(holdout_idx)]
    local_samples={name:[items[i] for i in indices] for name,indices in groups.items()}

    _,model=make_policy_action_model(REPRESENTATION,device="cpu",seed=0)
    model.load_state_dict(d3["policy"])
    opt=torch.optim.Adam(model.parameters(),lr=float(cfg["learning_rate"]))
    opt.load_state_dict(copy.deepcopy(d3["pol_opt"]))

    train_rng=random.Random(SHADOW_SEED^0x5A17)
    milestones=[]
    completed=0

    def snapshot(step):
        row={
            "extra_steps":int(step),
            "global_holdout":eval_samples(model,holdout),
            "exact_q8_884_policy":exact_probs(model,target_obs,target_legal),
            "trip_local":{name:eval_samples(model,samples) for name,samples in local_samples.items()},
        }
        milestones.append(row)
        print("SHADOW_MILESTONE "+json.dumps({
            "extra_steps":row["extra_steps"],
            "holdout_ce":row["global_holdout"].get("cross_entropy_iteration_weighted_mean"),
            "holdout_tv":row["global_holdout"].get("tv_mean"),
            "holdout_argmax_mismatch":row["global_holdout"].get("argmax_mismatch_rate"),
            "q8_fold":row["exact_q8_884_policy"].get("0"),
        },sort_keys=True),flush=True)

    snapshot(0)
    batch_size=int(cfg["batch_size"])
    for target_step in MILESTONES[1:]:
        while completed<int(target_step):
            indices=train_rng.sample(train_idx,min(batch_size,len(train_idx)))
            samples=[items[i] for i in indices]
            batch,target,weights=vectorized_batch(samples,"cpu")
            train_step(model,opt,batch,target,weights,"strategy")
            completed+=1
        snapshot(completed)

    base=milestones[0]["global_holdout"]
    final=milestones[-1]["global_holdout"]
    report={
        "schema":"SPINCORE_LT3_10105_AVERAGE_POLICY_SHADOW_REFIT_V1",
        "checkpoint_sha256":actual,
        "scope":"DIAGNOSTIC_SHADOW_ONLY_NO_CHECKPOINT_MUTATION_NO_DEPLOYMENT",
        "config":cfg,
        "checkpoint_policy_optimizer_steps":int((d3.get("counters") or {}).get("policy_optimizer_steps",0)),
        "strategy_reservoir_seen":int(mem.get("seen",0)),
        "strategy_reservoir_retained":len(items),
        "holdout_size":len(holdout),
        "protected_trip_samples":len(protected)-len(holdout_idx),
        "shadow_train_pool":len(train_idx),
        "milestones":milestones,
        "final_minus_base":{
            "weighted_cross_entropy":(
                final["cross_entropy_iteration_weighted_mean"]-base["cross_entropy_iteration_weighted_mean"]
            ),
            "tv_mean":final["tv_mean"]-base["tv_mean"],
            "argmax_mismatch_rate":final["argmax_mismatch_rate"]-base["argmax_mismatch_rate"],
            "exact_q8_fold":(
                milestones[-1]["exact_q8_884_policy"]["0"]-milestones[0]["exact_q8_884_policy"]["0"]
            ),
        },
        "interpretation":(
            "A material improvement on the fixed holdout after strategy-only shadow steps is direct evidence "
            "that the finalized AveragePolicy was not at its fit optimum on the current strategy reservoir. "
            "Little/no holdout improvement with persistent target mismatch points instead toward target "
            "nonstationarity/conflict, model capacity or representation/generalization limits. The shadow "
            "model is never a deployment candidate from this audit alone."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("report="+str(args.report.resolve()))
    print("LT3_10105_AVERAGE_POLICY_SHADOW_REFIT_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
