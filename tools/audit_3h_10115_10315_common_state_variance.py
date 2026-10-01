#!/usr/bin/env python3
from __future__ import annotations

"""Independent common-state attribution of 10115 vs 10315 policy drift.

Uses a fresh diagnostic seed and a frozen 10105 rollout policy to create a
common THREE_HANDED state bank. It does not use DC1 seed 20261001 and does not
train/select a model. On the same visited states it compares:
- deployed 10115 vs 10315 policy probabilities;
- 10115 vs 10315 Advantage-teacher sigma;
for 72o preflop and postflop trips-or-better/Fold-legal surfaces.
"""

import argparse, json, random, statistics, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python")); sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_semantic_stratified_specialist_moe_10115 as oldspec
import build_3h_semantic_stratified_specialist_10315 as newspec
from audit_3h_average_policy_semantic_continuation_10105 import V1SemanticPolicyNet,semantic_vector_from_obs
from audit_3h_semantic_sidecar_attribution_10105 import decode_obs,private_semantics,rank,suit
from evaluate_deepcrusher_dc1_semantic_candidate_10315 import StrongSpecialistMoEPolicyNet
from spincore.legacy_scenario import LegacyScenarioConfig,LegacyScenarioSampler
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import apply_lean,lean_legal_actions
from spincore.r7_5_action_cfr import legal_mask,sample_action
from spincore_nn.action_models import collate_action_observations
from spincore.solver import SolverLibrary
from spincore.lean_hybrid_deployment_agent import LeanHybridDeploymentAgent

DOMAIN="THREE_HANDED"; FOLD=0; ALL_IN=9
SEED=20261007

def mix64(*values):
    x=0x9E3779B97F4A7C15; mask=(1<<64)-1
    for value in values:
        y=int(value)&mask
        x^=(y+0x9E3779B97F4A7C15+((x<<6)&mask)+(x>>2))&mask
        x&=mask
    return x

def load_candidate(path):
    p=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!="SPINCORE_3H_SEMANTIC_STRATIFIED_STRONG_SPECIALIST_MOE_V1":
        raise RuntimeError("wrong specialist schema")
    b=V1SemanticPolicyNet(); b.load_state_dict(p["base_model_state"]); b.eval()
    s=V1SemanticPolicyNet(); s.load_state_dict(p["specialist_model_state"]); s.eval()
    return StrongSpecialistMoEPolicyNet(b,s).eval(),p

def candidate_probs(model,obs,legal):
    b=collate_action_observations("C0_V1_FROZEN_CONTROL",[obs],[legal_mask(legal)],device="cpu")
    sb=dict(b); sb["semantic"]=torch.tensor(np.asarray([semantic_vector_from_obs(obs)],dtype=np.float32))
    with torch.no_grad(): return [float(x) for x in model.probabilities(sb)[0].cpu().tolist()]

def is_72o(hole):
    return len(hole)==2 and {rank(hole[0]),rank(hole[1])}=={7,2} and suit(hole[0])!=suit(hole[1])

def q(vals,p):
    if not vals:return None
    return float(np.quantile(np.asarray(vals,dtype=np.float64),p))

def stats(rows,key):
    vals=[float(r[key]) for r in rows]
    if not vals:return {"count":0}
    return {"count":len(vals),"mean":statistics.fmean(vals),"median":statistics.median(vals),
            "p90":q(vals,.90),"p95":q(vals,.95),"max":max(vals)}

def summarize(rows,action):
    if not rows:return {"count":0}
    keys=["old_policy","new_policy","old_teacher","new_teacher",
          "teacher_delta","policy_delta","old_distill_gap","new_distill_gap"]
    out={k:stats(rows,k) for k in keys}
    out["count"]=len(rows)
    out["new_policy_gt_old_by_10pp"]=sum(r["policy_delta"]>=.10 for r in rows)
    out["new_teacher_gt_old_by_10pp"]=sum(r["teacher_delta"]>=.10 for r in rows)
    out["new_policy_ge_50pct"]=sum(r["new_policy"]>=.50 for r in rows)
    out["action"]=action
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--base-bundle",type=Path,required=True)
    ap.add_argument("--old-specialist",type=Path,required=True)
    ap.add_argument("--new-specialist",type=Path,required=True)
    ap.add_argument("--old-advantage",type=Path,required=True)
    ap.add_argument("--new-advantage",type=Path,required=True)
    ap.add_argument("--episodes",type=int,default=20000)
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--report",type=Path,required=True)
    a=ap.parse_args(); torch.set_num_threads(int(a.threads))

    solver=SolverLibrary(a.solver.resolve(strict=True))
    rollout=LeanHybridDeploymentAgent.from_bundle(a.base_bundle.resolve(strict=True),device="cpu",seed=0)
    oldm,oldp=load_candidate(a.old_specialist); newm,newp=load_candidate(a.new_specialist)
    oldadv=oldspec.load_adv(a.old_advantage); newadv=newspec.load_adv(a.new_advantage)
    if int(oldp.get("semantic_completed_iteration",-1))!=10115: raise RuntimeError("old iteration drift")
    if int(newp.get("semantic_completed_iteration",-1))!=10315: raise RuntimeError("new iteration drift")

    sampler=LegacyScenarioSampler(seed=SEED,config=LegacyScenarioConfig())
    rng=random.Random(SEED^0xAC710)
    weak=[]; strong=[]; decisions=0

    for ep_idx in range(int(a.episodes)):
        ep=sampler.sample_episode(force_domain=DOMAIN)
        state=solver.create(ep,int(mix64(SEED,ep_idx,0xDEC4)))
        try:
            while not state.terminal:
                street=int(state.neural_bytes_v2()[112])
                active=FIRST_RELEASE_ACTION_SPEC.active_mask(street)
                legal=tuple(int(x) for x in lean_legal_actions(state,active))
                obs=state.neural_bytes(); hole,board,_n,_c=decode_obs(obs)
                sem=private_semantics(hole,board)
                target=None; slot=None
                if street==0 and ALL_IN in legal and is_72o(hole):
                    target=weak; slot=ALL_IN
                elif street>0 and FOLD in legal and int(sem["made"])>=3:
                    target=strong; slot=FOLD
                if target is not None:
                    op=candidate_probs(oldm,obs,legal)[slot]
                    np_=candidate_probs(newm,obs,legal)[slot]
                    ot=float(distill.semantic_sigma(oldadv,obs,legal)[slot])
                    nt=float(distill.semantic_sigma(newadv,obs,legal)[slot])
                    target.append({
                        "old_policy":op,"new_policy":np_,
                        "old_teacher":ot,"new_teacher":nt,
                        "teacher_delta":nt-ot,"policy_delta":np_-op,
                        "old_distill_gap":op-ot,"new_distill_gap":np_-nt,
                        "street":street,"made":int(sem["made"]),
                    })
                _mask,rlegal,rprobs=rollout.distribution(state)
                if tuple(rlegal)!=legal: raise RuntimeError("rollout legal drift")
                action=sample_action(rprobs,legal,rng)
                apply_lean(state,active,action)
                decisions+=1
        finally:
            state.close()
        if (ep_idx+1)%2000==0:
            print(f"VARIANCE_DIAG_EPISODES {ep_idx+1}/{a.episodes} weak={len(weak)} strong={len(strong)}",flush=True)

    result={
        "schema":"SPINCORE_10115_10315_COMMON_STATE_VARIANCE_ATTRIBUTION_V1",
        "scope":"DIAGNOSTIC_ONLY_FRESH_SEED_NO_TRAINING_NO_SELECTION",
        "seed":SEED,"episodes":int(a.episodes),"decisions":decisions,
        "state_generator":"frozen 10105 hybrid policy on forced THREE_HANDED episodes",
        "scientific_dc1_seed_20261001_used":False,
        "preflop_72o_allin":summarize(weak,"ALL_IN"),
        "postflop_trips_plus_fold":summarize(strong,"FOLD"),
        "interpretation_contract":{
            "teacher_moves_policy_moves":"broad drift originates mainly in the changed Advantage teacher / online training estimate",
            "teacher_stable_policy_moves":"broad drift originates mainly in distillation/generalization of the rebuilt deployable policy",
            "only_sparse_tail_moves":"rare-state finite-sample instability is more plausible than broad strategic drift"
        }
    }
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("VARIANCE_DIAG_72O "+json.dumps(result["preflop_72o_allin"],sort_keys=True),flush=True)
    print("VARIANCE_DIAG_STRONG "+json.dumps(result["postflop_trips_plus_fold"],sort_keys=True),flush=True)
    print(f"report={a.report.resolve()}",flush=True)
    print("SPINCORE_10115_10315_COMMON_STATE_VARIANCE_ATTRIBUTION_COMPLETE",flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())
