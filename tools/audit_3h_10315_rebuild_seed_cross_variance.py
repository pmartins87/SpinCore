#!/usr/bin/env python3
from __future__ import annotations

"""Compare original and seed-crossed 10315 rebuilds on a fresh common state bank.

PROJECT_CONTRACT_IDS: VALID-025,VALID-030,VALID-031,VALID-032,SAFE-001,PERF-010,PERF-013,RNG-001,RNG-002,RNG-003,ART-001,SRC-003

Both 10315 policies use the same frozen 10315 Advantage teacher and the same
architecture/hyperparameters. They differ only in the distillation/rebuild data
seed family. The accepted 10115 policy is included as a reference. No DC1 seed
or state is reused and this report is diagnostic-only.
"""

import argparse,json,random,statistics,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python")); sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_10115_10315_common_state_variance as common
from audit_3h_semantic_sidecar_attribution_10105 import decode_obs,private_semantics
from spincore.legacy_scenario import LegacyScenarioConfig,LegacyScenarioSampler
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import apply_lean,lean_legal_actions
from spincore.r7_5_action_cfr import sample_action
from spincore.solver import SolverLibrary
from spincore.lean_hybrid_deployment_agent import LeanHybridDeploymentAgent

FOLD=0; ALL_IN=9; DOMAIN="THREE_HANDED"; SEED=20261012

def dist(rows,a,b):
    vals=[abs(float(r[a])-float(r[b])) for r in rows]
    signed=[float(r[a])-float(r[b]) for r in rows]
    if not vals:return {"count":0}
    arr=np.asarray(vals,dtype=np.float64)
    return {
      "count":len(vals),
      "mean_abs_delta":statistics.fmean(vals),
      "median_abs_delta":statistics.median(vals),
      "p90_abs_delta":float(np.quantile(arr,.90)),
      "p95_abs_delta":float(np.quantile(arr,.95)),
      "max_abs_delta":max(vals),
      "mean_signed_delta":statistics.fmean(signed),
      "count_abs_delta_ge_10pp":sum(x>=.10 for x in vals),
    }

def probs(rows,key):
    vals=[float(r[key]) for r in rows]
    if not vals:return {"count":0}
    arr=np.asarray(vals,dtype=np.float64)
    return {"count":len(vals),"mean":statistics.fmean(vals),"median":statistics.median(vals),
            "p90":float(np.quantile(arr,.90)),"p95":float(np.quantile(arr,.95)),
            "max":max(vals),"ge_50pct":sum(x>=.50 for x in vals)}

def summary(rows,action):
    if not rows:return {"count":0,"action":action}
    between=dist(rows,"original_10315","accepted_10115")
    rebuild=dist(rows,"seedcross_10315","original_10315")
    denom=float(between.get("mean_abs_delta") or 0.0)
    return {
      "count":len(rows),"action":action,
      "accepted_10115":probs(rows,"accepted_10115"),
      "original_10315":probs(rows,"original_10315"),
      "seedcross_10315":probs(rows,"seedcross_10315"),
      "original_10315_vs_10115":between,
      "same_teacher_seedcross_vs_original_10315":rebuild,
      "same_teacher_mae_over_original10315_vs10115_mae":
        (float(rebuild["mean_abs_delta"])/denom if denom>0 else None),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--base-bundle",type=Path,required=True)
    ap.add_argument("--accepted-10115",type=Path,required=True)
    ap.add_argument("--original-10315",type=Path,required=True)
    ap.add_argument("--seedcross-10315",type=Path,required=True)
    ap.add_argument("--episodes",type=int,default=20000)
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--report",type=Path,required=True)
    a=ap.parse_args(); torch.set_num_threads(int(a.threads))
    solver=SolverLibrary(a.solver.resolve(strict=True))
    rollout=LeanHybridDeploymentAgent.from_bundle(a.base_bundle.resolve(strict=True),device="cpu",seed=0)
    old,op=common.load_candidate(a.accepted_10115)
    orig,np=common.load_candidate(a.original_10315)
    cross,cp=common.load_candidate(a.seedcross_10315)
    if int(op.get("semantic_completed_iteration",-1))!=10115:raise RuntimeError("10115 iteration drift")
    if int(np.get("semantic_completed_iteration",-1))!=10315 or int(cp.get("semantic_completed_iteration",-1))!=10315:
        raise RuntimeError("10315 iteration drift")

    sampler=LegacyScenarioSampler(seed=SEED,config=LegacyScenarioConfig())
    rng=random.Random(SEED^0xAC710); weak=[]; strong=[]; decisions=0
    for ep_idx in range(int(a.episodes)):
        ep=sampler.sample_episode(force_domain=DOMAIN)
        state=solver.create(ep,int(common.mix64(SEED,ep_idx,0xDEC4)))
        try:
            while not state.terminal:
                street=int(state.neural_bytes_v2()[112])
                active=FIRST_RELEASE_ACTION_SPEC.active_mask(street)
                legal=tuple(int(x) for x in lean_legal_actions(state,active))
                obs=state.neural_bytes(); hole,board,_n,_c=decode_obs(obs)
                sem=private_semantics(hole,board)
                target=None; slot=None
                if street==0 and ALL_IN in legal and common.is_72o(hole):
                    target=weak;slot=ALL_IN
                elif street>0 and FOLD in legal and int(sem["made"])>=3:
                    target=strong;slot=FOLD
                if target is not None:
                    target.append({
                      "accepted_10115":common.candidate_probs(old,obs,legal)[slot],
                      "original_10315":common.candidate_probs(orig,obs,legal)[slot],
                      "seedcross_10315":common.candidate_probs(cross,obs,legal)[slot],
                    })
                _mask,rlegal,rprobs=rollout.distribution(state)
                if tuple(rlegal)!=legal:raise RuntimeError("rollout legal drift")
                apply_lean(state,active,sample_action(rprobs,legal,rng)); decisions+=1
        finally:
            state.close()
        if (ep_idx+1)%2000==0:
            print(f"SEED_CROSS_DIAG_EPISODES {ep_idx+1}/{a.episodes} weak={len(weak)} strong={len(strong)}",flush=True)

    w=summary(weak,"ALL_IN"); s=summary(strong,"FOLD")
    result={
      "schema":"SPINCORE_10315_REBUILD_SEED_CROSS_VARIANCE_V1",
      "scope":"DIAGNOSTIC_ONLY_NO_TRAINING_SELECTION_FROM_EVAL_STATES",
      "seed":SEED,"episodes":int(a.episodes),"decisions":decisions,
      "scientific_dc1_seed_20261001_used":False,
      "state_generator":"frozen 10105 hybrid policy; forced THREE_HANDED",
      "preflop_72o_allin":w,"postflop_trips_plus_fold":s,
      "interpretation_contract":{
        "ratio_ge_0_5":"same-teacher rebuild/data-seed variance is a material contributor to the observed 10115-to-10315 policy shift",
        "ratio_le_0_25":"same-teacher rebuild/data-seed variance is small relative to the observed 10115-to-10315 shift",
        "between":"ratios are descriptive diagnostics only and do not promote or reject a policy"
      }
    }
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("SEED_CROSS_72O "+json.dumps(w,sort_keys=True),flush=True)
    print("SEED_CROSS_STRONG "+json.dumps(s,sort_keys=True),flush=True)
    print(f"report={a.report.resolve()}",flush=True)
    print("SPINCORE_10315_REBUILD_SEED_CROSS_VARIANCE_COMPLETE",flush=True)
    return 0
if __name__=="__main__":raise SystemExit(main())
