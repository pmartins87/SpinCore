#!/usr/bin/env python3
from __future__ import annotations

"""Calibrate a confidence-gated strong-hand specialist MoE.

The pure strong specialist improved the rare strong-hand surface substantially,
but it over-corrected a small set of legitimate high-Fold teacher states and
missed the frozen strong-CE target.  This experiment does not retrain either
expert.

Routing contract on the rare strong/Fold-legal stratum:
- compute the validated full-pool base policy first;
- if base Fold probability <= calibrated threshold, use the learned specialist;
- otherwise keep the full-pool base policy.

This is a model-confidence router, not an action override.  Both experts still
produce full learned action distributions.

Threshold selection uses a fresh calibration stream. Final validation uses a
separate fresh stream that is untouched by training and calibration.
"""

import argparse, json, statistics, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_fresh_semantic_strategy_independent500_10105 as confirm
import audit_3h_semantic_tail_strong_hand_coverage_10115 as coverage
from audit_3h_average_policy_semantic_continuation_10105 import (
    V1SemanticPolicyNet,
    semantic_vector_from_obs,
)
from spincore.solver import SolverLibrary

EXPECTED_SHA=distill.EXPECTED_SHA
ADV_SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_ENSEMBLE_V1"
PURE_SCHEMA="SPINCORE_3H_SEMANTIC_STRONG_SPECIALIST_MOE_V1"
OUT_SCHEMA="SPINCORE_3H_SEMANTIC_CONFIDENCE_GATED_STRONG_MOE_V1"
FINAL_ITERATION=10115
DOMAIN="THREE_HANDED"
CAL_EPISODES=80000
CAL_SEED=20260928 ^ 0xCA11B4
EVAL_EPISODES=120000
EVAL_SEED=20260928 ^ 0xEFA115
THRESHOLDS=tuple(round(x,2) for x in np.arange(0.10,0.91,0.05))
FOLD=0

_D3=None
_CFG=None


class ConfidenceGatedMoE(torch.nn.Module):
    def __init__(self,base,specialist,threshold):
        super().__init__()
        self.base_model=base.eval()
        self.specialist_model=specialist.eval()
        self.threshold=float(threshold)

    def probabilities(self,batch):
        base=self.base_model.probabilities(batch)
        spec=self.specialist_model.probabilities(batch)
        semantic=batch["semantic"]
        strong=semantic[:,3:9].sum(dim=1)>0.5
        fold_legal=batch["legal"][:,FOLD]
        route=(strong & fold_legal & (base[:,FOLD] <= self.threshold)).unsqueeze(1)
        return torch.where(route,spec,base)

    def forward(self,batch):
        return torch.log(self.probabilities(batch).clamp_min(1e-30))


def load_adv(path):
    p=torch.load(Path(path).resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=ADV_SCHEMA or p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError("canonical Advantage artifact mismatch")
    if int(p.get("completed_iteration",-1))!=FINAL_ITERATION:
        raise RuntimeError("canonical Advantage iteration mismatch")
    states=list(p.get("members") or [])
    if len(states)!=8:
        raise RuntimeError("expected eight Advantage members")
    return [distill.load_semantic_advantage(s) for s in states]


def load_pure(path):
    p=torch.load(Path(path).resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=PURE_SCHEMA or p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError("pure specialist artifact mismatch")
    if int(p.get("semantic_completed_iteration",-1))!=FINAL_ITERATION:
        raise RuntimeError("pure specialist iteration mismatch")
    base=V1SemanticPolicyNet(); base.load_state_dict(p["base_model_state"]); base.eval()
    spec=V1SemanticPolicyNet(); spec.load_state_dict(p["specialist_model_state"]); spec.eval()
    return base,spec,p


def strong(sample):
    m=coverage.sample_meta(sample)
    return int(m["street"])>0 and int(m["made"])>=3 and bool(m["fold_legal"])


def key(sample):
    return (bytes(sample.observation),tuple(bool(x) for x in sample.legal))


def unique_strong(samples):
    out=[]; seen=set()
    for s in samples:
        if not strong(s):
            continue
        k=key(s)
        if k in seen:
            continue
        seen.add(k); out.append(s)
    return out


def dummy_v1():
    v1,_,_,_=confirm.initial_policy_pair(_D3,_CFG)
    return v1


def fold_rows(model,samples):
    return coverage.strong_rows(coverage.predict_rows(dummy_v1(),model,samples))


def subset_summary(rows,pred):
    return coverage.subset_summary([r for r in rows if pred(r)])


def low_target_tail(rows):
    rr=[r for r in rows if float(r["target_fold"])<=0.05]
    if not rr:
        return {"count":0}
    vals=[float(r["semantic_tail_fold"]) for r in rr]
    return {
        "count":len(rr),
        "pred_fold_mean":statistics.fmean(vals),
        "pred_fold_median":statistics.median(vals),
        "pred_fold_p90":float(np.quantile(np.asarray(vals),0.90)),
        "pred_fold_p95":float(np.quantile(np.asarray(vals),0.95)),
        "pred_fold_max":max(vals),
    }


def high_target(rows):
    rr=[r for r in rows if float(r["target_fold"])>=0.50]
    if not rr:
        return {"count":0,"abs_bias":None}
    t=statistics.fmean(float(r["target_fold"]) for r in rr)
    p=statistics.fmean(float(r["semantic_tail_fold"]) for r in rr)
    return {
        "count":len(rr),
        "target_mean":t,
        "pred_mean":p,
        "abs_bias":abs(p-t),
    }


def route_fraction(model,samples):
    routed=0
    for start in range(0,len(samples),4096):
        chunk=samples[start:start+4096]
        b,_,_=__import__("spincore_nn.lean_batch",fromlist=["vectorized_batch"]).vectorized_batch(chunk,"cpu")
        sb=dict(b)
        sb["semantic"]=torch.tensor(
            np.asarray([semantic_vector_from_obs(s.observation) for s in chunk],dtype=np.float32),
            dtype=torch.float32,
        )
        with torch.no_grad():
            base=model.base_model.probabilities(sb)
            semantic=sb["semantic"]
            mask=(semantic[:,3:9].sum(dim=1)>0.5) & sb["legal"][:,FOLD] & (base[:,FOLD]<=model.threshold)
        routed+=int(mask.sum().item())
    return routed/len(samples) if samples else 0.0


def eval_pack(model,samples):
    rows=fold_rows(model,samples)
    metrics=distill.metrics(dummy_v1(),model,samples)
    return {
        "weighted_ce":float(metrics["semantic_weighted_ce"]),
        "weighted_tv":float(metrics["semantic_weighted_tv"]),
        "fold_overall":coverage.subset_summary(rows),
        "fold_straight_flush":subset_summary(rows,lambda r:int(r["made"]) in (4,5)),
        "fold_river":subset_summary(rows,lambda r:int(r["street"])==3),
        "low_target_fold_tail":low_target_tail(rows),
        "high_target_fold":high_target(rows),
        "route_fraction":route_fraction(model,samples) if isinstance(model,ConfidenceGatedMoE) else None,
    }


def calibration_feasible(base,cand):
    return all([
        cand["weighted_ce"] <= base["weighted_ce"],
        cand["fold_overall"]["semantic_tail_fold_abs_bias"] <= 0.80*base["fold_overall"]["semantic_tail_fold_abs_bias"],
        cand["low_target_fold_tail"]["pred_fold_mean"] <= 0.80*base["low_target_fold_tail"]["pred_fold_mean"],
        cand["low_target_fold_tail"]["pred_fold_p95"] <= 0.75*base["low_target_fold_tail"]["pred_fold_p95"],
        cand["high_target_fold"]["abs_bias"] <= base["high_target_fold"]["abs_bias"] + 0.015,
    ])


def ratio(a,b):
    return None if b is None or float(b)==0 else float(a)/float(b)


def main():
    global _D3,_CFG
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage",type=Path,required=True)
    ap.add_argument("--pure-moe",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--out-model",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    args=ap.parse_args()
    torch.set_num_threads(int(args.threads))

    cp=args.checkpoint.resolve(strict=True)
    if distill.sha256(cp)!=EXPECTED_SHA:
        raise RuntimeError("checkpoint SHA mismatch")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    _D3=(payload.get("domains") or {}).get(DOMAIN) or {}
    _CFG=dict(payload.get("config") or {})

    adv=load_adv(args.semantic_advantage)
    base,spec,pure_payload=load_pure(args.pure_moe)
    solver=SolverLibrary(args.solver.resolve(strict=True))

    distill.MASTER_SEED=CAL_SEED
    ca,cb,cal_collection=distill.collect_fresh(solver,adv,CAL_EPISODES)
    cal=unique_strong(list(ca)+list(cb))
    if len(cal)<400:
        raise RuntimeError(f"calibration strong sample too small: {len(cal)}")

    base_cal=eval_pack(base,cal)
    threshold_rows=[]
    feasible=[]
    for tau in THRESHOLDS:
        model=ConfidenceGatedMoE(base,spec,tau).eval()
        row={"fold_threshold":tau,**eval_pack(model,cal)}
        row["calibration_feasible"]=calibration_feasible(base_cal,row)
        threshold_rows.append(row)
        if row["calibration_feasible"]:
            feasible.append(row)
        print("CONFIDENCE_GATE_CAL "+json.dumps({
            "tau":tau,
            "ce":row["weighted_ce"],
            "tv":row["weighted_tv"],
            "fold_bias":row["fold_overall"]["semantic_tail_fold_abs_bias"],
            "low_mean":row["low_target_fold_tail"]["pred_fold_mean"],
            "low_p95":row["low_target_fold_tail"]["pred_fold_p95"],
            "high_bias":row["high_target_fold"]["abs_bias"],
            "route_fraction":row["route_fraction"],
            "feasible":row["calibration_feasible"],
        },sort_keys=True),flush=True)

    if not feasible:
        raise RuntimeError("no confidence threshold satisfies calibration constraints")
    selected=min(feasible,key=lambda r:(r["weighted_ce"],r["weighted_tv"],r["fold_threshold"]))
    tau=float(selected["fold_threshold"])
    gated=ConfidenceGatedMoE(base,spec,tau).eval()

    distill.MASTER_SEED=EVAL_SEED
    ea,eb,eval_collection=distill.collect_fresh(solver,adv,EVAL_EPISODES)
    evaluation=unique_strong(list(ea)+list(eb))
    if len(evaluation)<600:
        raise RuntimeError(f"independent strong eval too small: {len(evaluation)}")

    base_eval=eval_pack(base,evaluation)
    gated_eval=eval_pack(gated,evaluation)

    bh=base_eval["high_target_fold"]["abs_bias"]
    gh=gated_eval["high_target_fold"]["abs_bias"]
    criteria={
        "new_eval_has_at_least_600_unique_strong_states":len(evaluation)>=600,
        "gated_strong_ce_at_least_10pct_better":gated_eval["weighted_ce"]<=0.90*base_eval["weighted_ce"],
        "gated_strong_tv_at_least_10pct_better":gated_eval["weighted_tv"]<=0.90*base_eval["weighted_tv"],
        "gated_fold_abs_bias_at_least_30pct_better":gated_eval["fold_overall"]["semantic_tail_fold_abs_bias"]<=0.70*base_eval["fold_overall"]["semantic_tail_fold_abs_bias"],
        "gated_low_target_fold_mean_at_least_30pct_better":gated_eval["low_target_fold_tail"]["pred_fold_mean"]<=0.70*base_eval["low_target_fold_tail"]["pred_fold_mean"],
        "gated_low_target_fold_p95_at_least_30pct_better":gated_eval["low_target_fold_tail"]["pred_fold_p95"]<=0.70*base_eval["low_target_fold_tail"]["pred_fold_p95"],
        "gated_straight_flush_bias_at_least_30pct_better":gated_eval["fold_straight_flush"]["semantic_tail_fold_abs_bias"]<=0.70*base_eval["fold_straight_flush"]["semantic_tail_fold_abs_bias"],
        "gated_river_bias_at_least_30pct_better":gated_eval["fold_river"]["semantic_tail_fold_abs_bias"]<=0.70*base_eval["fold_river"]["semantic_tail_fold_abs_bias"],
        "legitimate_high_fold_bias_not_worse_by_over_1_5pp":bh is not None and gh is not None and gh<=bh+0.015,
    }
    passed=all(criteria.values())

    args.out_model.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema":OUT_SCHEMA,
        "source_checkpoint_sha256":EXPECTED_SHA,
        "semantic_completed_iteration":FINAL_ITERATION,
        "selected_steps":int(pure_payload["selected_steps"]),
        "fold_threshold":tau,
        "base_model_state":{k:v.detach().cpu() for k,v in base.state_dict().items()},
        "specialist_model_state":{k:v.detach().cpu() for k,v in spec.state_dict().items()},
        "route_contract":"postflop made_category>=TRIPS AND Fold legal AND base_Fold<=threshold -> specialist; otherwise fullpool general",
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    },args.out_model)

    report={
        "schema":"SPINCORE_3H_SEMANTIC_CONFIDENCE_GATED_STRONG_MOE_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_CONFIDENCE_ROUTER_NO_ACTION_HARDCODE",
        "source_checkpoint_sha256":EXPECTED_SHA,
        "pure_specialist_selected_steps":int(pure_payload["selected_steps"]),
        "calibration_collection":cal_collection,
        "calibration_unique_strong_samples":len(cal),
        "calibration_base":base_cal,
        "threshold_grid":threshold_rows,
        "selected_fold_threshold":tau,
        "selected_calibration_row":selected,
        "independent_eval_collection":eval_collection,
        "independent_unique_strong_samples":len(evaluation),
        "fullpool_general":base_eval,
        "confidence_gated_moe":gated_eval,
        "effect_sizes":{
            "ce_ratio":ratio(gated_eval["weighted_ce"],base_eval["weighted_ce"]),
            "tv_ratio":ratio(gated_eval["weighted_tv"],base_eval["weighted_tv"]),
            "fold_bias_ratio":ratio(gated_eval["fold_overall"]["semantic_tail_fold_abs_bias"],base_eval["fold_overall"]["semantic_tail_fold_abs_bias"]),
            "low_target_mean_ratio":ratio(gated_eval["low_target_fold_tail"]["pred_fold_mean"],base_eval["low_target_fold_tail"]["pred_fold_mean"]),
            "low_target_p95_ratio":ratio(gated_eval["low_target_fold_tail"]["pred_fold_p95"],base_eval["low_target_fold_tail"]["pred_fold_p95"]),
            "straight_flush_bias_ratio":ratio(gated_eval["fold_straight_flush"]["semantic_tail_fold_abs_bias"],base_eval["fold_straight_flush"]["semantic_tail_fold_abs_bias"]),
            "river_bias_ratio":ratio(gated_eval["fold_river"]["semantic_tail_fold_abs_bias"],base_eval["fold_river"]["semantic_tail_fold_abs_bias"]),
        },
        "precommitted_criteria":criteria,
        "semantic_confidence_gated_moe_pass":passed,
        "interpretation":"PASS authorizes one exact fixed-seed DC1 1k replay of the confidence-gated MoE. It does not authorize production, DC2 or 5k.",
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("=== confidence-gated MoE audit ===")
    print(f"selected_fold_threshold={tau}")
    print("base="+json.dumps(base_eval,sort_keys=True))
    print("gated="+json.dumps(gated_eval,sort_keys=True))
    print("criteria="+json.dumps(criteria,sort_keys=True))
    print(f"semantic_confidence_gated_moe_pass={passed}")
    print(f"report={args.report.resolve()}")
    print("SEMANTIC_CONFIDENCE_GATED_MOE_AUDIT_COMPLETE")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
