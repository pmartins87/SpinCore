#!/usr/bin/env python3
from __future__ import annotations

"""Deterministic HU preflop-root policy-drift audit for two checkpoints.

For every HU forensic scenario, evaluate the current Advantage-induced policy at
the initial preflop root before sampling any hero action.  This removes the
sampled-action noise from first-divergence counts and measures exactly where
policy probability mass moved.

Read only: no roots, optimizer steps, memory writes, or holdout seeds.
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch

import audit_lt2_hu_behavior_first_divergence as behfd
import audit_lt2_jammer_facing_allin_target_overlay as overlay
import audit_lt2_stage_a_b_first_divergence as fd
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore.solver import SolverLibrary

FORENSIC_SEEDS=(20260920,20260921,20260922,20260923,20260924,20260925)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--stage-a",type=Path,required=True)
    p.add_argument("--stage-b",type=Path,required=True)
    p.add_argument("--scenarios-per-seed",type=int,default=5000)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--report",type=Path,required=True)
    return p.parse_args()


def _mean_ci(vals):
    vals=[float(x) for x in vals]
    n=len(vals)
    mean=float(statistics.fmean(vals)) if vals else float("nan")
    sem=0.0 if n<=1 else float(statistics.stdev(vals)/math.sqrt(n))
    h=1.96*sem
    return {"n":n,"mean":mean,"sem":sem,"ci95_low":mean-h,"ci95_high":mean+h}


def _seed_cluster(rows, getter):
    by={}
    for r in rows:
        by.setdefault(int(r["seed"]),[]).append(float(getter(r)))
    return _mean_ci([statistics.fmean(v) for _,v in sorted(by.items())])


def _bucket_eff_stack(bb):
    x=float(bb)
    if x <= 3: return "<=3bb"
    if x <= 5: return ">3-5bb"
    if x <= 8: return ">5-8bb"
    if x <= 12: return ">8-12bb"
    return ">12bb"


def _aggregate(rows):
    out={
        "n":len(rows),
        "mean_tv_seed_cluster":_seed_cluster(rows,lambda r:r["tv"]),
        "argmax_disagreement_seed_cluster":_seed_cluster(rows,lambda r:1.0 if r["argmax_a"]!=r["argmax_b"] else 0.0),
        "mass_delta_b_minus_a_by_action":{},
        "stage_a_mass_by_action":{},
        "stage_b_mass_by_action":{},
    }
    for slot in range(10):
        name=NAME_BY_SLOT[slot]
        out["mass_delta_b_minus_a_by_action"][name]=_seed_cluster(rows,lambda r,s=slot:r["probs_b"][s]-r["probs_a"][s])
        out["stage_a_mass_by_action"][name]=_seed_cluster(rows,lambda r,s=slot:r["probs_a"][s])
        out["stage_b_mass_by_action"][name]=_seed_cluster(rows,lambda r,s=slot:r["probs_b"][s])

    out["by_blind"]={}
    for blind in sorted({r["blind"] for r in rows}):
        rr=[r for r in rows if r["blind"]==blind]
        out["by_blind"][blind]={
            "n":len(rr),
            "tv":_seed_cluster(rr,lambda r:r["tv"]),
            "all_in_delta":_seed_cluster(rr,lambda r:r["probs_b"][9]-r["probs_a"][9]),
            "pot33_delta":_seed_cluster(rr,lambda r:r["probs_b"][3]-r["probs_a"][3]),
            "check_call_delta":_seed_cluster(rr,lambda r:r["probs_b"][1]-r["probs_a"][1]),
            "fold_delta":_seed_cluster(rr,lambda r:r["probs_b"][0]-r["probs_a"][0]),
        }

    out["by_effective_stack_bucket"]={}
    order=("<=3bb",">3-5bb",">5-8bb",">8-12bb",">12bb")
    for bucket in order:
        rr=[r for r in rows if r["effective_stack_bucket"]==bucket]
        if not rr:
            continue
        out["by_effective_stack_bucket"][bucket]={
            "n":len(rr),
            "tv":_seed_cluster(rr,lambda r:r["tv"]),
            "all_in_delta":_seed_cluster(rr,lambda r:r["probs_b"][9]-r["probs_a"][9]),
            "pot33_delta":_seed_cluster(rr,lambda r:r["probs_b"][3]-r["probs_a"][3]),
            "check_call_delta":_seed_cluster(rr,lambda r:r["probs_b"][1]-r["probs_a"][1]),
            "fold_delta":_seed_cluster(rr,lambda r:r["probs_b"][0]-r["probs_a"][0]),
        }
    return out


def main():
    args=parse_args()
    if args.scenarios_per_seed<=0 or args.threads<=0:
        raise SystemExit("positive scenarios/threads required")
    args.report.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(int(args.threads))

    solver=SolverLibrary(args.solver.resolve(strict=True))
    snap_a=args.report.parent/"stage_a_hu_models.pt"
    snap_b=args.report.parent/"stage_b_hu_models.pt"
    meta_a=overlay._extract_stage_snapshot(args.stage_a.resolve(strict=True),snap_a)
    meta_b=overlay._extract_stage_snapshot(args.stage_b.resolve(strict=True),snap_b)
    stage_a=overlay.StageModels(snap_a,solver)
    stage_b=overlay.StageModels(snap_b,solver)

    rows=[]
    hu_counts={}
    for seed in FORENSIC_SEEDS:
        sampler=LegacyScenarioSampler(seed=int(seed)^0x5CE0A710,config=LegacyScenarioConfig())
        hu=0
        for scenario_index in range(int(args.scenarios_per_seed)):
            episode=sampler.sample_episode()
            if not episode.game_is_hu:
                continue
            hu+=1
            deal_seed=fd._mix64(int(seed),int(scenario_index),0xD34A1)
            state=solver.create(episode,int(deal_seed))
            try:
                actor=int(state.actor)
                ma,la,pa=behfd._behavior_distribution(stage_a,state)
                mb,lb,pb=behfd._behavior_distribution(stage_b,state)
                if ma!=mb or la!=lb:
                    raise RuntimeError("root legal-action drift")
                tv=0.5*sum(abs(float(pa[i])-float(pb[i])) for i in range(10))
                argmax_a=max(la,key=lambda i:(float(pa[i]),-int(i)))
                argmax_b=max(lb,key=lambda i:(float(pb[i]),-int(i)))
                live=[int(x) for x in episode.stacks if int(x)>0]
                eff=min(live)/float(episode.big_blind)
                rows.append({
                    "seed":int(seed),
                    "scenario":int(scenario_index),
                    "blind":f"{episode.small_blind}/{episode.big_blind}",
                    "actor":actor,
                    "effective_stack_bb":float(eff),
                    "effective_stack_bucket":_bucket_eff_stack(eff),
                    "legal":[int(x) for x in la],
                    "probs_a":[float(x) for x in pa],
                    "probs_b":[float(x) for x in pb],
                    "tv":float(tv),
                    "argmax_a":int(argmax_a),
                    "argmax_b":int(argmax_b),
                })
            finally:
                state.close()
        hu_counts[str(seed)]=hu

    summary=_aggregate(rows)
    report={
        "schema":"SPINCORE_LT2_HU_ROOT_POLICY_DRIFT_V1",
        "stage_a":{"checkpoint":str(args.stage_a.resolve()),"completed_iteration":int(meta_a["completed_iteration"])},
        "stage_b":{"checkpoint":str(args.stage_b.resolve()),"completed_iteration":int(meta_b["completed_iteration"])},
        "method":{
            "read_only":True,
            "new_training_roots":0,
            "optimizer_steps":0,
            "training_memory_writes":0,
            "forensic_seeds":list(FORENSIC_SEEDS),
            "future_holdout_seeds_touched":False,
            "scenarios_per_seed":int(args.scenarios_per_seed),
            "hu_scenarios_by_seed":hu_counts,
            "selection":"every HU scenario root before any sampled hero action; no divergence or terminal-outcome selection",
            "action_slots":{str(i):NAME_BY_SLOT[i] for i in range(10)},
        },
        "summary":summary,
    }
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== LT2 HU ROOT POLICY DRIFT ===")
    print(f"roots={summary['n']}")
    tv=summary["mean_tv_seed_cluster"]
    ad=summary["mass_delta_b_minus_a_by_action"]["ALL_IN"]
    p33=summary["mass_delta_b_minus_a_by_action"]["POT_33"]
    cc=summary["mass_delta_b_minus_a_by_action"]["CHECK_CALL"]
    fdlt=summary["mass_delta_b_minus_a_by_action"]["FOLD"]
    arg=summary["argmax_disagreement_seed_cluster"]
    print(f"TV B-vs-A={tv['mean']:.4f} CI95=[{tv['ci95_low']:.4f},{tv['ci95_high']:.4f}]")
    print(f"argmax_disagreement={arg['mean']:.4f}")
    print(f"ALL_IN mass delta={ad['mean']:+.4f} CI95=[{ad['ci95_low']:+.4f},{ad['ci95_high']:+.4f}]")
    print(f"POT_33 mass delta={p33['mean']:+.4f} CI95=[{p33['ci95_low']:+.4f},{p33['ci95_high']:+.4f}]")
    print(f"CHECK_CALL mass delta={cc['mean']:+.4f} CI95=[{cc['ci95_low']:+.4f},{cc['ci95_high']:+.4f}]")
    print(f"FOLD mass delta={fdlt['mean']:+.4f} CI95=[{fdlt['ci95_low']:+.4f},{fdlt['ci95_high']:+.4f}]")
    print("LT2_HU_ROOT_POLICY_DRIFT_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
