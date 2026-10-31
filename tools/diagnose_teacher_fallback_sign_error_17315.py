#!/usr/bin/env python3
from __future__ import annotations

"""Diagnose positive-regret sign loss in the pure teacher @17315.

PROJECT_CONTRACT_IDS:
MODEL-020,MODEL-025,VALID-025,VALID-034,VALID-035,SAFE-001,SAFE-010,RNG-001,RNG-003

Research-only. No training, no policy mutation.
"""

import argparse,json,math,random,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import run_3h_semantic_online_pilot_10105 as pilot
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore_nn.action_models import collate_action_observations
from spincore_nn.reservoir import UniformReservoir
from spincore_nn.semantic_v1 import semantic_vector_from_observation

SCHEMA="SPINCORE_TEACHER_FALLBACK_SIGN_ERROR_17315_V1"


def legal_actions(mask):
    return tuple(i for i,v in enumerate(mask) if bool(v))


def rm(raw,legal):
    return np.asarray(
        lean_regret_matching_policy([float(x) for x in raw],tuple(legal)),
        dtype=np.float64,
    )


def quantile_edges(values,n):
    arr=np.asarray(values,dtype=np.float64)
    return [float(np.quantile(arr,i/n)) for i in range(1,n)]


def bin_index(value,edges):
    return int(np.searchsorted(np.asarray(edges),float(value),side="right"))


def summarize_records(records):
    if not records:
        return {"count":0}
    votes=np.asarray([r["target_best_positive_votes"] for r in records],dtype=np.int64)
    indiv_fb=np.asarray([r["individual_fallback_count"] for r in records],dtype=np.int64)
    target_strength=np.asarray([r["target_strength"] for r in records],dtype=np.float64)
    target_prob=np.asarray([r["target_best_policy_probability"] for r in records],dtype=np.float64)
    mean_pred=np.asarray([r["mean_raw_target_best"] for r in records],dtype=np.float64)
    target=np.asarray([r["target_raw_target_best"] for r in records],dtype=np.float64)
    bprob=np.asarray([r["member_policy_mean_target_best_probability"] for r in records],dtype=np.float64)
    return {
        "count":len(records),
        "target_best_positive_vote_mean":float(votes.mean()),
        "zero_positive_vote_fraction":float(np.mean(votes==0)),
        "one_to_three_positive_vote_fraction":float(np.mean((votes>=1)&(votes<=3))),
        "four_to_seven_positive_vote_fraction":float(np.mean((votes>=4)&(votes<=7))),
        "eight_positive_vote_fraction":float(np.mean(votes==8)),
        "at_least_four_positive_vote_fraction":float(np.mean(votes>=4)),
        "all_members_individual_fallback_fraction":float(np.mean(indiv_fb==8)),
        "mean_individual_fallback_member_fraction":float(indiv_fb.mean()/8.0),
        "any_member_positive_legal_regret_fraction":float(np.mean(indiv_fb<8)),
        "target_strength_mean":float(target_strength.mean()),
        "target_strength_median":float(np.median(target_strength)),
        "target_strength_p10":float(np.quantile(target_strength,0.10)),
        "target_strength_p90":float(np.quantile(target_strength,0.90)),
        "target_best_policy_probability_mean":float(target_prob.mean()),
        "target_best_policy_probability_median":float(np.median(target_prob)),
        "mean_raw_target_best_bias":float(np.mean(mean_pred-target)),
        "mean_raw_target_best_mae":float(np.mean(np.abs(mean_pred-target))),
        "member_policy_mean_target_best_probability_mean":float(bprob.mean()),
        "member_policy_mean_target_best_probability_median":float(np.median(bprob)),
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume",type=Path,required=True)
    ap.add_argument("--ensemble",type=Path,required=True)
    ap.add_argument("--training-report",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--prereg",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--batch-size",type=int,default=512)
    args=ap.parse_args()

    prereg=json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("schema")!="SPINCORE_TEACHER_FALLBACK_SIGN_ERROR_17315_PREREG_V1":
        raise RuntimeError("wrong prereg schema")
    if prereg.get("status")!="FROZEN_BEFORE_RESULTS":
        raise RuntimeError("prereg not frozen")

    training=json.loads(args.training_report.read_text(encoding="utf-8"))
    if training.get("status")!="PASS" or int(training.get("completed_iteration",-1))!=17315:
        raise RuntimeError("training report is not PASS @17315")
    if training.get("final_ensemble_sha256")!=pilot.sha256(args.ensemble):
        raise RuntimeError("ensemble hash drift")
    if training.get("resume_state_sha256")!=pilot.sha256(args.resume):
        raise RuntimeError("resume hash drift")

    ensemble=torch.load(args.ensemble,map_location="cpu",weights_only=False)
    resume=torch.load(args.resume,map_location="cpu",weights_only=False)
    states=list(ensemble.get("members") or [])
    if len(states)!=8:
        raise RuntimeError("expected eight members")
    memory=UniformReservoir.from_state_dict(resume["adv_mem"])
    probe=torch.load(args.probe,map_location="cpu",weights_only=False)
    rng=random.Random(int(probe["holdout_seed"]))
    indices=sorted(rng.sample(range(len(memory.items)),int(probe["holdout_size"])))
    if len(indices)!=50000:
        raise RuntimeError("protected holdout size drift")

    models=[distill.load_semantic_advantage(s) for s in states]
    all_target_positive=[]
    current_fallback_target_positive=[]
    member_sign_correct=[0]*8
    member_primary_count=[0]*8
    by_legal=defaultdict(list)

    with torch.no_grad():
        for start in range(0,len(indices),int(args.batch_size)):
            pos=indices[start:start+int(args.batch_size)]
            samples=[memory.items[i] for i in pos]
            batch=collate_action_observations(
                distill.REPRESENTATION,
                [s.observation for s in samples],
                [s.legal for s in samples],
                device="cpu",
            )
            sem=np.asarray(
                [semantic_vector_from_observation(s.observation) for s in samples],
                dtype=np.float32,
            )
            sb=dict(batch)
            sb["semantic"]=torch.tensor(sem,dtype=torch.float32)
            raw_members=np.stack(
                [m(sb).detach().cpu().numpy() for m in models],axis=0
            )
            for j,sample in enumerate(samples):
                legal=legal_actions(sample.legal)
                target=np.asarray(sample.target,dtype=np.float64)
                legal_target=np.asarray([target[a] for a in legal],dtype=np.float64)
                max_target=float(legal_target.max())
                if max_target<=0:
                    continue
                best_candidates=[a for a in legal if abs(float(target[a])-max_target)<=1e-12]
                best=int(best_candidates[0])
                tp=rm(target,legal)
                mr=raw_members[:,j,:].astype(np.float64,copy=False)
                mean_raw=mr.mean(axis=0)
                current_fallback=max(float(mean_raw[a]) for a in legal)<=0.0
                votes=[float(mr[k,best])>0.0 for k in range(8)]
                indiv_fb=[
                    max(float(mr[k,a]) for a in legal)<=0.0
                    for k in range(8)
                ]
                member_policies=np.stack([rm(mr[k],legal) for k in range(8)],axis=0)
                rec={
                    "target_strength":max_target,
                    "target_best_policy_probability":float(tp[best]),
                    "target_best_positive_votes":int(sum(votes)),
                    "individual_fallback_count":int(sum(indiv_fb)),
                    "mean_raw_target_best":float(mean_raw[best]),
                    "target_raw_target_best":float(target[best]),
                    "member_policy_mean_target_best_probability":float(member_policies[:,best].mean()),
                    "legal_count":len(legal),
                }
                all_target_positive.append(rec)
                if current_fallback:
                    current_fallback_target_positive.append(rec)
                    by_legal[len(legal)].append(rec)
                    for k in range(8):
                        member_primary_count[k]+=1
                        member_sign_correct[k]+=int(votes[k])

    strengths=[r["target_strength"] for r in all_target_positive]
    q_edges=quantile_edges(strengths,4)
    d_edges=quantile_edges(strengths,10)
    quart=Counter()
    dec=Counter()
    for r in current_fallback_target_positive:
        quart[bin_index(r["target_strength"],q_edges)]+=1
        dec[bin_index(r["target_strength"],d_edges)]+=1

    primary=summarize_records(current_fallback_target_positive)
    expected=int(prereg["source"]["current_fallback_target_positive_regret_count"])
    if int(primary["count"])!=expected:
        raise RuntimeError(f"primary population drift: {primary['count']} != {expected}")

    zero=float(primary["zero_positive_vote_fraction"])
    indiv=float(primary["mean_individual_fallback_member_fraction"])
    four=float(primary["at_least_four_positive_vote_fraction"])
    if zero>=0.50 or indiv>=0.75:
        classification="ALL_MEMBERS_LOSE_SIGNAL"
    elif four>=0.50:
        classification="ENSEMBLE_CANCELLATION_DOMINANT"
    else:
        classification="MIXED_MODEL_AND_ENSEMBLE_FAILURE"

    bottom2=(quart.get(0,0)+quart.get(1,0))/float(expected)
    margin_fragile=bottom2>=0.80

    report={
        "schema":SCHEMA,
        "status":"PASS",
        "teacher_iteration":17315,
        "teacher_sha256":pilot.sha256(args.ensemble),
        "prereg_sha256":pilot.sha256(args.prereg),
        "protected_holdout_count":len(indices),
        "all_target_positive":summarize_records(all_target_positive),
        "current_fallback_target_positive":primary,
        "per_member_target_best_positive_sign_recall":[
            0.0 if member_primary_count[k]==0 else member_sign_correct[k]/float(member_primary_count[k])
            for k in range(8)
        ],
        "target_strength_quartile_edges":q_edges,
        "target_strength_decile_edges":d_edges,
        "primary_quartile_counts":{str(k):int(quart.get(k,0)) for k in range(4)},
        "primary_decile_counts":{str(k):int(dec.get(k,0)) for k in range(10)},
        "primary_bottom_two_quartiles_fraction":bottom2,
        "by_legal_action_count":{
            str(k):summarize_records(v) for k,v in sorted(by_legal.items())
        },
        "decision":{
            "classification":classification,
            "target_margin_fragile":bool(margin_fragile),
            "production_change_authorized":False,
            "long_training_unblocked":False,
        },
        "interpretation":{
            "zero_votes":"Fraction of current-fallback/target-positive states where none of the eight members predicts the CFR-best action raw Advantage >0.",
            "individual_fallback":"A member individually falls back when all of its legal raw Advantages are <=0.",
            "quartiles":"Target-strength bins are defined from all target-positive protected-holdout states, then the current-fallback/target-positive population is located within those frozen bins.",
            "scope":"Diagnostic only; no model or policy changes."
        }
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("=== Teacher @17315 fallback/sign-error diagnostic ===")
    print("classification="+classification)
    print("target_margin_fragile="+str(margin_fragile))
    print("primary="+json.dumps(primary,sort_keys=True))
    print("per_member_sign_recall="+json.dumps(report["per_member_target_best_positive_sign_recall"]))
    print("quartile_counts="+json.dumps(report["primary_quartile_counts"],sort_keys=True))
    print("bottom_two_quartiles_fraction="+str(bottom2))
    print(f"report={args.out.resolve()}")
    print("SPINCORE_TEACHER_FALLBACK_SIGN_ERROR_17315_PASS")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
