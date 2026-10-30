#!/usr/bin/env python3
from __future__ import annotations

"""Teacher @17315 aggregation/fallback architecture benchmark.

PROJECT_CONTRACT_IDS:
MODEL-020,MODEL-025,VALID-025,VALID-034,VALID-035,SAFE-001,SAFE-010,RNG-001,RNG-003

Research-only. Compares frozen aggregation operators on the exact same teacher
members and on the pre-existing protected 50k CFR Advantage holdout. No model
training or production behavior mutation occurs.
"""

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

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

SCHEMA="SPINCORE_TEACHER_ARCHITECTURE_BENCHMARK_17315_V1"
ENSEMBLE_SCHEMA="SPINCORE_3H_SEMANTIC_LONG_ENSEMBLE_V1"
ITERATION=17315
CANDIDATES=(
    "A_CURRENT_RAW_MEAN_RM",
    "B_MEMBER_POLICY_MEAN",
    "C_MEDIAN_RAW_RM",
    "D_TRIMMED_RAW_MEAN_RM",
)
EPS=1e-12


def _legal_actions(mask) -> tuple[int,...]:
    legal=tuple(i for i,v in enumerate(mask) if bool(v))
    if not legal:
        raise RuntimeError("empty legal action set")
    return legal


def _rm(raw,legal):
    return np.asarray(
        lean_regret_matching_policy(
            [float(x) for x in raw],
            tuple(int(a) for a in legal),
        ),
        dtype=np.float64,
    )


def candidate_policy(member_raw:np.ndarray,legal:tuple[int,...],candidate:str)->np.ndarray:
    arr=np.asarray(member_raw,dtype=np.float64)
    if arr.ndim!=2 or arr.shape[0]!=8 or arr.shape[1]!=10:
        raise ValueError(f"expected 8x10 member raw matrix, got {arr.shape}")
    if candidate=="A_CURRENT_RAW_MEAN_RM":
        return _rm(arr.mean(axis=0),legal)
    if candidate=="B_MEMBER_POLICY_MEAN":
        return np.stack([_rm(row,legal) for row in arr],axis=0).mean(axis=0)
    if candidate=="C_MEDIAN_RAW_RM":
        return _rm(np.median(arr,axis=0),legal)
    if candidate=="D_TRIMMED_RAW_MEAN_RM":
        ordered=np.sort(arr,axis=0)
        return _rm(ordered[1:-1].mean(axis=0),legal)
    raise ValueError(f"unknown candidate: {candidate}")


def _weighted_quantile(values,weights,q):
    order=np.argsort(values)
    v=np.asarray(values,dtype=np.float64)[order]
    w=np.asarray(weights,dtype=np.float64)[order]
    total=float(w.sum())
    if total<=0:
        raise RuntimeError("nonpositive total weight")
    cutoff=float(q)*total
    idx=int(np.searchsorted(np.cumsum(w),cutoff,side="left"))
    return float(v[min(idx,len(v)-1)])


def _finalize_metric(acc:dict[str,Any])->dict[str,Any]:
    w=np.asarray(acc["weights"],dtype=np.float64)
    tv=np.asarray(acc["tv"],dtype=np.float64)
    ce=np.asarray(acc["ce"],dtype=np.float64)
    ev=np.asarray(acc["expected_target_advantage"],dtype=np.float64)
    agree=np.asarray(acc["argmax_agree"],dtype=np.float64)
    total=float(w.sum())
    return {
        "count":len(tv),
        "weight_sum":total,
        "weighted_mean_tv":float(np.dot(w,tv)/total),
        "weighted_median_tv":_weighted_quantile(tv,w,0.5),
        "weighted_p95_tv":_weighted_quantile(tv,w,0.95),
        "unweighted_mean_tv":float(tv.mean()),
        "unweighted_median_tv":float(np.median(tv)),
        "unweighted_p95_tv":float(np.quantile(tv,0.95)),
        "weighted_cross_entropy":float(np.dot(w,ce)/total),
        "weighted_expected_stored_target_advantage":float(np.dot(w,ev)/total),
        "weighted_target_argmax_agreement":float(np.dot(w,agree)/total),
    }


def _target_argmax_agree(policy,target,legal):
    vals=[float(target[a]) for a in legal]
    mx=max(vals)
    target_best={a for a in legal if abs(float(target[a])-mx)<=1e-12}
    pvals=[float(policy[a]) for a in legal]
    pmx=max(pvals)
    policy_best={a for a in legal if abs(float(policy[a])-pmx)<=1e-12}
    return float(bool(target_best & policy_best))


def _target_branch(raw,legal)->str:
    positive=sum(max(0.0,float(raw[a])) for a in legal)
    return "POSITIVE_REGRET_MATCHING" if positive>0 else "SOFTMAX_FALLBACK"


def evaluate_holdout(*,memory,indices,models,batch_size:int)->dict[str,Any]:
    acc={
        c:{
            "weights":[],
            "tv":[],
            "ce":[],
            "expected_target_advantage":[],
            "argmax_agree":[],
        }
        for c in CANDIDATES
    }
    fallback={
        "current_fallback_count":0,
        "current_fallback_weight":0.0,
        "current_fallback_target_positive_regret_count":0,
        "current_fallback_target_fallback_count":0,
        "current_fallback_target_max_probabilities":[],
        "current_fallback_current_tv":[],
        "current_fallback_weights":[],
    }

    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            chunk_idx=indices[start:start+batch_size]
            samples=[memory.items[i] for i in chunk_idx]
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
                [m(sb).detach().cpu().numpy() for m in models],
                axis=0,
            )  # [8,B,10]

            for j,sample in enumerate(samples):
                legal=_legal_actions(sample.legal)
                target_raw=np.asarray(sample.target,dtype=np.float64)
                target_policy=_rm(target_raw,legal)
                weight=float(sample.weight)
                if not math.isfinite(weight) or weight<=0:
                    raise RuntimeError("nonpositive/nonfinite holdout sample weight")
                member_raw=raw_members[:,j,:].astype(np.float64,copy=False)

                policies={}
                for c in CANDIDATES:
                    p=candidate_policy(member_raw,legal,c)
                    if abs(float(p.sum())-1.0)>1e-8:
                        raise RuntimeError(f"candidate policy not normalized: {c}")
                    policies[c]=p
                    tv=0.5*float(np.abs(p-target_policy).sum())
                    ce=-sum(
                        float(target_policy[a])*math.log(max(EPS,float(p[a])))
                        for a in legal
                    )
                    expected=sum(float(p[a])*float(target_raw[a]) for a in legal)
                    acc[c]["weights"].append(weight)
                    acc[c]["tv"].append(tv)
                    acc[c]["ce"].append(ce)
                    acc[c]["expected_target_advantage"].append(expected)
                    acc[c]["argmax_agree"].append(
                        _target_argmax_agree(p,target_policy,legal)
                    )

                current_mean=member_raw.mean(axis=0)
                if _target_branch(current_mean,legal)=="SOFTMAX_FALLBACK":
                    fallback["current_fallback_count"]+=1
                    fallback["current_fallback_weight"]+=weight
                    tb=_target_branch(target_raw,legal)
                    if tb=="POSITIVE_REGRET_MATCHING":
                        fallback["current_fallback_target_positive_regret_count"]+=1
                    else:
                        fallback["current_fallback_target_fallback_count"]+=1
                    fallback["current_fallback_target_max_probabilities"].append(
                        max(float(target_policy[a]) for a in legal)
                    )
                    fallback["current_fallback_current_tv"].append(
                        0.5*float(np.abs(policies["A_CURRENT_RAW_MEAN_RM"]-target_policy).sum())
                    )
                    fallback["current_fallback_weights"].append(weight)

    metrics={c:_finalize_metric(acc[c]) for c in CANDIDATES}
    n=int(fallback["current_fallback_count"])
    if n:
        fw=np.asarray(fallback["current_fallback_weights"],dtype=np.float64)
        ftv=np.asarray(fallback["current_fallback_current_tv"],dtype=np.float64)
        fmax=np.asarray(fallback["current_fallback_target_max_probabilities"],dtype=np.float64)
        fallback_summary={
            "current_fallback_count":n,
            "current_fallback_fraction":n/float(len(indices)),
            "current_fallback_weight":float(fallback["current_fallback_weight"]),
            "target_positive_regret_count":int(
                fallback["current_fallback_target_positive_regret_count"]
            ),
            "target_positive_regret_fraction":(
                fallback["current_fallback_target_positive_regret_count"]/float(n)
            ),
            "target_fallback_count":int(
                fallback["current_fallback_target_fallback_count"]
            ),
            "target_policy_max_probability_mean":float(fmax.mean()),
            "target_policy_max_probability_median":float(np.median(fmax)),
            "target_policy_max_probability_p95":float(np.quantile(fmax,0.95)),
            "target_policy_max_probability_ge80_fraction":float(np.mean(fmax>=0.80)),
            "current_weighted_mean_tv_to_target_on_fallback":float(
                np.dot(fw,ftv)/fw.sum()
            ),
        }
    else:
        fallback_summary={"current_fallback_count":0,"current_fallback_fraction":0.0}
    return {"candidate_metrics":metrics,"fallback_diagnostic":fallback_summary}


def _row_member_raw(detail:dict[str,Any])->tuple[np.ndarray,tuple[int,...]]:
    legal_slots=[int(x) for x in detail.get("legal_slots") or []]
    legal_names=[str(x) for x in detail.get("legal_names") or []]
    if not legal_slots or len(legal_slots)!=len(legal_names):
        raise RuntimeError("raw replay row lacks legal mapping")
    name_to_slot=dict(zip(legal_names,legal_slots))
    members=list(detail.get("member_raw_advantages") or [])
    if len(members)!=8:
        raise RuntimeError("raw replay row lacks eight members")
    arr=np.zeros((8,10),dtype=np.float64)
    for i,row in enumerate(members):
        for name,slot in name_to_slot.items():
            arr[i,int(slot)]=float(row[name])
    return arr,tuple(legal_slots)


def evaluate_pathologies(raw_report:dict[str,Any])->dict[str,Any]:
    groups:dict[str,list[dict[str,Any]]]={}
    for row in raw_report.get("rows") or []:
        for reason in row.get("audit_reasons") or []:
            if not str(reason).startswith("POLICY_SURFACE:"):
                continue
            groups.setdefault(str(reason),[]).append(row)

    out={}
    for reason,rows in sorted(groups.items()):
        per_candidate={c:[] for c in CANDIDATES}
        target_actions=[]
        for row in rows:
            detail=row["policy_detail"]
            member_raw,legal=_row_member_raw(detail)
            target_action=str(row["diagnostic_target_action"])
            legal_names=[str(x) for x in detail["legal_names"]]
            legal_slots=[int(x) for x in detail["legal_slots"]]
            name_to_slot=dict(zip(legal_names,legal_slots))
            if target_action not in name_to_slot:
                raise RuntimeError("pathology target action is not legal")
            slot=int(name_to_slot[target_action])
            target_actions.append(target_action)
            for c in CANDIDATES:
                p=candidate_policy(member_raw,legal,c)
                per_candidate[c].append(float(p[slot]))
        out[reason]={
            "count":len(rows),
            "target_actions":sorted(set(target_actions)),
            "candidates":{
                c:{
                    "mean_target_action_probability":float(np.mean(vals)),
                    "median_target_action_probability":float(np.median(vals)),
                    "max_target_action_probability":float(np.max(vals)),
                    "ge50_count":int(np.sum(np.asarray(vals)>=0.50)),
                    "ge80_count":int(np.sum(np.asarray(vals)>=0.80)),
                    "ge95_count":int(np.sum(np.asarray(vals)>=0.95)),
                }
                for c,vals in per_candidate.items()
            },
        }
    return out


def adjudicate(holdout:dict[str,Any],pathologies:dict[str,Any],prereg:dict[str,Any])->dict[str,Any]:
    current=holdout["candidate_metrics"]["A_CURRENT_RAW_MEAN_RM"]
    hc=pathologies["POLICY_SURFACE:POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW"]["candidates"]
    aa=pathologies["POLICY_SURFACE:PREFLOP_AA_FOLD"]["candidates"]
    monster=pathologies["POLICY_SURFACE:POSTFLOP_MONSTER_FOLD"]["candidates"]
    trips=pathologies["POLICY_SURFACE:POSTFLOP_TRIPS_PLUS_FOLD"]["candidates"]
    seven=pathologies["POLICY_SURFACE:PREFLOP_DEEP_72O_JAM"]["candidates"]
    passes={}
    detail={}
    for c in CANDIDATES[1:]:
        m=holdout["candidate_metrics"][c]
        checks={
            "holdout_weighted_mean_tv":m["weighted_mean_tv"]<=current["weighted_mean_tv"]+0.005,
            "holdout_p95_tv":m["unweighted_p95_tv"]<=current["unweighted_p95_tv"]+0.010,
            "high_card_no_draw_mean_allin":hc[c]["mean_target_action_probability"]<=0.50,
            "high_card_no_draw_ge80":hc[c]["ge80_count"]<=5,
            "aa_fold_max":aa[c]["max_target_action_probability"]<=0.05,
            "monster_fold_max":monster[c]["max_target_action_probability"]<=0.05,
            "trips_fold_mean":trips[c]["mean_target_action_probability"]<=(
                trips["A_CURRENT_RAW_MEAN_RM"]["mean_target_action_probability"]+0.05
            ),
            "deep_72o_mean_allin":seven[c]["mean_target_action_probability"]<=(
                seven["A_CURRENT_RAW_MEAN_RM"]["mean_target_action_probability"]+0.05
            ),
        }
        passes[c]=all(checks.values())
        detail[c]=checks
    passed=[c for c,v in passes.items() if v]
    if not passed:
        classification="AGGREGATION_ONLY_INSUFFICIENT"
    elif len(passed)==1:
        classification="AGGREGATION_CANDIDATE_IDENTIFIED"
    else:
        classification="MULTIPLE_AGGREGATION_CANDIDATES_PASS"
    return {
        "classification":classification,
        "passing_candidates":passed,
        "candidate_pass":passes,
        "checks":detail,
        "production_change_authorized":False,
        "next_gate_required":"DIRECT_PLAY_COMMON_STATE_VALIDATION" if passed else "FALLBACK_OR_MODEL_ARCHITECTURE_RESEARCH",
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume",type=Path,required=True)
    ap.add_argument("--ensemble",type=Path,required=True)
    ap.add_argument("--training-report",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--raw-report",type=Path,required=True)
    ap.add_argument("--prereg",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--batch-size",type=int,default=512)
    args=ap.parse_args()

    prereg=json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("schema")!="SPINCORE_TEACHER_ARCHITECTURE_BENCHMARK_17315_PREREG_V1":
        raise RuntimeError("wrong prereg schema")
    if prereg.get("status")!="FROZEN_BEFORE_BENCHMARK_RESULTS":
        raise RuntimeError("architecture prereg is not frozen")

    training=json.loads(args.training_report.read_text(encoding="utf-8"))
    ensemble=torch.load(args.ensemble,map_location="cpu",weights_only=False)
    resume=torch.load(args.resume,map_location="cpu",weights_only=False)
    if training.get("status")!="PASS" or int(training.get("completed_iteration",-1))!=ITERATION:
        raise RuntimeError("teacher training report is not PASS @17315")
    if training.get("final_ensemble_sha256")!=pilot.sha256(args.ensemble):
        raise RuntimeError("teacher ensemble hash does not match PASS report")
    if training.get("resume_state_sha256")!=pilot.sha256(args.resume):
        raise RuntimeError("teacher resume hash does not match PASS report")
    if ensemble.get("schema")!=ENSEMBLE_SCHEMA or int(ensemble.get("completed_iteration",-1))!=ITERATION:
        raise RuntimeError("wrong teacher ensemble endpoint")
    if int(resume.get("completed_iteration",-1))!=ITERATION:
        raise RuntimeError("wrong teacher resume endpoint")
    states=list(ensemble.get("members") or [])
    if len(states)!=8:
        raise RuntimeError("expected eight teacher members")
    resume_states=list(resume.get("semantic_members") or [])
    if len(resume_states)!=8:
        raise RuntimeError("resume lacks eight semantic members")
    for a,b in zip(states,resume_states):
        if set(a)!=set(b) or any(not torch.equal(a[k].cpu(),b[k].cpu()) for k in a):
            raise RuntimeError("ensemble/resume member state mismatch")

    memory=UniformReservoir.from_state_dict(resume["adv_mem"])
    if len(memory.items)!=2_000_000:
        raise RuntimeError("expected saturated 2M 3H Advantage reservoir")

    probe=torch.load(args.probe,map_location="cpu",weights_only=False)
    if probe.get("schema")!=pilot.PROBE_SCHEMA:
        raise RuntimeError("wrong controlled-split probe schema")
    split_rng=random.Random(int(probe["holdout_seed"]))
    indices=sorted(split_rng.sample(range(len(memory.items)),int(probe["holdout_size"])))
    if len(indices)!=50000:
        raise RuntimeError("protected holdout size drift")

    models=[distill.load_semantic_advantage(state) for state in states]
    holdout=evaluate_holdout(
        memory=memory,
        indices=indices,
        models=models,
        batch_size=int(args.batch_size),
    )

    raw_report=json.loads(args.raw_report.read_text(encoding="utf-8"))
    if raw_report.get("schema")!="SPINCORE_3H_TEACHER_RAW_ADVANTAGE_DIAGNOSTIC_17315_V1":
        raise RuntimeError("wrong raw replay report schema")
    if raw_report.get("status")!="PASS":
        raise RuntimeError("raw replay report is not PASS")
    pathologies=evaluate_pathologies(raw_report)
    required={
        "POLICY_SURFACE:POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW",
        "POLICY_SURFACE:POSTFLOP_DEEP_HIGH_CARD_JAM",
        "POLICY_SURFACE:PREFLOP_DEEP_72O_JAM",
        "POLICY_SURFACE:POSTFLOP_TRIPS_PLUS_FOLD",
        "POLICY_SURFACE:PREFLOP_AA_FOLD",
        "POLICY_SURFACE:POSTFLOP_MONSTER_FOLD",
    }
    if not required.issubset(pathologies):
        raise RuntimeError("missing preregistered pathology surface")

    decision=adjudicate(holdout,pathologies,prereg)
    report={
        "schema":SCHEMA,
        "status":"PASS",
        "teacher_iteration":ITERATION,
        "teacher_ensemble_sha256":pilot.sha256(args.ensemble),
        "teacher_resume_sha256":pilot.sha256(args.resume),
        "prereg_sha256":pilot.sha256(args.prereg),
        "protected_holdout_count":len(indices),
        "candidates":list(CANDIDATES),
        "holdout":holdout,
        "pathologies":pathologies,
        "decision":decision,
        "production_change_authorized":False,
        "interpretation":{
            "holdout":"Measures approximation to stored CFR Advantage targets on the fixed protected 50k positions excluded from semantic fitting.",
            "pathology":"Recomputes frozen aggregation candidates on exactly replayed @17315 pathology states using the same eight member raw outputs.",
            "fallback":"If current fallback states usually have positive-regret stored targets, sign/approximation error is implicated. If stored targets are also fallback-heavy, the target/regret semantics require separate investigation.",
            "scope":"Research-only architecture gate. No model training or production policy mutation."
        },
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== Teacher @17315 architecture benchmark ===")
    print("classification="+decision["classification"])
    print("passing_candidates="+json.dumps(decision["passing_candidates"]))
    print("holdout="+json.dumps(holdout["candidate_metrics"],sort_keys=True))
    print("fallback="+json.dumps(holdout["fallback_diagnostic"],sort_keys=True))
    print("high_card_no_draw="+json.dumps(
        pathologies["POLICY_SURFACE:POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW"]["candidates"],
        sort_keys=True,
    ))
    print("deep_72o="+json.dumps(
        pathologies["POLICY_SURFACE:PREFLOP_DEEP_72O_JAM"]["candidates"],
        sort_keys=True,
    ))
    print(f"report={args.out.resolve()}")
    print("SPINCORE_TEACHER_ARCHITECTURE_BENCHMARK_17315_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
