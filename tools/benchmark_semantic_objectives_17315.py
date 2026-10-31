#!/usr/bin/env python3
from __future__ import annotations

"""Offline objective benchmark for Semantic ENS8 @17315.

PROJECT_CONTRACT_IDS:
MODEL-020,MODEL-025,TRAIN-020,VALID-025,VALID-034,VALID-035,SAFE-001,SAFE-010,RNG-001,RNG-003,PERF-001,PERF-002

No CFR collection occurs. Refit uses the frozen final 3H Advantage reservoir,
same protected split, exact member init/batch seeds, same architecture and same
optimizer schedule. The only experimental variable is the training objective.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as shadow
from run_3h_v1_semantic_shadow_10105 import make_semantic_model
import run_3h_semantic_online_pilot_10105 as pilot
import lt3_3h_semantic_parallel_fit as packed_fit

from spincore.lean_action_policy import lean_regret_matching_policy
from spincore_nn.action_models import collate_action_observations
from spincore_nn.reservoir import UniformReservoir
from spincore_nn.semantic_v1 import semantic_vector_from_observation
from spincore_nn.training import train_step

SCHEMA="SPINCORE_SEMANTIC_OBJECTIVE_BENCHMARK_17315_V1"
ITERATION=17315
ENSEMBLE_SIZE=8
MEMBER_STEPS=1600
CANDIDATES=(
    "A_CURRENT_WEIGHTED_MSE_REFIT",
    "B_STATE_SCALE_NORMALIZED_MSE",
    "C_POSNEG_BALANCED_MSE",
    "D_NORMALIZED_POSNEG_BALANCED_MSE",
)
_PACKED=None


def sha256(path:Path)->str:
    return pilot.sha256(path)


def state_equal(a,b)->bool:
    return set(a)==set(b) and all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a)


def clone_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def objective_step(model,opt,batch,target,weights,objective:str)->float:
    if objective=="A_CURRENT_WEIGHTED_MSE_REFIT":
        return train_step(model,opt,batch,target,weights,"advantage")

    model.train()
    opt.zero_grad(set_to_none=True)
    out=model(batch)
    mask=batch["legal"].bool()
    maskf=mask.float()
    y=target

    normalized=objective in {
        "B_STATE_SCALE_NORMALIZED_MSE",
        "D_NORMALIZED_POSNEG_BALANCED_MSE",
    }
    balanced=objective in {
        "C_POSNEG_BALANCED_MSE",
        "D_NORMALIZED_POSNEG_BALANCED_MSE",
    }

    if normalized:
        scale=(target.abs()*maskf).amax(dim=1).clamp_min(1e-6)
        y=target/scale[:,None]

    err=(out-y).pow(2)
    if balanced:
        pos=mask & (y>0.0)
        neg=mask & ~pos
        pos_n=pos.sum(dim=1).float()
        neg_n=neg.sum(dim=1).float()
        legal_n=mask.sum(dim=1).float().clamp_min(1.0)
        both=(pos_n>0.0)&(neg_n>0.0)
        uniform=maskf/legal_n[:,None]
        action_w=torch.zeros_like(err)
        pos_w=0.5/pos_n.clamp_min(1.0)
        neg_w=0.5/neg_n.clamp_min(1.0)
        action_w=torch.where(
            both[:,None],
            pos.float()*pos_w[:,None]+neg.float()*neg_w[:,None],
            uniform,
        )
        per=(err*action_w).sum(dim=1)
    else:
        per=(err*maskf).sum(dim=1)/maskf.sum(dim=1).clamp_min(1.0)

    w=weights/weights.mean().clamp_min(1e-12)
    loss=(per*w).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),10.0)
    opt.step()
    return float(loss.detach().cpu())


def worker_init(manifest_path:str,threads:int):
    global _PACKED
    os.environ["OMP_NUM_THREADS"]=str(int(threads))
    os.environ["MKL_NUM_THREADS"]=str(int(threads))
    os.environ["OPENBLAS_NUM_THREADS"]=str(int(threads))
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _PACKED=packed_fit.PackedSemanticAdvantageReservoir(Path(manifest_path),mode="r")


def worker_fit(task):
    global _PACKED
    if _PACKED is None:
        raise RuntimeError("packed reservoir worker not initialized")
    objective,member,meta,lr,batch_size=task
    member=int(member)
    init_seed=int(meta["init_seed"])
    batch_seed=int(meta["batch_seed"])
    model=make_semantic_model(init_seed)
    opt=torch.optim.Adam(model.parameters(),lr=float(lr))
    rng=random.Random(batch_seed)
    population=range(int(_PACKED.train_count))
    bs=min(int(batch_size),int(_PACKED.train_count))
    losses=[]
    started=time.perf_counter()
    for _ in range(MEMBER_STEPS):
        positions=rng.sample(population,bs)
        batch,target,weights=_PACKED.batch_from_positions(positions)
        losses.append(objective_step(model,opt,batch,target,weights,str(objective)))
    elapsed=float(time.perf_counter()-started)
    model.eval()
    return {
        "objective":str(objective),
        "member":member,
        "init_seed":init_seed,
        "batch_seed":batch_seed,
        "steps":MEMBER_STEPS,
        "loss_last":float(losses[-1]),
        "fit_seconds":elapsed,
        "maxrss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "pid":int(os.getpid()),
        "state":clone_state(model),
    }


def fit_objective(pool,objective,member_meta,lr,batch_size):
    tasks=[
        (objective,i,member_meta[i],float(lr),int(batch_size))
        for i in range(ENSEMBLE_SIZE)
    ]
    rows=list(pool.map(worker_fit,tasks,chunksize=1))
    rows.sort(key=lambda x:int(x["member"]))
    return rows


def rm(raw,legal):
    return np.asarray(
        lean_regret_matching_policy([float(x) for x in raw],tuple(int(a) for a in legal)),
        dtype=np.float64,
    )


def legal_actions(mask):
    return tuple(i for i,v in enumerate(mask) if bool(v))


def weighted_quantile(values,weights,q):
    order=np.argsort(values)
    v=np.asarray(values,dtype=np.float64)[order]
    w=np.asarray(weights,dtype=np.float64)[order]
    cutoff=float(q)*float(w.sum())
    idx=int(np.searchsorted(np.cumsum(w),cutoff,side="left"))
    return float(v[min(idx,len(v)-1)])


def finalize_acc(acc):
    w=np.asarray(acc["w"],dtype=np.float64)
    tv=np.asarray(acc["tv"],dtype=np.float64)
    ce=np.asarray(acc["ce"],dtype=np.float64)
    ev=np.asarray(acc["ev"],dtype=np.float64)
    agree=np.asarray(acc["agree"],dtype=np.float64)
    total=float(w.sum())
    return {
        "count":len(tv),
        "weight_sum":total,
        "weighted_mean_tv":float(np.dot(w,tv)/total),
        "weighted_median_tv":weighted_quantile(tv,w,0.5),
        "weighted_p95_tv":weighted_quantile(tv,w,0.95),
        "weighted_cross_entropy":float(np.dot(w,ce)/total),
        "weighted_expected_stored_target_advantage":float(np.dot(w,ev)/total),
        "weighted_target_argmax_agreement":float(np.dot(w,agree)/total),
        "unweighted_mean_tv":float(tv.mean()),
        "unweighted_p95_tv":float(np.quantile(tv,0.95)),
    }


def evaluate_models(memory,indices,models,batch_size,primary_indices=None):
    acc={k:[] for k in ("w","tv","ce","ev","agree")}
    own_fallback=0
    own_target_positive_fallback=0
    own_target_positive=0
    discovered_primary=[]
    primary_rows=[]

    with torch.no_grad():
        for start in range(0,len(indices),int(batch_size)):
            chunk=indices[start:start+int(batch_size)]
            samples=[memory.items[i] for i in chunk]
            batch=collate_action_observations(
                shadow.REPRESENTATION,
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
            member_raw=np.stack(
                [m(sb).detach().cpu().numpy() for m in models],axis=0
            )
            for j,sample in enumerate(samples):
                idx=int(chunk[j])
                legal=legal_actions(sample.legal)
                target=np.asarray(sample.target,dtype=np.float64)
                target_policy=rm(target,legal)
                raw=member_raw[:,j,:].astype(np.float64,copy=False)
                mean_raw=raw.mean(axis=0)
                policy=rm(mean_raw,legal)
                weight=float(sample.weight)
                tv=0.5*float(np.abs(policy-target_policy).sum())
                ce=-sum(
                    float(target_policy[a])*math.log(max(1e-12,float(policy[a])))
                    for a in legal
                )
                ev=sum(float(policy[a])*float(target[a]) for a in legal)
                tmax=max(float(target_policy[a]) for a in legal)
                pmax=max(float(policy[a]) for a in legal)
                tbest={a for a in legal if abs(float(target_policy[a])-tmax)<=1e-12}
                pbest={a for a in legal if abs(float(policy[a])-pmax)<=1e-12}
                acc["w"].append(weight)
                acc["tv"].append(tv)
                acc["ce"].append(ce)
                acc["ev"].append(ev)
                acc["agree"].append(float(bool(tbest&pbest)))

                target_strength=max(float(target[a]) for a in legal)
                target_positive=target_strength>0.0
                fallback=max(float(mean_raw[a]) for a in legal)<=0.0
                own_fallback+=int(fallback)
                own_target_positive+=int(target_positive)
                own_target_positive_fallback+=int(fallback and target_positive)
                if primary_indices is None and fallback and target_positive:
                    discovered_primary.append(idx)

                if primary_indices is not None and idx in primary_indices:
                    best=max(legal,key=lambda a:float(target[a]))
                    votes=int(sum(float(raw[k,best])>0.0 for k in range(ENSEMBLE_SIZE)))
                    member_policies=np.stack([rm(raw[k],legal) for k in range(ENSEMBLE_SIZE)],axis=0)
                    primary_rows.append({
                        "residual_fallback":bool(fallback),
                        "target_best_positive_votes":votes,
                        "member_policy_mean_target_best_probability":float(member_policies[:,best].mean()),
                        "ensemble_target_best_probability":float(policy[best]),
                    })

    metrics=finalize_acc(acc)
    metrics.update({
        "own_fallback_count":int(own_fallback),
        "own_fallback_fraction":own_fallback/float(len(indices)),
        "own_target_positive_count":int(own_target_positive),
        "own_target_positive_fallback_count":int(own_target_positive_fallback),
        "own_target_positive_fallback_fraction":(
            0.0 if own_target_positive==0 else own_target_positive_fallback/float(own_target_positive)
        ),
    })
    primary=None
    if primary_indices is not None:
        if len(primary_rows)!=len(primary_indices):
            raise RuntimeError("fixed-primary evaluation count drift")
        votes=np.asarray([r["target_best_positive_votes"] for r in primary_rows],dtype=np.float64)
        residual=np.asarray([r["residual_fallback"] for r in primary_rows],dtype=np.float64)
        mp=np.asarray([r["member_policy_mean_target_best_probability"] for r in primary_rows],dtype=np.float64)
        ep=np.asarray([r["ensemble_target_best_probability"] for r in primary_rows],dtype=np.float64)
        primary={
            "count":len(primary_rows),
            "residual_ensemble_fallback_fraction":float(residual.mean()),
            "target_best_positive_vote_mean":float(votes.mean()),
            "zero_positive_vote_fraction":float(np.mean(votes==0)),
            "at_least_four_positive_vote_fraction":float(np.mean(votes>=4)),
            "member_policy_mean_target_best_probability_mean":float(mp.mean()),
            "ensemble_target_best_probability_mean":float(ep.mean()),
        }
    return metrics,discovered_primary,primary


def load_models(states):
    return [shadow.load_semantic_advantage(s) for s in states]


def adjudicate(results):
    A=results["A_CURRENT_WEIGHTED_MSE_REFIT"]
    passed=[]
    checks={}
    for cid in CANDIDATES[1:]:
        m=results[cid]
        c={
            "weighted_mean_tv":m["holdout"]["weighted_mean_tv"]<=A["holdout"]["weighted_mean_tv"]+0.005,
            "weighted_cross_entropy":m["holdout"]["weighted_cross_entropy"]<=A["holdout"]["weighted_cross_entropy"]+0.50,
            "weighted_expected_stored_target_advantage":m["holdout"]["weighted_expected_stored_target_advantage"]>=A["holdout"]["weighted_expected_stored_target_advantage"]-0.001,
            "weighted_target_argmax_agreement":m["holdout"]["weighted_target_argmax_agreement"]>=A["holdout"]["weighted_target_argmax_agreement"]-0.005,
            "own_target_positive_fallback_count":m["holdout"]["own_target_positive_fallback_count"]<=0.60*A["holdout"]["own_target_positive_fallback_count"],
            "fixed_primary_residual_fallback_fraction":m["fixed_primary"]["residual_ensemble_fallback_fraction"]<=0.50,
        }
        checks[cid]=c
        if all(c.values()):
            passed.append(cid)
    return {
        "classification":(
            "NO_OBJECTIVE_CANDIDATE_PASS"
            if not passed
            else "OBJECTIVE_CANDIDATE_IDENTIFIED"
            if len(passed)==1
            else "MULTIPLE_OBJECTIVE_CANDIDATES_PASS"
        ),
        "passing_candidates":passed,
        "checks":checks,
        "production_change_authorized":False,
        "long_training_unblocked":False,
        "next_gate":"FROZEN_PATHOLOGY_AND_DIRECT_PLAY" if passed else "MODEL_REPRESENTATION_OR_OBJECTIVE_RESEARCH",
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--resume",type=Path,required=True)
    ap.add_argument("--ensemble",type=Path,required=True)
    ap.add_argument("--training-report",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--prereg",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--fit-concurrency",type=int,default=4)
    ap.add_argument("--threads-per-member",type=int,default=8)
    ap.add_argument("--eval-batch-size",type=int,default=512)
    args=ap.parse_args()

    prereg=json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("schema")!="SPINCORE_SEMANTIC_OBJECTIVE_BENCHMARK_17315_PREREG_V1":
        raise RuntimeError("wrong objective prereg schema")
    if prereg.get("status")!="FROZEN_BEFORE_RESULTS":
        raise RuntimeError("objective prereg is not frozen")

    training=json.loads(args.training_report.read_text(encoding="utf-8"))
    if training.get("status")!="PASS" or int(training.get("completed_iteration",-1))!=ITERATION:
        raise RuntimeError("source training report not PASS @17315")
    if training.get("final_ensemble_sha256")!=sha256(args.ensemble):
        raise RuntimeError("source ensemble hash drift")
    if training.get("resume_state_sha256")!=sha256(args.resume):
        raise RuntimeError("source resume hash drift")

    cp=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    cfg=dict(cp.get("config") or {})
    lr=float(cfg["learning_rate"])
    batch_size=int(cfg["batch_size"])

    ensemble=torch.load(args.ensemble,map_location="cpu",weights_only=False)
    resume=torch.load(args.resume,map_location="cpu",weights_only=False)
    source_states=list(ensemble.get("members") or [])
    member_meta=list(resume.get("member_seed_contract") or [])
    if len(source_states)!=ENSEMBLE_SIZE or len(member_meta)!=ENSEMBLE_SIZE:
        raise RuntimeError("source ensemble/member metadata drift")

    memory=UniformReservoir.from_state_dict(resume["adv_mem"])
    if len(memory.items)!=2_000_000:
        raise RuntimeError("expected saturated 2M Advantage reservoir")
    probe=torch.load(args.probe,map_location="cpu",weights_only=False)
    split_rng=random.Random(int(probe["holdout_seed"]))
    protected=sorted(split_rng.sample(range(len(memory.items)),int(probe["holdout_size"])))
    protected_set=set(protected)
    train_pool=[i for i in range(len(memory.items)) if i not in protected_set]
    if len(protected)!=50000 or len(train_pool)!=1950000:
        raise RuntimeError("controlled split drift")

    out_dir=args.out_dir.resolve()
    out_dir.mkdir(parents=True,exist_ok=True)
    pack_dir=out_dir/"packed_reservoir"
    if pack_dir.exists():
        shutil.rmtree(pack_dir)

    print("OBJECTIVE_BENCHMARK_PRECOMPUTE_BEGIN",flush=True)
    sem_rows,precompute_seconds=pilot.precompute_semantics(memory)
    print(f"OBJECTIVE_BENCHMARK_PRECOMPUTE_PASS seconds={precompute_seconds:.3f}",flush=True)
    packed,pack_meta=packed_fit.PackedSemanticAdvantageReservoir.build(
        memory,sem_rows,train_pool,pack_dir
    )
    manifest_path=packed.manifest_path
    packed.close()

    ctx=mp.get_context("spawn")
    fit_results={}
    candidate_states={}
    started=time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=int(args.fit_concurrency),
        mp_context=ctx,
        initializer=worker_init,
        initargs=(str(manifest_path),int(args.threads_per_member)),
    ) as pool:
        print("OBJECTIVE_BASELINE_REFIT_BEGIN",flush=True)
        rows=fit_objective(pool,CANDIDATES[0],member_meta,lr,batch_size)
        baseline_states=[r["state"] for r in rows]
        for i,(got,expected) in enumerate(zip(baseline_states,source_states)):
            if not state_equal(got,expected):
                raise RuntimeError(f"baseline exact-refit parity failed member {i}")
        print("OBJECTIVE_BASELINE_EXACT_PARITY_PASS",flush=True)
        fit_results[CANDIDATES[0]]=[
            {k:v for k,v in r.items() if k!="state"} for r in rows
        ]
        candidate_states[CANDIDATES[0]]=baseline_states

        for cid in CANDIDATES[1:]:
            print(f"OBJECTIVE_FIT_BEGIN candidate={cid}",flush=True)
            rows=fit_objective(pool,cid,member_meta,lr,batch_size)
            candidate_states[cid]=[r["state"] for r in rows]
            fit_results[cid]=[
                {k:v for k,v in r.items() if k!="state"} for r in rows
            ]
            print(f"OBJECTIVE_FIT_PASS candidate={cid}",flush=True)

    source_models=load_models(source_states)
    A_metrics,primary_indices,_=evaluate_models(
        memory,protected,source_models,int(args.eval_batch_size),None
    )
    if len(primary_indices)!=4676:
        raise RuntimeError(f"frozen primary population drift: {len(primary_indices)} != 4676")
    primary_set=set(primary_indices)

    results={}
    for cid in CANDIDATES:
        models=load_models(candidate_states[cid])
        metrics,_,primary=evaluate_models(
            memory,protected,models,int(args.eval_batch_size),primary_set
        )
        results[cid]={
            "holdout":metrics,
            "fixed_primary":primary,
            "fit":fit_results[cid],
        }
        torch.save({
            "schema":"SPINCORE_SEMANTIC_OBJECTIVE_CANDIDATE_ENSEMBLE_V1",
            "teacher_iteration":ITERATION,
            "candidate":cid,
            "members":candidate_states[cid],
            "member_seed_contract":member_meta,
            "member_steps":MEMBER_STEPS,
            "learning_rate":lr,
            "batch_size":batch_size,
            "production_status":"RESEARCH_ONLY_NOT_PROMOTED",
        },out_dir/f"{cid}.pt")
        print(
            "OBJECTIVE_EVAL "
            +json.dumps({
                "candidate":cid,
                "weighted_mean_tv":metrics["weighted_mean_tv"],
                "weighted_cross_entropy":metrics["weighted_cross_entropy"],
                "weighted_expected_advantage":metrics["weighted_expected_stored_target_advantage"],
                "target_positive_fallback_count":metrics["own_target_positive_fallback_count"],
                "primary_residual_fallback":primary["residual_ensemble_fallback_fraction"],
                "primary_vote_mean":primary["target_best_positive_vote_mean"],
            },sort_keys=True),
            flush=True,
        )

    # Source and reconstructed A must be inference-identical by state equality.
    if any(not state_equal(a,b) for a,b in zip(candidate_states[CANDIDATES[0]],source_states)):
        raise RuntimeError("baseline parity changed after evaluation")

    decision=adjudicate(results)
    report={
        "schema":SCHEMA,
        "status":"PASS",
        "teacher_iteration":ITERATION,
        "teacher_sha256":sha256(args.ensemble),
        "resume_sha256":sha256(args.resume),
        "checkpoint_sha256":sha256(args.checkpoint),
        "prereg_sha256":sha256(args.prereg),
        "protected_holdout_count":len(protected),
        "train_pool_count":len(train_pool),
        "baseline_exact_refit_parity":True,
        "learning_rate":lr,
        "batch_size":batch_size,
        "member_steps":MEMBER_STEPS,
        "fit_concurrency":int(args.fit_concurrency),
        "threads_per_member":int(args.threads_per_member),
        "precompute_seconds":precompute_seconds,
        "pack_build_seconds":float(pack_meta["build_seconds"]),
        "wall_seconds":float(time.perf_counter()-started),
        "results":results,
        "decision":decision,
        "production_change_authorized":False,
        "long_training_unblocked":False,
        "interpretation":{
            "scale_normalization":"Per-state positive scaling leaves ideal regret-matching policy invariant, so B isolates cross-state target-scale conditioning.",
            "posneg_balance":"C/D equalize aggregate loss mass between positive-regret and non-positive legal actions within a state when both groups exist.",
            "parity":"Bitwise baseline refit parity proves data, seeds, batches, optimizer and current objective were reconstructed exactly before alternate objectives were evaluated."
        }
    }
    report_path=out_dir/"semantic_objective_benchmark_17315.json"
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("=== Semantic objective benchmark @17315 ===")
    print("classification="+decision["classification"])
    print("passing_candidates="+json.dumps(decision["passing_candidates"]))
    print("checks="+json.dumps(decision["checks"],sort_keys=True))
    print(f"report={report_path}")
    print("SPINCORE_SEMANTIC_OBJECTIVE_BENCHMARK_17315_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
