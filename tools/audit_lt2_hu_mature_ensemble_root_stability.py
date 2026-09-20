#!/usr/bin/env python3
from __future__ import annotations

"""Mature-reservoir HU Advantage ensemble stability on the frozen iteration-8000 memory.

Eight independent 400-step fresh fits see the exact same HU Advantage reservoir.
For each model we cache raw Advantage predictions on the complete forensic HU
root corpus.  Disjoint 1-, 2- and 4-model ensembles average raw predictions
before the unchanged production lean regret-matching map.

Read only source checkpoint. No CFR roots, no source-memory writes, no holdout.
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import torch

import audit_lt2_hu_behavior_first_divergence as behfd
import audit_lt2_stage_a_b_first_divergence as fd
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_functional_training import load_checkpoint
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import legal_mask
from spincore.solver import SolverLibrary
from spincore_nn.action_models import collate_action_observations

FORENSIC_SEEDS=(20260920,20260921,20260922,20260923,20260924,20260925)
DOMAIN="TRUE_HEADS_UP"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
PAIR_GROUPS={
    "size_1":[((0,),(4,)),((1,),(5,)),((2,),(6,)),((3,),(7,))],
    "size_2":[((0,1),(4,5)),((2,3),(6,7))],
    "size_4":[((0,1,2,3),(4,5,6,7))],
}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--budget",type=int,default=400)
    p.add_argument("--replicas",type=int,default=8)
    p.add_argument("--scenarios-per-seed",type=int,default=5000)
    p.add_argument("--eval-batch-size",type=int,default=1024)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--report",type=Path,required=True)
    return p.parse_args()


def _mean_ci(xs):
    vals=[float(x) for x in xs]
    n=len(vals)
    mean=float(statistics.fmean(vals))
    sem=0.0 if n<=1 else float(statistics.stdev(vals)/math.sqrt(n))
    h=1.96*sem
    return {"n":n,"mean":mean,"sem":sem,"ci95_low":mean-h,"ci95_high":mean+h}


def _collect_corpus(solver, scenarios):
    rows=[]
    counts={}
    for seed in FORENSIC_SEEDS:
        sampler=LegacyScenarioSampler(seed=int(seed)^0x5CE0A710,config=LegacyScenarioConfig())
        hu=0
        for idx in range(int(scenarios)):
            ep=sampler.sample_episode()
            if not ep.game_is_hu:
                continue
            hu+=1
            deal_seed=fd._mix64(int(seed),int(idx),0xD34A1)
            state=solver.create(ep,int(deal_seed))
            try:
                _mask,legal=behfd._legal_context(state)
                rows.append({
                    "seed":int(seed),
                    "scenario":int(idx),
                    "observation":state.neural_bytes(),
                    "legal":tuple(int(x) for x in legal),
                    "legal_mask":legal_mask(legal),
                })
            finally:
                state.close()
        counts[str(seed)]=hu
    return rows,counts


def _predict_raw(model, corpus, batch_size):
    model.eval()
    chunks=[]
    with torch.no_grad():
        for start in range(0,len(corpus),int(batch_size)):
            block=corpus[start:start+int(batch_size)]
            batch=collate_action_observations(
                REPRESENTATION,
                [r["observation"] for r in block],
                [r["legal_mask"] for r in block],
                device="cpu",
            )
            chunks.append(model(batch).detach().cpu())
    return torch.cat(chunks,dim=0)


def _ensemble_policy(raw_predictions, members, corpus):
    raw=torch.stack([raw_predictions[i] for i in members],dim=0).mean(dim=0)
    out=[]
    for i,row in enumerate(corpus):
        out.append(lean_regret_matching_policy(raw[i].tolist(),row["legal"]))
    return torch.tensor(out,dtype=torch.float32)


def _pair_metrics(a,b):
    tv=0.5*torch.abs(a-b).sum(dim=1)
    arg=(torch.argmax(a,dim=1)!=torch.argmax(b,dim=1)).float()
    return {
        "mean_tv":float(tv.mean().item()),
        "p95_tv":float(torch.quantile(tv,torch.tensor(0.95)).item()),
        "argmax_disagreement":float(arg.mean().item()),
        "left_all_in_mass":float(a[:,9].mean().item()),
        "right_all_in_mass":float(b[:,9].mean().item()),
        "left_pot33_mass":float(a[:,3].mean().item()),
        "right_pot33_mass":float(b[:,3].mean().item()),
    }


def main():
    args=parse_args()
    if args.replicas!=8:
        raise SystemExit("this gate requires exactly 8 replicas")
    if args.budget<=0 or args.scenarios_per_seed<=0 or args.threads<=0:
        raise SystemExit("positive budget/scenarios/threads required")

    torch.set_num_threads(int(args.threads))
    solver=SolverLibrary(args.solver.resolve(strict=True))
    seed,config,iteration,_sampler,runtimes,_history,_finalized=load_checkpoint(
        args.checkpoint.resolve(strict=True),solver=solver
    )
    if int(iteration)!=8000:
        raise RuntimeError(f"expected iteration 8000, got {iteration}")
    runtime=runtimes[DOMAIN]
    runtime.session.batch_mode="vectorized"

    corpus,hu_counts=_collect_corpus(solver,args.scenarios_per_seed)

    raw_predictions=[]
    replica_meta=[]
    for rep in range(8):
        init_seed=fd._mix64(20260920,rep,0xE115E)&0x7fffffff
        batch_seed=fd._mix64(20260920,rep,0xEBA7C4)&0x7fffffff
        runtime.session.reset_advantage_network(
            init_seed=int(init_seed),lr=float(config.learning_rate)
        )
        runtime.bundle.batch_rng.seed(int(batch_seed))
        started=time.perf_counter()
        losses=runtime.session.train_advantage(
            steps=int(args.budget),batch_size=int(config.batch_size)
        )
        elapsed=time.perf_counter()-started
        raw=_predict_raw(runtime.bundle.advantage,corpus,args.eval_batch_size)
        raw_predictions.append(raw)
        policy=_ensemble_policy(raw_predictions,(rep,),corpus)
        meta={
            "replicate":rep,
            "init_seed":int(init_seed),
            "batch_seed":int(batch_seed),
            "steps":int(args.budget),
            "fit_seconds":float(elapsed),
            "loss_last":float(losses[-1]),
            "all_in_mass":float(policy[:,9].mean().item()),
            "pot33_mass":float(policy[:,3].mean().item()),
            "check_call_mass":float(policy[:,1].mean().item()),
            "fold_mass":float(policy[:,0].mean().item()),
        }
        replica_meta.append(meta)
        print(
            f"rep={rep+1}/8 loss={meta['loss_last']:.6f} "
            f"AI={meta['all_in_mass']:.4f} P33={meta['pot33_mass']:.4f}",
            flush=True,
        )

    results={}
    for label,pairs in PAIR_GROUPS.items():
        pair_rows=[]
        group_all_in=[]
        group_pot33=[]
        for left,right in pairs:
            pa=_ensemble_policy(raw_predictions,left,corpus)
            pb=_ensemble_policy(raw_predictions,right,corpus)
            row=_pair_metrics(pa,pb)
            row["left_members"]=list(left)
            row["right_members"]=list(right)
            pair_rows.append(row)
            group_all_in.extend([row["left_all_in_mass"],row["right_all_in_mass"]])
            group_pot33.extend([row["left_pot33_mass"],row["right_pot33_mass"]])
        results[label]={
            "ensemble_size":len(pairs[0][0]),
            "pairs":pair_rows,
            "pairwise_mean_tv":_mean_ci([x["mean_tv"] for x in pair_rows]),
            "pairwise_p95_tv":_mean_ci([x["p95_tv"] for x in pair_rows]),
            "pairwise_argmax_disagreement":_mean_ci([x["argmax_disagreement"] for x in pair_rows]),
            "all_in_mass_across_disjoint_groups":{
                "group_ci":_mean_ci(group_all_in),
                "min":min(group_all_in),
                "max":max(group_all_in),
                "range":max(group_all_in)-min(group_all_in),
            },
            "pot33_mass_across_disjoint_groups":{
                "group_ci":_mean_ci(group_pot33),
                "min":min(group_pot33),
                "max":max(group_pot33),
                "range":max(group_pot33)-min(group_pot33),
            },
        }

    s1=results["size_1"]
    ratios={}
    for label in ("size_2","size_4"):
        x=results[label]
        ratios[label]={
            "mean_tv_ratio_to_single":float(
                x["pairwise_mean_tv"]["mean"]/max(s1["pairwise_mean_tv"]["mean"],1e-12)
            ),
            "p95_tv_ratio_to_single":float(
                x["pairwise_p95_tv"]["mean"]/max(s1["pairwise_p95_tv"]["mean"],1e-12)
            ),
            "argmax_ratio_to_single":float(
                x["pairwise_argmax_disagreement"]["mean"]/
                max(s1["pairwise_argmax_disagreement"]["mean"],1e-12)
            ),
            "all_in_range_ratio_to_single":float(
                x["all_in_mass_across_disjoint_groups"]["range"]/
                max(s1["all_in_mass_across_disjoint_groups"]["range"],1e-12)
            ),
            "pot33_range_ratio_to_single":float(
                x["pot33_mass_across_disjoint_groups"]["range"]/
                max(s1["pot33_mass_across_disjoint_groups"]["range"],1e-12)
            ),
        }

    out={
        "schema":"SPINCORE_LT2_HU_MATURE_ENSEMBLE_ROOT_STABILITY_V1",
        "checkpoint":{
            "path":str(args.checkpoint.resolve()),
            "iteration":int(iteration),
            "seed":int(seed),
            "adv_mem_items":len(runtime.bundle.adv_mem.items),
            "adv_mem_seen":int(runtime.bundle.adv_mem.seen),
        },
        "method":{
            "read_only_source_checkpoint":True,
            "new_training_roots":0,
            "source_training_memory_writes":0,
            "future_holdout_seeds_touched":False,
            "replicas":8,
            "budget_per_replica":int(args.budget),
            "ensemble_semantics":"mean raw Advantage outputs, then unchanged lean_regret_matching_policy",
            "disjoint_pairs":{k:[[list(a),list(b)] for a,b in v] for k,v in PAIR_GROUPS.items()},
            "same_root_corpus_for_all_models":True,
            "forensic_seeds":list(FORENSIC_SEEDS),
            "scenarios_per_seed":int(args.scenarios_per_seed),
            "hu_scenarios_by_seed":hu_counts,
            "root_count":len(corpus),
        },
        "replicas":replica_meta,
        "results":results,
        "ratios_to_single":ratios,
    }

    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== MATURE HU ADVANTAGE ENSEMBLE ROOT STABILITY ===")
    for label in ("size_1","size_2","size_4"):
        x=results[label]
        print(
            f"{label}: meanTV={x['pairwise_mean_tv']['mean']:.4f} "
            f"p95={x['pairwise_p95_tv']['mean']:.4f} "
            f"argmax={x['pairwise_argmax_disagreement']['mean']:.4f} "
            f"AI_range={x['all_in_mass_across_disjoint_groups']['range']:.4f} "
            f"P33_range={x['pot33_mass_across_disjoint_groups']['range']:.4f}"
        )
    print("LT2_HU_MATURE_ENSEMBLE_ROOT_STABILITY_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
