#!/usr/bin/env python3
from __future__ import annotations

"""Same-memory HU root-policy stability across cumulative Advantage fit budgets."""

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


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--budgets",default="400,800,1600,3200")
    p.add_argument("--replicates",type=int,default=4)
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


def _evaluate_model(model, corpus, batch_size):
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
            raw=model(batch).detach().cpu().tolist()
            probs=[
                lean_regret_matching_policy(row,r["legal"])
                for row,r in zip(raw,block)
            ]
            chunks.append(torch.tensor(probs,dtype=torch.float32))
    return torch.cat(chunks,dim=0)


def _pair_metrics(a,b):
    tv=0.5*torch.abs(a-b).sum(dim=1)
    arg=(torch.argmax(a,dim=1)!=torch.argmax(b,dim=1)).float()
    return {
        "mean_tv":float(tv.mean().item()),
        "p95_tv":float(torch.quantile(tv,torch.tensor(0.95)).item()),
        "argmax_disagreement":float(arg.mean().item()),
    }


def _budget_summary(preds, replica_rows):
    pairs=[]
    for i in range(len(preds)):
        for j in range(i+1,len(preds)):
            row=_pair_metrics(preds[i],preds[j])
            row["left"]=i
            row["right"]=j
            pairs.append(row)
    all_in=[float(p[:,9].mean().item()) for p in preds]
    pot33=[float(p[:,3].mean().item()) for p in preds]
    check_call=[float(p[:,1].mean().item()) for p in preds]
    fold=[float(p[:,0].mean().item()) for p in preds]
    return {
        "replicas":replica_rows,
        "pairs":pairs,
        "pairwise_mean_tv":_mean_ci([p["mean_tv"] for p in pairs]),
        "pairwise_p95_tv":_mean_ci([p["p95_tv"] for p in pairs]),
        "pairwise_argmax_disagreement":_mean_ci([p["argmax_disagreement"] for p in pairs]),
        "all_in_mass":{
            "replicate_ci":_mean_ci(all_in),
            "min":min(all_in),"max":max(all_in),"range":max(all_in)-min(all_in),
        },
        "pot33_mass":{
            "replicate_ci":_mean_ci(pot33),
            "min":min(pot33),"max":max(pot33),"range":max(pot33)-min(pot33),
        },
        "check_call_mass":{"replicate_ci":_mean_ci(check_call)},
        "fold_mass":{"replicate_ci":_mean_ci(fold)},
    }


def main():
    args=parse_args()
    budgets=sorted({int(x) for x in str(args.budgets).split(",") if x.strip()})
    if not budgets or budgets[0]<=0 or args.replicates<2 or args.scenarios_per_seed<=0:
        raise SystemExit("invalid positive budgets/replicates/scenarios")
    if any(b<=a for a,b in zip(budgets,budgets[1:])):
        raise SystemExit("budgets must be strictly increasing")

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
    predictions={str(b):[] for b in budgets}
    replica_meta={str(b):[] for b in budgets}

    for rep in range(int(args.replicates)):
        init_seed=fd._mix64(20260920,rep,0xB0D637)&0x7fffffff
        batch_seed=fd._mix64(20260920,rep,0xBA7C4)&0x7fffffff
        runtime.session.reset_advantage_network(
            init_seed=int(init_seed),lr=float(config.learning_rate)
        )
        runtime.bundle.batch_rng.seed(int(batch_seed))
        previous=0
        elapsed_total=0.0
        for budget in budgets:
            segment=int(budget)-int(previous)
            started=time.perf_counter()
            losses=runtime.session.train_advantage(
                steps=segment,batch_size=int(config.batch_size)
            )
            elapsed_total+=time.perf_counter()-started
            pred=_evaluate_model(runtime.bundle.advantage,corpus,args.eval_batch_size)
            predictions[str(budget)].append(pred)
            replica_meta[str(budget)].append({
                "replicate":rep,
                "init_seed":int(init_seed),
                "batch_seed":int(batch_seed),
                "cumulative_steps":int(budget),
                "segment_steps":int(segment),
                "cumulative_fit_seconds":float(elapsed_total),
                "loss_last":float(losses[-1]),
                "all_in_mass":float(pred[:,9].mean().item()),
                "pot33_mass":float(pred[:,3].mean().item()),
                "check_call_mass":float(pred[:,1].mean().item()),
                "fold_mass":float(pred[:,0].mean().item()),
            })
            print(
                f"rep={rep+1}/{args.replicates} steps={budget} "
                f"loss={losses[-1]:.6f} "
                f"AI={pred[:,9].mean().item():.4f} "
                f"P33={pred[:,3].mean().item():.4f}",
                flush=True,
            )
            previous=int(budget)

    by_budget={}
    for b in budgets:
        by_budget[str(b)]=_budget_summary(
            predictions[str(b)],replica_meta[str(b)]
        )

    base=by_budget[str(budgets[0])]
    ratios={}
    for b in budgets[1:]:
        cur=by_budget[str(b)]
        ratios[str(b)]={
            "pairwise_mean_tv_ratio_to_400":float(
                cur["pairwise_mean_tv"]["mean"]/
                max(base["pairwise_mean_tv"]["mean"],1e-12)
            ),
            "pairwise_p95_tv_ratio_to_400":float(
                cur["pairwise_p95_tv"]["mean"]/
                max(base["pairwise_p95_tv"]["mean"],1e-12)
            ),
            "all_in_range_ratio_to_400":float(
                cur["all_in_mass"]["range"]/
                max(base["all_in_mass"]["range"],1e-12)
            ),
            "pot33_range_ratio_to_400":float(
                cur["pot33_mass"]["range"]/
                max(base["pot33_mass"]["range"],1e-12)
            ),
        }

    out={
        "schema":"SPINCORE_LT2_HU_REFIT_BUDGET_ROOT_STABILITY_V1",
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
            "budgets":budgets,
            "replicates":int(args.replicates),
            "same_fit_trajectory_across_cumulative_budgets":True,
            "same_root_corpus_for_all_models":True,
            "forensic_seeds":list(FORENSIC_SEEDS),
            "scenarios_per_seed":int(args.scenarios_per_seed),
            "hu_scenarios_by_seed":hu_counts,
            "root_count":len(corpus),
        },
        "by_budget":by_budget,
        "ratios_vs_400":ratios,
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== HU REFIT BUDGET ROOT STABILITY ===")
    for b in budgets:
        x=by_budget[str(b)]
        print(
            f"steps={b} pairTV={x['pairwise_mean_tv']['mean']:.4f} "
            f"p95={x['pairwise_p95_tv']['mean']:.4f} "
            f"argmax={x['pairwise_argmax_disagreement']['mean']:.4f} "
            f"AI_range={x['all_in_mass']['range']:.4f} "
            f"P33_range={x['pot33_mass']['range']:.4f}"
        )
    print("LT2_HU_REFIT_BUDGET_ROOT_STABILITY_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
