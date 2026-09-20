#!/usr/bin/env python3
from __future__ import annotations

"""Paired fresh-400-refit HU root-policy audit for iteration 7600 vs 8000 reservoirs."""

import argparse, json, math, statistics, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python")); sys.path.insert(0,str(ROOT/"tools"))

import torch
import audit_lt2_hu_behavior_first_divergence as behfd
import audit_lt2_stage_a_b_first_divergence as fd
from spincore.lean_functional_training import load_checkpoint
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore.solver import SolverLibrary

SEEDS=(20260920,20260921,20260922,20260923,20260924,20260925)
DOMAIN="TRUE_HEADS_UP"

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--stage-a",type=Path,required=True)
    p.add_argument("--stage-b",type=Path,required=True)
    p.add_argument("--replicates",type=int,default=3)
    p.add_argument("--budget",type=int,default=400)
    p.add_argument("--scenarios-per-seed",type=int,default=5000)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--report",type=Path,required=True)
    return p.parse_args()

def ci(xs):
    xs=[float(x) for x in xs]; n=len(xs); m=float(statistics.fmean(xs))
    se=0.0 if n<=1 else float(statistics.stdev(xs)/math.sqrt(n)); h=1.96*se
    return {"n":n,"mean":m,"sem":se,"ci95_low":m-h,"ci95_high":m+h}

def cluster(rows,getter):
    by={}
    for r in rows: by.setdefault(int(r["seed"]),[]).append(float(getter(r)))
    return ci([statistics.fmean(v) for _,v in sorted(by.items())])

def eval_pair(*,solver,stage_a,stage_b,scenarios):
    rows=[]; hu_counts={}
    for seed in SEEDS:
        sampler=LegacyScenarioSampler(seed=int(seed)^0x5CE0A710,config=LegacyScenarioConfig())
        n=0
        for idx in range(int(scenarios)):
            ep=sampler.sample_episode()
            if not ep.game_is_hu: continue
            n+=1
            deal_seed=fd._mix64(int(seed),int(idx),0xD34A1)
            st=solver.create(ep,int(deal_seed))
            try:
                _ma,la,pa=behfd._behavior_distribution(stage_a,st)
                _mb,lb,pb=behfd._behavior_distribution(stage_b,st)
                if la!=lb: raise RuntimeError("root legal drift")
                rows.append({
                    "seed":int(seed),"scenario":int(idx),
                    "pa":[float(x) for x in pa],"pb":[float(x) for x in pb],
                    "tv":0.5*sum(abs(float(pa[i])-float(pb[i])) for i in range(10)),
                    "argmax_a":max(la,key=lambda i:(float(pa[i]),-int(i))),
                    "argmax_b":max(lb,key=lambda i:(float(pb[i]),-int(i))),
                })
            finally: st.close()
        hu_counts[str(seed)]=n
    out={
        "n":len(rows),"hu_counts":hu_counts,
        "tv":cluster(rows,lambda r:r["tv"]),
        "argmax_disagreement":cluster(rows,lambda r:1.0 if r["argmax_a"]!=r["argmax_b"] else 0.0),
        "a_mass":{},"b_mass":{},"b_minus_a_mass":{},
    }
    for slot in (0,1,3,9):
        name=NAME_BY_SLOT[slot]
        out["a_mass"][name]=cluster(rows,lambda r,s=slot:r["pa"][s])
        out["b_mass"][name]=cluster(rows,lambda r,s=slot:r["pb"][s])
        out["b_minus_a_mass"][name]=cluster(rows,lambda r,s=slot:r["pb"][s]-r["pa"][s])
    return out

class View:
    def __init__(self,runtime): self.runtime=runtime

def fit(runtime,*,init_seed,batch_seed,budget,batch_size,lr):
    runtime.session.batch_mode="vectorized"
    runtime.session.reset_advantage_network(init_seed=int(init_seed),lr=float(lr))
    runtime.bundle.batch_rng.seed(int(batch_seed))
    t=time.perf_counter()
    losses=runtime.session.train_advantage(steps=int(budget),batch_size=int(batch_size))
    return {"seconds":time.perf_counter()-t,"loss_last":float(losses[-1])}

def main():
    args=parse_args()
    if min(args.replicates,args.budget,args.scenarios_per_seed,args.threads)<=0: raise SystemExit("positive args required")
    torch.set_num_threads(int(args.threads))
    solver=SolverLibrary(args.solver.resolve(strict=True))
    sa,ca,ia,_sama,ra,_ha,_fa=load_checkpoint(args.stage_a.resolve(strict=True),solver=solver)
    sb,cb,ib,_samb,rb,_hb,_fb=load_checkpoint(args.stage_b.resolve(strict=True),solver=solver)
    if int(ia)!=7600 or int(ib)!=8000: raise RuntimeError(f"unexpected iterations {ia},{ib}")
    a=ra[DOMAIN]; b=rb[DOMAIN]
    if int(ca.batch_size)!=int(cb.batch_size): raise RuntimeError("batch size drift")
    if abs(float(ca.learning_rate)-float(cb.learning_rate))>1e-15: raise RuntimeError("lr drift")
    va,vb=View(a),View(b)

    production=eval_pair(solver=solver,stage_a=va,stage_b=vb,scenarios=args.scenarios_per_seed)
    trials=[]
    for rep in range(int(args.replicates)):
        init=(fd._mix64(20260920,args.budget,rep,0xA11CE)&0x7fffffff)
        batch=(fd._mix64(20260920,args.budget,rep,0xBA7C4)&0x7fffffff)
        print(f"PAIRED_REFIT rep={rep+1}/{args.replicates} init={init} batch={batch}",flush=True)
        fita=fit(a,init_seed=init,batch_seed=batch,budget=args.budget,batch_size=ca.batch_size,lr=ca.learning_rate)
        fitb=fit(b,init_seed=init,batch_seed=batch,budget=args.budget,batch_size=cb.batch_size,lr=cb.learning_rate)
        ev=eval_pair(solver=solver,stage_a=va,stage_b=vb,scenarios=args.scenarios_per_seed)
        trials.append({"replicate":rep,"init_seed":init,"batch_seed":batch,"fit_a":fita,"fit_b":fitb,"root_eval":ev})
        d=ev["b_minus_a_mass"]
        print(f"  ALL_IN B-A={d['ALL_IN']['mean']:+.4f} POT_33 B-A={d['POT_33']['mean']:+.4f} TV={ev['tv']['mean']:.4f}",flush=True)

    def reps(path):
        vals=[]
        for t in trials:
            x=t["root_eval"]
            for k in path: x=x[k]
            vals.append(float(x))
        return ci(vals)

    aggregate={
        "paired_all_in_delta_replicate_ci":reps(["b_minus_a_mass","ALL_IN","mean"]),
        "paired_pot33_delta_replicate_ci":reps(["b_minus_a_mass","POT_33","mean"]),
        "paired_tv_replicate_ci":reps(["tv","mean"]),
        "stage_a_all_in_mass_across_refits":reps(["a_mass","ALL_IN","mean"]),
        "stage_b_all_in_mass_across_refits":reps(["b_mass","ALL_IN","mean"]),
        "stage_a_pot33_mass_across_refits":reps(["a_mass","POT_33","mean"]),
        "stage_b_pot33_mass_across_refits":reps(["b_mass","POT_33","mean"]),
    }
    out={
        "schema":"SPINCORE_LT2_HU_PAIRED_FRESH400_ROOT_REFIT_V1",
        "stage_a":{"checkpoint":str(args.stage_a.resolve()),"iteration":int(ia),"adv_mem_items":len(a.bundle.adv_mem.items),"adv_mem_seen":int(a.bundle.adv_mem.seen)},
        "stage_b":{"checkpoint":str(args.stage_b.resolve()),"iteration":int(ib),"adv_mem_items":len(b.bundle.adv_mem.items),"adv_mem_seen":int(b.bundle.adv_mem.seen)},
        "method":{"read_only_source_checkpoints":True,"new_training_roots":0,"source_training_memory_writes":0,"future_holdout_seeds_touched":False,"in_memory_optimizer_steps":2*int(args.budget)*int(args.replicates),"budget":int(args.budget),"replicates":int(args.replicates),"same_init_and_batch_seed_across_A_B_within_trial":True,"scenarios_per_seed":int(args.scenarios_per_seed),"forensic_seeds":list(SEEDS)},
        "production_checkpoint_pair":production,
        "trials":trials,
        "aggregate":aggregate,
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("LT2_HU_PAIRED_FRESH400_ROOT_REFIT_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0
if __name__=="__main__": raise SystemExit(main())
