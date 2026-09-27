#!/usr/bin/env python3
from __future__ import annotations

"""Independent confirmation of the post-hoc-selected 500-step fresh semantic distill.

The previous fresh-target bridge showed its best first-holdout point at 500
steps, but that milestone was selected after inspecting that holdout.  This
gate therefore regenerates the original deterministic training stream, trains
exactly 500 paired V1 / V1+semantic AveragePolicy steps, and evaluates only on
a completely new fresh-target episode stream produced by the same frozen
semantic Advantage ensemble.

No model selection is performed on the new stream.  Diagnostic only.
"""

import argparse
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as base
from audit_3h_average_policy_semantic_continuation_10105 import (
    build_semantic_from_v1,
    clone_optimizer_state_v1_to_semantic,
    semantic_vector_from_obs,
)
from spincore_nn.action_models import make_policy_action_model
from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.training import train_step
from spincore.solver import SolverLibrary

EXPECTED_SHA=base.EXPECTED_SHA
ADV_SCHEMA=base.ADV_SCHEMA
REPRESENTATION=base.REPRESENTATION
TRAIN_COLLECTION_EPISODES=8000
INDEPENDENT_EVAL_EPISODES=2000
SELECTED_STEPS=500
TRAIN_COLLECTION_SEED=base.MASTER_SEED
INDEPENDENT_EVAL_SEED=TRAIN_COLLECTION_SEED ^ 0x6EEDBEEF


def sha256(path:Path)->str:
    return base.sha256(path)


def load_adv_models(path:Path):
    payload=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if payload.get("schema")!=ADV_SCHEMA or int(payload.get("budget",-1))!=1600:
        raise SystemExit("wrong semantic Advantage artifact")
    models=[base.load_semantic_advantage(state) for state in payload["members"]]
    if len(models)!=8:
        raise SystemExit("expected eight semantic Advantage models")
    return payload,models


def initial_policy_pair(d3,cfg):
    _,v1=make_policy_action_model(REPRESENTATION,device="cpu",seed=0)
    v1.load_state_dict(d3["policy"])
    v1opt=torch.optim.Adam(v1.parameters(),lr=float(cfg["learning_rate"]))
    v1opt.load_state_dict(copy.deepcopy(d3["pol_opt"]))

    sem=build_semantic_from_v1(v1)
    semopt=torch.optim.Adam(sem.parameters(),lr=float(cfg["learning_rate"]))
    clone_optimizer_state_v1_to_semantic(v1,v1opt,sem,semopt)
    return v1,v1opt,sem,semopt


def train_500(v1,v1opt,sem,semopt,train,cfg):
    rng=random.Random(base.TRAIN_SEED)
    bs=int(cfg["batch_size"])
    for step in range(SELECTED_STEPS):
        idx=rng.sample(range(len(train)),min(bs,len(train)))
        samples=[train[i] for i in idx]
        b,t,w=vectorized_batch(samples,"cpu")
        sb=dict(b)
        sb["semantic"]=torch.tensor(
            np.asarray(
                [semantic_vector_from_obs(s.observation) for s in samples],
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )
        train_step(v1,v1opt,b,t,w,"strategy")
        train_step(sem,semopt,sb,t,w,"strategy")
        if (step+1)%100==0:
            print(f"INDEPENDENT500_TRAIN {step+1}/{SELECTED_STEPS}",flush=True)


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage-models",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--spin-bundle",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--out-model",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    args=ap.parse_args()

    torch.set_num_threads(int(args.threads))
    cp=args.checkpoint.resolve(strict=True)
    actual=sha256(cp)
    if actual!=EXPECTED_SHA:
        raise SystemExit("checkpoint SHA mismatch")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    d3=(payload.get("domains") or {}).get(base.DOMAIN) or {}
    cfg=dict(payload.get("config") or {})

    adv_payload,adv_models=load_adv_models(args.semantic_advantage_models)
    solver=SolverLibrary(args.solver.resolve(strict=True))

    # Reconstruct exactly the train side used by the milestone-selection run.
    base.MASTER_SEED=TRAIN_COLLECTION_SEED
    train,selection_holdout,train_collection=base.collect_fresh(
        solver,adv_models,TRAIN_COLLECTION_EPISODES
    )
    if len(train)!=23862 or len(selection_holdout)!=6095:
        raise RuntimeError(
            "original fresh-stream identity drift: "
            f"train={len(train)} holdout={len(selection_holdout)}"
        )

    # Build untouched step-0 reference plus the paired trainable arms.
    step0_v1,_,step0_sem,_=initial_policy_pair(d3,cfg)
    v1,v1opt,sem,semopt=initial_policy_pair(d3,cfg)
    train_500(v1,v1opt,sem,semopt,train,cfg)

    # Completely new chance/episode stream.  It is evaluation-only.
    base.MASTER_SEED=INDEPENDENT_EVAL_SEED
    eval_train,eval_hold,eval_collection=base.collect_fresh(
        solver,adv_models,INDEPENDENT_EVAL_EPISODES
    )
    independent=list(eval_train)+list(eval_hold)
    if len(independent)<5000:
        raise RuntimeError(f"independent eval too small: {len(independent)}")

    step0=base.metrics(step0_v1,step0_sem,independent)
    final=base.metrics(v1,sem,independent)
    hc=final["high_card_no_draw_allin"]
    if int(hc["count"])<150:
        raise RuntimeError(f"independent high-card subgroup too small: {hc['count']}")

    v1_ce=float(final["v1_weighted_ce"])
    sem_ce=float(final["semantic_weighted_ce"])
    v1_tv=float(final["v1_weighted_tv"])
    sem_tv=float(final["semantic_weighted_tv"])
    vb=float(hc["v1_abs_bias"])
    sb=float(hc["semantic_abs_bias"])
    step0_sem_ce=float(step0["semantic_weighted_ce"])

    criteria={
        "semantic_ce_at_least_10pct_better_than_v1":bool(sem_ce<=0.90*v1_ce),
        "semantic_tv_at_least_15pct_better_than_v1":bool(sem_tv<=0.85*v1_tv),
        "semantic_high_card_abs_bias_at_least_50pct_better":bool(
            vb>0.0 and sb<=0.50*vb
        ),
        "semantic_ce_at_least_10pct_better_than_step0":bool(
            sem_ce<=0.90*step0_sem_ce
        ),
    }
    passed=all(criteria.values())

    dc1=base.dc1(
        solver,
        args.spin_bundle.resolve(strict=True),
        v1,
        sem,
        200,
    )

    args.out_model.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema":"SPINCORE_3H_FRESH_SEMANTIC_STRATEGY_DISTILL_500_MODEL_V1",
        "checkpoint_sha256":actual,
        "semantic_advantage_schema":ADV_SCHEMA,
        "train_collection_seed":TRAIN_COLLECTION_SEED,
        "selected_steps":SELECTED_STEPS,
        "model_state":{k:v.detach().cpu() for k,v in sem.state_dict().items()},
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    },args.out_model)

    report={
        "schema":"SPINCORE_3H_FRESH_SEMANTIC_STRATEGY_DISTILL_INDEPENDENT500_V1",
        "scope":"DIAGNOSTIC_ONLY_PRECOMMITTED_500_INDEPENDENT_FRESH_TARGET_STREAM",
        "checkpoint_sha256":actual,
        "semantic_advantage_schema":ADV_SCHEMA,
        "selected_steps":SELECTED_STEPS,
        "train_collection_seed":TRAIN_COLLECTION_SEED,
        "independent_eval_seed":INDEPENDENT_EVAL_SEED,
        "train_collection":train_collection,
        "discarded_selection_holdout_samples":len(selection_holdout),
        "independent_eval_collection":eval_collection,
        "independent_eval_samples":len(independent),
        "step0_independent_eval":step0,
        "step500_independent_eval":final,
        "dc1":dc1,
        "precommitted_criteria":criteria,
        "independent500_pass":passed,
        "interpretation":(
            "The 500-step milestone was frozen before this independent stream was "
            "generated. PASS confirms that the semantic AveragePolicy advantage on "
            "fresh contemporaneous semantic-Advantage targets generalizes to new "
            "episodes rather than reflecting first-holdout milestone selection. "
            "PASS still authorizes only a bounded online semantic-CFR pilot."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )

    print("=== Independent 500-step fresh semantic distill ===")
    print("step0="+json.dumps(step0,sort_keys=True))
    print("step500="+json.dumps(final,sort_keys=True))
    print("criteria="+json.dumps(criteria,sort_keys=True))
    print(f"independent500_pass={passed}")
    print(f"model={args.out_model.resolve()}")
    print(f"report={args.report.resolve()}")
    print("3H_FRESH_SEMANTIC_STRATEGY_INDEPENDENT500_RUN_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
