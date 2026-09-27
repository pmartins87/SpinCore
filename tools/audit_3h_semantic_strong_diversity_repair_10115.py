#!/usr/bin/env python3
from __future__ import annotations

"""Controlled rare-strong-hand diversity repair for canonical semantic 10115.

The canonical coverage audit localized the remaining strong-made-hand Fold
failure to AveragePolicy generalization:
- only 44 strong Fold-legal states in the ordinary final training stream;
- all 44 were repeatedly sampled by the frozen 500-step optimizer;
- the tail policy fit those training states reasonably well but failed on new
  strong states.

This gate isolates *diversity* from mere class reweighting.

Three semantic AveragePolicy arms share the exact 10105 initialization,
optimizer state and 500-step budget:
1) BASELINE: the exact canonical 22,726-sample ordinary train stream;
2) REPEAT_CONTROL: BASELINE + 256 extra slots made by repeating the existing
   strong-hand train states;
3) DIVERSE_STRONG: BASELINE + 256 genuinely new strong-hand states collected
   from an independent 60,000-episode teacher stream.

REPEAT_CONTROL and DIVERSE_STRONG have identical dataset length and use the
same minibatch index sequence.  Therefore any difference between them is due
to unique strong-state content, not more strong-hand weight, more optimizer
steps, or a larger dataset.

A completely new 30,000-episode evaluation stream is generated after the arms
are frozen.  No model is promoted here.
"""

import argparse
import copy
import json
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_fresh_semantic_strategy_independent500_10105 as confirm
import audit_3h_semantic_tail_strong_hand_coverage_10115 as coverage
from audit_3h_average_policy_semantic_continuation_10105 import (
    semantic_vector_from_obs,
)

from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.training import train_step
from spincore.solver import SolverLibrary

EXPECTED_SHA=distill.EXPECTED_SHA
ADV_SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_ENSEMBLE_V1"
TAIL_SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_TAIL_POLICY_V1"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
DOMAIN="THREE_HANDED"
FINAL_ITERATION=10115

BASE_TRAIN_SEED=20260927 ^ 0x10115A
BASE_TRAIN_EPISODES=8000
EXPECTED_BASE_TRAIN=22726
EXPECTED_BASE_HOLD=5701

AUGMENT_EPISODES=60000
AUGMENT_SEED=20260928 ^ 0xD17E25
AUGMENT_SELECT_SEED=20260928 ^ 0x5E1EC7
EXTRA_STRONG_SLOTS=256

EVAL_EPISODES=30000
EVAL_SEED=20260928 ^ 0xE7A1D5
SELECTED_STEPS=500
FOLD=0


def load_adv(path:Path):
    p=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=ADV_SCHEMA:
        raise RuntimeError("wrong canonical semantic Advantage schema")
    if p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError("canonical semantic Advantage source mismatch")
    if int(p.get("completed_iteration",-1))!=FINAL_ITERATION:
        raise RuntimeError("canonical semantic Advantage iteration mismatch")
    states=list(p.get("members") or [])
    if len(states)!=8:
        raise RuntimeError("canonical semantic Advantage ensemble-size drift")
    return [distill.load_semantic_advantage(s) for s in states]


def load_tail_state(path:Path):
    p=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=TAIL_SCHEMA:
        raise RuntimeError("wrong canonical tail-policy schema")
    if p.get("source_checkpoint_sha256")!=EXPECTED_SHA:
        raise RuntimeError("canonical tail-policy source mismatch")
    if int(p.get("completed_iteration",-1))!=FINAL_ITERATION:
        raise RuntimeError("canonical tail-policy iteration mismatch")
    if int(p.get("selected_steps",-1))!=SELECTED_STEPS:
        raise RuntimeError("canonical tail-policy step-budget mismatch")
    return p["model_state"]


def strong_sample(sample)->bool:
    m=coverage.sample_meta(sample)
    return bool(
        int(m["street"])>0
        and int(m["made"])>=3
        and bool(m["fold_legal"])
    )


def state_key(sample):
    return (
        bytes(sample.observation),
        tuple(bool(x) for x in sample.legal),
    )


def describe_strong_samples(samples):
    metas=[coverage.sample_meta(s) for s in samples]
    cats={}
    streets={}
    target_fold=[]
    for s,m in zip(samples,metas):
        cats[str(int(m["made"]))]=cats.get(str(int(m["made"])),0)+1
        streets[str(int(m["street"]))]=streets.get(str(int(m["street"])),0)+1
        target_fold.append(float(s.target[FOLD]))
    return {
        "count":len(samples),
        "category_counts":cats,
        "street_counts":streets,
        "target_fold_mean":statistics.fmean(target_fold) if target_fold else None,
        "target_fold_max":max(target_fold) if target_fold else None,
        "target_fold_count_ge_0_10":sum(x>=0.10 for x in target_fold),
    }


def make_semantic_arm(d3,cfg):
    _v1,_v1opt,sem,semopt=confirm.initial_policy_pair(d3,cfg)
    return sem,semopt


def paired_train(
    repeat_model,
    repeat_opt,
    diverse_model,
    diverse_opt,
    repeat_train,
    diverse_train,
    cfg,
):
    if len(repeat_train)!=len(diverse_train):
        raise RuntimeError("paired diversity train lengths differ")
    rng=random.Random(distill.TRAIN_SEED)
    bs=int(cfg["batch_size"])
    for step in range(SELECTED_STEPS):
        idx=rng.sample(range(len(repeat_train)),min(bs,len(repeat_train)))

        rs=[repeat_train[i] for i in idx]
        rb,rt,rw=vectorized_batch(rs,"cpu")
        rsb=dict(rb)
        rsb["semantic"]=torch.tensor(
            np.asarray(
                [semantic_vector_from_obs(s.observation) for s in rs],
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )
        train_step(repeat_model,repeat_opt,rsb,rt,rw,"strategy")

        ds=[diverse_train[i] for i in idx]
        db,dt,dw=vectorized_batch(ds,"cpu")
        dsb=dict(db)
        dsb["semantic"]=torch.tensor(
            np.asarray(
                [semantic_vector_from_obs(s.observation) for s in ds],
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )
        train_step(diverse_model,diverse_opt,dsb,dt,dw,"strategy")

        if (step+1)%100==0:
            print(f"DIVERSITY_PAIRED_TRAIN {step+1}/{SELECTED_STEPS}",flush=True)


def max_state_diff(model,state):
    current=model.state_dict()
    if set(current)!=set(state):
        raise RuntimeError("tail reproduction state keys differ")
    out=0.0
    for key in current:
        a=current[key].detach().cpu()
        b=state[key].detach().cpu()
        out=max(out,float((a-b).abs().max().item()))
    return out


def eval_strong(v1,sem,samples):
    rows=coverage.predict_rows(v1,sem,samples)
    strong=coverage.strong_rows(rows)
    summary=coverage.subset_summary(strong)
    straight_flush=[
        r for r in strong if int(r["made"]) in (4,5)
    ]
    river=[r for r in strong if int(r["street"])==3]
    return {
        "overall":summary,
        "straight_or_flush":coverage.subset_summary(straight_flush),
        "river":coverage.subset_summary(river),
        "strata":coverage.strata(strong),
    }


def safe_ratio(a,b):
    if b is None or float(b)==0.0:
        return None
    return float(a)/float(b)


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage",type=Path,required=True)
    ap.add_argument("--canonical-tail-policy",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--out-model",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    args=ap.parse_args()
    torch.set_num_threads(int(args.threads))

    cp=args.checkpoint.resolve(strict=True)
    if distill.sha256(cp)!=EXPECTED_SHA:
        raise RuntimeError("source checkpoint SHA mismatch")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    d3=(payload.get("domains") or {}).get(DOMAIN) or {}
    cfg=dict(payload.get("config") or {})

    models=load_adv(args.semantic_advantage)
    canonical_tail_state=load_tail_state(args.canonical_tail_policy)
    solver=SolverLibrary(args.solver.resolve(strict=True))

    # Exact canonical ordinary train stream.
    distill.MASTER_SEED=BASE_TRAIN_SEED
    base_train,base_hold,base_collection=distill.collect_fresh(
        solver,models,BASE_TRAIN_EPISODES
    )
    if len(base_train)!=EXPECTED_BASE_TRAIN or len(base_hold)!=EXPECTED_BASE_HOLD:
        raise RuntimeError(
            f"canonical base-stream identity drift train={len(base_train)} "
            f"hold={len(base_hold)}"
        )
    base_strong=[s for s in base_train if strong_sample(s)]
    if len(base_strong)!=44:
        raise RuntimeError(
            f"canonical base strong-count drift: {len(base_strong)} != 44"
        )

    # Reproduce the exact canonical tail before applying any intervention.
    base_v1,base_v1opt,base_sem,base_semopt=confirm.initial_policy_pair(d3,cfg)
    confirm.train_500(
        base_v1,base_v1opt,base_sem,base_semopt,base_train,cfg
    )
    repro_diff=max_state_diff(base_sem,canonical_tail_state)
    if repro_diff>2e-6:
        raise RuntimeError(
            f"canonical tail 500-step reproduction drift: {repro_diff}"
        )
    print(f"DIVERSITY_BASELINE_REPRO_PASS max_state_diff={repro_diff:.9g}",flush=True)

    # Independent augmentation pool under the final canonical teacher.
    distill.MASTER_SEED=AUGMENT_SEED
    aug_a,aug_b,aug_collection=distill.collect_fresh(
        solver,models,AUGMENT_EPISODES
    )
    aug_all=list(aug_a)+list(aug_b)

    base_keys={state_key(s) for s in base_strong}
    diverse_candidates=[]
    seen=set()
    for s in aug_all:
        if not strong_sample(s):
            continue
        key=state_key(s)
        if key in base_keys or key in seen:
            continue
        seen.add(key)
        diverse_candidates.append(s)
    if len(diverse_candidates)<EXTRA_STRONG_SLOTS:
        raise RuntimeError(
            f"not enough novel strong states: {len(diverse_candidates)} "
            f"< {EXTRA_STRONG_SLOTS}"
        )

    srng=random.Random(AUGMENT_SELECT_SEED)
    chosen_idx=srng.sample(
        range(len(diverse_candidates)),EXTRA_STRONG_SLOTS
    )
    diverse_extra=[diverse_candidates[i] for i in chosen_idx]

    # Same class weight, same dataset length; repeat-control adds no new states.
    repeat_order=list(base_strong)
    srng.shuffle(repeat_order)
    repeat_extra=[
        repeat_order[i%len(repeat_order)]
        for i in range(EXTRA_STRONG_SLOTS)
    ]
    repeat_train=list(base_train)+repeat_extra
    diverse_train=list(base_train)+diverse_extra
    if len(repeat_train)!=len(diverse_train):
        raise RuntimeError("paired augmented dataset length drift")

    repeat_model,repeat_opt=make_semantic_arm(d3,cfg)
    diverse_model,diverse_opt=make_semantic_arm(d3,cfg)
    paired_train(
        repeat_model,repeat_opt,diverse_model,diverse_opt,
        repeat_train,diverse_train,cfg,
    )

    # Completely new evaluation stream, unseen by diagnosis and augmentation.
    distill.MASTER_SEED=EVAL_SEED
    eval_a,eval_b,eval_collection=distill.collect_fresh(
        solver,models,EVAL_EPISODES
    )
    evaluation=list(eval_a)+list(eval_b)
    strong_eval=[s for s in evaluation if strong_sample(s)]
    if len(strong_eval)<150:
        raise RuntimeError(
            f"new independent strong eval too small: {len(strong_eval)}"
        )

    baseline_global=distill.metrics(base_v1,base_sem,evaluation)
    # V1 reference is identical for all semantic arms; use base_v1.
    repeat_global=distill.metrics(base_v1,repeat_model,evaluation)
    diverse_global=distill.metrics(base_v1,diverse_model,evaluation)

    baseline_strong=eval_strong(base_v1,base_sem,strong_eval)
    repeat_strong=eval_strong(base_v1,repeat_model,strong_eval)
    diverse_strong=eval_strong(base_v1,diverse_model,strong_eval)

    bsb=float(baseline_strong["overall"]["semantic_tail_fold_abs_bias"])
    rsb=float(repeat_strong["overall"]["semantic_tail_fold_abs_bias"])
    dsb=float(diverse_strong["overall"]["semantic_tail_fold_abs_bias"])

    bs_sf=float(
        baseline_strong["straight_or_flush"]["semantic_tail_fold_abs_bias"]
    )
    ds_sf=float(
        diverse_strong["straight_or_flush"]["semantic_tail_fold_abs_bias"]
    )
    bs_r=float(baseline_strong["river"]["semantic_tail_fold_abs_bias"])
    ds_r=float(diverse_strong["river"]["semantic_tail_fold_abs_bias"])

    base_ce=float(baseline_global["semantic_weighted_ce"])
    div_ce=float(diverse_global["semantic_weighted_ce"])
    base_tv=float(baseline_global["semantic_weighted_tv"])
    div_tv=float(diverse_global["semantic_weighted_tv"])
    base_hc=float(
        baseline_global["high_card_no_draw_allin"]["semantic_abs_bias"]
    )
    div_hc=float(
        diverse_global["high_card_no_draw_allin"]["semantic_abs_bias"]
    )

    criteria={
        "new_eval_has_at_least_150_strong_states":len(strong_eval)>=150,
        "diverse_strong_bias_at_least_30pct_better_than_baseline":(
            bsb>0.0 and dsb<=0.70*bsb
        ),
        "diverse_strong_bias_at_least_15pct_better_than_repeat_control":(
            rsb>0.0 and dsb<=0.85*rsb
        ),
        "diverse_straight_flush_bias_at_least_30pct_better_than_baseline":(
            bs_sf>0.0 and ds_sf<=0.70*bs_sf
        ),
        "diverse_river_strong_bias_at_least_30pct_better_than_baseline":(
            bs_r>0.0 and ds_r<=0.70*bs_r
        ),
        "global_ce_not_worse_than_baseline_by_over_2pct":(
            div_ce<=1.02*base_ce
        ),
        "global_tv_not_worse_than_baseline_by_over_2pct":(
            div_tv<=1.02*base_tv
        ),
        "high_card_no_draw_abs_bias_not_worse_by_over_1pp":(
            div_hc<=base_hc+0.01
        ),
    }
    passed=all(criteria.values())

    args.out_model.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema":"SPINCORE_3H_SEMANTIC_STRONG_DIVERSITY_TAIL_CANDIDATE_V1",
        "source_checkpoint_sha256":EXPECTED_SHA,
        "semantic_completed_iteration":FINAL_ITERATION,
        "extra_unique_strong_states":EXTRA_STRONG_SLOTS,
        "selected_steps":SELECTED_STEPS,
        "model_state":{
            k:v.detach().cpu() for k,v in diverse_model.state_dict().items()
        },
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    },args.out_model)

    report={
        "schema":"SPINCORE_3H_SEMANTIC_STRONG_DIVERSITY_REPAIR_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_PAIRED_DIVERSITY_VS_REPEAT_CONTROL",
        "source_checkpoint_sha256":EXPECTED_SHA,
        "semantic_completed_iteration":FINAL_ITERATION,
        "base_collection":base_collection,
        "base_train_samples":len(base_train),
        "base_strong":describe_strong_samples(base_strong),
        "canonical_tail_reproduction_max_state_diff":repro_diff,
        "augmentation_collection":aug_collection,
        "augmentation_pool_samples":len(aug_all),
        "novel_strong_candidates":len(diverse_candidates),
        "extra_strong_slots":EXTRA_STRONG_SLOTS,
        "repeat_extra":describe_strong_samples(repeat_extra),
        "diverse_extra":describe_strong_samples(diverse_extra),
        "paired_augmented_train_samples":len(diverse_train),
        "independent_eval_collection":eval_collection,
        "independent_eval_samples":len(evaluation),
        "independent_strong_samples":len(strong_eval),
        "baseline":{
            "global":baseline_global,
            "strong":baseline_strong,
        },
        "repeat_control":{
            "global":repeat_global,
            "strong":repeat_strong,
        },
        "diverse_strong":{
            "global":diverse_global,
            "strong":diverse_strong,
        },
        "effect_sizes":{
            "diverse_vs_baseline_strong_abs_bias_ratio":safe_ratio(dsb,bsb),
            "diverse_vs_repeat_strong_abs_bias_ratio":safe_ratio(dsb,rsb),
            "diverse_vs_baseline_straight_flush_bias_ratio":safe_ratio(ds_sf,bs_sf),
            "diverse_vs_baseline_river_bias_ratio":safe_ratio(ds_r,bs_r),
            "diverse_vs_baseline_global_ce_ratio":safe_ratio(div_ce,base_ce),
            "diverse_vs_baseline_global_tv_ratio":safe_ratio(div_tv,base_tv),
            "diverse_minus_baseline_hcdn_abs_bias":div_hc-base_hc,
        },
        "precommitted_criteria":criteria,
        "semantic_strong_diversity_repair_pass":passed,
        "interpretation":(
            "PASS means adding genuinely new teacher-labelled strong-hand states "
            "improves unseen strong-hand Fold calibration beyond an equal-weight "
            "control that merely repeats already-known strong states, without "
            "materially regressing global or high-card/no-draw fit. PASS authorizes "
            "an exact DC1 1k candidate replay; it does not authorize production or DC2."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )

    print("=== semantic strong-hand diversity repair ===")
    print("baseline_strong="+json.dumps(baseline_strong["overall"],sort_keys=True))
    print("repeat_strong="+json.dumps(repeat_strong["overall"],sort_keys=True))
    print("diverse_strong="+json.dumps(diverse_strong["overall"],sort_keys=True))
    print("criteria="+json.dumps(criteria,sort_keys=True))
    print(f"semantic_strong_diversity_repair_pass={passed}")
    print(f"report={args.report.resolve()}")
    print("SEMANTIC_STRONG_DIVERSITY_REPAIR_AUDIT_COMPLETE")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
