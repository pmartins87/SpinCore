#!/usr/bin/env python3
from __future__ import annotations

"""Quantify the historical semantic-sigma fallback mismatch.

Before this audit, the diagnostic semantic_sigma helper used a uniform fallback
when every legal averaged raw Advantage was non-positive.  The functional lean
deployment contract instead uses lean_regret_matching_policy(), whose fallback
is softmax over the legal raw values.

This script does not train anything.  It replays legacy semantic behavior
trajectories and compares, on every visited state:
- the historical uniform-fallback distribution actually used by the semantic
  research lane;
- the canonical lean distribution.

The exact final 10115 8k target-generation stream is replayed under the old
mapping as an identity guard.
"""

import argparse
import json
import math
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
import run_3h_v1_semantic_shadow_10105 as shadow
from audit_3h_semantic_sidecar_attribution_10105 import decode_obs, private_semantics

from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import LeanSolverState, apply_lean
from spincore.r7_5_action_cfr import legal_mask, sample_action
from spincore_nn.action_models import collate_action_observations
from spincore.solver import SolverLibrary

REPRESENTATION="C0_V1_FROZEN_CONTROL"
DOMAIN="THREE_HANDED"
FINAL_SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_ENSEMBLE_V1"
SOURCE_SHA=distill.EXPECTED_SHA
FINAL_TRAIN_SEED=20260927 ^ 0x10115A
GENERIC_SEED=20260927 ^ 0xFA11BAC
FOLD=0


def mix64(*values):
    return distill.mix64(*values)


def load_models(path:Path,kind:str):
    p=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if kind=="initial":
        if p.get("schema")!=distill.ADV_SCHEMA:
            raise RuntimeError("wrong initial semantic shadow schema")
        if int(p.get("budget",-1))!=1600:
            raise RuntimeError("initial semantic shadow budget mismatch")
        states=list(p.get("members") or [])
    else:
        if p.get("schema")!=FINAL_SCHEMA:
            raise RuntimeError("wrong semantic research ensemble schema")
        if p.get("source_checkpoint_sha256")!=SOURCE_SHA:
            raise RuntimeError("semantic research source mismatch")
        states=list(p.get("members") or [])
    if len(states)!=8:
        raise RuntimeError("expected eight semantic Advantage members")
    return [distill.load_semantic_advantage(s) for s in states]


def raw_ensemble(models,obs,legal):
    b=collate_action_observations(
        REPRESENTATION,[obs],[legal_mask(legal)],device="cpu"
    )
    class S: pass
    s=S();s.observation=obs
    sv=shadow.semantic_vector(s)
    sb=dict(b)
    sb["semantic"]=torch.tensor(
        np.asarray([sv],dtype=np.float32),
        dtype=torch.float32,
    )
    with torch.no_grad():
        raw=torch.stack([m(sb)[0] for m in models],dim=0).mean(dim=0)
    return tuple(float(x) for x in raw.detach().cpu().tolist())


def legacy_policy(raw,legal):
    positive=sum(max(0.0,float(raw[a])) for a in legal)
    out=[0.0]*10
    if positive<=0:
        for a in legal:
            out[a]=1.0/len(legal)
    else:
        for a in legal:
            out[a]=max(0.0,float(raw[a]))/positive
    return tuple(out),bool(positive<=0)


def tv(a,b,legal):
    return 0.5*sum(abs(float(a[x])-float(b[x])) for x in legal)


def argmax(policy,legal):
    return max(legal,key=lambda a:(float(policy[a]),-int(a)))


def select(policy,legal,u):
    acc=0.0
    for a in legal:
        acc+=float(policy[a])
        if u<acc:
            return int(a)
    return int(legal[-1])


def subgroup(obs,legal):
    hole,board,_n,_c=decode_obs(obs)
    ps=private_semantics(hole,board)
    return {
        "strong_fold_legal":bool(
            len(board)>=3 and int(ps["made"])>=3 and FOLD in legal
        ),
        "high_card_no_draw":bool(ps["high_card_no_draw"]),
        "made":int(ps["made"]),
    }


def summarize(values):
    if not values:
        return {"count":0}
    arr=np.asarray(values,dtype=np.float64)
    return {
        "count":len(values),
        "mean":float(arr.mean()),
        "median":float(np.median(arr)),
        "p90":float(np.quantile(arr,0.90)),
        "p95":float(np.quantile(arr,0.95)),
        "max":float(arr.max()),
    }


def replay(solver,models,*,episodes:int,seed:int,label:str):
    sampler=LegacyScenarioSampler(seed=seed,config=LegacyScenarioConfig())
    action_rng=random.Random(seed^0xAC710)
    decisions=0
    fallback=0
    tvs=[]
    fallback_tvs=[]
    argdiff=0
    samplediff=0
    strong=0
    strong_fallback=0
    hc=0
    hc_fallback=0
    by_made={}
    for ep_idx in range(int(episodes)):
        ep=sampler.sample_episode(force_domain=DOMAIN)
        raw_state=solver.create(ep,int(mix64(seed,ep_idx,0xDEC4)))
        state=LeanSolverState(raw_state)
        try:
            while not state.terminal:
                street=int(state.inner.neural_bytes_v2()[112])
                active=FIRST_RELEASE_ACTION_SPEC.active_mask(street)
                legal=tuple(int(x) for x in state.universal_legal_actions(active))
                obs=state.neural_bytes()
                raw=raw_ensemble(models,obs,legal)
                legacy,is_fallback=legacy_policy(raw,legal)
                canonical=lean_regret_matching_policy(raw,legal)
                distance=tv(legacy,canonical,legal)
                u=action_rng.random()
                a_old=select(legacy,legal,u)
                a_new=select(canonical,legal,u)

                decisions+=1
                tvs.append(distance)
                if is_fallback:
                    fallback+=1
                    fallback_tvs.append(distance)
                if argmax(legacy,legal)!=argmax(canonical,legal):
                    argdiff+=1
                if a_old!=a_new:
                    samplediff+=1

                sg=subgroup(obs,legal)
                made=sg["made"]
                if made>=3:
                    row=by_made.setdefault(str(made),{
                        "decisions":0,"fallback":0,"tv_sum":0.0
                    })
                    row["decisions"]+=1
                    row["fallback"]+=int(is_fallback)
                    row["tv_sum"]+=distance
                if sg["strong_fold_legal"]:
                    strong+=1
                    strong_fallback+=int(is_fallback)
                if sg["high_card_no_draw"]:
                    hc+=1
                    hc_fallback+=int(is_fallback)

                # Reproduce the historical semantic-lane path.
                apply_lean(state.inner,active,a_old)
        finally:
            state.close()

        if (ep_idx+1)%1000==0:
            print(
                f"FALLBACK_AUDIT {label} {ep_idx+1}/{episodes} "
                f"decisions={decisions} fallback={fallback}",
                flush=True,
            )

    for row in by_made.values():
        row["fallback_fraction"]=(
            row["fallback"]/row["decisions"] if row["decisions"] else 0.0
        )
        row["mean_tv"]=(
            row["tv_sum"]/row["decisions"] if row["decisions"] else 0.0
        )

    return {
        "label":label,
        "episodes":int(episodes),
        "seed":int(seed),
        "decisions":decisions,
        "all_nonpositive_fallback_count":fallback,
        "all_nonpositive_fallback_fraction":fallback/decisions if decisions else 0.0,
        "legacy_vs_canonical_tv_all":summarize(tvs),
        "legacy_vs_canonical_tv_fallback_only":summarize(fallback_tvs),
        "argmax_disagreement_count":argdiff,
        "argmax_disagreement_fraction":argdiff/decisions if decisions else 0.0,
        "same_u_sampled_action_difference_count":samplediff,
        "same_u_sampled_action_difference_fraction":samplediff/decisions if decisions else 0.0,
        "strong_hand_fold_legal_decisions":strong,
        "strong_hand_fallback_count":strong_fallback,
        "strong_hand_fallback_fraction":strong_fallback/strong if strong else 0.0,
        "high_card_no_draw_decisions":hc,
        "high_card_no_draw_fallback_count":hc_fallback,
        "high_card_no_draw_fallback_fraction":hc_fallback/hc if hc else 0.0,
        "made_category_ge3":by_made,
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--initial-ensemble",type=Path,required=True)
    ap.add_argument("--ensemble-10107",type=Path,required=True)
    ap.add_argument("--ensemble-10110",type=Path,required=True)
    ap.add_argument("--ensemble-10115",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    args=ap.parse_args()
    torch.set_num_threads(int(args.threads))

    solver=SolverLibrary(args.solver.resolve(strict=True))
    initial=load_models(args.initial_ensemble,"initial")
    e107=load_models(args.ensemble_10107,"final")
    e110=load_models(args.ensemble_10110,"final")
    e115=load_models(args.ensemble_10115,"final")

    rows=[]
    rows.append(replay(
        solver,initial,episodes=4000,seed=GENERIC_SEED,label="initial_shadow"
    ))
    rows.append(replay(
        solver,e107,episodes=4000,seed=GENERIC_SEED,label="semantic_10107"
    ))
    rows.append(replay(
        solver,e110,episodes=4000,seed=GENERIC_SEED,label="semantic_10110"
    ))
    rows.append(replay(
        solver,e115,episodes=8000,seed=FINAL_TRAIN_SEED,
        label="semantic_10115_exact_old_tail_stream"
    ))

    final=rows[-1]
    if int(final["decisions"])!=28418:
        raise RuntimeError(
            f"historical 10115 tail-stream identity drift: {final['decisions']} != 28418"
        )

    max_fallback=max(float(r["all_nonpositive_fallback_fraction"]) for r in rows)
    max_samplediff=max(float(r["same_u_sampled_action_difference_fraction"]) for r in rows)
    report={
        "schema":"SPINCORE_3H_SEMANTIC_FALLBACK_CONTRACT_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_HISTORICAL_UNIFORM_VS_CANONICAL_LEAN_SOFTMAX",
        "historical_bug":(
            "semantic_sigma used a uniform fallback when all legal raw averaged "
            "Advantages were non-positive; canonical lean uses softmax."
        ),
        "rows":rows,
        "max_fallback_fraction":max_fallback,
        "max_same_u_action_difference_fraction":max_samplediff,
        "decision_rule":{
            "no_material_effect":(
                "All audited fallback fractions <=0.001 and same-u action "
                "difference fractions <=0.0005."
            ),
            "rerun_required":(
                "Otherwise the semantic online lane must be replayed from frozen "
                "10105 under the corrected canonical lean mapping before further "
                "promotion/repair decisions."
            ),
        },
        "fallback_contract_material":bool(
            max_fallback>0.001 or max_samplediff>0.0005
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print("=== semantic fallback contract audit ===")
    for row in rows:
        print(json.dumps({
            "label":row["label"],
            "decisions":row["decisions"],
            "fallback_fraction":row["all_nonpositive_fallback_fraction"],
            "mean_tv":row["legacy_vs_canonical_tv_all"]["mean"],
            "same_u_action_diff":row["same_u_sampled_action_difference_fraction"],
        },sort_keys=True))
    print(f"fallback_contract_material={report['fallback_contract_material']}")
    print(f"report={args.report.resolve()}")
    print("SEMANTIC_FALLBACK_CONTRACT_AUDIT_COMPLETE")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
