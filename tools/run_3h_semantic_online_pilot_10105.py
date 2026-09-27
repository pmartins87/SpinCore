#!/usr/bin/env python3
from __future__ import annotations

"""Bounded 3H online V1+semantic CFR feedback pilot from frozen 10105.

This is the first online feedback test after the representation diagnosis closed.

Scope:
- THREE_HANDED only;
- source 10105 checkpoint remains read-only;
- exact source 2M 3H Advantage reservoir is restored in memory;
- initial behavior is the independently validated eight-member semantic
  Advantage shadow;
- two bounded online iterations, 64 roots each;
- after each root block, all eight semantic members are refit fresh for 1600
  steps using the same fixed member init/batch seeds;
- old fixed 50k controlled-split positions remain excluded from Advantage
  minibatch sampling, matching the validated shadow fit contract;
- after the second online iteration, the final semantic ensemble generates a
  fresh strategy-target stream;
- the already independently confirmed 500-step AveragePolicy distillation
  budget is reused without new milestone selection;
- final policy is checked on another independent fresh-target stream and on
  fixed DC1 development states.

No HU training, no source checkpoint mutation, no sealed benchmark promotion,
and no production authorization occurs.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_fresh_semantic_strategy_independent500_10105 as confirm
import run_3h_v1_semantic_shadow_10105 as shadow

from spincore.decision_sanity import sanity_flags
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_functional_training import (
    _root_deck_seed,
    _root_policy_seed,
    load_checkpoint,
)
from spincore.lean_solver_actions import LeanLegacyActionCollector
from spincore.lean_training_scope import LeanTrainingScope
from spincore.r7_5_action_cfr import legal_mask
from spincore.solver import SolverLibrary
from spincore_nn.action_models import collate_action_observations
from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.training import train_step

EXPECTED_SHA=distill.EXPECTED_SHA
ADV_SCHEMA=distill.ADV_SCHEMA
PROBE_SCHEMA="SPINCORE_3H_ADVANTAGE_CONTROLLED_SPLIT_PROBE_V1"
DOMAIN="THREE_HANDED"
ONLINE_ITERATIONS=2
ROOTS_PER_ITERATION=64
MEMBERS=8
MEMBER_STEPS=1600
POLICY_TRAIN_EPISODES=8000
POLICY_EVAL_EPISODES=2000
POLICY_STEPS=500
SEM_DIM=30
ONLINE_POLICY_TRAIN_SEED=20260927 ^ 0x0A11CE
ONLINE_POLICY_EVAL_SEED=20260927 ^ 0xE7A1
ALL_IN="ALL_IN"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


class _Sink:
    def __init__(self):
        self.items=[]
        self.seen=0
    def add(self,item):
        self.items.append(item)
        self.seen+=1


class SemanticEnsemblePolicy:
    def __init__(self,models):
        self.models=list(models)
    def __call__(self,_state,observation:bytes,legal:tuple[int,...]):
        return distill.semantic_sigma(self.models,observation,legal)


def load_semantic_artifact(path:Path):
    p=torch.load(path.resolve(strict=True),map_location="cpu",weights_only=False)
    if p.get("schema")!=ADV_SCHEMA:
        raise RuntimeError(f"wrong semantic Advantage schema: {p.get('schema')!r}")
    if str(p.get("source_checkpoint_sha256"))!=EXPECTED_SHA:
        raise RuntimeError("semantic Advantage source checkpoint mismatch")
    if int(p.get("budget",-1))!=MEMBER_STEPS:
        raise RuntimeError("semantic Advantage member-step mismatch")
    states=list(p.get("members") or [])
    meta=list(p.get("member_meta") or [])
    if len(states)!=MEMBERS or len(meta)!=MEMBERS:
        raise RuntimeError("semantic Advantage ensemble size mismatch")
    models=[distill.load_semantic_advantage(s) for s in states]
    for row in meta:
        if int(row.get("steps",-1))!=MEMBER_STEPS:
            raise RuntimeError("semantic member metadata step drift")
    return p,models,meta


def precompute_semantics(memory):
    rows=np.empty((len(memory.items),SEM_DIM),dtype=np.float32)
    started=time.perf_counter()
    for i,s in enumerate(memory.items):
        rows[i]=shadow.semantic_vector(s)
        if (i+1)%200000==0:
            print(f"ONLINE_SEMANTIC_PRECOMPUTE {i+1}/{len(memory.items)}",flush=True)
    def observer(index,sample):
        rows[int(index)]=shadow.semantic_vector(sample)
    memory.set_write_observer(observer)
    return rows,float(time.perf_counter()-started)


def fit_ensemble(memory,sem_rows,train_pool,member_meta,*,lr:float,batch_size:int,iteration:int):
    models=[]
    states=[]
    reports=[]
    overall=time.perf_counter()
    for member,row in enumerate(member_meta):
        init_seed=int(row["init_seed"])
        batch_seed=int(row["batch_seed"])
        model=shadow.make_semantic_model(init_seed)
        opt=torch.optim.Adam(model.parameters(),lr=float(lr))
        rng=random.Random(batch_seed)
        losses=[]
        started=time.perf_counter()
        for step in range(MEMBER_STEPS):
            pos=rng.sample(range(len(train_pool)),min(int(batch_size),len(train_pool)))
            idx=[train_pool[p] for p in pos]
            samples=[memory.items[i] for i in idx]
            b,t,w=vectorized_batch(samples,"cpu")
            sb=dict(b)
            sb["semantic"]=torch.tensor(
                np.asarray(sem_rows[idx],dtype=np.float32),
                dtype=torch.float32,
            )
            losses.append(train_step(model,opt,sb,t,w,"advantage"))
        elapsed=float(time.perf_counter()-started)
        model.eval()
        models.append(model)
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        reports.append({
            "member":member,
            "init_seed":init_seed,
            "batch_seed":batch_seed,
            "steps":MEMBER_STEPS,
            "loss_last":float(losses[-1]),
            "fit_seconds":elapsed,
        })
        print(
            "ONLINE_SEMANTIC_MEMBER "
            +json.dumps({
                "iteration":int(iteration),
                "member":member,
                "loss_last":float(losses[-1]),
                "fit_seconds":elapsed,
            },sort_keys=True),
            flush=True,
        )
    return models,states,reports,float(time.perf_counter()-overall)


def collect_roots(
    *,
    solver,
    sampler,
    memory,
    models,
    seed:int,
    source_config,
    iteration:int,
):
    behavior=SemanticEnsemblePolicy(models)
    dummy=_Sink()
    collector=LeanLegacyActionCollector(
        action_spec=FIRST_RELEASE_ACTION_SPEC,
        selected_representation="C0_V1_FROZEN_CONTROL",
        policy=behavior,
        terminal_utility=LeanTrainingScope().terminal_utility,
        rng=random.Random(0),
        advantage_memory=memory,
        strategy_memory=dummy,
    )
    nodes=0
    samples=0
    started=time.perf_counter()
    seen_before=int(memory.seen)
    for local_root in range(ROOTS_PER_ITERATION):
        episode=sampler.sample_episode(force_domain=DOMAIN)
        deck_seed=_root_deck_seed(seed,DOMAIN,iteration,local_root)
        policy_seed=_root_policy_seed(seed,DOMAIN,iteration,local_root)
        collector.rng=random.Random(int(policy_seed))
        live=[i for i,stack in enumerate(episode.stacks) if stack>0]
        root_samples=0
        root_nodes=0
        for player in live:
            root=solver.create(episode,int(deck_seed))
            try:
                result=collector.collect_advantage_partial_exact(
                    root,
                    traverser=int(player),
                    iteration=int(iteration),
                    exact_opponent_levels=int(source_config.exact_opponent_levels),
                )
            finally:
                root.close()
            root_nodes+=int(result.nodes)
            root_samples+=int(result.samples_added)
        nodes+=root_nodes
        samples+=root_samples
        if (local_root+1)%8==0:
            elapsed=time.perf_counter()-started
            projected=elapsed/(local_root+1)*ROOTS_PER_ITERATION
            print(
                "ONLINE_SEMANTIC_ROOTS "
                +json.dumps({
                    "iteration":int(iteration),
                    "completed":local_root+1,
                    "total":ROOTS_PER_ITERATION,
                    "nodes":nodes,
                    "samples":samples,
                    "projected_tree_minutes":projected/60.0,
                },sort_keys=True),
                flush=True,
            )
    if int(memory.seen)-seen_before!=samples:
        raise RuntimeError("online Advantage reservoir seen-count drift")
    return {
        "roots":ROOTS_PER_ITERATION,
        "nodes":nodes,
        "advantage_samples":samples,
        "seconds":float(time.perf_counter()-started),
    }


def semantic_batch_from_obs(obs:bytes,legal):
    b=collate_action_observations(
        "C0_V1_FROZEN_CONTROL",[obs],[legal_mask(tuple(legal))],device="cpu"
    )
    class S: pass
    s=S();s.observation=obs
    sv=shadow.semantic_vector(s)
    sb=dict(b)
    sb["semantic"]=torch.tensor(
        np.asarray([sv],dtype=np.float32),
        dtype=torch.float32,
    )
    return sb


def advantage_dc1_summary(models,traces):
    all_rows=[]
    high=[]
    nodraw=[]
    trips=[]
    for tr in traces:
        if tr.policy_id!=shadow.SPINCORE_POLICY_ID or tr.domain!=DOMAIN:
            continue
        d=dict(tr.policy_detail or {})
        obs_hex=d.get("observation_hex")
        legal=tuple(int(x) for x in d.get("legal_slots") or ())
        if not obs_hex or not legal:
            raise RuntimeError("missing captured 3H state in online pilot")
        sb=semantic_batch_from_obs(bytes.fromhex(obs_hex),legal)
        ps=shadow.policy_stats(models,sb,legal)
        all_rows.append(ps)
        for flag in sanity_flags(tr):
            if flag.code=="POSTFLOP_DEEP_HIGH_CARD_JAM":
                high.append(ps)
                if not bool((flag.context or {}).get("has_immediate_straight_or_flush_draw")):
                    nodraw.append(ps)
            elif flag.code=="POSTFLOP_TRIPS_PLUS_FOLD":
                trips.append(ps)

    def weird(rows,action):
        if not rows:
            return {"count":0}
        return {
            "count":len(rows),
            "raw_ensemble_action_probability_mean":statistics.fmean(
                r["raw_ensemble_policy"].get(action,0.0) for r in rows
            ),
            "policy_mixture_action_probability_mean":statistics.fmean(
                r["policy_mixture"].get(action,0.0) for r in rows
            ),
            "majority_member_argmax_action_count":sum(
                sum(x==action for x in r["member_argmax"])>=5 for r in rows
            ),
            "unanimous_member_argmax_action_count":sum(
                all(x==action for x in r["member_argmax"]) for r in rows
            ),
        }

    return {
        "three_handed_decisions":len(all_rows),
        "pairwise_tv_mean":statistics.fmean(r["pairwise_tv_mean"] for r in all_rows),
        "pairwise_argmax_disagreement_mean":statistics.fmean(
            r["pairwise_argmax_disagreement"] for r in all_rows
        ),
        "high_card_jam_all":weird(high,"ALL_IN"),
        "high_card_jam_no_immediate_draw":weird(nodraw,"ALL_IN"),
        "trips_plus_fold":weird(trips,"FOLD"),
    }


def policy_pair_from_source(source_bundle,source_config):
    mini={
        "policy":{k:v.detach().cpu().clone() for k,v in source_bundle.policy.state_dict().items()},
        "pol_opt":copy.deepcopy(source_bundle.pol_opt.state_dict()),
    }
    cfg={
        "learning_rate":float(source_config.learning_rate),
        "batch_size":int(source_config.batch_size),
    }
    return mini,cfg


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--semantic-advantage-models",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--spin-bundle",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--out-ensemble",type=Path,required=True)
    ap.add_argument("--out-policy",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    args=ap.parse_args()

    torch.set_num_threads(int(args.threads))
    source=args.checkpoint.resolve(strict=True)
    actual=sha256(source)
    if actual!=EXPECTED_SHA:
        raise SystemExit(f"checkpoint SHA mismatch: {actual}")

    solver=SolverLibrary(args.solver.resolve(strict=True))
    seed,source_config,completed,sampler,runtimes,_history,_finalized=load_checkpoint(
        source,solver=solver
    )
    if int(completed)!=10105:
        raise RuntimeError(f"expected source iteration 10105, got {completed}")
    r3=runtimes[DOMAIN]
    if len(r3.bundle.adv_mem.items)!=2_000_000:
        raise RuntimeError("expected saturated 2M 3H Advantage reservoir")

    sem_payload,models,member_meta=load_semantic_artifact(
        args.semantic_advantage_models
    )

    probe=torch.load(args.probe.resolve(strict=True),map_location="cpu",weights_only=False)
    if probe.get("schema")!=PROBE_SCHEMA:
        raise RuntimeError("controlled split probe schema mismatch")
    if str(probe.get("source_checkpoint_sha256"))!=EXPECTED_SHA:
        raise RuntimeError("controlled split probe checkpoint mismatch")
    split_rng=random.Random(int(probe["holdout_seed"]))
    protected=set(split_rng.sample(
        range(len(r3.bundle.adv_mem.items)),
        int(probe["holdout_size"]),
    ))
    train_pool=[i for i in range(len(r3.bundle.adv_mem.items)) if i not in protected]
    if len(train_pool)!=1_950_000:
        raise RuntimeError("controlled split train-pool size drift")

    print("ONLINE_SEMANTIC_PRECOMPUTE_BEGIN",flush=True)
    sem_rows,precompute_seconds=precompute_semantics(r3.bundle.adv_mem)
    print(
        f"ONLINE_SEMANTIC_PRECOMPUTE_PASS seconds={precompute_seconds:.3f}",
        flush=True,
    )

    traces,games=shadow.replay_states(
        solver,args.spin_bundle.resolve(strict=True),200
    )
    initial_dc1=advantage_dc1_summary(models,traces)

    online_rows=[]
    final_states=[{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for m in models]
    overall_started=time.perf_counter()
    for offset in range(1,ONLINE_ITERATIONS+1):
        iteration=int(completed)+offset
        before_seen=int(r3.bundle.adv_mem.seen)
        tree=collect_roots(
            solver=solver,
            sampler=sampler,
            memory=r3.bundle.adv_mem,
            models=models,
            seed=int(seed),
            source_config=source_config,
            iteration=iteration,
        )
        models,final_states,fit_meta,fit_seconds=fit_ensemble(
            r3.bundle.adv_mem,
            sem_rows,
            train_pool,
            member_meta,
            lr=float(source_config.learning_rate),
            batch_size=int(source_config.batch_size),
            iteration=iteration,
        )
        row={
            "iteration":iteration,
            "adv_seen_before":before_seen,
            "adv_seen_after":int(r3.bundle.adv_mem.seen),
            "tree":tree,
            "fit_seconds":fit_seconds,
            "member_meta":fit_meta,
        }
        online_rows.append(row)
        print("ONLINE_SEMANTIC_ITERATION "+json.dumps({
            "iteration":iteration,
            "adv_seen_after":row["adv_seen_after"],
            "tree_seconds":tree["seconds"],
            "fit_seconds":fit_seconds,
        },sort_keys=True),flush=True)

    final_dc1=advantage_dc1_summary(models,traces)

    # Final online behavior -> fresh strategy targets.
    distill.MASTER_SEED=ONLINE_POLICY_TRAIN_SEED
    policy_train,discarded_holdout,train_collection=distill.collect_fresh(
        solver,models,POLICY_TRAIN_EPISODES
    )
    if len(policy_train)<20000:
        raise RuntimeError(f"online policy train stream too small: {len(policy_train)}")

    mini,cfg=policy_pair_from_source(r3.bundle,source_config)
    v1,v1opt,sem,semopt=confirm.initial_policy_pair(mini,cfg)
    confirm.train_500(v1,v1opt,sem,semopt,policy_train,cfg)

    # Independent post-online target stream; no model selection on it.
    step0_v1,_,step0_sem,_=confirm.initial_policy_pair(mini,cfg)
    distill.MASTER_SEED=ONLINE_POLICY_EVAL_SEED
    eval_a,eval_b,eval_collection=distill.collect_fresh(
        solver,models,POLICY_EVAL_EPISODES
    )
    independent=list(eval_a)+list(eval_b)
    if len(independent)<5000:
        raise RuntimeError(f"online independent eval too small: {len(independent)}")
    step0=distill.metrics(step0_v1,step0_sem,independent)
    final_policy_eval=distill.metrics(v1,sem,independent)
    hc=final_policy_eval["high_card_no_draw_allin"]
    if int(hc["count"])<150:
        raise RuntimeError("online independent high-card subgroup too small")

    policy_dc1=distill.dc1(
        solver,args.spin_bundle.resolve(strict=True),v1,sem,200
    )

    initial_nodraw=initial_dc1["high_card_jam_no_immediate_draw"]
    final_nodraw=final_dc1["high_card_jam_no_immediate_draw"]
    policy_criteria={
        "semantic_ce_at_least_10pct_better_than_v1":bool(
            final_policy_eval["semantic_weighted_ce"]
            <=0.90*final_policy_eval["v1_weighted_ce"]
        ),
        "semantic_tv_at_least_15pct_better_than_v1":bool(
            final_policy_eval["semantic_weighted_tv"]
            <=0.85*final_policy_eval["v1_weighted_tv"]
        ),
        "semantic_high_card_abs_bias_at_least_50pct_better":bool(
            float(hc["v1_abs_bias"])>0
            and float(hc["semantic_abs_bias"])<=0.50*float(hc["v1_abs_bias"])
        ),
        "semantic_ce_at_least_10pct_better_than_step0":bool(
            final_policy_eval["semantic_weighted_ce"]
            <=0.90*step0["semantic_weighted_ce"]
        ),
    }
    advantage_criteria={
        "final_pairwise_tv_not_worse_by_over_10pct":bool(
            final_dc1["pairwise_tv_mean"]
            <=1.10*initial_dc1["pairwise_tv_mean"]
        ),
        "final_argmax_disagreement_not_worse_by_over_10pct":bool(
            final_dc1["pairwise_argmax_disagreement_mean"]
            <=1.10*initial_dc1["pairwise_argmax_disagreement_mean"]
        ),
        "final_no_draw_raw_allin_not_up_by_over_10pp":bool(
            final_nodraw["raw_ensemble_action_probability_mean"]
            <=initial_nodraw["raw_ensemble_action_probability_mean"]+0.10
        ),
        "final_no_draw_member_mix_allin_not_up_by_over_10pp":bool(
            final_nodraw["policy_mixture_action_probability_mean"]
            <=initial_nodraw["policy_mixture_action_probability_mean"]+0.10
        ),
    }
    criteria={**advantage_criteria,**policy_criteria}
    passed=all(criteria.values())

    args.out_ensemble.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema":"SPINCORE_3H_SEMANTIC_ONLINE_PILOT_ENSEMBLE_V1",
        "source_checkpoint_sha256":actual,
        "source_iteration":10105,
        "completed_iteration":10105+ONLINE_ITERATIONS,
        "ensemble_size":MEMBERS,
        "member_steps":MEMBER_STEPS,
        "members":final_states,
        "member_seed_contract":member_meta,
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    },args.out_ensemble)
    torch.save({
        "schema":"SPINCORE_3H_SEMANTIC_ONLINE_PILOT_TAIL_POLICY_V1",
        "source_checkpoint_sha256":actual,
        "selected_steps":POLICY_STEPS,
        "model_state":{k:v.detach().cpu() for k,v in sem.state_dict().items()},
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    },args.out_policy)

    report={
        "schema":"SPINCORE_3H_SEMANTIC_ONLINE_PILOT_V1",
        "scope":"DIAGNOSTIC_ONLY_BOUNDED_3H_ONLINE_FEEDBACK",
        "source_checkpoint_sha256":actual,
        "source_iteration":10105,
        "completed_iteration":10105+ONLINE_ITERATIONS,
        "source_checkpoint_mutated":False,
        "hu_training_performed":False,
        "online_iterations":ONLINE_ITERATIONS,
        "roots_per_iteration":ROOTS_PER_ITERATION,
        "ensemble_size":MEMBERS,
        "member_steps":MEMBER_STEPS,
        "protected_controlled_split_positions":len(protected),
        "semantic_precompute_seconds":precompute_seconds,
        "online_rows":online_rows,
        "advantage_dc1":{
            "balanced_games":games,
            "trace_decisions":len(traces),
            "initial":initial_dc1,
            "final":final_dc1,
        },
        "tail_policy":{
            "train_collection":train_collection,
            "discarded_selection_holdout_samples":len(discarded_holdout),
            "selected_steps":POLICY_STEPS,
            "independent_eval_collection":eval_collection,
            "independent_eval_samples":len(independent),
            "step0_independent_eval":step0,
            "step500_independent_eval":final_policy_eval,
            "dc1":policy_dc1,
        },
        "precommitted_criteria":criteria,
        "semantic_online_pilot_pass":passed,
        "wall_seconds":float(time.perf_counter()-overall_started),
        "interpretation":(
            "PASS means the independently validated V1+semantic Advantage/strategy "
            "bridge survives two real online 3H CFR feedback iterations without a "
            "material DC1 stability regression, and the precommitted 500-step semantic "
            "tail AveragePolicy continues to dominate V1 on a new post-online target "
            "stream. PASS authorizes a longer research-only semantic continuation; it "
            "does not authorize production or DC2."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )

    print("=== 3H semantic online pilot ===")
    print("advantage_initial="+json.dumps(initial_dc1,sort_keys=True))
    print("advantage_final="+json.dumps(final_dc1,sort_keys=True))
    print("policy_final="+json.dumps(final_policy_eval,sort_keys=True))
    print("criteria="+json.dumps(criteria,sort_keys=True))
    print(f"semantic_online_pilot_pass={passed}")
    print(f"report={args.report.resolve()}")
    print("3H_SEMANTIC_ONLINE_PILOT_RUN_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
