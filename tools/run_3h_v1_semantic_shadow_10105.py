#!/usr/bin/env python3
from __future__ import annotations

"""Matched shadow ablation: frozen V1 vs V1 + general poker semantics at 10105.

The candidate keeps the complete SPNNIV1 observation path and V1 architecture,
adding only a compact semantic vector derived deterministically from card tokens
already present in SPNNIV1:
- made-hand category;
- pair relation;
- overcard count;
- pocket-pair / flush-draw / straight-draw / backdoor flags;
- board pairedness, suit density, straight-window occupancy, broadway density.

Critically, there is NO explicit high_card_no_draw feature in the candidate.
The network must learn that conjunction from general semantics.

Training is paired to the existing controlled-split V1 probe:
- exact same frozen 10105 3H Advantage reservoir;
- exact same 50k holdout;
- exact same per-replica init seed and minibatch RNG seed;
- exact same 1600 optimizer-step budget;
- exact same Adam/lr/batch size.

The original V1 1600 snapshots from the probe are the baseline. No CFR roots,
AveragePolicy training, checkpoint mutation, or deployment change occurs.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

from spincore.deepcrusher_benchmark import (
    SPINCORE_POLICY_ID,
    ExternalExactAction,
    OfflineHeadToHeadEngine,
)
from spincore.deepcrusher_policy import DeepCrusherR8Policy
from spincore.decision_sanity import sanity_flags
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_solver_actions import resolve_lean_exact
from spincore.lean_hybrid_deployment_agent import LeanHybridDeploymentAgent
from spincore.solver import SolverLibrary
from spincore_nn.action_models import (
    ActionNetworkConfigV1,
    collate_action_observations,
    make_advantage_action_model,
)
from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.training import train_step

from audit_3h_semantic_sidecar_attribution_10105 import (
    board_semantics,
    decode_obs,
    private_semantics,
)

EXPECTED_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
PROBE_SCHEMA="SPINCORE_3H_ADVANTAGE_CONTROLLED_SPLIT_PROBE_V1"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
DOMAIN="THREE_HANDED"
BUDGET=1600
REPLICAS=8
SEED=20260923
ALL_IN=9
SEM_DIM=30


def mix64(*values:int)->int:
    x=0x9E3779B97F4A7C15
    mask=(1<<64)-1
    for value in values:
        y=int(value)&mask
        x^=(y+0x9E3779B97F4A7C15+((x<<6)&mask)+(x>>2))&mask
        x&=mask
    return x


def semantic_vector(sample)->np.ndarray:
    hole,board,_numeric,_cat=decode_obs(sample.observation)
    ps=private_semantics(hole,board)
    bs=board_semantics(board)
    out=[]
    out.extend(float(ps["made"]==k) for k in range(9))
    out.extend(float(ps["pair_relation"]==k) for k in range(9))
    out.extend(float(ps["overcards"]==k) for k in range(3))
    out.extend([
        float(ps["pocket_pair"]),
        float(ps["flush_draw"]),
        float(ps["straight_draw"]),
        float(ps["backdoor_flush"]),
        float(ps["backdoor_straight"]),
        float(bs["paired"]),
        float(bs["max_suit"]),
        float(bs["straight_occ"]),
        float(bs["broadway"]),
    ])
    arr=np.asarray(out,dtype=np.float32)
    if arr.shape!=(SEM_DIM,):
        raise RuntimeError(f"semantic dim drift: {arr.shape}")
    return arr


class V1SemanticAdvantageNet(nn.Module):
    def __init__(self,cfg:ActionNetworkConfigV1|None=None):
        super().__init__()
        self.cfg=cfg or ActionNetworkConfigV1()
        c=self.cfg
        self.card_emb=nn.Embedding(53,c.card_emb,padding_idx=0)
        self.cat_emb=nn.Embedding(32,c.cat_emb)
        self.hist_emb=nn.Embedding(64,c.cat_emb,padding_idx=0)
        self.gru=nn.GRU(c.cat_emb,c.gru_hidden,batch_first=True)
        input_dim=7*c.card_emb+8*c.cat_emb+16+c.gru_hidden+SEM_DIM
        self.body=nn.Sequential(
            nn.Linear(input_dim,c.hidden),
            nn.ReLU(),
            nn.Linear(c.hidden,c.head_hidden),
            nn.ReLU(),
        )
        self.head=nn.Linear(c.head_hidden,c.actions)

    def forward(self,batch):
        cards=self.card_emb(batch["cards"]).flatten(1)
        categorical=self.cat_emb(batch["categorical"].clamp(0,31)).flatten(1)
        history=self.hist_emb(batch["history"].clamp(0,63))
        _,hidden=self.gru(history)
        x=torch.cat([
            cards,categorical,batch["numeric"],hidden[-1],batch["semantic"]
        ],dim=1)
        return self.head(self.body(x))


def make_semantic_model(seed:int):
    cfg=ActionNetworkConfigV1()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model=V1SemanticAdvantageNet(cfg)
    return model


def load_v1_model(state):
    _,m=make_advantage_action_model(REPRESENTATION,device="cpu",seed=0)
    m.load_state_dict(state)
    m.eval()
    return m


def base_batch(samples,device="cpu"):
    return vectorized_batch(samples,device)


def semantic_batch(samples,semantic_rows,device="cpu"):
    batch,target,weights=vectorized_batch(samples,device)
    batch["semantic"]=torch.tensor(
        np.asarray(semantic_rows,dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    return batch,target,weights


class Weighted:
    def __init__(self):
        self.n=0;self.w=0.0;self.sq=0.0;self.bias=0.0;self.pred=0.0;self.target=0.0
    def add(self,p,t,w):
        p=float(p);t=float(t);w=float(w)
        self.n+=1;self.w+=w
        e=p-t
        self.sq+=w*e*e
        self.bias+=w*e
        self.pred+=w*p
        self.target+=w*t
    def out(self):
        return {
            "count":self.n,
            "weight_sum":self.w,
            "prediction_weighted_mean":self.pred/self.w if self.w else None,
            "target_weighted_mean":self.target/self.w if self.w else None,
            "weighted_bias_prediction_minus_target":self.bias/self.w if self.w else None,
            "weighted_mse":self.sq/self.w if self.w else None,
        }


def named_policy(values,legal):
    return {
        str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")):float(values[a])
        for a in legal
    }


def argmax_name(p):
    return max(p,key=p.get) if p else None


def tv(a,b):
    keys=set(a)|set(b)
    return 0.5*sum(abs(a.get(k,0.0)-b.get(k,0.0)) for k in keys)


def policy_stats(models,batch,legal):
    raws=[]
    with torch.no_grad():
        for m in models:
            raws.append(m(batch)[0].detach().cpu().tolist())
    members=[
        named_policy(lean_regret_matching_policy(r,tuple(legal)),tuple(legal))
        for r in raws
    ]
    raw_mean=[statistics.fmean([r[a] for r in raws]) for a in range(10)]
    raw_ens=named_policy(
        lean_regret_matching_policy(raw_mean,tuple(legal)),tuple(legal)
    )
    mix={
        k:statistics.fmean([p.get(k,0.0) for p in members])
        for k in set().union(*(p.keys() for p in members))
    }
    pair_tvs=[];pair_arg=[]
    for i in range(len(members)):
        for j in range(i+1,len(members)):
            pair_tvs.append(tv(members[i],members[j]))
            pair_arg.append(float(argmax_name(members[i])!=argmax_name(members[j])))
    return {
        "raw_ensemble_policy":raw_ens,
        "policy_mixture":mix,
        "member_argmax":[argmax_name(p) for p in members],
        "pairwise_tv_mean":statistics.fmean(pair_tvs),
        "pairwise_argmax_disagreement":statistics.fmean(pair_arg),
    }


class CapturePolicy:
    policy_id=SPINCORE_POLICY_ID
    def __init__(self,agent):
        self.agent=agent
        self._last={}
    def choose_exact(self,state,*,seat:int,rng:random.Random)->ExternalExactAction:
        active_mask,legal,probs=self.agent.distribution(state)
        domain=self.agent.domain_for_state(state)
        x=rng.random()
        cumulative=0.0
        slot=int(legal[-1])
        for a in legal:
            cumulative+=float(probs[a])
            if x<cumulative:
                slot=int(a);break
        action_type,amount_to=resolve_lean_exact(state,active_mask,slot)
        self._last[int(seat)]={
            "domain":domain,
            "observation_hex":state.neural_bytes().hex() if domain==DOMAIN else None,
            "legal_slots":[int(a) for a in legal] if domain==DOMAIN else None,
        }
        return ExternalExactAction(int(action_type),int(amount_to))
    def decision_metadata(self,*,seat:int):
        r=self._last.get(int(seat))
        return None if r is None else dict(r)


def replay_states(solver,bundle_path,scenarios):
    agent=LeanHybridDeploymentAgent.from_bundle(bundle_path,device="cpu",seed=0)
    spin=CapturePolicy(agent)
    deep=DeepCrusherR8Policy.from_repository(ROOT)
    sampler=LegacyScenarioSampler(seed=SEED^0x5CE0A710,config=LegacyScenarioConfig())
    traces=[]
    games=0
    for index in range(int(scenarios)):
        episode=sampler.sample_episode()
        engine=OfflineHeadToHeadEngine(
            solver,spincore_policy=spin,deepcrusher_policy=deep,
            master_seed=SEED,max_decisions=200,decision_sink=traces.append,
        )
        obs=engine.play_balanced_block(
            episode,deal_seed=int(mix64(SEED,index,0xD34A1)),
            scenario_index=int(index),
        )
        games+=len(obs)
    return traces,games


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--probe",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--spin-bundle",type=Path,required=True)
    ap.add_argument("--out-models",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--scenarios",type=int,default=200)
    ap.add_argument("--max-projected-minutes",type=float,default=70.0)
    args=ap.parse_args()

    torch.set_num_threads(int(args.threads))

    import hashlib
    cp=args.checkpoint.resolve(strict=True)
    h=hashlib.sha256()
    with cp.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    if h.hexdigest()!=EXPECTED_SHA:raise SystemExit("checkpoint SHA mismatch")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    mem=((payload.get("domains") or {}).get(DOMAIN) or {}).get("adv_mem") or {}
    items=list(mem.get("items") or [])
    if len(items)!=2_000_000:raise SystemExit("expected saturated 2M 3H reservoir")

    probe=torch.load(args.probe.resolve(strict=True),map_location="cpu",weights_only=False)
    if probe.get("schema")!=PROBE_SCHEMA:raise SystemExit("wrong probe schema")
    baseline_states=list(probe["snapshots"][str(BUDGET)])
    meta=list(probe.get("member_meta") or [])
    if len(baseline_states)!=REPLICAS or len(meta)!=REPLICAS:
        raise SystemExit("expected eight baseline members/meta")
    batch_size=int(probe["batch_size"])
    lr=float(probe["learning_rate"])

    split_rng=random.Random(int(probe["holdout_seed"]))
    holdout_idx=set(split_rng.sample(range(len(items)),int(probe["holdout_size"])))
    train_idx=[i for i in range(len(items)) if i not in holdout_idx]
    holdout_indices=sorted(holdout_idx)

    print("SEMANTIC_PRECOMPUTE_BEGIN",flush=True)
    sem=np.empty((len(items),SEM_DIM),dtype=np.float32)
    started=time.perf_counter()
    for i,s in enumerate(items):
        sem[i]=semantic_vector(s)
        if (i+1)%200000==0:
            print(f"SEMANTIC_PRECOMPUTE {i+1}/{len(items)}",flush=True)
    semantic_seconds=time.perf_counter()-started
    print(f"SEMANTIC_PRECOMPUTE_PASS seconds={semantic_seconds:.3f}",flush=True)

    candidate_states=[]
    candidate_meta=[]
    for rep,row in enumerate(meta):
        init_seed=int(row["init_seed"])
        batch_seed=int(row["batch_seed"])
        model=make_semantic_model(init_seed)
        opt=torch.optim.Adam(model.parameters(),lr=lr)
        rng=random.Random(batch_seed)
        fit_started=time.perf_counter()
        losses=[]
        for step in range(BUDGET):
            pos=rng.sample(range(len(train_idx)),min(batch_size,len(train_idx)))
            idx=[train_idx[p] for p in pos]
            samples=[items[i] for i in idx]
            b,t,w=semantic_batch(samples,sem[idx],"cpu")
            losses.append(train_step(model,opt,b,t,w,"advantage"))
            if rep==0 and step+1==400:
                elapsed=time.perf_counter()-fit_started
                projected=(elapsed/400.0)*(BUDGET*REPLICAS)/60.0
                print(
                    f"SEMANTIC_PERF_GATE projected_fit_minutes={projected:.3f} "
                    f"limit={float(args.max_projected_minutes):.3f}",
                    flush=True,
                )
                if projected>float(args.max_projected_minutes):
                    raise RuntimeError("semantic shadow projected fit exceeds performance gate")
        elapsed=time.perf_counter()-fit_started
        candidate_states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        candidate_meta.append({
            "replica":rep,"init_seed":init_seed,"batch_seed":batch_seed,
            "steps":BUDGET,"fit_seconds":elapsed,"loss_last":float(losses[-1]),
        })
        print(
            f"SEMANTIC_REPLICA rep={rep+1}/{REPLICAS} "
            f"loss={losses[-1]:.8f} seconds={elapsed:.3f}",
            flush=True,
        )

    args.out_models.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema":"SPINCORE_3H_V1_SEMANTIC_SHADOW_MODELS_V1",
        "source_checkpoint_sha256":EXPECTED_SHA,
        "probe_schema":PROBE_SCHEMA,
        "budget":BUDGET,
        "semantic_dim":SEM_DIM,
        "explicit_high_card_no_draw_feature":False,
        "members":candidate_states,
        "member_meta":candidate_meta,
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    },args.out_models)

    baseline=[load_v1_model(s) for s in baseline_states]
    candidate=[make_semantic_model(0) for _ in candidate_states]
    for m,state in zip(candidate,candidate_states):
        m.load_state_dict(state);m.eval()

    full_base=Weighted();full_cand=Weighted()
    ai_base=Weighted();ai_cand=Weighted()
    hc_base=Weighted();hc_cand=Weighted()

    eval_bs=4096
    with torch.no_grad():
        for start in range(0,len(holdout_indices),eval_bs):
            idx=holdout_indices[start:start+eval_bs]
            samples=[items[i] for i in idx]
            bb,t,w=base_batch(samples,"cpu")
            sb=dict(bb)
            sb["semantic"]=torch.tensor(sem[idx],dtype=torch.float32)
            bout=torch.stack([m(bb).float() for m in baseline],dim=0).mean(dim=0)
            cout=torch.stack([m(sb).float() for m in candidate],dim=0).mean(dim=0)
            legal=bb["legal"].float()
            denom=legal.sum(1).clamp_min(1.0)
            per_b=(((bout-t)**2)*legal).sum(1)/denom
            per_c=(((cout-t)**2)*legal).sum(1)/denom
            for j,s in enumerate(samples):
                ww=float(w[j])
                full_base.add(0.0,math.sqrt(max(float(per_b[j]),0.0)),ww)
                full_cand.add(0.0,math.sqrt(max(float(per_c[j]),0.0)),ww)
                if not bool(s.legal[ALL_IN]):continue
                pb=float(bout[j,ALL_IN]);pc=float(cout[j,ALL_IN]);yy=float(t[j,ALL_IN])
                ai_base.add(pb,yy,ww);ai_cand.add(pc,yy,ww)
                _h,_b,_n,_c=decode_obs(s.observation)
                ps=private_semantics(_h,_b)
                # Exact frozen population contract from semantic-sidecar V2.
                if ps["high_card_no_draw"]:
                    hc_base.add(pb,yy,ww);hc_cand.add(pc,yy,ww)

    if hc_base.n!=6639:
        raise RuntimeError(f"holdout high-card population drift: {hc_base.n}")

    solver=SolverLibrary(args.solver.resolve(strict=True))
    traces,games=replay_states(solver,args.spin_bundle.resolve(strict=True),args.scenarios)
    global_rows=[]
    high_rows=[]
    trips_rows=[]
    for tr in traces:
        if tr.policy_id!=SPINCORE_POLICY_ID or tr.domain!=DOMAIN:
            continue
        d=dict(tr.policy_detail or {})
        obs_hex=d.get("observation_hex");legal=tuple(int(x) for x in d.get("legal_slots") or ())
        if not obs_hex or not legal:raise RuntimeError("missing captured 3H state")
        obs=bytes.fromhex(obs_hex)
        base_b=collate_action_observations(
            REPRESENTATION,[obs],[legal_mask(legal)],device="cpu"
        )
        # semantic vector from a light sample-like wrapper
        class S:pass
        tmp=S();tmp.observation=obs
        sv=semantic_vector(tmp)
        cand_b=dict(base_b)
        cand_b["semantic"]=torch.tensor([sv],dtype=torch.float32)
        bp=policy_stats(baseline,base_b,legal)
        cp=policy_stats(candidate,cand_b,legal)
        global_rows.append((bp,cp))
        for flag in sanity_flags(tr):
            row={"baseline":bp,"semantic":cp,"context":flag.context}
            if flag.code=="POSTFLOP_DEEP_HIGH_CARD_JAM":
                high_rows.append(row)
            elif flag.code=="POSTFLOP_TRIPS_PLUS_FOLD":
                trips_rows.append(row)

    no_draw=[r for r in high_rows if not bool((r["context"] or {}).get("has_immediate_straight_or_flush_draw"))]

    def stability(rows,key):
        return {
            "decision_count":len(rows),
            "pairwise_tv_mean":statistics.fmean([r[0 if key=="baseline" else 1]["pairwise_tv_mean"] for r in rows]),
            "pairwise_argmax_disagreement_mean":statistics.fmean([r[0 if key=="baseline" else 1]["pairwise_argmax_disagreement"] for r in rows]),
        }

    def weird(rows,key,action):
        if not rows:return {"count":0}
        vals=[r[key] for r in rows]
        return {
            "count":len(vals),
            "raw_ensemble_action_probability_mean":statistics.fmean([
                x["raw_ensemble_policy"].get(action,0.0) for x in vals
            ]),
            "policy_mixture_action_probability_mean":statistics.fmean([
                x["policy_mixture"].get(action,0.0) for x in vals
            ]),
            "majority_member_argmax_action_count":sum(
                sum(a==action for a in x["member_argmax"])>=5 for x in vals
            ),
            "unanimous_member_argmax_action_count":sum(
                all(a==action for a in x["member_argmax"]) for x in vals
            ),
        }

    fb=full_base.out();fc=full_cand.out()
    # Weighted helper above stores y=sqrt(perMSE), p=0, so weighted_mse equals
    # the desired weighted mean per-sample legal-action MSE.
    full_base_mse=fb["weighted_mse"]
    full_cand_mse=fc["weighted_mse"]
    aib=ai_base.out();aic=ai_cand.out()
    hcb=hc_base.out();hcc=hc_cand.out()

    base_stability=stability(global_rows,"baseline")
    cand_stability=stability(global_rows,"semantic")

    criteria={
        "global_legal_action_mse_not_worse_by_over_2pct":bool(
            full_cand_mse<=1.02*full_base_mse
        ),
        "global_allin_mse_not_worse_by_over_2pct":bool(
            aic["weighted_mse"]<=1.02*aib["weighted_mse"]
        ),
        "high_card_abs_allin_bias_reduced_by_at_least_50pct":bool(
            abs(hcc["weighted_bias_prediction_minus_target"])
            <=0.5*abs(hcb["weighted_bias_prediction_minus_target"])
        ),
        "high_card_allin_mse_not_worse":bool(
            hcc["weighted_mse"]<=hcb["weighted_mse"]
        ),
        "dc1_pairwise_policy_tv_not_worse_by_over_10pct":bool(
            cand_stability["pairwise_tv_mean"]
            <=1.10*base_stability["pairwise_tv_mean"]
        ),
    }
    passed=all(criteria.values())

    report={
        "schema":"SPINCORE_3H_V1_SEMANTIC_SHADOW_ABLATION_V1",
        "scope":"DIAGNOSTIC_ONLY_MATCHED_FROZEN_10105_RESERVOIR",
        "checkpoint_sha256":EXPECTED_SHA,
        "probe_schema":PROBE_SCHEMA,
        "budget":BUDGET,
        "replicas":REPLICAS,
        "semantic_dim":SEM_DIM,
        "explicit_high_card_no_draw_feature":False,
        "semantic_precompute_seconds":semantic_seconds,
        "baseline_parameter_count":sum(p.numel() for p in baseline[0].parameters()),
        "semantic_parameter_count":sum(p.numel() for p in candidate[0].parameters()),
        "holdout":{
            "size":len(holdout_indices),
            "global_legal_action_weighted_mse":{
                "baseline":full_base_mse,"semantic":full_cand_mse,
            },
            "allin_legal":{
                "baseline":aib,"semantic":aic,
            },
            "postflop_high_card_no_draw_allin":{
                "baseline":hcb,"semantic":hcc,
            },
        },
        "dc1":{
            "scenarios":int(args.scenarios),
            "balanced_games":games,
            "trace_decisions":len(traces),
            "three_handed_decisions":len(global_rows),
            "stability":{
                "baseline":base_stability,
                "semantic":cand_stability,
            },
            "high_card_jam_all":{
                "baseline":weird(high_rows,"baseline","ALL_IN"),
                "semantic":weird(high_rows,"semantic","ALL_IN"),
            },
            "high_card_jam_no_immediate_draw":{
                "baseline":weird(no_draw,"baseline","ALL_IN"),
                "semantic":weird(no_draw,"semantic","ALL_IN"),
            },
            "trips_plus_fold":{
                "baseline":weird(trips_rows,"baseline","FOLD"),
                "semantic":weird(trips_rows,"semantic","FOLD"),
            },
        },
        "precommitted_shadow_criteria":criteria,
        "v1_semantic_shadow_pass":passed,
        "interpretation":(
            "PASS means a general semantic augmentation derived from information already "
            "present in SPNNIV1 improves the specific high-card/no-draw ALL_IN calibration "
            "without material global-regression or DC1-stability degradation. It does not "
            "promote the candidate. FAIL means the linear sidecar attribution did not "
            "survive end-to-end neural refitting under a matched budget."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== 3H V1 + semantic shadow ===")
    print("holdout="+json.dumps(report["holdout"],sort_keys=True))
    print("dc1="+json.dumps(report["dc1"],sort_keys=True))
    print("criteria="+json.dumps(criteria,sort_keys=True))
    print(f"v1_semantic_shadow_pass={passed}")
    print(f"models={args.out_models.resolve()}")
    print(f"report={args.report.resolve()}")
    print("3H_V1_SEMANTIC_SHADOW_RUN_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
