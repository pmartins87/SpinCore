#!/usr/bin/env python3
from __future__ import annotations

"""Paired 3H AveragePolicy continuation: V1 control vs V1+semantic candidate.

Both arms start from the exact finalized 10105 THREE_HANDED AveragePolicy and
its Adam state. The semantic arm is functionally identical at step 0: all V1
weights and optimizer moments are copied, the extra semantic input columns are
zero-initialized, and their Adam moments start at zero.

Both arms then consume the exact same strategy-reservoir minibatches for 4000
additional optimizer steps. This isolates whether the already-supported general
poker semantics also help the AveragePolicy distillation problem.

Diagnostic only. No CFR roots, strategy-target generation, checkpoint mutation,
deployment change or sealed holdout access occurs.
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
from spincore.lean_solver_actions import resolve_lean_exact
from spincore.lean_hybrid_deployment_agent import LeanHybridDeploymentAgent
from spincore.solver import SolverLibrary
from spincore_nn.action_models import (
    ActionNetworkConfigV1,
    collate_action_observations,
    make_policy_action_model,
)
from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.training import train_step

from audit_3h_semantic_sidecar_attribution_10105 import (
    board_semantics,
    decode_obs,
    private_semantics,
)

EXPECTED_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
DOMAIN="THREE_HANDED"
MASTER_SEED=20260923
SHADOW_SEED=20260926 ^ 0xA9E901
MILESTONES=(0,500,1000,2000,4000)
SEM_DIM=30
ALL_IN=9
FOLD=0


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def mix64(*values:int)->int:
    x=0x9E3779B97F4A7C15
    mask=(1<<64)-1
    for value in values:
        y=int(value)&mask
        x^=(y+0x9E3779B97F4A7C15+((x<<6)&mask)+(x>>2))&mask
        x&=mask
    return x


def semantic_vector_from_obs(obs:bytes)->np.ndarray:
    hole,board,_numeric,_cat=decode_obs(obs)
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
        raise RuntimeError(f"semantic dimension drift: {arr.shape}")
    return arr


class V1SemanticPolicyNet(nn.Module):
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

    def probabilities(self,batch):
        logits=self.forward(batch).masked_fill(~batch["legal"],-1e9)
        return torch.softmax(logits,dim=-1)


def build_semantic_from_v1(v1):
    sem=V1SemanticPolicyNet(v1.cfg)
    with torch.no_grad():
        sem.card_emb.weight.copy_(v1.card_emb.weight)
        sem.cat_emb.weight.copy_(v1.cat_emb.weight)
        sem.hist_emb.weight.copy_(v1.hist_emb.weight)
        for dst,src in zip(sem.gru.parameters(),v1.gru.parameters()):
            dst.copy_(src)
        old0=v1.body[0]
        new0=sem.body[0]
        new0.weight.zero_()
        new0.weight[:,:old0.weight.shape[1]].copy_(old0.weight)
        new0.bias.copy_(old0.bias)
        sem.body[2].weight.copy_(v1.body[2].weight)
        sem.body[2].bias.copy_(v1.body[2].bias)
        sem.head.weight.copy_(v1.head.weight)
        sem.head.bias.copy_(v1.head.bias)
    return sem


def clone_optimizer_state_v1_to_semantic(v1,v1_opt,sem,sem_opt):
    old_by_name=dict(v1.named_parameters())
    new_by_name=dict(sem.named_parameters())
    for name,new_p in new_by_name.items():
        old_p=old_by_name[name]
        old_state=v1_opt.state.get(old_p,{})
        if not old_state:
            continue
        ns={}
        for key,value in old_state.items():
            if torch.is_tensor(value):
                if value.ndim==0:
                    ns[key]=value.detach().clone()
                elif tuple(value.shape)==tuple(new_p.shape):
                    ns[key]=value.detach().clone()
                elif name=="body.0.weight" and value.ndim==2:
                    z=torch.zeros_like(new_p)
                    z[:,:value.shape[1]].copy_(value)
                    ns[key]=z
                else:
                    raise RuntimeError(
                        f"optimizer-state shape drift {name}/{key}: "
                        f"{tuple(value.shape)} -> {tuple(new_p.shape)}"
                    )
            else:
                ns[key]=copy.deepcopy(value)
        sem_opt.state[new_p]=ns
    old_group=v1_opt.param_groups[0]
    new_group=sem_opt.param_groups[0]
    for key,value in old_group.items():
        if key!="params":
            new_group[key]=copy.deepcopy(value)


def semantic_batch(samples,semantic_rows):
    batch,target,weights=vectorized_batch(samples,"cpu")
    batch["semantic"]=torch.tensor(
        np.asarray(semantic_rows,dtype=np.float32),
        dtype=torch.float32,
    )
    return batch,target,weights


def high_card_no_draw_from_obs(obs:bytes)->bool:
    hole,board,_n,_c=decode_obs(obs)
    ps=private_semantics(hole,board)
    return bool(ps["high_card_no_draw"])


def policy_metrics(v1,sem,samples,semantic_rows,batch_size=4096):
    ce_v=[];ce_s=[];tv_v=[];tv_s=[];weights=[]
    hc_target=[];hc_v=[];hc_s=[];hc_w=[]
    v1.eval();sem.eval()
    for start in range(0,len(samples),batch_size):
        chunk=samples[start:start+batch_size]
        rows=semantic_rows[start:start+batch_size]
        b,t,w=vectorized_batch(chunk,"cpu")
        sb=dict(b)
        sb["semantic"]=torch.tensor(np.asarray(rows,dtype=np.float32),dtype=torch.float32)
        with torch.no_grad():
            lv=v1(b).masked_fill(~b["legal"],-1e9)
            ls=sem(sb).masked_fill(~b["legal"],-1e9)
            pv=torch.softmax(lv,dim=-1).cpu()
            ps=torch.softmax(ls,dim=-1).cpu()
            logv=torch.log_softmax(lv,dim=-1).cpu()
            logs=torch.log_softmax(ls,dim=-1).cpu()
        tc=t.cpu();legal=b["legal"].cpu();ww=w.cpu()
        for i,s in enumerate(chunk):
            la=[a for a,x in enumerate(legal[i].tolist()) if bool(x)]
            tgt=[float(x) for x in tc[i].tolist()]
            pvi=[float(x) for x in pv[i].tolist()]
            psi=[float(x) for x in ps[i].tolist()]
            ce_v.append(-sum(tgt[a]*float(logv[i,a]) for a in la))
            ce_s.append(-sum(tgt[a]*float(logs[i,a]) for a in la))
            tv_v.append(0.5*sum(abs(tgt[a]-pvi[a]) for a in la))
            tv_s.append(0.5*sum(abs(tgt[a]-psi[a]) for a in la))
            weights.append(float(ww[i]))
            if ALL_IN in la and high_card_no_draw_from_obs(s.observation):
                hc_target.append(tgt[ALL_IN])
                hc_v.append(pvi[ALL_IN])
                hc_s.append(psi[ALL_IN])
                hc_w.append(float(ww[i]))
    sw=sum(weights)
    hsw=sum(hc_w)
    def wm(vals,ws):
        return sum(v*w for v,w in zip(vals,ws))/sum(ws) if ws else None
    return {
        "count":len(samples),
        "v1_weighted_ce":wm(ce_v,weights),
        "semantic_weighted_ce":wm(ce_s,weights),
        "v1_tv_mean":statistics.fmean(tv_v),
        "semantic_tv_mean":statistics.fmean(tv_s),
        "v1_weighted_tv":wm(tv_v,weights),
        "semantic_weighted_tv":wm(tv_s,weights),
        "high_card_no_draw_allin":{
            "count":len(hc_target),
            "target_weighted_mean":wm(hc_target,hc_w),
            "v1_weighted_mean":wm(hc_v,hc_w),
            "semantic_weighted_mean":wm(hc_s,hc_w),
            "v1_abs_bias":abs(wm(hc_v,hc_w)-wm(hc_target,hc_w)) if hc_w else None,
            "semantic_abs_bias":abs(wm(hc_s,hc_w)-wm(hc_target,hc_w)) if hc_w else None,
        },
    }


def named_probs(values,legal):
    return {
        str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")):float(values[a])
        for a in legal
    }


class CapturePolicy:
    policy_id=SPINCORE_POLICY_ID
    def __init__(self,agent):
        self.agent=agent;self._last={}
    def choose_exact(self,state,*,seat:int,rng:random.Random)->ExternalExactAction:
        active_mask,legal,probs=self.agent.distribution(state)
        domain=self.agent.domain_for_state(state)
        x=rng.random();cum=0.0;slot=int(legal[-1])
        for a in legal:
            cum+=float(probs[a])
            if x<cum:
                slot=int(a);break
        action_type,amount_to=resolve_lean_exact(state,active_mask,slot)
        self._last[int(seat)]={
            "domain":domain,
            "observation_hex":state.neural_bytes().hex() if domain==DOMAIN else None,
            "legal_slots":[int(a) for a in legal] if domain==DOMAIN else None,
        }
        return ExternalExactAction(int(action_type),int(amount_to))
    def decision_metadata(self,*,seat:int):
        x=self._last.get(int(seat))
        return None if x is None else dict(x)


def replay_dc1(solver,bundle_path,scenarios):
    agent=LeanHybridDeploymentAgent.from_bundle(bundle_path,device="cpu",seed=0)
    spin=CapturePolicy(agent)
    deep=DeepCrusherR8Policy.from_repository(ROOT)
    sampler=LegacyScenarioSampler(seed=MASTER_SEED^0x5CE0A710,config=LegacyScenarioConfig())
    traces=[];games=0
    for index in range(int(scenarios)):
        ep=sampler.sample_episode()
        engine=OfflineHeadToHeadEngine(
            solver,spincore_policy=spin,deepcrusher_policy=deep,
            master_seed=MASTER_SEED,max_decisions=200,decision_sink=traces.append,
        )
        obs=engine.play_balanced_block(
            ep,deal_seed=int(mix64(MASTER_SEED,index,0xD34A1)),
            scenario_index=int(index),
        )
        games+=len(obs)
    return traces,games


def exact_policy(original,v1,sem,obs,legal):
    b=collate_action_observations(
        REPRESENTATION,[obs],[legal_mask(tuple(legal))],device="cpu"
    )
    sb=dict(b)
    sb["semantic"]=torch.tensor([semantic_vector_from_obs(obs)],dtype=torch.float32)
    with torch.no_grad():
        po=original.probabilities(b)[0].cpu().tolist()
        pv=v1.probabilities(b)[0].cpu().tolist()
        ps=sem.probabilities(sb)[0].cpu().tolist()
    return named_probs(po,legal),named_probs(pv,legal),named_probs(ps,legal)


def summarize_flag(rows,action):
    if not rows:return {"count":0}
    return {
        "count":len(rows),
        "production_reference_action_probability_mean":statistics.fmean(
            r["production_reference"].get(action,0.0) for r in rows
        ),
        "v1_control_action_probability_mean":statistics.fmean(
            r["v1"].get(action,0.0) for r in rows
        ),
        "semantic_action_probability_mean":statistics.fmean(
            r["semantic"].get(action,0.0) for r in rows
        ),
        "production_reference_argmax_action_count":sum(
            max(r["production_reference"],key=r["production_reference"].get)==action for r in rows
        ),
        "v1_control_argmax_action_count":sum(
            max(r["v1"],key=r["v1"].get)==action for r in rows
        ),
        "semantic_argmax_action_count":sum(
            max(r["semantic"],key=r["semantic"].get)==action for r in rows
        ),
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--spin-bundle",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--out-model",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--holdout-size",type=int,default=50000)
    ap.add_argument("--scenarios",type=int,default=200)
    ap.add_argument("--max-projected-minutes",type=float,default=65.0)
    args=ap.parse_args()

    torch.set_num_threads(int(args.threads))
    cp=args.checkpoint.resolve(strict=True)
    actual=sha256(cp)
    if actual!=EXPECTED_SHA:raise SystemExit("checkpoint SHA mismatch")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    d3=(payload.get("domains") or {}).get(DOMAIN) or {}
    cfg=dict(payload.get("config") or {})
    mem=d3.get("pol_mem") or {}
    items=list(mem.get("items") or [])
    if len(items)!=int(mem.get("capacity",len(items))):
        raise SystemExit("strategy reservoir must be saturated")

    _,original=make_policy_action_model(REPRESENTATION,device="cpu",seed=0)
    original.load_state_dict(d3["policy"])
    original.eval()

    _,v1=make_policy_action_model(REPRESENTATION,device="cpu",seed=0)
    v1.load_state_dict(d3["policy"])
    v1_opt=torch.optim.Adam(v1.parameters(),lr=float(cfg["learning_rate"]))
    v1_opt.load_state_dict(copy.deepcopy(d3["pol_opt"]))

    sem=build_semantic_from_v1(v1)
    sem_opt=torch.optim.Adam(sem.parameters(),lr=float(cfg["learning_rate"]))
    clone_optimizer_state_v1_to_semantic(v1,v1_opt,sem,sem_opt)

    # Verify exact step-0 functional identity on a deterministic sample.
    ident_rng=random.Random(SHADOW_SEED^0x1D)
    ident_idx=ident_rng.sample(range(len(items)),min(1024,len(items)))
    ident_samples=[items[i] for i in ident_idx]
    b,_,_=vectorized_batch(ident_samples,"cpu")
    sb=dict(b)
    sb["semantic"]=torch.tensor(
        np.asarray([semantic_vector_from_obs(s.observation) for s in ident_samples],dtype=np.float32)
    )
    # Structural identity must be exact: every pre-existing V1 parameter is
    # copied bit-for-bit and every newly-added semantic input weight is zero.
    old_named=dict(v1.named_parameters())
    new_named=dict(sem.named_parameters())
    for name,old_p in old_named.items():
        new_p=new_named[name]
        if name=="body.0.weight":
            old_width=old_p.shape[1]
            if not torch.equal(new_p[:,:old_width],old_p):
                raise RuntimeError("step0 structural drift in body.0 V1 columns")
            if not torch.count_nonzero(new_p[:,old_width:]).item()==0:
                raise RuntimeError("step0 semantic columns are not exactly zero")
        elif not torch.equal(new_p,old_p):
            raise RuntimeError(f"step0 structural parameter drift: {name}")

    # A wider GEMM can change float32 accumulation order even when the appended
    # semantic columns are exactly zero.  Accept only the tiny numerical drift
    # expected from that kernel-shape change, while the structural checks above
    # remain exact.
    with torch.no_grad():
        v1_logits=v1(b)
        sem_logits=sem(sb)
        diff=(v1_logits-sem_logits).abs().max().item()
        prob_diff=(
            torch.softmax(v1_logits.masked_fill(~b["legal"],-1e9),dim=-1)
            - torch.softmax(sem_logits.masked_fill(~b["legal"],-1e9),dim=-1)
        ).abs().max().item()
    if diff>2e-6 or prob_diff>5e-7:
        raise RuntimeError(
            f"step0 numerical identity drift: logits={diff} probs={prob_diff}"
        )

    split_rng=random.Random(SHADOW_SEED)
    holdout_idx=set(split_rng.sample(range(len(items)),min(int(args.holdout_size),len(items))))
    train_idx=[i for i in range(len(items)) if i not in holdout_idx]
    holdout_indices=sorted(holdout_idx)
    holdout=[items[i] for i in holdout_indices]

    print("POLICY_SEMANTIC_PRECOMPUTE_BEGIN",flush=True)
    sem_rows=np.empty((len(items),SEM_DIM),dtype=np.float32)
    t0=time.perf_counter()
    for i,s in enumerate(items):
        sem_rows[i]=semantic_vector_from_obs(s.observation)
        if (i+1)%200000==0:
            print(f"POLICY_SEMANTIC_PRECOMPUTE {i+1}/{len(items)}",flush=True)
    precompute_seconds=time.perf_counter()-t0
    print(f"POLICY_SEMANTIC_PRECOMPUTE_PASS seconds={precompute_seconds:.3f}",flush=True)

    holdout_sem=sem_rows[holdout_indices]
    batch_size=int(cfg["batch_size"])
    train_rng=random.Random(SHADOW_SEED^0x5A17)
    milestones=[]
    completed=0
    fit_start=time.perf_counter()

    def snapshot(step):
        m=policy_metrics(v1,sem,holdout,holdout_sem)
        m["extra_steps"]=int(step)
        milestones.append(m)
        print("POLICY_SEMANTIC_MILESTONE "+json.dumps({
            "step":step,
            "v1_ce":m["v1_weighted_ce"],
            "semantic_ce":m["semantic_weighted_ce"],
            "v1_tv":m["v1_weighted_tv"],
            "semantic_tv":m["semantic_weighted_tv"],
            "hc":m["high_card_no_draw_allin"],
        },sort_keys=True),flush=True)

    snapshot(0)
    for target_step in MILESTONES[1:]:
        while completed<int(target_step):
            pos=train_rng.sample(range(len(train_idx)),min(batch_size,len(train_idx)))
            idx=[train_idx[p] for p in pos]
            samples=[items[i] for i in idx]
            b,t,w=vectorized_batch(samples,"cpu")
            sb=dict(b)
            sb["semantic"]=torch.tensor(sem_rows[idx],dtype=torch.float32)
            train_step(v1,v1_opt,b,t,w,"strategy")
            train_step(sem,sem_opt,sb,t,w,"strategy")
            completed+=1
            if completed==500:
                elapsed=time.perf_counter()-fit_start
                projected=(elapsed/500.0)*MILESTONES[-1]/60.0
                print(
                    f"POLICY_SEMANTIC_PERF_GATE projected_pair_minutes={projected:.3f} "
                    f"limit={float(args.max_projected_minutes):.3f}",
                    flush=True,
                )
                if projected>float(args.max_projected_minutes):
                    raise RuntimeError("paired policy continuation exceeds performance gate")
        snapshot(completed)

    final=milestones[-1]
    hc=final["high_card_no_draw_allin"]

    solver=SolverLibrary(args.solver.resolve(strict=True))
    traces,games=replay_dc1(solver,args.spin_bundle.resolve(strict=True),args.scenarios)
    high=[];trips=[];three_handed=0
    for tr in traces:
        if tr.policy_id!=SPINCORE_POLICY_ID or tr.domain!=DOMAIN:
            continue
        three_handed+=1
        d=dict(tr.policy_detail or {})
        obs_hex=d.get("observation_hex");legal=tuple(int(x) for x in d.get("legal_slots") or ())
        if not obs_hex or not legal:raise RuntimeError("missing captured 3H state")
        op,vp,sp=exact_policy(original,v1,sem,bytes.fromhex(obs_hex),legal)
        for flag in sanity_flags(tr):
            row={
                "production_reference":op,
                "v1":vp,
                "semantic":sp,
                "context":flag.context,
            }
            if flag.code=="POSTFLOP_DEEP_HIGH_CARD_JAM":high.append(row)
            elif flag.code=="POSTFLOP_TRIPS_PLUS_FOLD":trips.append(row)
    nodraw=[r for r in high if not bool((r["context"] or {}).get("has_immediate_straight_or_flush_draw"))]

    base_bias=float(hc["v1_abs_bias"])
    sem_bias=float(hc["semantic_abs_bias"])
    criteria={
        "semantic_weighted_ce_not_worse_than_v1":bool(
            final["semantic_weighted_ce"]<=final["v1_weighted_ce"]
        ),
        "semantic_weighted_tv_not_worse_than_v1":bool(
            final["semantic_weighted_tv"]<=final["v1_weighted_tv"]
        ),
        "semantic_high_card_allin_abs_bias_reduced_by_at_least_50pct":bool(
            base_bias>0 and sem_bias<=0.5*base_bias
        ),
    }
    passed=all(criteria.values())

    args.out_model.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "schema":"SPINCORE_3H_AVERAGE_POLICY_V1_SEMANTIC_CONTINUATION_MODEL_V1",
        "checkpoint_sha256":actual,
        "extra_steps":MILESTONES[-1],
        "semantic_dim":SEM_DIM,
        "model_state":{k:v.detach().cpu() for k,v in sem.state_dict().items()},
        "production_status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
    },args.out_model)

    report={
        "schema":"SPINCORE_3H_AVERAGE_POLICY_V1_SEMANTIC_CONTINUATION_AUDIT_V1",
        "scope":"DIAGNOSTIC_ONLY_PAIRED_STRATEGY_REFIT_NO_CFR_ROOTS",
        "checkpoint_sha256":actual,
        "strategy_reservoir_retained":len(items),
        "strategy_reservoir_seen":int(mem.get("seen",0)),
        "holdout_size":len(holdout),
        "train_size":len(train_idx),
        "batch_size":batch_size,
        "learning_rate":float(cfg["learning_rate"]),
        "semantic_dim":SEM_DIM,
        "step0_max_abs_logit_diff":float(diff),
        "step0_max_abs_probability_diff":float(prob_diff),
        "step0_structural_identity_exact":True,
        "semantic_precompute_seconds":precompute_seconds,
        "milestones":milestones,
        "final_holdout":final,
        "dc1":{
            "scenarios":int(args.scenarios),
            "balanced_games":games,
            "trace_decisions":len(traces),
            "three_handed_decisions":three_handed,
            "high_card_jam_all":summarize_flag(high,"ALL_IN"),
            "high_card_jam_no_immediate_draw":summarize_flag(nodraw,"ALL_IN"),
            "trips_plus_fold":summarize_flag(trips,"FOLD"),
        },
        "precommitted_criteria":criteria,
        "average_policy_semantic_continuation_pass":passed,
        "interpretation":(
            "Because both arms start functionally identical and share the same optimizer "
            "history, train split and minibatch stream, a semantic advantage on strategy "
            "holdout CE/TV isolates representational benefit for AveragePolicy distillation. "
            "DC1 reports the untouched finalized production-reference AveragePolicy in "
            "addition to the paired 4000-step V1 control and semantic continuation. "
            "The weird-action rows are descriptive development diagnostics, not gate "
            "optimization targets. PASS still does not authorize deployment because the "
            "strategy reservoir was generated historically under the V1 learner."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("=== 3H AveragePolicy V1 vs V1+semantic continuation ===")
    print("final_holdout="+json.dumps(final,sort_keys=True))
    print("dc1="+json.dumps(report["dc1"],sort_keys=True))
    print("criteria="+json.dumps(criteria,sort_keys=True))
    print(f"average_policy_semantic_continuation_pass={passed}")
    print(f"model={args.out_model.resolve()}")
    print(f"report={args.report.resolve()}")
    print("3H_AVERAGE_POLICY_SEMANTIC_CONTINUATION_RUN_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
