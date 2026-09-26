#!/usr/bin/env python3
from __future__ import annotations

"""Audit frozen 10105 3H Advantage targets around DC1 high-card jams.

The fixed 200-scenario DC1 trajectory is replayed with the unchanged actual
hybrid policy only to recover the 3H flagged state observations.  The complete
2M retained 3H Advantage reservoir is then scanned once.

For each flagged high-card jam, progressively tighter morphology/geometry
subsets report whether the *stored Advantage targets themselves* favor ALL_IN.
This separates neural-fit/generalization from target-level signal.

No training, CFR roots, checkpoint mutation or strategy patch occurs.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import sys
from typing import Any

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))

from spincore.deepcrusher_benchmark import (
    SPINCORE_POLICY_ID,
    ExternalExactAction,
    OfflineHeadToHeadEngine,
)
from spincore.deepcrusher_policy import DeepCrusherR8Policy
from spincore.decision_sanity import sanity_flags
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore.lean_solver_actions import resolve_lean_exact
from spincore.lean_hybrid_deployment_agent import LeanHybridDeploymentAgent
from spincore.solver import SolverLibrary

EXPECTED_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
DOMAIN="THREE_HANDED"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
SEED=20260923
ALL_IN=9


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


def rank(card:int)->int:
    return 2+int(card)//4


def suit(card:int)->int:
    return int(card)%4


def straight_templates():
    return [
        {14,2,3,4,5},
        {2,3,4,5,6},
        {3,4,5,6,7},
        {4,5,6,7,8},
        {5,6,7,8,9},
        {6,7,8,9,10},
        {7,8,9,10,11},
        {8,9,10,11,12},
        {9,10,11,12,13},
        {10,11,12,13,14},
    ]
STRAIGHTS=straight_templates()


def high_card_no_immediate_draw(hole,board):
    cards=tuple(hole)+tuple(board)
    ranks=[rank(c) for c in cards]
    if len(set(ranks))!=len(ranks):
        return False
    rset=set(ranks)
    if any(t.issubset(rset) for t in STRAIGHTS):
        return False
    suits=[suit(c) for c in cards]
    if max(suits.count(s) for s in set(suits))>=5:
        return False
    if len(board)>=5:
        return True
    # One unseen next card makes a flush if four cards already share a suit.
    if max(suits.count(s) for s in set(suits))>=4:
        return False
    # One unseen next card makes a straight iff a 5-rank template already has 4.
    if any(len(t & rset)>=4 for t in STRAIGHTS):
        return False
    return True


def decode_obs(obs:bytes):
    if len(obs)!=126 or obs[:8]!=b"SPNNIV1\x00":
        raise ValueError("bad SPNNIV1 observation")
    cards=tuple(int(x) for x in obs[8:15])
    numeric=struct.unpack_from("<16f",obs,15)
    cat=tuple(int(x) for x in obs[79:87])
    hlen=int(obs[93])
    hist=tuple(int(x) for x in obs[94:94+hlen])
    hole=tuple(x-1 for x in cards[:2] if x)
    board=tuple(x-1 for x in cards[2:] if x)
    return hole,board,tuple(float(x) for x in numeric),cat,hist


def effective_stack(numeric,cat):
    hero=float(numeric[3])
    opp=[]
    for rel in (1,2):
        status=int(cat[4+rel])
        if status!=1:  # folded=1; active=0; all-in=2 still contests the pot
            opp.append(float(numeric[3+rel]))
    if not opp:
        return 0.0
    return min(hero,max(opp))


def facing_class(to_call_bb:float)->str:
    return "FACING_ACTION" if to_call_bb>1e-9 else "CHECKED_TO"


def ratio(a,b):
    return float(a)/float(b) if float(b)>1e-9 else 0.0


def near_ratio(value,target,tol=0.15):
    return abs(float(value)-float(target))<=float(tol)


def near_scale(value,target,lo=0.70,hi=1.43):
    value=float(value);target=float(target)
    if target<=1e-9:
        return abs(value)<=0.25
    return lo*target<=value<=hi*target


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
            "oracle":"HighCardTargetCapture",
            "selection":"SAMPLED_ACTUAL_HYBRID_POLICY",
            "domain":domain,
            "sample_u":float(x),
            "selected_slot":int(slot),
            "selected_slot_name":str(NAME_BY_SLOT.get(slot,f"SLOT_{slot}")),
            "legal_slots":[int(a) for a in legal],
            "observation_hex":state.neural_bytes().hex() if domain==DOMAIN else None,
        }
        return ExternalExactAction(int(action_type),int(amount_to))

    def decision_metadata(self,*,seat:int):
        row=self._last.get(int(seat))
        return None if row is None else dict(row)


def replay_targets(solver,bundle_path,scenarios):
    agent=LeanHybridDeploymentAgent.from_bundle(bundle_path,device="cpu",seed=0)
    spin=CapturePolicy(agent)
    deep=DeepCrusherR8Policy.from_repository(ROOT)
    sampler=LegacyScenarioSampler(seed=SEED^0x5CE0A710,config=LegacyScenarioConfig())
    traces=[]
    for index in range(int(scenarios)):
        episode=sampler.sample_episode()
        engine=OfflineHeadToHeadEngine(
            solver,
            spincore_policy=spin,
            deepcrusher_policy=deep,
            master_seed=SEED,
            max_decisions=200,
            decision_sink=traces.append,
        )
        engine.play_balanced_block(
            episode,
            deal_seed=int(mix64(SEED,index,0xD34A1)),
            scenario_index=int(index),
        )

    targets=[]
    for t in traces:
        if t.policy_id!=SPINCORE_POLICY_ID or t.domain!=DOMAIN:
            continue
        flags=[f for f in sanity_flags(t) if f.code=="POSTFLOP_DEEP_HIGH_CARD_JAM"]
        if not flags:
            continue
        d=dict(t.policy_detail or {})
        obs_hex=d.get("observation_hex")
        legal=tuple(int(x) for x in d.get("legal_slots") or ())
        if not obs_hex or ALL_IN not in legal:
            raise RuntimeError("flagged jam missing captured 3H observation/legal set")
        hole,board,n,c,hist=decode_obs(bytes.fromhex(obs_hex))
        target={
            "scenario_index":int(t.scenario_index),
            "lineup_index":int(t.lineup_index),
            "decision_index":int(t.decision_index),
            "street":int(c[1]),
            "hole":[int(x) for x in hole],
            "board":[int(x) for x in board],
            "pot_bb":float(n[0]),
            "to_call_bb":float(n[1]),
            "current_bet_bb":float(n[2]),
            "hero_stack_bb":float(n[3]),
            "sanity_effective_stack_bb":float(flags[0].context.get("effective_stack_bb")),
            "contesting_effective_stack_bb":effective_stack(n,c),
            "to_call_over_pot":ratio(n[1],n[0]),
            "current_bet_over_pot":ratio(n[2],n[0]),
            "dealer_rel":int(c[2]),
            "live_count":int(c[3]),
            "statuses":[int(x) for x in c[4:7]],
            "legal":[int(x) for x in legal],
            "history":[int(x) for x in hist],
            "history_suffix2":[int(x) for x in hist[-2:]],
            "facing_class":facing_class(n[1]),
            "has_immediate_draw":bool(flags[0].context.get("has_immediate_straight_or_flush_draw")),
        }
        targets.append(target)
    return targets,len(traces)


class Acc:
    def __init__(self):
        self.n=0;self.w=0.0;self.ai=0.0;self.aiw=0.0
        self.pos=0;self.argmax=0;self.solepos=0
        self.cohorts={k:[0,0.0,0.0,0] for k in ("LE_8100","8101_9105","9106_10105")}
    def add(self,sample):
        legal=[i for i,v in enumerate(sample.legal) if bool(v)]
        if ALL_IN not in legal:return
        val=float(sample.target[ALL_IN]);w=float(sample.weight);it=int(sample.iteration)
        self.n+=1;self.w+=w;self.ai+=val;self.aiw+=val*w
        self.pos+=int(val>0.0)
        self.argmax+=int(max(legal,key=lambda a:float(sample.target[a]))==ALL_IN)
        self.solepos+=int(val>0.0 and all(float(sample.target[a])<=0.0 for a in legal if a!=ALL_IN))
        key="LE_8100" if it<=8100 else ("8101_9105" if it<=9105 else "9106_10105")
        c=self.cohorts[key];c[0]+=1;c[1]+=val;c[2]+=val*w;c[3]+=int(val>0.0)
    def out(self):
        cohorts={}
        for k,(n,s,sw,p) in self.cohorts.items():
            cohorts[k]={
                "count":n,
                "all_in_target_mean":s/n if n else None,
                "all_in_target_positive_fraction":p/n if n else None,
            }
        return {
            "count":self.n,
            "all_in_target_mean":self.ai/self.n if self.n else None,
            "all_in_target_iteration_weighted_mean":self.aiw/self.w if self.w else None,
            "all_in_target_positive_fraction":self.pos/self.n if self.n else None,
            "all_in_target_argmax_fraction":self.argmax/self.n if self.n else None,
            "all_in_only_positive_legal_fraction":self.solepos/self.n if self.n else None,
            "iteration_cohorts":cohorts,
        }


def candidate_features(sample):
    if not bool(sample.legal[ALL_IN]):
        return None
    hole,board,n,c,hist=decode_obs(sample.observation)
    if len(hole)!=2 or len(board)<3:
        return None
    if int(c[0])!=0 or int(c[1])==0 or int(c[3])!=3:
        return None
    eff=effective_stack(n,c)
    if not high_card_no_immediate_draw(hole,board):
        return None
    return {
        "hole":hole,"board":board,"numeric":n,"cat":c,"hist":hist,
        "street":int(c[1]),"dealer_rel":int(c[2]),"live_count":int(c[3]),
        "statuses":tuple(int(x) for x in c[4:7]),
        "legal_mask":tuple(int(x) for x in sample.legal),
        "legal_slots":tuple(i for i,v in enumerate(sample.legal) if bool(v)),
        "pot_bb":float(n[0]),"to_call_bb":float(n[1]),"current_bet_bb":float(n[2]),
        "hero_stack_bb":float(n[3]),"effective_stack_bb":eff,
        "to_call_over_pot":ratio(n[1],n[0]),
        "current_bet_over_pot":ratio(n[2],n[0]),
        "facing_class":facing_class(n[1]),
        "hole_ranks":tuple(sorted((rank(hole[0]),rank(hole[1])),reverse=True)),
    }


def match_stage(f,t,stage):
    if stage=="A_all_deep_high_card_no_draw":
        return True
    if f["street"]!=t["street"] or f["facing_class"]!=t["facing_class"]:
        return False
    if stage=="B_same_street_facing":
        return True
    if (
        f["dealer_rel"]!=t["dealer_rel"]
        or list(f["statuses"])!=list(t["statuses"])
        or list(f["legal_slots"])!=list(t["legal"])
    ):
        return False
    if stage=="C_same_public_structure":
        return True
    if not (
        near_scale(f["effective_stack_bb"],t["contesting_effective_stack_bb"])
        and near_scale(f["pot_bb"],t["pot_bb"])
        and near_ratio(f["to_call_over_pot"],t["to_call_over_pot"])
        and near_ratio(f["current_bet_over_pot"],t["current_bet_over_pot"])
    ):
        return False
    if stage=="D_near_geometry":
        return True
    target_suffix=tuple(t["history_suffix2"])
    if target_suffix and tuple(f["hist"][-len(target_suffix):])!=target_suffix:
        return False
    if stage=="E_near_geometry_history_suffix2":
        return True
    target_hole=tuple(sorted((rank(t["hole"][0]),rank(t["hole"][1])),reverse=True))
    if f["hole_ranks"]!=target_hole:
        return False
    if stage=="F_same_hole_ranks":
        return True
    raise ValueError(stage)


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--spin-bundle",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--scenarios",type=int,default=200)
    args=ap.parse_args()

    cp=args.checkpoint.resolve(strict=True)
    actual=sha256(cp)
    if actual!=EXPECTED_SHA:
        raise SystemExit(f"checkpoint SHA mismatch: {actual}")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    d3=(payload.get("domains") or {}).get(DOMAIN) or {}
    mem=d3.get("adv_mem") or {}
    items=list(mem.get("items") or [])
    if not items:raise SystemExit("empty 3H Advantage reservoir")

    solver=SolverLibrary(args.solver.resolve(strict=True))
    targets,trace_count=replay_targets(solver,args.spin_bundle.resolve(strict=True),args.scenarios)
    nodraw=[t for t in targets if not t["has_immediate_draw"]]
    if len(targets)!=23 or len(nodraw)!=17:
        raise RuntimeError(f"expected 23/17 fixed DC1 targets, got {len(targets)}/{len(nodraw)}")

    stages=(
        "A_all_deep_high_card_no_draw",
        "B_same_street_facing",
        "C_same_public_structure",
        "D_near_geometry",
        "E_near_geometry_history_suffix2",
        "F_same_hole_ranks",
    )
    broad=Acc()
    accs=[{stage:Acc() for stage in stages} for _ in nodraw]
    scanned=0;base=0
    for sample in items:
        scanned+=1
        f=candidate_features(sample)
        if f is None:
            continue
        base+=1
        broad.add(sample)
        for i,t in enumerate(nodraw):
            for stage in stages:
                if match_stage(f,t,stage):
                    accs[i][stage].add(sample)
                else:
                    # Stages are progressive/nested after A.
                    if stage!="A_all_deep_high_card_no_draw":
                        break

    per_target=[]
    for t,rows in zip(nodraw,accs):
        per_target.append({
            "target":t,
            "subsets":{stage:rows[stage].out() for stage in stages},
        })

    result={
        "schema":"SPINCORE_3H_HIGH_CARD_ADVANTAGE_TARGET_AUDIT_V2",
        "scope":"DIAGNOSTIC_ONLY_NO_TRAINING",
        "checkpoint_sha256":actual,
        "scenarios":int(args.scenarios),
        "trace_decisions":trace_count,
        "all_high_card_jam_flags":len(targets),
        "no_immediate_draw_flags":len(nodraw),
        "advantage_reservoir_retained":len(items),
        "advantage_reservoir_seen":int(mem.get("seen",0)),
        "base_no_draw_deep_high_card_allin_legal":broad.out(),
        "subset_contract":{
            "A":"all 3H postflop HIGH_CARD, no immediate straight/flush draw, ALL_IN legal; no stack cutoff so the fixed DC1 flags are not reclassified by a different effective-stack definition",
            "B":"A + same street + same facing-action/check-to class as target",
            "C":"B + same dealer_rel + actor-relative statuses + exact universal legal-slot set",
            "D":"C + contesting-opponent effective stack and pot within x0.70..x1.43 + call/pot and current-bet/pot within +/-0.15",
            "E":"D + exact last two frozen V1 public-history tokens (when target history nonempty)",
            "F":"E + same two hole-card ranks ignoring suits",
        },
        "targets":per_target,
        "interpretation":(
            "V2 fixes V1's mask-vs-slot comparison bug at subset C and removes V1's inconsistent "
            "secondary >=10bb cutoff. The original sanity flag's effective_stack_bb uses all non-DEAD "
            "lineup opponents because DecisionTrace does not carry folded status; V2 separately records "
            "that provenance value and uses actor-relative folded statuses to compute contesting-opponent "
            "effective stack for neighborhood geometry. Advantage targets are signed sampled counterfactual regrets: positive ALL_IN means "
            "the traversal estimated ALL_IN above the node value for that stored sample. "
            "Persistent positive/argmax ALL_IN target mass in recent tight neighborhoods means "
            "the aggressive signal is already present in the training targets, not created only "
            "by neural fitting. Sparse or contradictory tight neighborhoods instead point toward "
            "generalization and/or external-sampling variance. This audit does not prove poker "
            "optimality of a jam."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== 3H high-card Advantage target audit ===")
    print("broad="+json.dumps(result["base_no_draw_deep_high_card_allin_legal"],sort_keys=True))
    for i,row in enumerate(per_target):
        t=row["target"]
        print(
            f"TARGET_{i+1:02d} scenario={t['scenario_index']} decision={t['decision_index']} "
            + " ".join(
                f"{stage.split('_',1)[0]}_n={row['subsets'][stage]['count']}"
                for stage in stages
            )
            + f" D_pos={row['subsets']['D_near_geometry']['all_in_target_positive_fraction']}"
            + f" E_pos={row['subsets']['E_near_geometry_history_suffix2']['all_in_target_positive_fraction']}",
            flush=True,
        )
    print(f"report={args.report.resolve()}")
    print("3H_HIGH_CARD_ADVANTAGE_TARGET_AUDIT_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
