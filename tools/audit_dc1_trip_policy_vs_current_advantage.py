#!/usr/bin/env python3
from __future__ import annotations

"""Replay the exact DC1 trips-fold hand and compare 3H AveragePolicy vs current Advantage.

Diagnostic only.  The action actually played is still sampled from the frozen
AveragePolicy with the original DC1 RNG contract.  The iteration-10105 3H
Advantage model is evaluated on the same observation without consuming RNG or
changing the hand trajectory.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))

from spincore.deepcrusher_benchmark import (
    DEEPC_RUSHER_POLICY_ID,
    SPINCORE_POLICY_ID,
    ExternalExactAction,
    OfflineHeadToHeadEngine,
    balanced_three_handed_lineups,
)
from spincore.deepcrusher_policy import DeepCrusherR8Policy
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_hybrid_deployment_agent import _street_from_state
from spincore.lean_solver_actions import lean_legal_actions, resolve_lean_exact
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore.solver import SolverLibrary
from spincore_nn.action_models import (
    collate_action_observations,
    make_advantage_action_model,
    make_policy_action_model,
)

EXPECTED_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
REPRESENTATION="C0_V1_FROZEN_CONTROL"
SCENARIO_INDEX=86
LINEUP_INDEX=2
MASTER_SEED=20260923
TARGET_HOLE=(40,26)
TARGET_BOARD=(24,27,11)


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


class PolicyVsAdvantageProbe:
    policy_id=SPINCORE_POLICY_ID

    def __init__(self,policy_model,advantage_model):
        self.policy_model=policy_model.eval()
        self.advantage_model=advantage_model.eval()
        self._last:dict[int,dict[str,Any]]={}

    def choose_exact(self,state,*,seat:int,rng:random.Random)->ExternalExactAction:
        street=_street_from_state(state)
        active_mask=FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal=tuple(int(x) for x in lean_legal_actions(state,active_mask))
        obs=state.neural_bytes()
        batch=collate_action_observations(
            REPRESENTATION,[obs],[legal_mask(legal)],device="cpu"
        )
        with torch.no_grad():
            p=self.policy_model.probabilities(batch)[0].detach().cpu().tolist()
            raw=self.advantage_model(batch)[0].detach().cpu().tolist()
        ap=tuple(float(x) for x in p)
        adv=tuple(float(x) for x in lean_regret_matching_policy(raw,legal))

        x=rng.random()
        cumulative=0.0
        slot=int(legal[-1])
        for candidate in legal:
            cumulative+=float(ap[candidate])
            if x<cumulative:
                slot=int(candidate)
                break
        action_type,amount_to=resolve_lean_exact(state,active_mask,slot)

        self._last[int(seat)]={
            "oracle":"PolicyVsAdvantageProbe",
            "selection":"SAMPLED_AVERAGE_POLICY",
            "active_mask":int(active_mask),
            "sample_u":float(x),
            "selected_slot":int(slot),
            "selected_slot_name":str(NAME_BY_SLOT.get(slot,f"SLOT_{slot}")),
            "resolved_action_type":int(action_type),
            "resolved_amount_to":int(amount_to),
            "average_policy":[
                {
                    "slot":int(a),
                    "name":str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")),
                    "probability":float(ap[a]),
                }
                for a in legal
            ],
            "current_advantage_raw":[
                {
                    "slot":int(a),
                    "name":str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")),
                    "raw":float(raw[a]),
                }
                for a in legal
            ],
            "current_advantage_policy":[
                {
                    "slot":int(a),
                    "name":str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")),
                    "probability":float(adv[a]),
                }
                for a in legal
            ],
        }
        return ExternalExactAction(int(action_type),int(amount_to))

    def decision_metadata(self,*,seat:int):
        value=self._last.get(int(seat))
        return None if value is None else dict(value)


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    args=ap.parse_args()

    cp=args.checkpoint.resolve(strict=True)
    actual=sha256(cp)
    if actual!=EXPECTED_SHA:
        raise SystemExit(f"checkpoint SHA mismatch: {actual}")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    if int(payload.get("completed_iteration",-1))!=10105:
        raise SystemExit("expected iteration 10105")
    d3=(payload.get("domains") or {}).get("THREE_HANDED") or {}
    if "policy" not in d3 or "advantage" not in d3:
        raise SystemExit("checkpoint missing 3H policy/advantage")

    _,policy=make_policy_action_model(REPRESENTATION,device="cpu",seed=0)
    policy.load_state_dict(d3["policy"])
    _,advantage=make_advantage_action_model(REPRESENTATION,device="cpu",seed=0)
    advantage.load_state_dict(d3["advantage"])

    solver=SolverLibrary(args.solver.resolve(strict=True))
    spin=PolicyVsAdvantageProbe(policy,advantage)
    deep=DeepCrusherR8Policy.from_repository(ROOT)

    sampler=LegacyScenarioSampler(
        seed=MASTER_SEED^0x5CE0A710,
        config=LegacyScenarioConfig(),
    )
    episode=None
    for i in range(SCENARIO_INDEX+1):
        episode=sampler.sample_episode()
    assert episode is not None
    if bool(episode.game_is_hu):
        raise RuntimeError("scenario 86 unexpectedly became HU")

    traces=[]
    engine=OfflineHeadToHeadEngine(
        solver,
        spincore_policy=spin,
        deepcrusher_policy=deep,
        master_seed=MASTER_SEED,
        max_decisions=200,
        decision_sink=traces.append,
    )
    lineup=balanced_three_handed_lineups()[LINEUP_INDEX]
    obs=engine.play_hand(
        episode,
        deal_seed=mix64(MASTER_SEED,SCENARIO_INDEX,0xD34A1),
        scenario_index=SCENARIO_INDEX,
        lineup=lineup,
        lineup_index=LINEUP_INDEX,
    )

    matches=[
        t for t in traces
        if int(t.actor)==1
        and int(t.street)==1
        and tuple(t.hole_cards or ())==TARGET_HOLE
        and tuple(t.board)==TARGET_BOARD
        and int(t.pot)==120
        and int(t.to_call)==60
    ]
    if len(matches)!=1:
        raise RuntimeError(f"expected one target trips decision, got {len(matches)}")
    t=matches[0]
    detail=dict(t.policy_detail or {})

    avg={row["name"]:float(row["probability"]) for row in detail["average_policy"]}
    cur={row["name"]:float(row["probability"]) for row in detail["current_advantage_policy"]}
    known=0.07327636331319809
    if abs(float(avg.get("FOLD",0.0))-known)>1e-7:
        raise RuntimeError(
            "target AveragePolicy distribution failed exact DC1 reproducibility gate: "
            f"fold={avg.get('FOLD')}"
        )
    if str(detail.get("selected_slot_name"))!="FOLD":
        raise RuntimeError("target sampled action no longer reproduces FOLD")

    legal_names=sorted(set(avg)|set(cur))
    tv=0.5*sum(abs(avg.get(k,0.0)-cur.get(k,0.0)) for k in legal_names)
    report={
        "schema":"SPINCORE_DC1_TRIPS_POLICY_VS_CURRENT_ADVANTAGE_V1",
        "checkpoint_sha256":actual,
        "completed_iteration":10105,
        "scope":"DIAGNOSTIC_ONLY_NO_DEPLOYMENT_CHANGE",
        "scenario_index":SCENARIO_INDEX,
        "lineup_index":LINEUP_INDEX,
        "lineup":list(lineup.seats),
        "terminal_chip_delta":list(obs.chip_delta),
        "target_trace":asdict(t),
        "average_policy":avg,
        "current_advantage_policy":cur,
        "current_advantage_raw":{
            row["name"]:float(row["raw"]) for row in detail["current_advantage_raw"]
        },
        "total_variation":float(tv),
        "average_policy_argmax":max(avg,key=avg.get),
        "current_advantage_argmax":max(cur,key=cur.get),
        "interpretation":(
            "AveragePolicy is the actual 3H DC1/deployment behavior. Current Advantage is the "
            "single iteration-10105 3H Advantage network and is diagnostic only; it is not an "
            "ensemble and must not be promoted from this one state."
        ),
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("=== DC1 exact trips state: AveragePolicy vs current 3H Advantage ===")
    print("average_policy="+json.dumps(avg,sort_keys=True))
    print("current_advantage_policy="+json.dumps(cur,sort_keys=True))
    print(f"total_variation={tv:.9f}")
    print(f"average_policy_argmax={report['average_policy_argmax']}")
    print(f"current_advantage_argmax={report['current_advantage_argmax']}")
    print(f"report={args.report.resolve()}")
    print("DC1_TRIPS_POLICY_VS_CURRENT_ADVANTAGE_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
