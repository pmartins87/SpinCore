#!/usr/bin/env python3
from __future__ import annotations

"""Mechanical parity gate for the frozen LT2 hybrid deployment bundle.

No EV measurement and no policy selection. This checks that the compact
deployment artifact reproduces source inference semantics exactly enough across
3H and HU trajectories on already-seen forensic sampler seeds.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import torch

import audit_lt2_stage_a_b_first_divergence as fd
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_hybrid_deployment_agent import LeanHybridDeploymentAgent
from spincore.lean_solver_actions import apply_lean, lean_legal_actions, resolve_lean_exact
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import legal_mask
from spincore.solver import SolverLibrary
from spincore_nn.action_models import (
    collate_action_observations,
    make_advantage_action_model,
    make_policy_action_model,
)

REPRESENTATION="C0_V1_FROZEN_CONTROL"
FORENSIC_SEEDS=(20260920,20260921,20260922)
ENSEMBLE_SCHEMA="SPINCORE_LT2_HU_ENS8_CURRENT_STATE_V1"


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--ensemble",type=Path,required=True)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--scenarios-per-seed",type=int,default=1000)
    p.add_argument("--max-decisions-per-hand",type=int,default=200)
    p.add_argument("--tolerance",type=float,default=1e-6)
    p.add_argument("--report",type=Path,required=True)
    return p.parse_args()


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


class Reference:
    def __init__(self,checkpoint:Path,ensemble:Path):
        cp=torch.load(checkpoint,map_location="cpu",weights_only=False)
        ens=torch.load(ensemble,map_location="cpu",weights_only=False)
        _,self.p3=make_policy_action_model(REPRESENTATION,device="cpu",seed=0)
        self.p3.load_state_dict(cp["domains"]["THREE_HANDED"]["policy"])
        self.p3.eval()
        if ens.get("schema")!=ENSEMBLE_SCHEMA or len(ens.get("members") or [])!=8:
            raise RuntimeError("bad reference ensemble")
        self.hu=[]
        for i,state in enumerate(ens["members"]):
            _,m=make_advantage_action_model(REPRESENTATION,device="cpu",seed=i)
            m.load_state_dict(state)
            m.eval()
            self.hu.append(m)

    def distribution(self,state):
        payload=state.neural_bytes_v2()
        street=int(payload[112])
        active_mask=FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal=tuple(int(x) for x in lean_legal_actions(state,active_mask))
        obs=state.neural_bytes()
        batch=collate_action_observations(
            REPRESENTATION,[obs],[legal_mask(legal)],device="cpu"
        )
        with torch.no_grad():
            if int(state.domain)==0:
                probs=self.p3.probabilities(batch)[0].detach().cpu().tolist()
                out=tuple(float(x) for x in probs)
            elif int(state.domain)==1:
                raw=torch.stack([m(batch)[0] for m in self.hu],dim=0).mean(dim=0).detach().cpu().tolist()
                out=tuple(float(x) for x in lean_regret_matching_policy(raw,legal))
            else:
                raise RuntimeError("unsupported domain")
        return int(active_mask),legal,out


def sample(legal,probs,rng):
    x=rng.random()
    c=0.0
    for a in legal:
        c+=float(probs[a])
        if x<c:
            return int(a)
    return int(legal[-1])


def main():
    args=parse_args()
    if args.scenarios_per_seed<=0 or args.tolerance<=0:
        raise SystemExit("positive scenarios/tolerance required")
    torch.set_num_threads(8)
    solver=SolverLibrary(args.solver.resolve(strict=True))
    checkpoint=args.checkpoint.resolve(strict=True)
    ensemble=args.ensemble.resolve(strict=True)
    bundle=args.bundle.resolve(strict=True)
    agent=LeanHybridDeploymentAgent.from_bundle(bundle,device="cpu",seed=0)
    ref=Reference(checkpoint,ensemble)

    max_abs=0.0
    sum_abs=0.0
    compared_probs=0
    decisions=0
    hands={"THREE_HANDED":0,"TRUE_HEADS_UP":0}
    decisions_by_domain={"THREE_HANDED":0,"TRUE_HEADS_UP":0}
    legal_mismatch=0
    argmax_mismatch=0
    exact_resolution_mismatch=0

    for seed in FORENSIC_SEEDS:
        sampler=LegacyScenarioSampler(
            seed=int(seed)^0x5CE0A710,config=LegacyScenarioConfig()
        )
        for scenario in range(int(args.scenarios_per_seed)):
            episode=sampler.sample_episode()
            domain="TRUE_HEADS_UP" if episode.game_is_hu else "THREE_HANDED"
            hands[domain]+=1
            deal_seed=fd._mix64(int(seed),int(scenario),0xD34A1)
            state=solver.create(episode,int(deal_seed))
            rng=random.Random(fd._mix64(int(seed),int(scenario),0xD3A10))
            try:
                local=0
                while not state.terminal:
                    am,al,ap=agent.distribution(state)
                    rm,rl,rp=ref.distribution(state)
                    if am!=rm or al!=rl:
                        legal_mismatch+=1
                        raise RuntimeError("deployment/reference legal context mismatch")
                    diffs=[abs(float(ap[i])-float(rp[i])) for i in range(len(ap))]
                    d=max(diffs)
                    max_abs=max(max_abs,d)
                    sum_abs+=sum(diffs)
                    compared_probs+=len(diffs)
                    if max(al,key=lambda a:ap[a])!=max(rl,key=lambda a:rp[a]):
                        argmax_mismatch+=1

                    slot=sample(rl,rp,rng)
                    e1=resolve_lean_exact(state,am,slot)
                    e2=resolve_lean_exact(state,rm,slot)
                    if e1!=e2:
                        exact_resolution_mismatch+=1
                        raise RuntimeError("exact action resolution mismatch")
                    apply_lean(state,rm,slot)
                    decisions+=1
                    decisions_by_domain[domain]+=1
                    local+=1
                    if local>int(args.max_decisions_per_hand):
                        raise RuntimeError("parity trajectory exceeded decision cap")
            finally:
                state.close()

    passed=(
        legal_mismatch==0
        and argmax_mismatch==0
        and exact_resolution_mismatch==0
        and max_abs<=float(args.tolerance)
    )
    report={
        "schema":"SPINCORE_LT2_HYBRID_DEPLOYMENT_PARITY_V1",
        "verdict":"PASS" if passed else "FAIL",
        "artifacts":{
            "checkpoint_sha256":sha256(checkpoint),
            "ensemble_sha256":sha256(ensemble),
            "bundle_sha256":sha256(bundle),
        },
        "method":{
            "forensic_seeds":list(FORENSIC_SEEDS),
            "scenarios_per_seed":int(args.scenarios_per_seed),
            "training_roots":0,
            "optimizer_steps":0,
            "strategic_ev_evaluations":0,
            "holdout_reused":False,
            "tolerance":float(args.tolerance),
        },
        "hands":hands,
        "decisions":decisions,
        "decisions_by_domain":decisions_by_domain,
        "compared_probabilities":compared_probs,
        "max_abs_probability_diff":max_abs,
        "mean_abs_probability_diff":sum_abs/max(compared_probs,1),
        "legal_context_mismatches":legal_mismatch,
        "argmax_mismatches":argmax_mismatch,
        "exact_resolution_mismatches":exact_resolution_mismatch,
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("=== LT2 HYBRID DEPLOYMENT PARITY ===")
    print(f"VERDICT={report['verdict']}")
    print(f"hands={hands} decisions={decisions}")
    print(f"max_abs_probability_diff={max_abs:.12g}")
    print(f"argmax_mismatches={argmax_mismatch}")
    print("LT2_HYBRID_DEPLOYMENT_PARITY_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0 if passed else 2


if __name__=="__main__":
    raise SystemExit(main())
