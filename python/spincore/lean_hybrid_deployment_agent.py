from __future__ import annotations

"""Hybrid inference agent for the validated LT2 deployment candidate.

THREE_HANDED uses the checkpoint's finalized AveragePolicy.
TRUE_HEADS_UP uses the frozen current ENS8 behavior:
mean eight raw Advantage outputs, then unchanged lean regret matching.

The bundle is intentionally inference-only and contains no optimizer state or
training reservoirs.
"""

from pathlib import Path
import random
from typing import Any

import torch

from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import lean_legal_actions, resolve_lean_exact
from spincore.r7_5_action_cfr import legal_mask
from spincore_nn.action_models import (
    collate_action_observations,
    make_advantage_action_model,
    make_policy_action_model,
)

DEPLOYMENT_SCHEMA="SPINCORE_LT2_HYBRID_DEPLOYMENT_V1"
GENERIC_INFERENCE_SCHEMA="SPINCORE_HYBRID_INFERENCE_V1"
ALLOWED_BUNDLE_SCHEMAS={DEPLOYMENT_SCHEMA,GENERIC_INFERENCE_SCHEMA}
REPRESENTATION="C0_V1_FROZEN_CONTROL"
DOMAIN_BY_ID={0:"THREE_HANDED",1:"TRUE_HEADS_UP"}
MODE_AVERAGE_POLICY="AVERAGE_POLICY"
MODE_HU_ENS8="ADVANTAGE_ENSEMBLE_RAW_MEAN_RM"


def _street_from_state(state)->int:
    payload=state.neural_bytes_v2()
    if len(payload)!=830 or not payload.startswith(b"SPNNIV2\x00"):
        raise RuntimeError("hybrid inference requires valid SPNNIV2 metadata")
    street=int(payload[112])
    if street not in (0,1,2,3):
        raise RuntimeError(f"invalid street id {street}")
    return street


class LeanHybridDeploymentAgent:
    def __init__(
        self,
        *,
        three_handed_policy:Any,
        hu_advantage_members:list[Any],
        device:str="cpu",
        seed:int=0,
        metadata:dict[str,Any]|None=None,
    ):
        if not hu_advantage_members:
            raise ValueError("HU ensemble must contain at least one member")
        self.three_handed_policy=three_handed_policy
        self.hu_advantage_members=list(hu_advantage_members)
        self.device=str(device)
        self.rng=random.Random(int(seed))
        self.metadata=dict(metadata or {})
        self.three_handed_policy.eval()
        for model in self.hu_advantage_members:
            model.eval()

    @classmethod
    def from_bundle(
        cls,
        path:str|Path,
        *,
        device:str="cpu",
        seed:int=0,
    )->"LeanHybridDeploymentAgent":
        payload=torch.load(Path(path),map_location=device,weights_only=False)
        if payload.get("schema") not in ALLOWED_BUNDLE_SCHEMAS:
            raise ValueError("wrong hybrid inference/deployment schema")
        if payload.get("representation")!=REPRESENTATION:
            raise ValueError("hybrid deployment representation drift")
        if payload.get("action_candidate")!=FIRST_RELEASE_ACTION_SPEC.candidate_id:
            raise ValueError("hybrid deployment action-scope drift")

        domains=dict(payload.get("domains") or {})
        d3=domains.get("THREE_HANDED") or {}
        dhu=domains.get("TRUE_HEADS_UP") or {}
        if d3.get("mode")!=MODE_AVERAGE_POLICY:
            raise ValueError("THREE_HANDED deployment mode drift")
        if dhu.get("mode")!=MODE_HU_ENS8:
            raise ValueError("TRUE_HEADS_UP deployment mode drift")

        _,policy=make_policy_action_model(REPRESENTATION,device=device,seed=0)
        policy.load_state_dict(d3["policy"])
        policy.eval()

        members=[]
        states=list(dhu.get("members") or [])
        expected=int(dhu.get("ensemble_size",0))
        if expected<=0 or len(states)!=expected:
            raise ValueError("HU ensemble member-count drift")
        for index,state in enumerate(states):
            _,model=make_advantage_action_model(
                REPRESENTATION,device=device,seed=index
            )
            model.load_state_dict(state)
            model.eval()
            members.append(model)

        return cls(
            three_handed_policy=policy,
            hu_advantage_members=members,
            device=device,
            seed=seed,
            metadata={
                "completed_iteration":int(payload.get("completed_iteration",-1)),
                "source_checkpoint_sha256":payload.get("source_checkpoint_sha256"),
                "source_ensemble_sha256":payload.get("source_ensemble_sha256"),
                "ensemble_size":expected,
            },
        )

    @staticmethod
    def domain_for_state(state)->str:
        try:
            return DOMAIN_BY_ID[int(state.domain)]
        except KeyError as exc:
            raise RuntimeError(f"unsupported solver domain id {state.domain}") from exc

    def distribution(self,state)->tuple[int,tuple[int,...],tuple[float,...]]:
        if state.terminal:
            raise ValueError("cannot infer action on terminal state")
        street=_street_from_state(state)
        active_mask=FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal=tuple(int(x) for x in lean_legal_actions(state,active_mask))
        if not legal:
            raise RuntimeError("nonterminal state has no legal action")

        obs=state.neural_bytes()
        batch=collate_action_observations(
            REPRESENTATION,[obs],[legal_mask(legal)],device=self.device
        )

        domain=self.domain_for_state(state)
        with torch.no_grad():
            if domain=="THREE_HANDED":
                probs=self.three_handed_policy.probabilities(batch)[0].detach().cpu().tolist()
                out=tuple(float(x) for x in probs)
            else:
                raw=torch.stack(
                    [m(batch)[0] for m in self.hu_advantage_members],dim=0
                ).mean(dim=0).detach().cpu().tolist()
                out=tuple(float(x) for x in lean_regret_matching_policy(raw,legal))

        total=sum(out[a] for a in legal)
        if not (0.999<=total<=1.001):
            raise RuntimeError(f"deployment probability mass drift: {total}")
        if any(out[a]<0.0 for a in legal):
            raise RuntimeError("negative deployment probability")
        return int(active_mask),legal,out

    def choose_slot(self,state,*,greedy:bool=False)->int:
        _mask,legal,probs=self.distribution(state)
        if greedy:
            return int(max(legal,key=lambda a:probs[a]))
        x=self.rng.random()
        cumulative=0.0
        for action in legal:
            cumulative+=float(probs[action])
            if x<cumulative:
                return int(action)
        return int(legal[-1])

    def choose_exact(self,state,*,greedy:bool=False)->dict[str,int]:
        active_mask,_legal,_probs=self.distribution(state)
        slot=self.choose_slot(state,greedy=greedy)
        action_type,amount_to=resolve_lean_exact(state,active_mask,slot)
        return {
            "slot":int(slot),
            "action_type":int(action_type),
            "amount_to":int(amount_to),
            "active_mask":int(active_mask),
        }
