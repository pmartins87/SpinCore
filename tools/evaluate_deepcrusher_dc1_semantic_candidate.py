#!/usr/bin/env python3
from __future__ import annotations

"""DC1 development benchmark for the 10115 3H semantic AveragePolicy candidate.

THREE_HANDED uses the research-only V1+general-semantic tail AveragePolicy
produced by the 10105->10115 semantic continuation.
TRUE_HEADS_UP is unchanged and uses the frozen 10105 HU ENS8 from the ordinary
hybrid inference bundle.

This remains DEVELOPMENT_ONLY until DC0 real-OpenHoldem action+sizing parity is
complete.  It is intended for paired comparison against the unchanged 10105
baseline on identical scenario/deal seeds.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import evaluate_deepcrusher_dc1 as base
from audit_3h_average_policy_semantic_continuation_10105 import (
    V1SemanticPolicyNet,
    semantic_vector_from_obs,
)

from spincore.deepcrusher_benchmark import (
    SPINCORE_POLICY_ID,
    ExternalExactAction,
    OfflineHeadToHeadEngine,
)
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import lean_legal_actions, resolve_lean_exact
from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore_nn.action_models import collate_action_observations

EXPECTED_SOURCE_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
SEMANTIC_POLICY_SCHEMA="SPINCORE_3H_SEMANTIC_RESEARCH_TAIL_POLICY_V1"
SEMANTIC_DIVERSITY_SCHEMA="SPINCORE_3H_SEMANTIC_STRONG_DIVERSITY_TAIL_CANDIDATE_V1"
SEMANTIC_STRATIFIED_DIVERSITY_SCHEMA="SPINCORE_3H_SEMANTIC_STRATIFIED_STRONG_DIVERSITY_TAIL_CANDIDATE_V1"
SEMANTIC_FULLPOOL_DIVERSITY_SCHEMA="SPINCORE_3H_SEMANTIC_FULLPOOL_STRONG_DIVERSITY_TAIL_CANDIDATE_V1"
SEMANTIC_STRONG_SPECIALIST_MOE_SCHEMA="SPINCORE_3H_SEMANTIC_STRONG_SPECIALIST_MOE_V1"
SEMANTIC_CONFIDENCE_GATED_MOE_SCHEMA="SPINCORE_3H_SEMANTIC_CONFIDENCE_GATED_STRONG_MOE_V1"
SEMANTIC_STRATIFIED_SPECIALIST_MOE_SCHEMA="SPINCORE_3H_SEMANTIC_STRATIFIED_STRONG_SPECIALIST_MOE_V1"
SEMANTIC_FOLD_LOGIT_CALIBRATED_SCHEMA="SPINCORE_3H_SEMANTIC_FOLD_LOGIT_CALIBRATED_STRONG_SPECIALIST_MOE_V1"
SEMANTIC_COMPLETED_ITERATION=10115
REPRESENTATION="C0_V1_FROZEN_CONTROL"

_SOLVER=None
_SPIN=None
_DC=None
_ENGINE_SEED=None


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,required=True)
    p.add_argument("--base-spin-bundle",type=Path,required=True)
    p.add_argument("--semantic-policy",type=Path,required=True)
    p.add_argument("--scenarios",type=int,default=1000)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--seed",type=int,default=20260923)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--traces",type=Path,required=True)
    p.add_argument("--max-decisions",type=int,default=200)
    return p.parse_args()


def street_from_state(state):
    payload=state.neural_bytes_v2()
    if len(payload)!=830 or not payload.startswith(b"SPNNIV2\x00"):
        raise RuntimeError("semantic benchmark requires valid SPNNIV2 metadata")
    street=int(payload[112])
    if street not in (0,1,2,3):
        raise RuntimeError(f"invalid street id {street}")
    return street


class StrongSpecialistMoEPolicyNet(torch.nn.Module):
    def __init__(self,base_model,specialist_model):
        super().__init__()
        self.base_model=base_model.eval()
        self.specialist_model=specialist_model.eval()
        self._spincore_strong_specialist_moe=True

    def probabilities(self,batch):
        base=self.base_model.probabilities(batch)
        specialist=self.specialist_model.probabilities(batch)
        semantic=batch["semantic"]
        made_ge_trips=semantic[:,3:9].sum(dim=1)>0.5
        fold_legal=batch["legal"][:,0]
        route=(made_ge_trips & fold_legal).unsqueeze(1)
        return torch.where(route,specialist,base)


class StrongConfidenceGatedMoEPolicyNet(torch.nn.Module):
    def __init__(self,base_model,specialist_model,fold_threshold:float):
        super().__init__()
        self.base_model=base_model.eval()
        self.specialist_model=specialist_model.eval()
        self.fold_threshold=float(fold_threshold)
        self._spincore_confidence_gated_moe=True

    def probabilities(self,batch):
        base=self.base_model.probabilities(batch)
        specialist=self.specialist_model.probabilities(batch)
        semantic=batch["semantic"]
        made_ge_trips=semantic[:,3:9].sum(dim=1)>0.5
        fold_legal=batch["legal"][:,0]
        route=(
            made_ge_trips
            & fold_legal
            & (base[:,0] <= self.fold_threshold)
        ).unsqueeze(1)
        return torch.where(route,specialist,base)

    def forward(self,batch):
        probs=self.probabilities(batch).clamp_min(1e-30)
        return torch.log(probs)


class FoldLogitCalibratedStrongMoEPolicyNet(torch.nn.Module):
    def __init__(self,base_model,specialist_model,fold_logit_scale:float,fold_logit_bias:float):
        super().__init__()
        self.base_model=base_model.eval()
        self.specialist_model=specialist_model.eval()
        self.fold_logit_scale=float(fold_logit_scale)
        self.fold_logit_bias=float(fold_logit_bias)
        self._spincore_fold_logit_calibrated_moe=True

    def probabilities(self,batch):
        base=self.base_model.probabilities(batch)
        specialist=self.specialist_model.probabilities(batch)
        semantic=batch["semantic"]
        strong=semantic[:,3:9].sum(dim=1)>0.5
        fold_legal=batch["legal"][:,0]
        route=(strong & fold_legal).unsqueeze(1)

        pf=specialist[:,0].clamp(1e-8,1.0-1e-8)
        logit=torch.log(pf)-torch.log1p(-pf)
        q=torch.sigmoid(
            self.fold_logit_scale*logit+self.fold_logit_bias
        )
        remain=(1.0-q).clamp_min(0.0)
        nonfold=specialist[:,1:]
        nonfold_sum=nonfold.sum(dim=1).clamp_min(1e-8)
        calibrated_nonfold=nonfold*(remain/nonfold_sum).unsqueeze(1)
        calibrated=torch.cat([q.unsqueeze(1),calibrated_nonfold],dim=1)
        return torch.where(route,calibrated,base)

    def forward(self,batch):
        return torch.log(self.probabilities(batch).clamp_min(1e-30))


class SemanticHybridBenchmarkPolicy:
    policy_id=SPINCORE_POLICY_ID

    def __init__(self,base_agent,semantic_policy):
        self.base_agent=base_agent
        self.semantic_policy=semantic_policy.eval()
        self._last={}

    def domain_for_state(self,state):
        return self.base_agent.domain_for_state(state)

    def three_handed_mode_label(self,domain):
        if domain!="THREE_HANDED":
            return "UNCHANGED_HU_ENS8_10105"
        if getattr(self.semantic_policy,"_spincore_fold_logit_calibrated_moe",False):
            return "V1_GENERAL_SEMANTIC_FOLD_LOGIT_CALIBRATED_STRONG_MOE_10115"
        if getattr(self.semantic_policy,"_spincore_confidence_gated_moe",False):
            return "V1_GENERAL_SEMANTIC_CONFIDENCE_GATED_STRONG_MOE_10115"
        if getattr(self.semantic_policy,"_spincore_stratified_specialist_moe",False):
            return "V1_GENERAL_SEMANTIC_STRATIFIED_STRONG_SPECIALIST_MOE_10115"
        if getattr(self.semantic_policy,"_spincore_strong_specialist_moe",False):
            return "V1_GENERAL_SEMANTIC_STRONG_SPECIALIST_MOE_10115"
        if getattr(self.semantic_policy,"_spincore_fullpool_diversity_candidate",False):
            return "V1_GENERAL_SEMANTIC_FULLPOOL_STRONG_DIVERSITY_AVERAGE_POLICY_10115"
        if getattr(self.semantic_policy,"_spincore_stratified_diversity_candidate",False):
            return "V1_GENERAL_SEMANTIC_STRATIFIED_STRONG_DIVERSITY_AVERAGE_POLICY_10115"
        if getattr(self.semantic_policy,"_spincore_diversity_candidate",False):
            return "V1_GENERAL_SEMANTIC_STRONG_DIVERSITY_AVERAGE_POLICY_10115"
        return "V1_GENERAL_SEMANTIC_AVERAGE_POLICY_10115"

    def distribution(self,state):
        domain=self.domain_for_state(state)
        if domain=="TRUE_HEADS_UP":
            return self.base_agent.distribution(state)

        street=street_from_state(state)
        active=FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal=tuple(int(x) for x in lean_legal_actions(state,active))
        if not legal:
            raise RuntimeError("nonterminal 3H state has no legal action")
        obs=state.neural_bytes()
        batch=collate_action_observations(
            REPRESENTATION,[obs],[legal_mask(legal)],device="cpu"
        )
        sb=dict(batch)
        sb["semantic"]=torch.tensor(
            np.asarray([semantic_vector_from_obs(obs)],dtype=np.float32),
            dtype=torch.float32,
        )
        with torch.no_grad():
            probs=self.semantic_policy.probabilities(sb)[0].detach().cpu().tolist()
        out=tuple(float(x) for x in probs)
        mass=sum(out[a] for a in legal)
        if not (0.999<=mass<=1.001):
            raise RuntimeError(f"semantic 3H probability mass drift: {mass}")
        return int(active),legal,out

    def choose_exact(self,state,*,seat:int,rng:random.Random):
        active,legal,probs=self.distribution(state)
        domain=self.domain_for_state(state)
        x=rng.random()
        cumulative=0.0
        slot=int(legal[-1])
        for a in legal:
            cumulative+=float(probs[a])
            if x<cumulative:
                slot=int(a)
                break
        action_type,amount_to=resolve_lean_exact(state,active,slot)
        self._last[int(seat)]={
            "oracle":"SemanticHybridBenchmarkPolicy",
            "selection":"SAMPLED_POLICY",
            "domain":domain,
            "active_mask":int(active),
            "sample_u":float(x),
            "legal_actions":[
                {
                    "slot":int(a),
                    "name":str(NAME_BY_SLOT.get(int(a),f"SLOT_{int(a)}")),
                    "probability":float(probs[a]),
                }
                for a in legal
            ],
            "selected_slot":int(slot),
            "selected_slot_name":str(NAME_BY_SLOT.get(int(slot),f"SLOT_{int(slot)}")),
            "selected_probability":float(probs[slot]),
            "resolved_action_type":int(action_type),
            "resolved_amount_to":int(amount_to),
            "three_handed_mode":self.three_handed_mode_label(domain),
        }
        return ExternalExactAction(int(action_type),int(amount_to))

    def decision_metadata(self,*,seat:int):
        row=self._last.get(int(seat))
        return None if row is None else dict(row)


def init_worker(solver_path,bundle_path,semantic_policy_path,root_path,seed):
    global _SOLVER,_SPIN,_DC,_ENGINE_SEED
    import os
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    os.environ["SPINCORE_TORCH_THREADS"]="1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    from spincore.solver import SolverLibrary
    from spincore.deepcrusher_policy import DeepCrusherR8Policy
    from spincore.lean_hybrid_deployment_agent import LeanHybridDeploymentAgent

    _SOLVER=SolverLibrary(solver_path)
    if not _SOLVER.explicit_deal_available:
        raise RuntimeError("semantic DC1 requires explicit-deal solver ABI")

    agent=LeanHybridDeploymentAgent.from_bundle(bundle_path,device="cpu",seed=0)
    payload=torch.load(semantic_policy_path,map_location="cpu",weights_only=False)
    schema=str(payload.get("schema"))
    if schema not in (
        SEMANTIC_POLICY_SCHEMA,
        SEMANTIC_DIVERSITY_SCHEMA,
        SEMANTIC_STRATIFIED_DIVERSITY_SCHEMA,
        SEMANTIC_FULLPOOL_DIVERSITY_SCHEMA,
        SEMANTIC_STRONG_SPECIALIST_MOE_SCHEMA,
        SEMANTIC_CONFIDENCE_GATED_MOE_SCHEMA,
        SEMANTIC_STRATIFIED_SPECIALIST_MOE_SCHEMA,
        SEMANTIC_FOLD_LOGIT_CALIBRATED_SCHEMA,
    ):
        raise RuntimeError(f"wrong semantic tail-policy schema: {schema!r}")
    if payload.get("source_checkpoint_sha256")!=EXPECTED_SOURCE_SHA:
        raise RuntimeError("semantic tail-policy source mismatch")
    completed_iteration=int(
        payload.get(
            "completed_iteration",
            payload.get("semantic_completed_iteration",-1),
        )
    )
    if completed_iteration!=SEMANTIC_COMPLETED_ITERATION:
        raise RuntimeError("semantic tail-policy iteration mismatch")
    selected_steps=int(payload.get("selected_steps",-1))
    if schema in (
        SEMANTIC_STRONG_SPECIALIST_MOE_SCHEMA,
        SEMANTIC_CONFIDENCE_GATED_MOE_SCHEMA,
        SEMANTIC_STRATIFIED_SPECIALIST_MOE_SCHEMA,
    ):
        if selected_steps not in (10,25,50,100,200):
            raise RuntimeError("semantic specialist step-budget mismatch")
    elif selected_steps!=500:
        raise RuntimeError("semantic tail-policy step-budget mismatch")

    if schema in (
        SEMANTIC_STRONG_SPECIALIST_MOE_SCHEMA,
        SEMANTIC_CONFIDENCE_GATED_MOE_SCHEMA,
    ):
        base_model=V1SemanticPolicyNet()
        base_model.load_state_dict(payload["base_model_state"])
        specialist_model=V1SemanticPolicyNet()
        specialist_model.load_state_dict(payload["specialist_model_state"])
        if schema==SEMANTIC_CONFIDENCE_GATED_MOE_SCHEMA:
            model=StrongConfidenceGatedMoEPolicyNet(
                base_model,
                specialist_model,
                float(payload["fold_threshold"]),
            ).eval()
        elif schema==SEMANTIC_FOLD_LOGIT_CALIBRATED_SCHEMA:
            model=FoldLogitCalibratedStrongMoEPolicyNet(
                base_model,
                specialist_model,
                float(payload["fold_logit_scale"]),
                float(payload["fold_logit_bias"]),
            ).eval()
        else:
            model=StrongSpecialistMoEPolicyNet(
                base_model,specialist_model
            ).eval()
            if schema==SEMANTIC_STRATIFIED_SPECIALIST_MOE_SCHEMA:
                model._spincore_stratified_specialist_moe=True
    else:
        model=V1SemanticPolicyNet()
        model.load_state_dict(payload["model_state"])
        model.eval()
        model._spincore_diversity_candidate=(schema==SEMANTIC_DIVERSITY_SCHEMA)
        model._spincore_stratified_diversity_candidate=(
            schema==SEMANTIC_STRATIFIED_DIVERSITY_SCHEMA
        )
        model._spincore_fullpool_diversity_candidate=(
            schema==SEMANTIC_FULLPOOL_DIVERSITY_SCHEMA
        )

    _SPIN=SemanticHybridBenchmarkPolicy(agent,model)
    _DC=DeepCrusherR8Policy.from_repository(root_path)
    _ENGINE_SEED=int(seed)


def worker(task):
    index,episode,deal_seed,max_decisions=task
    traces=[]
    engine=OfflineHeadToHeadEngine(
        _SOLVER,
        spincore_policy=_SPIN,
        deepcrusher_policy=_DC,
        master_seed=int(_ENGINE_SEED),
        max_decisions=int(max_decisions),
        decision_sink=traces.append,
    )
    observations=engine.play_balanced_block(
        episode,deal_seed=int(deal_seed),scenario_index=int(index)
    )
    summary=base._scenario_summary(observations)
    summary.update({
        "scenario":int(index),
        "domain":"TRUE_HEADS_UP" if bool(episode.game_is_hu) else "THREE_HANDED",
        "blind":f"{int(episode.small_blind)}/{int(episode.big_blind)}",
        "dealer_id":int(episode.dealer_id),
        "deal_seed":int(deal_seed),
    })
    return {"summary":summary,"traces":[asdict(t) for t in traces]}


def main()->int:
    args=parse_args()
    for p in (
        args.solver,args.base_spin_bundle,args.semantic_policy,
        base.DC_SOURCE,base.LIB1,base.LIB2,
    ):
        if not p.is_file():
            raise SystemExit(f"missing input: {p}")

    sampler=LegacyScenarioSampler(
        seed=int(args.seed)^0x5CE0A710,
        config=LegacyScenarioConfig(),
    )
    tasks=[]
    domain_counts={"THREE_HANDED":0,"TRUE_HEADS_UP":0}
    blind_counts={}
    for index in range(int(args.scenarios)):
        episode=sampler.sample_episode()
        domain="TRUE_HEADS_UP" if bool(episode.game_is_hu) else "THREE_HANDED"
        blind=f"{int(episode.small_blind)}/{int(episode.big_blind)}"
        domain_counts[domain]+=1
        key=f"{domain}:{blind}"
        blind_counts[key]=blind_counts.get(key,0)+1
        tasks.append((
            int(index),episode,int(base._mix64(args.seed,index,0xD34A1)),
            int(args.max_decisions),
        ))

    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.traces.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    trace_count=0
    policy_decision_counts={base.SPINCORE_POLICY_ID:0,base.DEEPC_RUSHER_POLICY_ID:0}
    action_counts={base.SPINCORE_POLICY_ID:{},base.DEEPC_RUSHER_POLICY_ID:{}}

    ctx=mp.get_context("spawn")
    worker_count=min(int(args.workers),int(args.scenarios))
    with args.traces.open("w",encoding="utf-8") as out:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
            initializer=init_worker,
            initargs=(
                str(args.solver.resolve()),
                str(args.base_spin_bundle.resolve()),
                str(args.semantic_policy.resolve()),
                str(ROOT.resolve()),
                int(args.seed),
            ),
        ) as pool:
            for chunk in pool.map(worker,tasks,chunksize=1):
                rows.append(chunk["summary"])
                for trace in chunk["traces"]:
                    out.write(json.dumps(trace,sort_keys=True,separators=(",",":"))+"\n")
                    trace_count+=1
                    policy=str(trace["policy_id"])
                    policy_decision_counts[policy]+=1
                    action=str(trace["action_type"])
                    bucket=action_counts[policy]
                    bucket[action]=bucket.get(action,0)+1

    by_domain,by_blind=base._summaries(rows)
    report={
        "schema":"SPINCORE_DEEPCRUSHER_DC1_SEMANTIC_DEVELOPMENT_V1",
        "status":"PASS",
        "stage":"DC1_DEVELOPMENT_SEMANTIC_1K",
        "canonical_quality_claim_authorized":False,
        "oracle_parity_status":"REAL_OPENHOLDEM_FIXTURES_PENDING",
        "warning":(
            "Development benchmark only. Relative comparison to the unchanged "
            "10105 baseline uses identical scenario/deal seeds, but no canonical "
            "strength claim is authorized until DC0 parity is complete."
        ),
        "spin_core":{
            "base_hybrid_bundle":str(args.base_spin_bundle.resolve()),
            "base_hybrid_bundle_sha256":base._sha256(args.base_spin_bundle),
            "semantic_policy":str(args.semantic_policy.resolve()),
            "semantic_policy_sha256":base._sha256(args.semantic_policy),
            "semantics":(
                "THREE_HANDED V1+general-semantic AveragePolicy@10115 + "
                "TRUE_HEADS_UP unchanged current ENS8@10105"
            ),
        },
        "deepcrusher":{
            "source":str(base.DC_SOURCE.resolve()),
            "source_sha256":base._sha256(base.DC_SOURCE),
            "library_files":[str(base.LIB1.resolve()),str(base.LIB2.resolve())],
        },
        "spin_core_git_head":base._git_head(),
        "seed":int(args.seed),
        "scenarios":int(args.scenarios),
        "workers":worker_count,
        "domain_counts":domain_counts,
        "blind_counts":dict(sorted(blind_counts.items())),
        "trace_file":str(args.traces.resolve()),
        "trace_decisions":trace_count,
        "policy_decision_counts":policy_decision_counts,
        "exact_action_type_counts":action_counts,
        "method":{
            "sampler":"legacy empirical full SpinGo 3H/HU blind-conditioned sampler",
            "deal_pairing":"same scenario/deal reused across balanced lineup block",
            "hu":"unchanged 10105 ENS8; two games; policies swap live seats",
            "three_handed":"semantic 10115 tail AveragePolicy; six AAB/ABB balanced games",
            "action_application":"exact Fold/Check/Call/BetTo/RaiseTo/AllIn",
            "ci":"normal 95% CI over scenario-cluster values",
        },
        "summary":by_domain,
        "by_blind":by_blind,
        "scenario_rows":rows,
    }
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    overall=by_domain["ALL"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"]
    three=by_domain["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"]
    print("=== Semantic SpinCore vs DeepCrusher DC1 development ===")
    print(
        f"ALL paired={overall['mean']:+.3f} "
        f"CI95=[{overall['ci95_low']:+.3f},{overall['ci95_high']:+.3f}]"
    )
    print(
        f"3H paired={three['mean']:+.3f} "
        f"CI95=[{three['ci95_low']:+.3f},{three['ci95_high']:+.3f}]"
    )
    print(f"report={args.report.resolve()}")
    print(f"traces={args.traces.resolve()}")
    print("DC1_SEMANTIC_DEVELOPMENT_ONLY_NO_QUALITY_CLAIM")
    print("DEEPC_RUSHER_DC1_SEMANTIC_DEVELOPMENT_PASS")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
