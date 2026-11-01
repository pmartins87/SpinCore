#!/usr/bin/env python3
from __future__ import annotations

"""Read-only direct-play gate: @17315 baseline vs published Deep CFR argmax fallback.

Uses frozen ENS8, existing SpinCore/DeepCrusher common-chance engine and exact
existing opportunity/sanity instrumentation. No CFR collection or training.
PROJECT_CONTRACT_IDS: MODEL-020,MODEL-025,VALID-025,VALID-031,VALID-034,VALID-035,SAFE-001,SAFE-010,RNG-001,RNG-002,RNG-003
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_3h_teacher_sanity_17315 as sanity
import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_reference_fallback_17315 as ref
import validate_teacher_member_policy_mean_17315 as previous
import evaluate_deepcrusher_dc1 as dcbase

from spincore.legacy_scenario import LegacyScenarioConfig,LegacyScenarioSampler
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import lean_legal_actions
from spincore.r7_5_action_cfr import legal_mask
from spincore_nn.action_models import collate_action_observations
from spincore_nn.semantic_v1 import semantic_vector_from_observation

SCHEMA="SPINCORE_REFERENCE_FALLBACK_DIRECT_PLAY_17315_V1"
PREREG_SCHEMA="SPINCORE_REFERENCE_FALLBACK_DIRECT_PLAY_17315_PREREG_V1"
ENSEMBLE_SCHEMA="SPINCORE_3H_SEMANTIC_LONG_ENSEMBLE_V1"
ITERATION=17315
MODE_A="A_CURRENT_SOFTMAX_FALLBACK"
MODE_B="B_PUBLISHED_ARGMAX_FALLBACK"


class PublishedFallbackPolicy(sanity.SemanticAdvantageTeacherPolicy):
    """Only change the all-nonpositive branch after averaging the same eight raw outputs."""
    def distribution(self,state):
        street=sanity.street_from_state(state)
        active=FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal=tuple(int(x) for x in lean_legal_actions(state,active))
        if not legal:
            raise RuntimeError("empty legal set")
        obs=state.neural_bytes()
        batch=collate_action_observations(
            distill.REPRESENTATION,[obs],[legal_mask(legal)],device="cpu"
        )
        semantic=semantic_vector_from_observation(obs)
        sb=dict(batch)
        sb["semantic"]=torch.tensor(
            np.asarray([semantic],dtype=np.float32),dtype=torch.float32
        )
        with torch.no_grad():
            raw=torch.stack(
                [model(sb)[0] for model in self.models],dim=0
            ).mean(dim=0).detach().cpu().tolist()
        return int(active),legal,ref.reference_policy(raw,legal)


def policy_selftest():
    ref.self_test()
    # Positive-Advantage action behavior is exactly identical to the current baseline.
    from spincore.lean_action_policy import lean_regret_matching_policy
    examples=[
        ([0.1,-0.2,0.4]+[0.0]*7,(0,1,2)),
        ([-0.1,-0.1,-0.1,1.2]+[0.0]*6,(0,1,2,3)),
        ([-2.0,-1.0,-0.4]+[0.0]*7,(0,1,2)),
    ]
    for raw,legal in examples:
        a=lean_regret_matching_policy(raw,legal)
        b=ref.reference_policy(raw,legal)
        if any(raw[x]>0 for x in legal):
            assert a==b
        else:
            assert sum(abs(a[x]-b[x]) for x in legal)>0
    print("G3_FALLBACK_ONLY_SELFTEST_PASS",flush=True)


def init_worker(solver_path,ensemble_path,root_path,seed,mode):
    previous.init_worker(
        solver_path,ensemble_path,root_path,seed,
        "A_CURRENT_RAW_MEAN_RM",
    )
    if mode==MODE_B:
        base=previous._POLICY
        previous._POLICY=PublishedFallbackPolicy(
            base.models, "17315_B_DEEP_CFR_ARGMAX_FALLBACK"
        )
    elif mode!=MODE_A:
        raise RuntimeError("unknown mode")


def run_candidate(args,tasks,mode):
    rows=[]
    ctx=mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers),len(tasks)),
        mp_context=ctx,
        initializer=init_worker,
        initargs=(
            str(args.solver.resolve()),
            str(args.teacher.resolve()),
            str(ROOT.resolve()),
            int(args.seed),
            mode,
        ),
    ) as pool:
        for row in pool.map(previous.worker,tasks,chunksize=1):
            rows.append(row)
    return previous.summarize_candidate(rows)


def mean_probability(report,code):
    entry=(report["opportunity_policy"] or {}).get(code) or {}
    value=entry.get("mean_target_probability")
    return None if value is None else float(value)


def adjudicate(A,B,paired):
    ah=A["opportunity_policy"].get("POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW") or {}
    bh=B["opportunity_policy"].get("POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW") or {}
    def high_fraction(x):
        count=int(x.get("opportunities",0))
        if count<=0:return None
        return int(x.get("ge_80pct_count",0))/float(count)
    af=high_fraction(ah)
    bf=high_fraction(bh)
    a72=mean_probability(A,"PREFLOP_DEEP_72O_JAM")
    b72=mean_probability(B,"PREFLOP_DEEP_72O_JAM")
    at=mean_probability(A,"POSTFLOP_TRIPS_PLUS_FOLD")
    bt=mean_probability(B,"POSTFLOP_TRIPS_PLUS_FOLD")
    am=mean_probability(A,"POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW")
    bm=mean_probability(B,"POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW")
    a72n=int((A["opportunity_policy"].get("PREFLOP_DEEP_72O_JAM") or {}).get("opportunities",0))
    b72n=int((B["opportunity_policy"].get("PREFLOP_DEEP_72O_JAM") or {}).get("opportunities",0))
    checks={
        "ev_noninferiority":math.isfinite(float(paired["ci95_low"])) and float(paired["ci95_low"])>=-0.50,
        "zero_aa_fold":int(B["flag_counts"].get("PREFLOP_AA_FOLD",0))==0,
        "zero_monster_fold":int(B["flag_counts"].get("POSTFLOP_MONSTER_FOLD",0))==0,
        "high_card_no_draw_ge80":af is not None and bf is not None and bf<=af+0.001,
        "high_card_no_draw_mean":am is not None and bm is not None and bm<=am+0.02,
        "deep_72o_sufficient_opportunities":a72n>=25 and b72n>=25,
        "deep_72o_mean":a72 is not None and b72 is not None and b72<=a72+0.005 and b72<=0.05 and b72<=a72-0.01,
        "trips_plus_fold_mean":at is not None and bt is not None and bt<=at+0.02,
    }
    return {
        "status":"PASS" if all(checks.values()) else "FAIL",
        "checks":checks,
        "parameters":{"a72":a72,"b72":b72,"a72n":a72n,"b72n":b72n,
            "a_no_draw":am,"b_no_draw":bm,"a_no_draw_ge80":af,"b_no_draw_ge80":bf,
            "a_trips_fold":at,"b_trips_fold":bt},
        "production_change_authorized":False,
        "long_training_unblocked":False,
        "candidate_status":"NEEDS_SEPARATE_INTEGRATION_GATE" if all(checks.values()) else "REJECT_AS_DIRECT_REPLACEMENT",
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver",type=Path)
    ap.add_argument("--teacher",type=Path)
    ap.add_argument("--training-report",type=Path)
    ap.add_argument("--prereg",type=Path)
    ap.add_argument("--report",type=Path)
    ap.add_argument("--seed",type=int,default=2026110103)
    ap.add_argument("--scenarios",type=int,default=10000)
    ap.add_argument("--workers",type=int,default=16)
    ap.add_argument("--max-decisions",type=int,default=300)
    ap.add_argument("--smoke-only",action="store_true")
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()

    if args.self_test:
        policy_selftest()
        if args.solver is None:
            return 0

    if not all((args.solver,args.teacher,args.training_report,args.prereg)):
        ap.error("--solver, --teacher, --training-report and --prereg are required")

    prereg=json.loads(args.prereg.read_text(encoding="utf-8"))
    if prereg.get("schema")!=PREREG_SCHEMA or prereg.get("status")!="FROZEN_BEFORE_RESULTS":
        raise RuntimeError("prereg invalid or unfrozen")
    frozen=prereg["design"]
    if args.smoke_only:
        # Lightweight integration gate does not generate a selected scientific outcome.
        scenarios=4
        workers=2
    else:
        scenarios=int(args.scenarios)
        workers=int(args.workers)
        for key,actual in (
            ("scenarios",scenarios),("workers",workers),
            ("seed",int(args.seed)),("max_decisions",int(args.max_decisions))
        ):
            if int(frozen[key])!=actual:
                raise RuntimeError(f"preregistered {key} drift")
        if args.report is None:
            ap.error("--report is required for full gate")

    training=json.loads(args.training_report.read_text(encoding="utf-8"))
    if training.get("status")!="PASS" or int(training.get("completed_iteration",-1))!=ITERATION:
        raise RuntimeError("source training must be PASS at 17315")
    teacher_sha=dcbase._sha256(args.teacher)
    if teacher_sha!=prereg["context"]["source_teacher_sha256"] or teacher_sha!=training.get("final_ensemble_sha256"):
        raise RuntimeError("source teacher SHA drift")

    sampler=LegacyScenarioSampler(seed=int(args.seed)^0x5CE0A710,config=LegacyScenarioConfig())
    tasks=[]
    for i in range(scenarios):
        ep=sampler.sample_episode(force_domain="THREE_HANDED")
        tasks.append((i,ep,int(dcbase._mix64(int(args.seed),i,0xD34A1)),int(args.max_decisions)))

    args.workers=workers
    print(f"G3_BASELINE_BEGIN scenarios={scenarios} smoke={args.smoke_only}",flush=True)
    A=run_candidate(args,tasks,MODE_A)
    print("G3_BASELINE_PASS",flush=True)
    B=run_candidate(args,tasks,MODE_B)
    print("G3_REFERENCE_FALLBACK_PASS",flush=True)

    arows={int(r["scenario"]):r for r in A["rows"]}
    brows={int(r["scenario"]):r for r in B["rows"]}
    if set(arows)!=set(brows) or len(arows)!=scenarios:
        raise RuntimeError("paired scenario alignment mismatch")
    diffs=[
        float(brows[i]["summary"]["spincore_chips_per_exposure"]) -
        float(arows[i]["summary"]["spincore_chips_per_exposure"])
        for i in sorted(arows)
    ]
    paired=previous.mean_ci(diffs)

    if args.smoke_only:
        print("SPINCORE_REFERENCE_FALLBACK_DIRECT_PLAY_17315_SMOKE_PASS")
        return 0

    verdict=adjudicate(A,B,paired)
    report={
        "schema":SCHEMA,"status":"PASS","iteration":ITERATION,
        "teacher_sha256":teacher_sha,"prereg_sha256":dcbase._sha256(args.prereg),
        "seed":int(args.seed),"scenarios":scenarios,"workers":workers,
        "candidate_A":{k:v for k,v in A.items() if k!="rows"},
        "candidate_B":{k:v for k,v in B.items() if k!="rows"},
        "paired_B_minus_A_spincore_chips_per_policy_seat_hand":paired,
        "decision":verdict,
        "production_change_authorized":False,
        "long_training_unblocked":False,
        "interpretation":"Fresh common-chance paired development play only. Same frozen weights; no training. A PASS does not fix positive-regret high-card-no-draw pathology and does not authorize production promotion."
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("G3_DECISION="+verdict["status"])
    print("G3_CHECKS="+json.dumps(verdict["checks"],sort_keys=True))
    print("G3_B_MINUS_A="+json.dumps(paired,sort_keys=True))
    print("G3_PARAMS="+json.dumps(verdict["parameters"],sort_keys=True))
    print("G3_FLAGS_A="+json.dumps(A["flag_counts"],sort_keys=True))
    print("G3_FLAGS_B="+json.dumps(B["flag_counts"],sort_keys=True))
    print("SPINCORE_REFERENCE_FALLBACK_DIRECT_PLAY_17315_REPORT_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
