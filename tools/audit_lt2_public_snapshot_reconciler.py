#!/usr/bin/env python3
from __future__ import annotations

"""Mechanical gate for public-snapshot -> canonical exact-action reconciliation.

Uses old forensic scenario seeds only. It generates diverse canonical and alias
exact actions, observes only the before/after public snapshots, and asks the
runtime reconciler to infer the canonical LT2 transcript action.

Also injects skipped-action and corrupted snapshots; those must fail closed.

No model inference, EV, training, optimizer work or holdout reuse.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_lt2_stage_a_b_first_divergence as fd
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.runtime_transition_reconciler import (
    RuntimeReconciliationError,
    infer_single_exact_action,
)
from spincore.solver import ResolvedExactAction, SolverLibrary

SEEDS=(20260920,20260921,20260922,20260923,20260924,20260925)
ACTION_NAMES={0:"FOLD",1:"CHECK",2:"CALL",3:"BET_TO",4:"RAISE_TO",5:"ALL_IN"}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--scenarios-per-seed",type=int,default=900)
    p.add_argument("--max-transitions",type=int,default=12000)
    p.add_argument("--fault-trials",type=int,default=1500)
    p.add_argument("--report",type=Path,required=True)
    return p.parse_args()


def _canonical_options(snapshot,rng):
    actor=int(snapshot.actor)
    stack=int(snapshot.stacks[actor])
    options=[]

    if snapshot.legal_fold:
        options.append(ResolvedExactAction(0,0))
    if snapshot.legal_check:
        options.append(ResolvedExactAction(1,0))
    if snapshot.legal_call:
        options.append(ResolvedExactAction(2,0))

    # Canonical aggressive all-in. Do not expose it as a second label when it
    # can only call; the validated lean strategy canonicalizes that to Call.
    if snapshot.legal_all_in and snapshot.to_call < stack:
        options.append(ResolvedExactAction(5,0))

    if snapshot.legal_bet or snapshot.legal_raise:
        lo=int(snapshot.min_raise_to)
        hi=int(snapshot.max_raise_to)
        # BetTo/RaiseTo is canonical only when chips remain after the action.
        max_nonallin=hi-1
        if lo<=max_nonallin:
            targets={lo,max_nonallin,(lo+max_nonallin)//2}
            if max_nonallin-lo>3:
                targets.add(rng.randint(lo,max_nonallin))
            action_type=3 if snapshot.current_bet==0 else 4
            for target in sorted(targets):
                if lo<=target<=max_nonallin:
                    options.append(ResolvedExactAction(action_type,int(target)))

    # Keep deterministic stable order and remove duplicates.
    unique=[]
    seen=set()
    for action in options:
        key=(action.action_type,action.amount_to)
        if key not in seen:
            seen.add(key)
            unique.append(action)
    return unique


def _choose_action(snapshot,rng):
    options=_canonical_options(snapshot,rng)
    if not options:
        raise RuntimeError("actionable state produced no canonical exact actions")
    # Favor nonterminal/passive continuation for street coverage while still
    # exercising arbitrary raise sizes and all-in.
    weights=[]
    for action in options:
        if action.action_type in (1,2):
            weights.append(5)
        elif action.action_type in (3,4):
            weights.append(4)
        elif action.action_type==5:
            weights.append(2)
        else:
            weights.append(1)
    total=sum(weights)
    pick=rng.randrange(total)
    for action,w in zip(options,weights):
        if pick<w:
            return action
        pick-=w
    return options[-1]


def _normalized_expected(before, action, after):
    actor=int(before.actor)
    paid=int(before.stacks[actor]-after.stacks[actor])
    target=int(before.street_commitments[actor]+paid)
    if action.action_type==5 and before.to_call>0 and target<=before.current_bet:
        return ResolvedExactAction(2,0)
    if action.action_type in (3,4) and int(after.stacks[actor])==0:
        return ResolvedExactAction(5,0)
    return action


def _alias_action(before):
    actor=int(before.actor)
    stack=int(before.stacks[actor])
    # Alias 1: all-in call must canonicalize to Call.
    if before.legal_all_in and before.legal_call and before.to_call>=stack and stack>0:
        return ResolvedExactAction(5,0),ResolvedExactAction(2,0),"ALLIN_CALL_TO_CALL"

    # Alias 2: BetTo/RaiseTo exactly to max empties the stack and must
    # canonicalize to AllIn.
    hi=int(before.max_raise_to)
    lo=int(before.min_raise_to)
    if (before.legal_bet or before.legal_raise) and hi>=lo and hi>before.current_bet:
        typ=3 if before.current_bet==0 else 4
        return ResolvedExactAction(typ,hi),ResolvedExactAction(5,0),"TO_MAX_TO_ALLIN"
    return None


def _expect_reject(state, observed):
    try:
        infer_single_exact_action(state,observed)
    except RuntimeReconciliationError:
        return True
    return False


def main():
    args=parse_args()
    if min(args.scenarios_per_seed,args.max_transitions,args.fault_trials)<=0:
        raise SystemExit("positive arguments required")
    solver_path=args.solver.resolve(strict=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    solver=SolverLibrary(solver_path)

    tested=0
    action_counts={name:0 for name in ACTION_NAMES.values()}
    street_counts={str(i):0 for i in range(4)}
    domain_counts={"THREE_HANDED":0,"TRUE_HEADS_UP":0}
    alias_counts={"ALLIN_CALL_TO_CALL":0,"TO_MAX_TO_ALLIN":0}
    alias_failures=[]
    failures=[]
    skipped_rejections=0
    corrupt_rejections=0
    noop_rejections=0
    fault_attempts=0

    for seed in SEEDS:
        sampler=LegacyScenarioSampler(
            seed=int(seed)^0x5CE0A710,
            config=LegacyScenarioConfig(),
        )
        for scenario in range(int(args.scenarios_per_seed)):
            if tested>=int(args.max_transitions):
                break
            episode=sampler.sample_episode()
            deal_seed=fd._mix64(int(seed),int(scenario),0xD34A1)
            state=solver.create(episode,int(deal_seed))
            rng=random.Random(fd._mix64(seed,scenario,0x51A7E))
            try:
                for decision in range(80):
                    if state.terminal or tested>=int(args.max_transitions):
                        break
                    before=state.public_snapshot()
                    if before.actor<0:
                        break
                    options=_canonical_options(before,rng)
                    if not options:
                        raise RuntimeError("no canonical options")

                    # Periodically test a public alias that should normalize to
                    # the validated LT2 transcript convention.
                    if decision%3==0:
                        alias=_alias_action(before)
                        if alias is not None:
                            raw,expected,label=alias
                            alias_state=state.clone()
                            try:
                                alias_state.apply_exact(raw.action_type,raw.amount_to)
                                observed=alias_state.public_snapshot()
                            except Exception:
                                observed=None
                            finally:
                                alias_state.close()
                            if observed is not None:
                                try:
                                    inferred=infer_single_exact_action(state,observed)
                                    if inferred!=expected:
                                        alias_failures.append({
                                            "seed":seed,"scenario":scenario,
                                            "label":label,
                                            "expected":(expected.action_type,expected.amount_to),
                                            "got":(inferred.action_type,inferred.amount_to),
                                        })
                                    else:
                                        alias_counts[label]+=1
                                except Exception as exc:
                                    alias_failures.append({
                                        "seed":seed,"scenario":scenario,
                                        "label":label,"error":str(exc),
                                    })

                    action=_choose_action(before,rng)
                    next_state=state.clone()
                    try:
                        next_state.apply_exact(action.action_type,action.amount_to)
                        observed=next_state.public_snapshot()
                        expected=_normalized_expected(before,action,observed)
                        inferred=infer_single_exact_action(state,observed)
                        if inferred!=expected:
                            failures.append({
                                "seed":seed,"scenario":scenario,
                                "decision":decision,
                                "expected":(expected.action_type,expected.amount_to),
                                "got":(inferred.action_type,inferred.amount_to),
                            })
                            break

                        action_counts[ACTION_NAMES[inferred.action_type]]+=1
                        street_counts[str(before.street)]+=1
                        domain_counts["TRUE_HEADS_UP" if before.domain==1 else "THREE_HANDED"]+=1
                        tested+=1

                        if fault_attempts<int(args.fault_trials):
                            fault_attempts+=1
                            if _expect_reject(state,before):
                                noop_rejections+=1
                            corrupted=replace(observed,pot=int(observed.pot)+1)
                            if _expect_reject(state,corrupted):
                                corrupt_rejections+=1

                            # Skip one extra action when possible. A one-action
                            # reconciler must reject the later snapshot.
                            if not next_state.terminal:
                                second_before=next_state.public_snapshot()
                                second_options=_canonical_options(second_before,rng)
                                if second_options:
                                    second=_choose_action(second_before,rng)
                                    skipped=next_state.clone()
                                    try:
                                        skipped.apply_exact(second.action_type,second.amount_to)
                                        if _expect_reject(state,skipped.public_snapshot()):
                                            skipped_rejections+=1
                                    finally:
                                        skipped.close()

                        state.close()
                        state=next_state
                        next_state=None
                    finally:
                        if next_state is not None:
                            next_state.close()
            finally:
                state.close()
        if tested>=int(args.max_transitions):
            break

    verdict=(
        not failures
        and not alias_failures
        and tested>0
        and domain_counts["THREE_HANDED"]>0
        and domain_counts["TRUE_HEADS_UP"]>0
        and street_counts["0"]>0
        and sum(street_counts[str(i)] for i in (1,2,3))>0
        and action_counts["CALL"]>0
        and action_counts["ALL_IN"]>0
        and (action_counts["BET_TO"]+action_counts["RAISE_TO"])>0
        and noop_rejections==fault_attempts
        and corrupt_rejections==fault_attempts
        and skipped_rejections>0
    )

    out={
        "schema":"SPINCORE_LT2_PUBLIC_SNAPSHOT_RECONCILER_V1",
        "verdict":"PASS" if verdict else "FAIL",
        "method":{
            "forensic_seeds":list(SEEDS),
            "scenarios_per_seed":int(args.scenarios_per_seed),
            "max_transitions":int(args.max_transitions),
            "fault_trials":int(args.fault_trials),
            "training_roots":0,
            "optimizer_steps":0,
            "strategic_ev_evaluations":0,
            "holdout_reused":False,
            "canonicalization":{
                "allin_call":"CALL",
                "stack_emptying_aggression":"ALL_IN",
                "nonallin_open_aggression":"BET_TO",
                "nonallin_facing_bet_aggression":"RAISE_TO",
            },
        },
        "transitions_checked":tested,
        "actions":action_counts,
        "states_by_street":street_counts,
        "states_by_domain":domain_counts,
        "alias_normalizations":alias_counts,
        "alias_failure_count":len(alias_failures),
        "first_alias_failure":alias_failures[0] if alias_failures else None,
        "fault_attempts":fault_attempts,
        "noop_snapshot_rejections":noop_rejections,
        "corrupt_snapshot_rejections":corrupt_rejections,
        "skipped_action_snapshot_rejections":skipped_rejections,
        "failure_count":len(failures),
        "first_failure":failures[0] if failures else None,
    }
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== LT2 PUBLIC SNAPSHOT RECONCILER ===")
    print(f"VERDICT={out['verdict']}")
    print(f"transitions={tested} actions={action_counts}")
    print(f"by_domain={domain_counts} by_street={street_counts}")
    print(f"aliases={alias_counts} alias_failures={len(alias_failures)}")
    print(
        f"faults={fault_attempts} noop_reject={noop_rejections} "
        f"corrupt_reject={corrupt_rejections} skipped_reject={skipped_rejections}"
    )
    print("LT2_PUBLIC_SNAPSHOT_RECONCILER_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0 if verdict else 2


if __name__=="__main__":
    raise SystemExit(main())
