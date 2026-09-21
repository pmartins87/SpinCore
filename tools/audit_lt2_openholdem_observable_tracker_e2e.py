#!/usr/bin/env python3
from __future__ import annotations

"""End-to-end observable OpenHoldem -> canonical SpinCore state gate.

Pipeline under test:
authoritative truth
 -> synthetic OpenHoldem raw frame
 -> strict symbol adapter
 -> observable tracker
 -> one-action reconciliation
 -> exact transcript append
 -> from-scratch rebuild with actual visible board + hidden fillers
 -> canonical solver state

At every Hero decision the rebuilt canonical SPNNIV1/SPNNIV2 and lean action
semantics must exactly match truth.

This explicitly exercises street reveals where the real flop/turn/river replaces
previous hidden filler cards.

Old forensic seeds only. No deployment model inference, EV, training, optimizer
work or holdout reuse.
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
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import lean_legal_actions, resolve_lean_exact
from spincore.openholdem_symbol_adapter import (
    OpenHoldemRawFrame,
    OpenHoldemSymbolAdapter,
    openholdem_betround_from_visible_count,
)
from spincore.runtime_observable_tracker import RuntimeObservableTracker
from spincore.solver import Episode, ResolvedExactAction, SolverLibrary

SEEDS=(20260920,20260921,20260922,20260923,20260924,20260925)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--scenarios-per-seed",type=int,default=700)
    p.add_argument("--max-transitions",type=int,default=10000)
    p.add_argument("--fault-hands",type=int,default=500)
    p.add_argument("--report",type=Path,required=True)
    return p.parse_args()


def _normalize_episode(e:Episode)->Episode:
    live=[i for i,s in enumerate(e.stacks) if int(s)>0]
    dealer=int(e.dealer_id)
    order=[]
    for step in range(3):
        seat=(dealer+step)%3
        if seat in live:
            order.append(seat)
    if e.game_is_hu:
        stacks=(int(e.stacks[order[0]]),int(e.stacks[order[1]]),0)
        dead=(2,)
    else:
        stacks=tuple(int(e.stacks[x]) for x in order)
        dead=()
    return Episode(
        total_chips=int(e.total_chips),game_is_hu=bool(e.game_is_hu),
        blind_index=int(e.blind_index),small_blind=int(e.small_blind),
        big_blind=int(e.big_blind),stacks=stacks,dealer_id=0,dead_players=dead,
    )


def _chairs(e,rng):
    dealer=rng.randrange(10)
    if e.game_is_hu:
        return (dealer,(dealer+rng.randrange(1,10))%10,-1)
    gaps=sorted(rng.sample(range(1,10),2))
    return (dealer,(dealer+gaps[0])%10,(dealer+gaps[1])%10)


def _frame(hand_id,e,state,chairs,hero):
    p=state.public_snapshot()
    deal=state.deal_snapshot()
    balances=[0.0]*10
    current=[0.0]*10
    dealt=playing=allin=0
    for logical,chair in enumerate(chairs):
        if chair<0:
            continue
        dealt|=1<<chair
        balances[chair]=float(p.stacks[logical])
        current[chair]=float(p.street_commitments[logical])
        if not p.folded[logical]:
            playing|=1<<chair
        if p.all_in[logical]:
            allin|=1<<chair
            playing|=1<<chair
    board=[-1]*5
    for i in range(int(p.visible_board_count)):
        board[i]=int(deal.board[i])
    return OpenHoldemRawFrame(
        hand_id=str(hand_id),
        user_chair=int(chairs[hero]),
        dealer_chair=int(chairs[0]),
        betround=openholdem_betround_from_visible_count(int(p.visible_board_count)),
        sblind=float(e.small_blind),
        bblind=float(e.big_blind),
        playersdealtbits=dealt,
        playersplayingbits=playing,
        playersallinbits=allin,
        balances=tuple(balances),
        currentbets=tuple(current),
        pot=float(p.pot),
        ncommoncardsknown=int(p.visible_board_count),
        hero_cards=tuple(int(x) for x in deal.holes[hero]),
        board_cards=tuple(board),
    )


def _options(s,rng):
    actor=int(s.actor);stack=int(s.stacks[actor]);out=[]
    if s.legal_fold: out.append(ResolvedExactAction(0,0))
    if s.legal_check: out.append(ResolvedExactAction(1,0))
    if s.legal_call: out.append(ResolvedExactAction(2,0))
    if s.legal_all_in and s.to_call<stack:
        out.append(ResolvedExactAction(5,0))
    if s.legal_bet or s.legal_raise:
        lo=int(s.min_raise_to);hi=int(s.max_raise_to)-1
        if lo<=hi:
            typ=3 if s.current_bet==0 else 4
            vals={lo,hi,(lo+hi)//2}
            if hi-lo>2: vals.add(rng.randint(lo,hi))
            for target in sorted(vals):
                out.append(ResolvedExactAction(typ,int(target)))
    return out


def _choose(s,rng):
    opts=_options(s,rng)
    if not opts: raise RuntimeError("no action options")
    passive=[a for a in opts if a.action_type in (1,2)]
    if passive and rng.random()<0.60:
        return passive[rng.randrange(len(passive))]
    nonfold=[a for a in opts if a.action_type!=0]
    pool=nonfold if nonfold and rng.random()<0.90 else opts
    return pool[rng.randrange(len(pool))]


def _normalized(before,raw,after):
    actor=int(before.actor)
    paid=int(before.stacks[actor]-after.stacks[actor])
    target=int(before.street_commitments[actor]+paid)
    if raw.action_type==5 and before.to_call>0 and target<=before.current_bet:
        return ResolvedExactAction(2,0)
    if raw.action_type in (3,4) and int(after.stacks[actor])==0:
        return ResolvedExactAction(5,0)
    return raw


def _hero_context(state):
    v2=state.neural_bytes_v2()
    street=int(v2[112])
    mask=FIRST_RELEASE_ACTION_SPEC.active_mask(street)
    legal=tuple(int(x) for x in lean_legal_actions(state,mask))
    resolved=tuple(
        (slot,tuple(int(x) for x in resolve_lean_exact(state,mask,slot)))
        for slot in legal
    )
    return (
        state.neural_bytes(),
        v2,
        int(state.actor),
        int(state.domain),
        int(mask),
        legal,
        resolved,
    )


def main():
    args=parse_args()
    if min(args.scenarios_per_seed,args.max_transitions,args.fault_hands)<=0:
        raise SystemExit("positive arguments required")

    solver=SolverLibrary(args.solver.resolve(strict=True))
    adapter=OpenHoldemSymbolAdapter()
    args.report.parent.mkdir(parents=True,exist_ok=True)

    transitions=0
    hero_state_checks=0
    street_reveals=0
    exact_action_mismatches=0
    canonical_mismatches=0
    transcript_mismatches=0
    duplicate_checks=0
    silent_check_deferrals=0
    delayed_actions_reconciled_at_myturn=0
    multi_action_sync_events=0
    failures=[]
    domain_counts={"THREE_HANDED":0,"TRUE_HEADS_UP":0}
    street_counts={str(i):0 for i in range(4)}

    corrupt_attempts=corrupt_rejections=0
    skipped_attempts=skipped_rejections=0
    fault_budget=int(args.fault_hands)

    for seed in SEEDS:
        sampler=LegacyScenarioSampler(seed=int(seed)^0x5CE0A710,config=LegacyScenarioConfig())
        for scenario in range(int(args.scenarios_per_seed)):
            if transitions>=int(args.max_transitions): break
            e=_normalize_episode(sampler.sample_episode())
            truth=solver.create(e,fd._mix64(seed,scenario,0xD34A1))
            rng=random.Random(fd._mix64(seed,scenario,0x0B5E2))
            chairs=_chairs(e,rng)
            live=[i for i,s in enumerate(e.stacks) if int(s)>0]
            hero=live[int(fd._mix64(seed,scenario,0xA11CE)%len(live))]
            hand_id=f"E2E-{seed}-{scenario}"
            tracker=RuntimeObservableTracker(solver)
            expected_transcript=[]
            try:
                start_raw=_frame(hand_id,e,truth,chairs,hero)
                anchor=adapter.start_hand(start_raw)
                start_obs=adapter.normalize(start_raw,anchor)
                ev=tracker.start_hand(anchor,start_obs)
                if ev.kind!="START":
                    failures.append({"seed":seed,"scenario":scenario,"kind":"start","reason":ev.reason})
                    break

                for decision in range(80):
                    if truth.terminal or transitions>=int(args.max_transitions):
                        break
                    before=truth.public_snapshot()
                    raw_frame=_frame(hand_id,e,truth,chairs,hero)
                    observed=adapter.normalize(raw_frame,anchor)

                    # Duplicate scrape must remain a no-op end-to-end.
                    prior=tracker.transcript_tuples()
                    dup=tracker.on_heartbeat(anchor,observed)
                    duplicate_checks+=1
                    if dup.kind!="NO_CHANGE" or tracker.transcript_tuples()!=prior:
                        failures.append({"seed":seed,"scenario":scenario,"kind":"duplicate"})
                        break

                    if int(before.actor)==int(hero):
                        transcript_before_myturn=len(tracker.transcript)
                        # OpenHoldem chip/card snapshots cannot reveal an
                        # opponent CHECK. DLLUpdateOnMyTurn is the synchronizing
                        # evidence that actor order has reached Hero, so force
                        # any pending silent CHECKs before comparing canonical
                        # state.
                        marker=tracker.on_my_turn(
                            anchor,
                            observed,
                            lambda t: ("E2E",t.generation,len(t.transcript)),
                        )
                        if marker is None:
                            failures.append({
                                "seed":seed,"scenario":scenario,
                                "kind":"myturn_sync",
                                "reason":tracker.failure_reason,
                            })
                            break
                        delayed_actions_reconciled_at_myturn += (
                            len(tracker.transcript)-transcript_before_myturn
                        )
                        if tracker.transcript_tuples()!=tuple(expected_transcript):
                            transcript_mismatches+=1
                            failures.append({
                                "seed":seed,"scenario":scenario,
                                "kind":"myturn_transcript",
                            })
                            break
                        rebuilt=tracker.canonical_state()
                        if rebuilt is None:
                            failures.append({"seed":seed,"scenario":scenario,"kind":"missing_state"})
                            break
                        if _hero_context(rebuilt)!=_hero_context(truth):
                            canonical_mismatches+=1
                            failures.append({
                                "seed":seed,"scenario":scenario,
                                "kind":"hero_canonical_context",
                                "street":int(before.street),
                            })
                            break
                        hero_state_checks+=1
                        domain_counts["TRUE_HEADS_UP" if e.game_is_hu else "THREE_HANDED"]+=1
                        street_counts[str(before.street)]+=1

                    action=_choose(before,rng)
                    old_street=int(before.street)
                    truth.apply_exact(action.action_type,action.amount_to)
                    after=truth.public_snapshot()
                    expected=_normalized(before,action,after)
                    expected_transcript.append((expected.action_type,expected.amount_to))
                    if not truth.terminal and int(after.street)!=old_street:
                        street_reveals+=1

                    new_frame=_frame(hand_id,e,truth,chairs,hero)
                    new_obs=adapter.normalize(new_frame,anchor)
                    observable_changed=(new_obs!=observed)
                    transcript_before_event=len(tracker.transcript)
                    event=tracker.on_heartbeat(anchor,new_obs)
                    transitions+=1

                    if not observable_changed:
                        # A CHECK that does not complete the street is
                        # intentionally invisible in OH balance/bet/pot/card
                        # snapshots. It remains pending until later observable
                        # evidence or DLLUpdateOnMyTurn.
                        if expected.action_type==1 and event.kind=="NO_CHANGE":
                            silent_check_deferrals+=1
                        if expected.action_type!=1 or event.kind!="NO_CHANGE":
                            failures.append({
                                "seed":seed,"scenario":scenario,
                                "kind":"silent_transition",
                                "expected":(expected.action_type,expected.amount_to),
                                "event":event.kind,
                                "reason":event.reason,
                            })
                            break
                    else:
                        if event.kind!="ACTION":
                            failures.append({
                                "seed":seed,"scenario":scenario,
                                "kind":"transition","reason":event.reason,
                            })
                            break
                        if len(tracker.transcript)-transcript_before_event>1:
                            multi_action_sync_events+=1
                        if event.action!=expected:
                            exact_action_mismatches+=1
                            failures.append({
                                "seed":seed,"scenario":scenario,
                                "kind":"action",
                                "expected":(expected.action_type,expected.amount_to),
                                "got":None if event.action is None else (event.action.action_type,event.action.amount_to),
                            })
                            break
                        if tracker.transcript_tuples()!=tuple(expected_transcript):
                            transcript_mismatches+=1
                            failures.append({"seed":seed,"scenario":scenario,"kind":"transcript"})
                            break

                        # Once an observable change synchronizes the tracker,
                        # its canonical public state must match truth, including
                        # a real street reveal replacing old filler cards.
                        rebuilt=tracker.canonical_state()
                        if rebuilt is None or rebuilt.public_snapshot()!=truth.public_snapshot():
                            canonical_mismatches+=1
                            failures.append({
                                "seed":seed,"scenario":scenario,
                                "kind":"public_canonical_after_action",
                            })
                            break

                if failures:
                    break
            finally:
                tracker.close()
                truth.close()

            if failures:
                break

            # End-to-end malformed/skip checks on fresh hands.
            if fault_budget>0:
                fault_budget-=1
                truth=solver.create(e,fd._mix64(seed,scenario,0xD34A1))
                tracker=RuntimeObservableTracker(solver)
                try:
                    start_raw=_frame(hand_id+"-F",e,truth,chairs,hero)
                    anchor=adapter.start_hand(start_raw)
                    obs0=adapter.normalize(start_raw,anchor)
                    if tracker.start_hand(anchor,obs0).kind!="START":
                        failures.append({"seed":seed,"scenario":scenario,"kind":"fault_start"})
                        break

                    if not truth.terminal:
                        b=truth.public_snapshot()
                        a=_choose(b,rng)
                        one=truth.clone()
                        try:
                            one.apply_exact(a.action_type,a.amount_to)
                            frame1=_frame(hand_id+"-F",e,one,chairs,hero)
                            obs1=adapter.normalize(frame1,anchor)
                        finally:
                            one.close()

                        corrupt_attempts+=1
                        bad=replace(obs1,pot=int(obs1.pot)+1)
                        if tracker.on_heartbeat(anchor,bad).kind=="FAILED":
                            corrupt_rejections+=1

                    tracker.close()

                    # Fresh tracker for skipped two-action transition.
                    truth.close()
                    truth=solver.create(e,fd._mix64(seed,scenario,0xD34A1))
                    tracker=RuntimeObservableTracker(solver)
                    start_raw=_frame(hand_id+"-S",e,truth,chairs,hero)
                    anchor=adapter.start_hand(start_raw)
                    obs0=adapter.normalize(start_raw,anchor)
                    if tracker.start_hand(anchor,obs0).kind!="START":
                        failures.append({"seed":seed,"scenario":scenario,"kind":"skip_start"})
                        break
                    first=truth.public_snapshot()
                    first_visible=[a for a in _options(first,rng) if a.action_type not in (0,1)]
                    rng.shuffle(first_visible)
                    applied_first=False
                    for a1 in first_visible:
                        probe=truth.clone()
                        try:
                            probe.apply_exact(a1.action_type,a1.amount_to)
                            if probe.terminal:
                                continue
                            second=probe.public_snapshot()
                            second_visible=[a for a in _options(second,rng) if a.action_type!=1]
                            if not second_visible:
                                continue
                            a2=second_visible[rng.randrange(len(second_visible))]
                            probe.apply_exact(a2.action_type,a2.amount_to)
                            skip_frame=_frame(hand_id+"-S",e,probe,chairs,hero)
                            skip_obs=adapter.normalize(skip_frame,anchor)
                            skipped_attempts+=1
                            if tracker.on_heartbeat(anchor,skip_obs).kind=="FAILED":
                                skipped_rejections+=1
                            applied_first=True
                            break
                        finally:
                            probe.close()
                    # No attempt is counted when this hand cannot produce two
                    # observable-impacting actions. Silent CHECKs are not
                    # considered skipped observable transitions.
                finally:
                    tracker.close()
                    truth.close()

        if failures or transitions>=int(args.max_transitions):
            break

    verdict=(
        not failures
        and transitions>0
        and hero_state_checks>0
        and street_reveals>0
        and exact_action_mismatches==0
        and canonical_mismatches==0
        and transcript_mismatches==0
        and duplicate_checks>0
        and silent_check_deferrals>0
        and (delayed_actions_reconciled_at_myturn>0 or multi_action_sync_events>0)
        and domain_counts["THREE_HANDED"]>0
        and domain_counts["TRUE_HEADS_UP"]>0
        and street_counts["0"]>0
        and sum(street_counts[str(i)] for i in (1,2,3))>0
        and corrupt_attempts==corrupt_rejections
        and skipped_attempts>0
        and skipped_attempts==skipped_rejections
    )

    out={
        "schema":"SPINCORE_LT2_OPENHOLDEM_OBSERVABLE_TRACKER_E2E_V1",
        "verdict":"PASS" if verdict else "FAIL",
        "method":{
            "forensic_seeds":list(SEEDS),
            "scenarios_per_seed":int(args.scenarios_per_seed),
            "max_transitions":int(args.max_transitions),
            "fault_hands":int(args.fault_hands),
            "training_roots":0,
            "optimizer_steps":0,
            "strategic_ev_evaluations":0,
            "deployment_model_inference":0,
            "holdout_reused":False,
            "pipeline":[
                "synthetic OpenHoldem raw frame",
                "strict OH symbol adapter",
                "observable one-action reconciliation",
                "canonical exact transcript",
                "from-scratch solver rebuild with actual visible board",
                "Hero canonical observation/action parity",
            ],
        },
        "transitions_checked":transitions,
        "duplicate_heartbeats_checked":duplicate_checks,
        "silent_check_deferrals":silent_check_deferrals,
        "delayed_actions_reconciled_at_myturn":delayed_actions_reconciled_at_myturn,
        "multi_action_sync_events":multi_action_sync_events,
        "street_reveals_checked":street_reveals,
        "hero_canonical_state_checks":hero_state_checks,
        "hero_states_by_domain":domain_counts,
        "hero_states_by_street":street_counts,
        "exact_action_mismatches":exact_action_mismatches,
        "canonical_state_mismatches":canonical_mismatches,
        "transcript_mismatches":transcript_mismatches,
        "faults":{
            "corrupt_attempts":corrupt_attempts,
            "corrupt_rejections":corrupt_rejections,
            "skipped_attempts":skipped_attempts,
            "skipped_rejections":skipped_rejections,
        },
        "failure_count":len(failures),
        "first_failure":failures[0] if failures else None,
    }
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== LT2 OPENHOLDEM OBSERVABLE TRACKER E2E ===")
    print(f"VERDICT={out['verdict']}")
    print(
        f"transitions={transitions} hero_checks={hero_state_checks} "
        f"street_reveals={street_reveals} duplicates={duplicate_checks} "
        f"silent_checks={silent_check_deferrals} "
        f"myturn_delayed={delayed_actions_reconciled_at_myturn} "
        f"multi_sync={multi_action_sync_events}"
    )
    print(f"faults={out['faults']}")
    print("LT2_OPENHOLDEM_OBSERVABLE_TRACKER_E2E_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0 if verdict else 2


if __name__=="__main__":
    raise SystemExit(main())
