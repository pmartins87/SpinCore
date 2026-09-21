#!/usr/bin/env python3
from __future__ import annotations

"""Mechanical lifecycle/cache gate for the LT2 OpenHoldem heartbeat tracker.

Uses old forensic seeds only.

Checks:
- hand start identity;
- duplicate heartbeat acceptance without transcript mutation;
- exactly-one-action reconciliation;
- transcript persistence across NewRound callbacks;
- MyTurn decision computation exactly once per canonical state;
- repeated ProcessQuery returns the same cached decision;
- action transition invalidates decision cache;
- skipped/corrupt/wrong-hand input latches fail-closed;
- HandReset clears the failure latch and allows a clean next hand.

No deployment model inference, EV, training, optimizer work or holdout reuse.
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
from spincore.runtime_heartbeat_tracker import RuntimeHeartbeatTracker
from spincore.solver import Episode, ResolvedExactAction, SolverLibrary

SEEDS=(20260920,20260921,20260922,20260923,20260924,20260925)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--scenarios-per-seed",type=int,default=650)
    p.add_argument("--max-transitions",type=int,default=10000)
    p.add_argument("--fault-hands",type=int,default=500)
    p.add_argument("--report",type=Path,required=True)
    return p.parse_args()


def _dead(episode):
    d=set(int(x) for x in episode.dead_players)
    if not d and episode.game_is_hu:
        d={i for i,x in enumerate(episode.stacks) if int(x)<=0}
    return d


def _runtime_deal(snapshot,hero,seed,episode):
    # Hand start has no visible community cards. Preserve Hero only; all other
    # cards are deterministic legal fillers.
    fixed=set(int(x) for x in snapshot.holes[int(hero)])
    holes=[[-1,-1] for _ in range(3)]
    holes[int(hero)]=[int(snapshot.holes[int(hero)][0]),int(snapshot.holes[int(hero)][1])]
    remaining=[x for x in range(52) if x not in fixed]
    rng=random.Random(int(seed))
    rng.shuffle(remaining)
    cursor=0
    dead=_dead(episode)
    for seat in range(3):
        if seat in dead:
            holes[seat]=[-1,-1]
        elif seat!=int(hero):
            holes[seat]=[remaining[cursor],remaining[cursor+1]]
            cursor+=2
    board=[remaining[cursor+i] for i in range(5)]
    return tuple(tuple(x) for x in holes),tuple(board)


def _options(s,rng):
    actor=int(s.actor)
    stack=int(s.stacks[actor])
    out=[]
    if s.legal_fold: out.append(ResolvedExactAction(0,0))
    if s.legal_check: out.append(ResolvedExactAction(1,0))
    if s.legal_call: out.append(ResolvedExactAction(2,0))
    if s.legal_all_in and s.to_call<stack:
        out.append(ResolvedExactAction(5,0))
    if s.legal_bet or s.legal_raise:
        lo=int(s.min_raise_to);hi=int(s.max_raise_to)-1
        if lo<=hi:
            typ=3 if s.current_bet==0 else 4
            targets={lo,hi,(lo+hi)//2}
            if hi-lo>2:
                targets.add(rng.randint(lo,hi))
            for target in sorted(targets):
                out.append(ResolvedExactAction(typ,int(target)))
    uniq=[];seen=set()
    for a in out:
        key=(a.action_type,a.amount_to)
        if key not in seen:
            seen.add(key);uniq.append(a)
    return uniq


def _choose(s,rng):
    opts=_options(s,rng)
    if not opts:
        raise RuntimeError("no exact options")
    weighted=[]
    for a in opts:
        w=6 if a.action_type in (1,2) else 4 if a.action_type in (3,4) else 2 if a.action_type==5 else 1
        weighted.extend([a]*w)
    return weighted[rng.randrange(len(weighted))]


def _normalized(before,raw,after):
    actor=int(before.actor)
    paid=int(before.stacks[actor]-after.stacks[actor])
    target=int(before.street_commitments[actor]+paid)
    if raw.action_type==5 and before.to_call>0 and target<=before.current_bet:
        return ResolvedExactAction(2,0)
    if raw.action_type in (3,4) and int(after.stacks[actor])==0:
        return ResolvedExactAction(5,0)
    return raw


def _new_tracker(solver,episode,truth,hero,hand_id,seed):
    snap=truth.deal_snapshot()
    holes,board=_runtime_deal(snap,hero,seed,episode)
    tracker=RuntimeHeartbeatTracker(solver)
    event=tracker.start_hand(
        hand_id=hand_id,episode=episode,holes=holes,board=board,
        hero_seat=hero,observed_initial=truth.public_snapshot(),
    )
    if event.kind!="START":
        raise RuntimeError(f"tracker start failed: {event}")
    return tracker


def main():
    args=parse_args()
    if min(args.scenarios_per_seed,args.max_transitions,args.fault_hands)<=0:
        raise SystemExit("positive arguments required")
    solver_path=args.solver.resolve(strict=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    solver=SolverLibrary(solver_path)

    transitions=0
    duplicate_heartbeats=0
    action_events=0
    new_round_callbacks=0
    myturn_callbacks=0
    decision_computations=0
    repeated_myturn_cache_hits=0
    process_queries=0
    cache_invalidation_checks=0
    cache_invalidation_pass=0
    transcript_mismatches=0
    action_mismatches=0
    lifecycle_failures=[]

    skipped_attempts=skipped_rejections=0
    corrupt_attempts=corrupt_rejections=0
    wrong_hand_attempts=wrong_hand_rejections=0
    failure_latch_checks=failure_latch_pass=0
    reset_recovery_attempts=reset_recovery_pass=0

    fault_budget=int(args.fault_hands)

    for seed in SEEDS:
        sampler=LegacyScenarioSampler(
            seed=int(seed)^0x5CE0A710,
            config=LegacyScenarioConfig(),
        )
        for scenario in range(int(args.scenarios_per_seed)):
            if transitions>=int(args.max_transitions):
                break
            episode=sampler.sample_episode()
            live=[s for s,stack in enumerate(episode.stacks) if int(stack)>0]
            if len(live) not in (2,3):
                continue
            hero=live[int(fd._mix64(seed,scenario,0xA11CE)%len(live))]
            deal_seed=fd._mix64(seed,scenario,0xD34A1)
            truth=solver.create(episode,int(deal_seed))
            hand_id=f"H-{seed}-{scenario}"
            tracker=_new_tracker(
                solver,episode,truth,hero,hand_id,
                fd._mix64(seed,scenario,0xF111E2),
            )
            expected_transcript=[]
            rng=random.Random(fd._mix64(seed,scenario,0xC011BAC))
            compute_counter=[0]
            try:
                for decision_index in range(80):
                    if truth.terminal or transitions>=int(args.max_transitions):
                        break

                    before=truth.public_snapshot()

                    # Duplicate scrape/heartbeat is an accepted no-op.
                    prior_len=len(tracker.transcript)
                    for _ in range(2):
                        event=tracker.on_heartbeat(hand_id,before)
                        duplicate_heartbeats+=1
                        if event.kind!="NO_CHANGE" or len(tracker.transcript)!=prior_len or tracker.failed:
                            lifecycle_failures.append({
                                "seed":seed,"scenario":scenario,
                                "kind":"duplicate_heartbeat",
                            })
                            break
                    if lifecycle_failures:
                        break

                    if int(before.actor)==int(hero):
                        def compute(t):
                            compute_counter[0]+=1
                            return {
                                "generation":t.generation,
                                "transcript_len":len(t.transcript),
                                "nonce":fd._mix64(seed,scenario,decision_index,0xDEC1DE),
                            }

                        compute_before=compute_counter[0]
                        first=tracker.on_my_turn(hand_id,before,compute)
                        myturn_callbacks+=1
                        second=tracker.on_my_turn(hand_id,before,compute)
                        myturn_callbacks+=1
                        if first is None or second!=first:
                            lifecycle_failures.append({
                                "seed":seed,"scenario":scenario,
                                "kind":"myturn_cache_value",
                            })
                            break
                        expected_compute_count=compute_before+1
                        if compute_counter[0] != expected_compute_count:
                            lifecycle_failures.append({
                                "seed":seed,"scenario":scenario,
                                "kind":"myturn_recomputed",
                                "counter":compute_counter[0],
                                "expected":expected_compute_count,
                            })
                            break
                        decision_computations+=1
                        repeated_myturn_cache_hits+=1
                        for _ in range(3):
                            process_queries+=1
                            if tracker.process_query(hand_id)!=first:
                                lifecycle_failures.append({
                                    "seed":seed,"scenario":scenario,
                                    "kind":"process_query_cache",
                                })
                                break
                        if lifecycle_failures:
                            break

                    raw=_choose(before,rng)
                    old_street=int(before.street)
                    truth.apply_exact(raw.action_type,raw.amount_to)
                    after=truth.public_snapshot()
                    expected=_normalized(before,raw,after)
                    expected_transcript.append((expected.action_type,expected.amount_to))

                    event=tracker.on_heartbeat(hand_id,after)
                    transitions+=1
                    if event.kind!="ACTION":
                        lifecycle_failures.append({
                            "seed":seed,"scenario":scenario,
                            "kind":"action_not_reconciled",
                            "event":event.kind,
                            "reason":event.reason,
                        })
                        break
                    action_events+=1
                    if event.action!=expected:
                        action_mismatches+=1
                        lifecycle_failures.append({
                            "seed":seed,"scenario":scenario,
                            "kind":"action_mismatch",
                            "expected":(expected.action_type,expected.amount_to),
                            "got":None if event.action is None else (event.action.action_type,event.action.amount_to),
                        })
                        break
                    if tracker.transcript_tuples()!=tuple(expected_transcript):
                        transcript_mismatches+=1
                        lifecycle_failures.append({
                            "seed":seed,"scenario":scenario,
                            "kind":"transcript_mismatch",
                        })
                        break

                    if int(before.actor)==int(hero):
                        cache_invalidation_checks+=1
                        if tracker.process_query(hand_id) is None:
                            cache_invalidation_pass+=1
                        else:
                            lifecycle_failures.append({
                                "seed":seed,"scenario":scenario,
                                "kind":"cache_not_invalidated",
                            })
                            break

                    if not truth.terminal and int(after.street)!=old_street:
                        before_len=len(tracker.transcript)
                        nr=tracker.on_new_round(hand_id)
                        new_round_callbacks+=1
                        if nr.kind!="NO_CHANGE" or len(tracker.transcript)!=before_len or tracker.failed:
                            lifecycle_failures.append({
                                "seed":seed,"scenario":scenario,
                                "kind":"new_round_mutated_hand",
                            })
                            break
            finally:
                tracker.close()
                truth.close()

            if lifecycle_failures:
                break

            # Fault injections use fresh independent tracker instances and must
            # latch until HandReset. Count exact attempts, not only rejections.
            if fault_budget>0:
                fault_budget-=1
                truth=solver.create(episode,int(deal_seed))
                try:
                    # Corrupted one-step snapshot.
                    if not truth.terminal:
                        before=truth.public_snapshot()
                        raw=_choose(before,rng)
                        one=truth.clone()
                        try:
                            one.apply_exact(raw.action_type,raw.amount_to)
                            observed=one.public_snapshot()
                        finally:
                            one.close()
                        ft=_new_tracker(
                            solver,episode,truth,hero,hand_id+"-C",
                            fd._mix64(seed,scenario,0xCC01),
                        )
                        corrupt_attempts+=1
                        bad=replace(observed,pot=int(observed.pot)+1)
                        ev=ft.on_heartbeat(hand_id+"-C",bad)
                        if ev.kind=="FAILED" and ft.failed:
                            corrupt_rejections+=1
                        failure_latch_checks+=1
                        if ft.on_heartbeat(hand_id+"-C",observed).kind=="FAILED":
                            failure_latch_pass+=1
                        ft.on_handreset()
                        reset_recovery_attempts+=1
                        start=ft.start_hand(
                            hand_id=hand_id+"-C2",episode=episode,
                            holes=_runtime_deal(truth.deal_snapshot(),hero,fd._mix64(seed,scenario,0xCC02),episode)[0],
                            board=_runtime_deal(truth.deal_snapshot(),hero,fd._mix64(seed,scenario,0xCC02),episode)[1],
                            hero_seat=hero,observed_initial=truth.public_snapshot(),
                        )
                        if start.kind=="START" and not ft.failed:
                            reset_recovery_pass+=1
                        ft.close()

                        # Wrong hand identity.
                        wt=_new_tracker(
                            solver,episode,truth,hero,hand_id+"-W",
                            fd._mix64(seed,scenario,0xDD01),
                        )
                        wrong_hand_attempts+=1
                        ev=wt.on_heartbeat("WRONG-"+hand_id,truth.public_snapshot())
                        if ev.kind=="FAILED" and wt.failed:
                            wrong_hand_rejections+=1
                        wt.close()

                        # Skipped transition: apply two actions if possible.
                        first=truth.clone()
                        try:
                            b1=first.public_snapshot()
                            a1=_choose(b1,rng)
                            first.apply_exact(a1.action_type,a1.amount_to)
                            if not first.terminal:
                                b2=first.public_snapshot()
                                a2=_choose(b2,rng)
                                second=first.clone()
                                try:
                                    second.apply_exact(a2.action_type,a2.amount_to)
                                    skipped=second.public_snapshot()
                                finally:
                                    second.close()
                                st=_new_tracker(
                                    solver,episode,truth,hero,hand_id+"-S",
                                    fd._mix64(seed,scenario,0xEE01),
                                )
                                skipped_attempts+=1
                                ev=st.on_heartbeat(hand_id+"-S",skipped)
                                if ev.kind=="FAILED" and st.failed:
                                    skipped_rejections+=1
                                st.close()
                        finally:
                            first.close()
                finally:
                    truth.close()

        if lifecycle_failures or transitions>=int(args.max_transitions):
            break

    verdict=(
        not lifecycle_failures
        and transitions>0
        and action_events==transitions
        and action_mismatches==0
        and transcript_mismatches==0
        and duplicate_heartbeats>0
        and myturn_callbacks>0
        and decision_computations>0
        and repeated_myturn_cache_hits==decision_computations
        and process_queries>0
        and cache_invalidation_checks==cache_invalidation_pass
        and corrupt_attempts==corrupt_rejections
        and wrong_hand_attempts==wrong_hand_rejections
        and skipped_attempts>0
        and skipped_attempts==skipped_rejections
        and failure_latch_checks==failure_latch_pass
        and reset_recovery_attempts==reset_recovery_pass
    )

    out={
        "schema":"SPINCORE_LT2_RUNTIME_HEARTBEAT_TRACKER_V1",
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
            "lifecycle_contract":[
                "duplicate heartbeat = no-op",
                "one public transition = exactly one canonical transcript action",
                "NewRound preserves hand transcript",
                "MyTurn computes once per canonical state",
                "ProcessQuery returns cached decision only",
                "state change invalidates decision cache",
                "failure latches until HandReset",
            ],
        },
        "transitions_checked":transitions,
        "action_events":action_events,
        "duplicate_heartbeats_checked":duplicate_heartbeats,
        "new_round_callbacks_checked":new_round_callbacks,
        "myturn_callbacks":myturn_callbacks,
        "decision_computations":decision_computations,
        "repeated_myturn_cache_hits":repeated_myturn_cache_hits,
        "process_queries_checked":process_queries,
        "cache_invalidation_checks":cache_invalidation_checks,
        "cache_invalidation_pass":cache_invalidation_pass,
        "action_mismatches":action_mismatches,
        "transcript_mismatches":transcript_mismatches,
        "faults":{
            "corrupt_attempts":corrupt_attempts,
            "corrupt_rejections":corrupt_rejections,
            "wrong_hand_attempts":wrong_hand_attempts,
            "wrong_hand_rejections":wrong_hand_rejections,
            "skipped_attempts":skipped_attempts,
            "skipped_rejections":skipped_rejections,
            "failure_latch_checks":failure_latch_checks,
            "failure_latch_pass":failure_latch_pass,
            "reset_recovery_attempts":reset_recovery_attempts,
            "reset_recovery_pass":reset_recovery_pass,
        },
        "failure_count":len(lifecycle_failures),
        "first_failure":lifecycle_failures[0] if lifecycle_failures else None,
    }
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== LT2 RUNTIME HEARTBEAT/LIFECYCLE TRACKER ===")
    print(f"VERDICT={out['verdict']}")
    print(
        f"transitions={transitions} duplicate_heartbeats={duplicate_heartbeats} "
        f"myturn={myturn_callbacks} computations={decision_computations} "
        f"process_queries={process_queries}"
    )
    print(f"faults={out['faults']}")
    print("LT2_RUNTIME_HEARTBEAT_TRACKER_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0 if verdict else 2


if __name__=="__main__":
    raise SystemExit(main())
