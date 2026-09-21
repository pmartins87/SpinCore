#!/usr/bin/env python3
from __future__ import annotations

"""Mechanical round-trip gate for OpenHoldem symbol/scrape normalization.

The test starts from authoritative solver states, renders them into synthetic
OpenHoldem symbol frames using the documented OH meanings of:
- balanceN;
- currentbetN;
- pot;
- playersdealtbits / playersplayingbits / playersallinbits;
- userchair / dealerchair;
- betround;
- visible board/card symbols.

It then throws away the canonical representation and asks the strict adapter to
recover the runtime hand anchor + observable public snapshot.

Old forensic seeds only. No model inference, EV, training, optimizer work or
holdout reuse.
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
from spincore.openholdem_symbol_adapter import (
    OpenHoldemAdapterError,
    OpenHoldemRawFrame,
    OpenHoldemSymbolAdapter,
    canonical_observable_projection,
    openholdem_betround_from_visible_count,
)
from spincore.solver import Episode, ResolvedExactAction, SolverLibrary

SEEDS=(20260920,20260921,20260922,20260923,20260924,20260925)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--scenarios-per-seed",type=int,default=650)
    p.add_argument("--max-frames",type=int,default=9000)
    p.add_argument("--fault-trials",type=int,default=1200)
    p.add_argument("--report",type=Path,required=True)
    return p.parse_args()


def _normalize_episode(e:Episode)->Episode:
    live=[i for i,s in enumerate(e.stacks) if int(s)>0]
    dealer=int(e.dealer_id)
    if dealer not in live:
        raise RuntimeError("sampled dealer not live")
    ordered=[]
    for step in range(3):
        seat=(dealer+step)%3
        if seat in live:
            ordered.append(seat)
    if e.game_is_hu:
        if len(ordered)!=2:
            raise RuntimeError("HU sample without two live seats")
        stacks=(int(e.stacks[ordered[0]]),int(e.stacks[ordered[1]]),0)
        dead=(2,)
    else:
        if len(ordered)!=3:
            raise RuntimeError("3H sample without three live seats")
        stacks=tuple(int(e.stacks[x]) for x in ordered)
        dead=()
    return Episode(
        total_chips=int(e.total_chips),
        game_is_hu=bool(e.game_is_hu),
        blind_index=int(e.blind_index),
        small_blind=int(e.small_blind),
        big_blind=int(e.big_blind),
        stacks=stacks,
        dealer_id=0,
        dead_players=dead,
    )


def _chairs_for_episode(e:Episode,rng):
    dealer=rng.randrange(10)
    if e.game_is_hu:
        gap=rng.randrange(1,10)
        opp=(dealer+gap)%10
        return (dealer,opp,-1)
    gaps=sorted(rng.sample(range(1,10),2))
    return (dealer,(dealer+gaps[0])%10,(dealer+gaps[1])%10)


def _frame(
    *,
    hand_id,
    e,
    state,
    logical_to_chair,
    hero,
):
    public=state.public_snapshot()
    deal=state.deal_snapshot()
    balances=[0.0]*10
    currentbets=[0.0]*10
    dealt=playing=allin=0
    for logical,chair in enumerate(logical_to_chair):
        if chair<0:
            continue
        dealt|=1<<chair
        balances[chair]=float(public.stacks[logical])
        currentbets[chair]=float(public.street_commitments[logical])
        if not public.folded[logical]:
            playing|=1<<chair
        if public.all_in[logical]:
            allin|=1<<chair
            playing|=1<<chair

    board=[-1]*5
    for i in range(int(public.visible_board_count)):
        board[i]=int(deal.board[i])
    hero_cards=tuple(int(x) for x in deal.holes[int(hero)])
    return OpenHoldemRawFrame(
        hand_id=str(hand_id),
        user_chair=int(logical_to_chair[hero]),
        dealer_chair=int(logical_to_chair[0]),
        betround=openholdem_betround_from_visible_count(int(public.visible_board_count)),
        sblind=float(e.small_blind),
        bblind=float(e.big_blind),
        playersdealtbits=dealt,
        playersplayingbits=playing,
        playersallinbits=allin,
        balances=tuple(balances),
        currentbets=tuple(currentbets),
        pot=float(public.pot),
        ncommoncardsknown=int(public.visible_board_count),
        hero_cards=hero_cards,
        board_cards=tuple(board),
    )


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
            for target in sorted({lo,hi,(lo+hi)//2}):
                out.append(ResolvedExactAction(typ,int(target)))
    return out


def _choose(s,rng):
    opts=_options(s,rng)
    if not opts:
        raise RuntimeError("no action options")
    passive=[a for a in opts if a.action_type in (1,2)]
    if passive and rng.random()<0.58:
        return passive[rng.randrange(len(passive))]
    nonfold=[a for a in opts if a.action_type!=0]
    pool=nonfold if nonfold and rng.random()<0.88 else opts
    return pool[rng.randrange(len(pool))]


def _expect_reject(adapter,frame,anchor=None,start=False):
    try:
        if start:
            adapter.start_hand(frame)
        else:
            adapter.normalize(frame,anchor)
    except OpenHoldemAdapterError:
        return True
    return False


def main():
    args=parse_args()
    if min(args.scenarios_per_seed,args.max_frames,args.fault_trials)<=0:
        raise SystemExit("positive arguments required")
    solver=SolverLibrary(args.solver.resolve(strict=True))
    adapter=OpenHoldemSymbolAdapter()
    args.report.parent.mkdir(parents=True,exist_ok=True)

    frames=0
    anchors=0
    mismatches=[]
    domain_counts={"THREE_HANDED":0,"TRUE_HEADS_UP":0}
    street_counts={str(i):0 for i in range(4)}
    hero_positions={str(i):0 for i in range(3)}
    chair_gap_layouts=set()

    fault_attempts=0
    fault_rejections=0
    fault_types={}

    for seed in SEEDS:
        sampler=LegacyScenarioSampler(seed=int(seed)^0x5CE0A710,config=LegacyScenarioConfig())
        for scenario in range(int(args.scenarios_per_seed)):
            if frames>=int(args.max_frames):
                break
            e=_normalize_episode(sampler.sample_episode())
            truth=solver.create(e,fd._mix64(seed,scenario,0xD34A1))
            rng=random.Random(fd._mix64(seed,scenario,0x0A0A0A))
            logical_to_chair=_chairs_for_episode(e,rng)
            hero=rng.choice([i for i,s in enumerate(e.stacks) if int(s)>0])
            hand_id=f"OH-{seed}-{scenario}"
            chair_gap_layouts.add(tuple(logical_to_chair))
            try:
                start_frame=_frame(
                    hand_id=hand_id,e=e,state=truth,
                    logical_to_chair=logical_to_chair,hero=hero,
                )
                anchor=adapter.start_hand(start_frame)
                anchors+=1
                if anchor.episode!=e or anchor.hero_logical_seat!=hero:
                    mismatches.append({
                        "seed":seed,"scenario":scenario,"kind":"anchor",
                        "expected_episode":repr(e),"got_episode":repr(anchor.episode),
                        "expected_hero":hero,"got_hero":anchor.hero_logical_seat,
                    })
                    break

                for decision in range(80):
                    if truth.terminal or frames>=int(args.max_frames):
                        break
                    frame=_frame(
                        hand_id=hand_id,e=e,state=truth,
                        logical_to_chair=logical_to_chair,hero=hero,
                    )
                    got=adapter.normalize(frame,anchor)
                    public=truth.public_snapshot()
                    board=tuple(
                        int(x) if i<int(public.visible_board_count) else -1
                        for i,x in enumerate(truth.deal_snapshot().board)
                    )
                    expected=canonical_observable_projection(public,board)
                    if got!=expected:
                        mismatches.append({
                            "seed":seed,"scenario":scenario,
                            "decision":decision,"kind":"snapshot",
                            "expected":repr(expected),"got":repr(got),
                        })
                        break

                    frames+=1
                    domain_counts["TRUE_HEADS_UP" if e.game_is_hu else "THREE_HANDED"]+=1
                    street_counts[str(public.street)]+=1
                    hero_positions[str(hero)]+=1

                    if fault_attempts<int(args.fault_trials):
                        candidates=[
                            ("wrong_hand",replace(frame,hand_id="WRONG-"+frame.hand_id),False),
                            ("unsupported_blinds",replace(frame,bblind=999.0),False),
                            ("fractional_balance",replace(
                                frame,
                                balances=tuple(
                                    (x+0.25 if i==logical_to_chair[hero] else x)
                                    for i,x in enumerate(frame.balances)
                                ),
                            ),False),
                            ("bad_board_count",replace(
                                frame,ncommoncardsknown=(1 if frame.betround==1 else 0)
                            ),False),
                        ]
                        name,bad,is_start=candidates[fault_attempts%len(candidates)]
                        fault_attempts+=1
                        ok=_expect_reject(adapter,bad,anchor,start=is_start)
                        fault_types.setdefault(name,{"attempts":0,"rejections":0})
                        fault_types[name]["attempts"]+=1
                        if ok:
                            fault_rejections+=1
                            fault_types[name]["rejections"]+=1

                    opts=_options(public,rng)
                    if not opts:
                        break
                    action=_choose(public,rng)
                    truth.apply_exact(action.action_type,action.amount_to)
            finally:
                truth.close()
            if mismatches:
                break
        if mismatches or frames>=int(args.max_frames):
            break

    verdict=(
        not mismatches
        and frames>0
        and anchors>0
        and domain_counts["THREE_HANDED"]>0
        and domain_counts["TRUE_HEADS_UP"]>0
        and all(street_counts[str(i)]>0 for i in range(4))
        and hero_positions["0"]>0
        and hero_positions["1"]>0
        and fault_attempts==fault_rejections
        and len(chair_gap_layouts)>20
    )
    out={
        "schema":"SPINCORE_LT2_OPENHOLDEM_SYMBOL_ADAPTER_V1",
        "verdict":"PASS" if verdict else "FAIL",
        "method":{
            "forensic_seeds":list(SEEDS),
            "scenarios_per_seed":int(args.scenarios_per_seed),
            "max_frames":int(args.max_frames),
            "fault_trials":int(args.fault_trials),
            "training_roots":0,
            "optimizer_steps":0,
            "strategic_ev_evaluations":0,
            "deployment_model_inference":0,
            "holdout_reused":False,
            "openholdem_semantics":[
                "balanceN = stack behind",
                "currentbetN = current-round chips in play",
                "pot = total chips in play including player bets",
                "playersdealtbits = hand-start participants",
                "playersplayingbits = not-folded participants",
                "playersallinbits = all-in participants",
                "betround 1..4 = preflop..river",
            ],
        },
        "anchors_checked":anchors,
        "frames_checked":frames,
        "states_by_domain":domain_counts,
        "states_by_street":street_counts,
        "hero_logical_positions":hero_positions,
        "distinct_physical_chair_layouts":len(chair_gap_layouts),
        "fault_attempts":fault_attempts,
        "fault_rejections":fault_rejections,
        "fault_types":fault_types,
        "failure_count":len(mismatches),
        "first_failure":mismatches[0] if mismatches else None,
    }
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== LT2 OPENHOLDEM SYMBOL/SCRAPE ADAPTER ===")
    print(f"VERDICT={out['verdict']}")
    print(
        f"anchors={anchors} frames={frames} layouts={len(chair_gap_layouts)} "
        f"faults={fault_rejections}/{fault_attempts}"
    )
    print(f"by_domain={domain_counts} by_street={street_counts} hero={hero_positions}")
    print("LT2_OPENHOLDEM_SYMBOL_ADAPTER_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0 if verdict else 2


if __name__=="__main__":
    raise SystemExit(main())
