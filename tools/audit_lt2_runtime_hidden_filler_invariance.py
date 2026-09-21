#!/usr/bin/env python3
from __future__ import annotations

"""Prove runtime hidden-card filler invariance for LT2 canonical state replay.

A real OpenHoldem runtime knows Hero's hole cards and visible board cards, but
not opponent holes or unrevealed future board cards. The intended bridge will
create deterministic legal filler cards for those hidden positions and replay
public lean actions into the authoritative solver.

This read-only mechanical audit checks that, for already-seen forensic states,
changing only hidden opponent holes and unrevealed future board cards leaves:
- current actor/domain;
- SPNNIV1 observation bytes;
- SPNNIV2 public metadata bytes;
- lean legal actions;
- exact resolution of every current lean legal action
identical.

No model inference, EV evaluation, optimizer step, training root or holdout.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_lt2_stage_a_b_first_divergence as fd
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import (
    apply_lean,
    lean_legal_actions,
    resolve_lean_exact,
)
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.solver import Episode, SolverLibrary

FORENSIC_SEEDS=(20260920,20260921,20260922,20260923,20260924,20260925)

_SOLVER=None


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--scenarios-per-seed",type=int,default=500)
    p.add_argument("--fillers-per-state",type=int,default=6)
    p.add_argument("--max-target-states",type=int,default=6000)
    p.add_argument("--workers",type=int,default=31)
    p.add_argument("--report",type=Path,required=True)
    return p.parse_args()


def _street(state)->int:
    payload=state.neural_bytes_v2()
    if len(payload)!=830 or not payload.startswith(b"SPNNIV2\x00"):
        raise RuntimeError("unexpected SPNNIV2 payload")
    street=int(payload[112])
    if street not in (0,1,2,3):
        raise RuntimeError(f"invalid street {street}")
    return street


def _context(state):
    street=_street(state)
    active_mask=FIRST_RELEASE_ACTION_SPEC.active_mask(street)
    legal=tuple(int(x) for x in lean_legal_actions(state,active_mask))
    resolved={
        int(slot):tuple(int(x) for x in resolve_lean_exact(state,active_mask,slot))
        for slot in legal
    }
    return {
        "street":street,
        "actor":int(state.actor),
        "domain":int(state.domain),
        "spnniv1":state.neural_bytes(),
        "spnniv2":state.neural_bytes_v2(),
        "active_mask":int(active_mask),
        "legal":legal,
        "resolved":resolved,
        "visible_board_count":int(state.deal_snapshot().visible_board_count),
    }


def _alternate_deal(snapshot,hero_seat:int,visible:int,seed:int,dead:set[int]):
    fixed=set()
    holes=[[-1,-1] for _ in range(3)]
    for card in snapshot.holes[int(hero_seat)]:
        holes[int(hero_seat)][0 if holes[int(hero_seat)][0]==-1 else 1]=int(card)
        fixed.add(int(card))

    board=[-1]*5
    for i in range(int(visible)):
        card=int(snapshot.board[i])
        board[i]=card
        fixed.add(card)

    remaining=[c for c in range(52) if c not in fixed]
    rng=random.Random(int(seed))
    rng.shuffle(remaining)
    cursor=0
    for seat in range(3):
        if seat in dead:
            holes[seat]=[-1,-1]
            continue
        if seat==int(hero_seat):
            continue
        holes[seat]=[remaining[cursor],remaining[cursor+1]]
        cursor+=2
    for i in range(int(visible),5):
        board[i]=remaining[cursor]
        cursor+=1
    return tuple(tuple(x) for x in holes),tuple(board)


def _init_worker(solver_path:str):
    global _SOLVER
    import os
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    os.environ["SPINCORE_TORCH_THREADS"]="1"
    _SOLVER=SolverLibrary(solver_path)


def _worker(task):
    seed,scenario,episode,deal_seed,path,fillers=task
    state=_SOLVER.create(episode,int(deal_seed))
    try:
        for mask,slot in path:
            apply_lean(state,int(mask),int(slot))
        if state.terminal:
            return None
        reference=_context(state)
        snap=state.deal_snapshot()
        hero=int(reference["actor"])
        dead=set(int(x) for x in episode.dead_players)
        if not dead and episode.game_is_hu:
            dead={i for i,x in enumerate(episode.stacks) if int(x)<=0}

        checked=0
        for rep in range(int(fillers)):
            holes,board=_alternate_deal(
                snap,hero,reference["visible_board_count"],
                fd._mix64(seed,scenario,len(path),rep,0xF111E2),
                dead,
            )
            alt=_SOLVER.create_with_deal(episode,holes,board)
            try:
                for mask,slot in path:
                    apply_lean(alt,int(mask),int(slot))
                if alt.terminal:
                    raise RuntimeError("alternate filler caused terminal drift")
                got=_context(alt)
            finally:
                alt.close()

            for key in ("actor","domain","spnniv1","spnniv2","active_mask","legal","resolved","visible_board_count"):
                if got[key]!=reference[key]:
                    return {
                        "pass":False,
                        "seed":seed,
                        "scenario":scenario,
                        "path_length":len(path),
                        "filler_rep":rep,
                        "field":key,
                        "street":reference["street"],
                    }
            checked+=1

        return {
            "pass":True,
            "seed":seed,
            "scenario":scenario,
            "path_length":len(path),
            "street":reference["street"],
            "domain":reference["domain"],
            "fillers_checked":checked,
            "legal_actions_checked":len(reference["legal"])*checked,
        }
    finally:
        state.close()


def _choose_slot(legal,rng):
    # Keep trajectories broad while avoiding pointless all-in saturation.
    if 1 in legal and rng.random()<0.52:
        return 1
    nonjam=[x for x in legal if x!=9]
    pool=nonjam if nonjam and rng.random()<0.82 else list(legal)
    return int(pool[rng.randrange(len(pool))])


def main():
    args=parse_args()
    if min(args.scenarios_per_seed,args.fillers_per_state,args.max_target_states,args.workers)<=0:
        raise SystemExit("positive arguments required")
    solver_path=args.solver.resolve(strict=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)

    sampler_tasks=[]
    builder=SolverLibrary(solver_path)
    street_candidates={0:0,1:0,2:0,3:0}
    for seed in FORENSIC_SEEDS:
        sampler=LegacyScenarioSampler(
            seed=int(seed)^0x5CE0A710,
            config=LegacyScenarioConfig(),
        )
        for scenario in range(int(args.scenarios_per_seed)):
            if len(sampler_tasks)>=int(args.max_target_states):
                break
            episode=sampler.sample_episode()
            deal_seed=fd._mix64(int(seed),int(scenario),0xD34A1)
            state=builder.create(episode,int(deal_seed))
            path=[]
            rng=random.Random(fd._mix64(seed,scenario,0xF177E2))
            try:
                for _decision in range(18):
                    if state.terminal or len(sampler_tasks)>=int(args.max_target_states):
                        break
                    st=_street(state)
                    sampler_tasks.append((
                        int(seed),int(scenario),episode,int(deal_seed),
                        tuple(path),int(args.fillers_per_state),
                    ))
                    street_candidates[st]+=1
                    active_mask=FIRST_RELEASE_ACTION_SPEC.active_mask(st)
                    legal=tuple(int(x) for x in lean_legal_actions(state,active_mask))
                    if not legal:
                        raise RuntimeError("nonterminal target without legal actions")
                    slot=_choose_slot(legal,rng)
                    apply_lean(state,active_mask,slot)
                    path.append((int(active_mask),int(slot)))
            finally:
                state.close()
        if len(sampler_tasks)>=int(args.max_target_states):
            break

    if not sampler_tasks:
        raise RuntimeError("no target states generated")

    rows=[]
    ctx=mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers),len(sampler_tasks)),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(solver_path),),
    ) as pool:
        for row in pool.map(_worker,sampler_tasks,chunksize=1):
            if row is not None:
                rows.append(row)

    failures=[r for r in rows if not r["pass"]]
    tested=[r for r in rows if r["pass"]]
    by_street={str(s):sum(1 for r in tested if int(r["street"])==s) for s in range(4)}
    by_domain={
        "THREE_HANDED":sum(1 for r in tested if int(r["domain"])==0),
        "TRUE_HEADS_UP":sum(1 for r in tested if int(r["domain"])==1),
    }
    out={
        "schema":"SPINCORE_LT2_RUNTIME_HIDDEN_FILLER_INVARIANCE_V1",
        "verdict":"PASS" if not failures else "FAIL",
        "method":{
            "forensic_seeds":list(FORENSIC_SEEDS),
            "scenarios_per_seed":int(args.scenarios_per_seed),
            "fillers_per_state":int(args.fillers_per_state),
            "max_target_states":int(args.max_target_states),
            "training_roots":0,
            "optimizer_steps":0,
            "strategic_ev_evaluations":0,
            "holdout_reused":False,
            "changed_information":"opponent private holes + unrevealed future board only",
            "preserved_information":"current actor Hero hole cards + visible board + public action path",
        },
        "target_states_generated":len(sampler_tasks),
        "states_checked":len(tested),
        "alternate_fillers_checked":sum(int(r["fillers_checked"]) for r in tested),
        "legal_exact_resolutions_checked":sum(int(r["legal_actions_checked"]) for r in tested),
        "states_by_street":by_street,
        "states_by_domain":by_domain,
        "generation_street_candidates":{str(k):v for k,v in street_candidates.items()},
        "failure_count":len(failures),
        "first_failure":failures[0] if failures else None,
    }
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== LT2 RUNTIME HIDDEN-FILLER INVARIANCE ===")
    print(f"VERDICT={out['verdict']}")
    print(
        f"states={out['states_checked']} fillers={out['alternate_fillers_checked']} "
        f"exact_resolutions={out['legal_exact_resolutions_checked']}"
    )
    print(f"by_domain={by_domain} by_street={by_street}")
    print("LT2_RUNTIME_HIDDEN_FILLER_INVARIANCE_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0 if not failures else 2


if __name__=="__main__":
    raise SystemExit(main())
