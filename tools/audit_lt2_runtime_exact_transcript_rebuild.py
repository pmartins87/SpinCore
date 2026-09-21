#!/usr/bin/env python3
from __future__ import annotations

"""Canonical runtime rebuild parity from an exact public action transcript.

This is a deployment-engineering gate for OpenHoldem integration.

For already-seen forensic episodes:
- choose one Hero seat;
- generate a public action transcript using canonical lean actions;
- at each Hero decision, throw away the live solver state;
- reconstruct from the original scenario using only:
  * Hero hole cards,
  * currently visible board cards,
  * deterministic legal fillers for hidden opponent holes/future board,
  * the exact public action transcript observed so far;
- replay that transcript through the authoritative exact-action solver API.

The rebuilt state must exactly reproduce actor/domain, SPNNIV1, SPNNIV2,
lean legal actions and exact resolution of every current legal lean action.

No model inference, EV, training, optimizer work or holdout reuse.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import audit_lt2_stage_a_b_first_divergence as fd
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import apply_lean, lean_legal_actions, resolve_lean_exact
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.solver import SolverLibrary

FORENSIC_SEEDS=(20260920,20260921,20260922,20260923,20260924,20260925)

_SOLVER=None


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,default=ROOT/"build/libspincore_solver_c.so")
    p.add_argument("--scenarios-per-seed",type=int,default=700)
    p.add_argument("--fillers-per-state",type=int,default=4)
    p.add_argument("--max-target-states",type=int,default=5000)
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


def _dead_seats(episode):
    dead=set(int(x) for x in episode.dead_players)
    if not dead and episode.game_is_hu:
        dead={i for i,x in enumerate(episode.stacks) if int(x)<=0}
    return dead


def _runtime_deal(snapshot,hero_seat:int,visible:int,seed:int,dead:set[int]):
    fixed=set()
    holes=[[-1,-1] for _ in range(3)]
    hero_cards=tuple(int(x) for x in snapshot.holes[int(hero_seat)])
    holes[int(hero_seat)]=[hero_cards[0],hero_cards[1]]
    fixed.update(hero_cards)

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
    return tuple(tuple(row) for row in holes),tuple(board)


def _choose_slot(legal,rng):
    # Encourage calls/checks so all streets receive meaningful coverage.
    if 1 in legal and rng.random()<0.58:
        return 1
    nonjam=[a for a in legal if a!=9]
    pool=nonjam if nonjam and rng.random()<0.90 else list(legal)
    return int(pool[rng.randrange(len(pool))])


def _init_worker(solver_path):
    global _SOLVER
    import os
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    os.environ["SPINCORE_TORCH_THREADS"]="1"
    _SOLVER=SolverLibrary(solver_path)


def _worker(task):
    seed,scenario,episode,deal_seed,hero,path,fillers=task
    reference=_SOLVER.create(episode,int(deal_seed))
    try:
        for action_type,amount_to in path:
            reference.apply_exact(int(action_type),int(amount_to))
        if reference.terminal:
            return None
        if int(reference.actor)!=int(hero):
            raise RuntimeError("target transcript does not end on Hero")
        expected=_context(reference)
        snap=reference.deal_snapshot()
        dead=_dead_seats(episode)

        checked=0
        exact_checks=0
        for rep in range(int(fillers)):
            holes,board=_runtime_deal(
                snap,int(hero),expected["visible_board_count"],
                fd._mix64(seed,scenario,len(path),rep,0xB017D),
                dead,
            )
            rebuilt=_SOLVER.create_with_deal(episode,holes,board)
            try:
                for action_type,amount_to in path:
                    rebuilt.apply_exact(int(action_type),int(amount_to))
                if rebuilt.terminal:
                    return {
                        "pass":False,
                        "seed":seed,
                        "scenario":scenario,
                        "field":"unexpected_terminal",
                        "path_length":len(path),
                        "filler_rep":rep,
                    }
                got=_context(rebuilt)
            except Exception as exc:
                return {
                    "pass":False,
                    "seed":seed,
                    "scenario":scenario,
                    "field":"replay_exception",
                    "error":str(exc),
                    "path_length":len(path),
                    "filler_rep":rep,
                }
            finally:
                rebuilt.close()

            for key in (
                "street","actor","domain","spnniv1","spnniv2",
                "active_mask","legal","resolved","visible_board_count",
            ):
                if got[key]!=expected[key]:
                    return {
                        "pass":False,
                        "seed":seed,
                        "scenario":scenario,
                        "field":key,
                        "street":expected["street"],
                        "path_length":len(path),
                        "filler_rep":rep,
                    }
            checked+=1
            exact_checks+=len(expected["legal"])

        return {
            "pass":True,
            "seed":seed,
            "scenario":scenario,
            "street":expected["street"],
            "domain":expected["domain"],
            "path_length":len(path),
            "fillers_checked":checked,
            "legal_exact_resolutions_checked":exact_checks,
        }
    finally:
        reference.close()


def main():
    args=parse_args()
    if min(args.scenarios_per_seed,args.fillers_per_state,args.max_target_states,args.workers)<=0:
        raise SystemExit("positive arguments required")

    solver_path=args.solver.resolve(strict=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    builder=SolverLibrary(solver_path)

    tasks=[]
    generated_by_street={0:0,1:0,2:0,3:0}
    generated_by_domain={0:0,1:0}

    for seed in FORENSIC_SEEDS:
        sampler=LegacyScenarioSampler(
            seed=int(seed)^0x5CE0A710,
            config=LegacyScenarioConfig(),
        )
        for scenario in range(int(args.scenarios_per_seed)):
            if len(tasks)>=int(args.max_target_states):
                break
            episode=sampler.sample_episode()
            live=[s for s,stack in enumerate(episode.stacks) if int(stack)>0]
            if len(live) not in (2,3):
                continue
            hero=live[int(fd._mix64(seed,scenario,0xA11CE)%len(live))]
            deal_seed=fd._mix64(int(seed),int(scenario),0xD34A1)
            state=builder.create(episode,int(deal_seed))
            rng=random.Random(fd._mix64(seed,scenario,0xBADA55))
            transcript=[]
            try:
                for _decision in range(80):
                    if state.terminal or len(tasks)>=int(args.max_target_states):
                        break
                    st=_street(state)
                    active_mask=FIRST_RELEASE_ACTION_SPEC.active_mask(st)
                    legal=tuple(int(x) for x in lean_legal_actions(state,active_mask))
                    if not legal:
                        raise RuntimeError("nonterminal state without lean legal actions")

                    if int(state.actor)==int(hero):
                        tasks.append((
                            int(seed),int(scenario),episode,int(deal_seed),int(hero),
                            tuple(transcript),int(args.fillers_per_state),
                        ))
                        generated_by_street[st]+=1
                        generated_by_domain[int(state.domain)]+=1
                        if len(tasks)>=int(args.max_target_states):
                            break

                    slot=_choose_slot(legal,rng)
                    action_type,amount_to=resolve_lean_exact(state,active_mask,slot)
                    apply_lean(state,active_mask,slot)
                    transcript.append((int(action_type),int(amount_to)))
            finally:
                state.close()
        if len(tasks)>=int(args.max_target_states):
            break

    if not tasks:
        raise RuntimeError("no Hero target decisions generated")

    rows=[]
    ctx=mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers),len(tasks)),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(solver_path),),
    ) as pool:
        for row in pool.map(_worker,tasks,chunksize=1):
            if row is not None:
                rows.append(row)

    failures=[r for r in rows if not r["pass"]]
    passed=[r for r in rows if r["pass"]]
    by_street={str(s):sum(1 for r in passed if int(r["street"])==s) for s in range(4)}
    by_domain={
        "THREE_HANDED":sum(1 for r in passed if int(r["domain"])==0),
        "TRUE_HEADS_UP":sum(1 for r in passed if int(r["domain"])==1),
    }
    path_lengths=[int(r["path_length"]) for r in passed]
    out={
        "schema":"SPINCORE_LT2_RUNTIME_EXACT_TRANSCRIPT_REBUILD_V1",
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
            "rebuild_inputs":[
                "original tournament scenario",
                "Hero hole cards",
                "currently visible board",
                "deterministic hidden-card fillers",
                "exact public voluntary action transcript",
            ],
            "replay_api":"spincore_solver_state_apply_exact",
        },
        "target_states_generated":len(tasks),
        "states_checked":len(passed),
        "alternate_rebuilds_checked":sum(int(r["fillers_checked"]) for r in passed),
        "legal_exact_resolutions_checked":sum(int(r["legal_exact_resolutions_checked"]) for r in passed),
        "states_by_street":by_street,
        "states_by_domain":by_domain,
        "generated_by_street":{str(k):v for k,v in generated_by_street.items()},
        "generated_by_domain":{
            "THREE_HANDED":generated_by_domain[0],
            "TRUE_HEADS_UP":generated_by_domain[1],
        },
        "path_length":{
            "min":min(path_lengths) if path_lengths else None,
            "max":max(path_lengths) if path_lengths else None,
            "mean":(sum(path_lengths)/len(path_lengths)) if path_lengths else None,
        },
        "failure_count":len(failures),
        "first_failure":failures[0] if failures else None,
    }
    args.report.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== LT2 RUNTIME EXACT-TRANSCRIPT REBUILD ===")
    print(f"VERDICT={out['verdict']}")
    print(
        f"states={out['states_checked']} rebuilds={out['alternate_rebuilds_checked']} "
        f"exact_resolutions={out['legal_exact_resolutions_checked']}"
    )
    print(f"by_domain={by_domain} by_street={by_street}")
    print(f"path_length={out['path_length']}")
    print("LT2_RUNTIME_EXACT_TRANSCRIPT_REBUILD_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0 if not failures else 2


if __name__=="__main__":
    raise SystemExit(main())
