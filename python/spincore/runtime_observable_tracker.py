from __future__ import annotations

"""End-to-end observable OpenHoldem tracker for the frozen LT2 runtime.

Consumes only:
- a validated OpenHoldemHandAnchor;
- normalized ObservedTableSnapshot values from the strict symbol adapter.

It never trusts hidden opponent cards or unrevealed future board cards.
After each accepted public action it rebuilds the authoritative SpinCore state
from scratch using:
- the frozen hand-start Episode;
- Hero's real hole cards;
- every currently visible real board card;
- deterministic lowest-card fillers for still-hidden positions;
- the accumulated canonical exact public transcript.

This is the production architecture needed to survive street reveals whose real
board cards differ from earlier filler cards.
"""

from dataclasses import dataclass
from typing import Any, Callable

from spincore.openholdem_symbol_adapter import (
    OpenHoldemHandAnchor,
    ObservedTableSnapshot,
    canonical_observable_projection,
)
from spincore.runtime_heartbeat_tracker import RuntimeSyncEvent
from spincore.solver import ResolvedExactAction, SolverLibrary


class ObservableTrackerError(RuntimeError):
    pass


def _runtime_deal(
    anchor: OpenHoldemHandAnchor,
    observed: ObservedTableSnapshot,
):
    hero=int(anchor.hero_logical_seat)
    dead=set(int(x) for x in anchor.episode.dead_players)

    holes=[[-1,-1] for _ in range(3)]
    holes[hero]=[int(anchor.hero_cards[0]),int(anchor.hero_cards[1])]

    fixed={int(anchor.hero_cards[0]),int(anchor.hero_cards[1])}
    board=[-1]*5
    visible=int(observed.visible_board_count)
    for i in range(visible):
        card=int(observed.board_cards[i])
        if card<0 or card>=52:
            raise ObservableTrackerError("visible board card missing/invalid")
        if card in fixed:
            raise ObservableTrackerError("duplicate visible/Hero card")
        fixed.add(card)
        board[i]=card

    remaining=[card for card in range(52) if card not in fixed]
    cursor=0
    for seat in range(3):
        if seat in dead:
            holes[seat]=[-1,-1]
            continue
        if seat==hero:
            continue
        holes[seat]=[remaining[cursor],remaining[cursor+1]]
        cursor+=2

    for i in range(visible,5):
        board[i]=remaining[cursor]
        cursor+=1

    return tuple(tuple(row) for row in holes),tuple(board)


def _infer_observable_action(
    state,
    observed: ObservedTableSnapshot,
)->ResolvedExactAction:
    before=state.public_snapshot()
    if before.terminal:
        raise ObservableTrackerError("cannot reconcile from terminal state")
    actor=int(before.actor)
    if actor not in (0,1,2):
        raise ObservableTrackerError("invalid canonical actor")

    if int(observed.street)<int(before.street):
        raise ObservableTrackerError("observed street moved backwards")
    if int(observed.street)>int(before.street)+1:
        raise ObservableTrackerError("observed street skipped")
    if observed.stacks[actor]>before.stacks[actor]:
        raise ObservableTrackerError("acting stack increased")
    if before.folded[actor] or before.all_in[actor]:
        raise ObservableTrackerError("canonical actor is not actionable")

    paid=int(before.stacks[actor]-observed.stacks[actor])
    folded_now=bool(observed.folded[actor]) and not bool(before.folded[actor])

    if folded_now:
        if paid!=0:
            raise ObservableTrackerError("fold transition paid chips")
        candidate=ResolvedExactAction(0,0)
    elif paid==0:
        candidate=ResolvedExactAction(1,0)
    else:
        target=int(before.street_commitments[actor]+paid)
        if before.to_call>0 and target<=before.current_bet:
            candidate=ResolvedExactAction(2,0)
        elif int(observed.stacks[actor])==0:
            candidate=ResolvedExactAction(5,0)
        elif before.current_bet==0:
            candidate=ResolvedExactAction(3,target)
        else:
            candidate=ResolvedExactAction(4,target)

    probe=state.clone()
    try:
        try:
            probe.apply_exact(candidate.action_type,candidate.amount_to)
        except Exception as exc:
            raise ObservableTrackerError(
                f"inferred exact action is illegal: {candidate}"
            ) from exc
        got=canonical_observable_projection(
            probe.public_snapshot(),
            observed.board_cards,
        )
    finally:
        probe.close()

    if got!=observed:
        raise ObservableTrackerError(
            "one-action candidate does not reproduce observable table snapshot"
        )
    return candidate


class RuntimeObservableTracker:
    def __init__(self, solver: SolverLibrary):
        self.solver=solver
        self.anchor: OpenHoldemHandAnchor | None=None
        self._state=None
        self.observed: ObservedTableSnapshot | None=None
        self.transcript: list[ResolvedExactAction]=[]
        self.failed=False
        self.failure_reason: str | None=None
        self.generation=0
        self._cached_generation: int | None=None
        self._cached_decision: Any=None

    def close(self)->None:
        if self._state is not None:
            self._state.close()
            self._state=None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _clear_cache(self)->None:
        self._cached_generation=None
        self._cached_decision=None

    def on_handreset(self)->None:
        self.close()
        self.anchor=None
        self.observed=None
        self.transcript=[]
        self.failed=False
        self.failure_reason=None
        self.generation=0
        self._clear_cache()

    def _fail(self,reason:str)->RuntimeSyncEvent:
        self.failed=True
        self.failure_reason=str(reason)
        self._clear_cache()
        return RuntimeSyncEvent("FAILED",reason=self.failure_reason)

    def _rebuild(
        self,
        anchor:OpenHoldemHandAnchor,
        observed:ObservedTableSnapshot,
        transcript:list[ResolvedExactAction],
    ):
        holes,board=_runtime_deal(anchor,observed)
        state=self.solver.create_with_deal(anchor.episode,holes,board)
        try:
            for action in transcript:
                state.apply_exact(action.action_type,action.amount_to)
            got=canonical_observable_projection(
                state.public_snapshot(),
                observed.board_cards,
            )
            if got!=observed:
                raise ObservableTrackerError(
                    "from-scratch transcript rebuild does not reproduce observed snapshot"
                )
            return state
        except Exception:
            state.close()
            raise

    def start_hand(
        self,
        anchor:OpenHoldemHandAnchor,
        observed_initial:ObservedTableSnapshot,
    )->RuntimeSyncEvent:
        self.on_handreset()
        try:
            state=self._rebuild(anchor,observed_initial,[])
        except Exception as exc:
            return self._fail(f"observable hand initialization failed: {exc}")
        self.anchor=anchor
        self.observed=observed_initial
        self._state=state
        return RuntimeSyncEvent("START")

    def on_new_round(self)->RuntimeSyncEvent:
        if self.failed:
            return RuntimeSyncEvent("FAILED",reason=self.failure_reason)
        if self._state is None or self.anchor is None:
            return self._fail("NewRound without active hand")
        return RuntimeSyncEvent("NO_CHANGE")

    def on_heartbeat(
        self,
        anchor:OpenHoldemHandAnchor,
        observed:ObservedTableSnapshot,
    )->RuntimeSyncEvent:
        if self.failed:
            return RuntimeSyncEvent("FAILED",reason=self.failure_reason)
        if self.anchor is None or self._state is None:
            return self._fail("heartbeat without active hand")
        if anchor!=self.anchor:
            return self._fail("OpenHoldem hand anchor changed without HandReset")

        if observed==self.observed:
            return RuntimeSyncEvent("NO_CHANGE")

        try:
            action=_infer_observable_action(self._state,observed)
            new_transcript=list(self.transcript)
            new_transcript.append(action)
            rebuilt=self._rebuild(self.anchor,observed,new_transcript)
        except Exception as exc:
            return self._fail(f"observable reconciliation/rebuild failed: {exc}")

        old=self._state
        self._state=rebuilt
        old.close()
        self.transcript=new_transcript
        self.observed=observed
        self.generation+=1
        self._clear_cache()
        return RuntimeSyncEvent("ACTION",action=action)

    def on_my_turn(
        self,
        anchor:OpenHoldemHandAnchor,
        observed:ObservedTableSnapshot,
        compute_decision:Callable[["RuntimeObservableTracker"],Any],
    ):
        event=self.on_heartbeat(anchor,observed)
        if event.kind=="FAILED":
            return None
        current=self._state.public_snapshot()
        if current.terminal:
            self._fail("MyTurn on terminal state")
            return None
        if int(current.actor)!=int(self.anchor.hero_logical_seat):
            self._fail(
                f"MyTurn actor mismatch canonical={current.actor} "
                f"hero={self.anchor.hero_logical_seat}"
            )
            return None

        if self._cached_generation==self.generation:
            return self._cached_decision

        try:
            decision=compute_decision(self)
        except Exception as exc:
            self._fail(f"decision computation failed: {exc}")
            return None
        if decision is None:
            self._fail("decision computation returned no action")
            return None

        self._cached_generation=self.generation
        self._cached_decision=decision
        return decision

    def process_query(self):
        if self.failed or self._state is None:
            return None
        if self._cached_generation!=self.generation:
            return None
        return self._cached_decision

    def canonical_state(self):
        if self.failed or self._state is None:
            return None
        return self._state

    def transcript_tuples(self)->tuple[tuple[int,int],...]:
        return tuple((a.action_type,a.amount_to) for a in self.transcript)
