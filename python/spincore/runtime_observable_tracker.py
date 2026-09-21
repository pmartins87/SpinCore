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


def _infer_observable_actions(
    state,
    observed: ObservedTableSnapshot,
    *,
    target_actor: int | None = None,
    max_steps: int = 6,
)->tuple[ResolvedExactAction, ...]:
    """Infer the unique canonical sequence evidenced by one observable frame.

    OpenHoldem balance/currentbet/pot/fold snapshots do not expose a CHECK:
    a check changes neither chips nor cards. Therefore the tracker may lag
    behind truth by one or more checks until later evidence appears.

    Safe rule:
    - permit any number of forced CHECKs required by actor order;
    - permit at most one observable-impacting action (fold/call/bet/raise/all-in);
    - stop as soon as the projected public snapshot matches;
    - when target_actor is supplied (DLLUpdateOnMyTurn), continue through
      otherwise invisible CHECKs until that actor is canonical.

    Two chip/fold-changing actions in one scrape interval remain a hard fail:
    that is a genuinely skipped observable transition.
    """
    probe=state.clone()
    actions:list[ResolvedExactAction]=[]
    visible_actions=0
    try:
        for _ in range(int(max_steps)+1):
            public=probe.public_snapshot()
            projected=canonical_observable_projection(public,observed.board_cards)
            actor_ok=(
                target_actor is None
                or (not public.terminal and int(public.actor)==int(target_actor))
            )
            if projected==observed and actor_ok:
                return tuple(actions)

            if public.terminal:
                raise ObservableTrackerError(
                    "terminal canonical state does not match observed snapshot"
                )

            actor=int(public.actor)
            if actor not in (0,1,2):
                raise ObservableTrackerError("invalid canonical actor")
            if int(observed.street)<int(public.street):
                raise ObservableTrackerError("observed street moved backwards")
            if int(observed.street)>int(public.street)+1:
                raise ObservableTrackerError("observed street skipped")
            if observed.stacks[actor]>public.stacks[actor]:
                raise ObservableTrackerError("acting stack increased")
            if public.folded[actor] or public.all_in[actor]:
                raise ObservableTrackerError("canonical actor is not actionable")

            paid=int(public.stacks[actor]-observed.stacks[actor])
            folded_now=bool(observed.folded[actor]) and not bool(public.folded[actor])

            if folded_now:
                if paid!=0:
                    raise ObservableTrackerError("fold transition paid chips")
                candidate=ResolvedExactAction(0,0)
                visible_actions+=1
            elif paid==0:
                # The only economically silent voluntary action is CHECK.
                candidate=ResolvedExactAction(1,0)
            else:
                target=int(public.street_commitments[actor]+paid)
                if public.to_call>0 and target<=public.current_bet:
                    candidate=ResolvedExactAction(2,0)
                elif int(observed.stacks[actor])==0:
                    candidate=ResolvedExactAction(5,0)
                elif public.current_bet==0:
                    candidate=ResolvedExactAction(3,target)
                else:
                    candidate=ResolvedExactAction(4,target)
                visible_actions+=1

            if visible_actions>1:
                raise ObservableTrackerError(
                    "more than one chip/fold-changing action occurred between observable frames"
                )

            try:
                probe.apply_exact(candidate.action_type,candidate.amount_to)
            except Exception as exc:
                raise ObservableTrackerError(
                    f"inferred exact action is illegal: {candidate}"
                ) from exc
            actions.append(candidate)

        raise ObservableTrackerError(
            f"observable reconciliation exceeded {max_steps} canonical actions"
        )
    finally:
        probe.close()


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

    def _sync(
        self,
        anchor:OpenHoldemHandAnchor,
        observed:ObservedTableSnapshot,
        *,
        target_actor:int | None=None,
    )->RuntimeSyncEvent:
        if self.failed:
            return RuntimeSyncEvent("FAILED",reason=self.failure_reason)
        if self.anchor is None or self._state is None:
            return self._fail("sync without active hand")
        if anchor!=self.anchor:
            return self._fail("OpenHoldem hand anchor changed without HandReset")

        current=self._state.public_snapshot()
        if (
            observed==self.observed
            and (
                target_actor is None
                or (not current.terminal and int(current.actor)==int(target_actor))
            )
        ):
            return RuntimeSyncEvent("NO_CHANGE")

        try:
            actions=_infer_observable_actions(
                self._state,
                observed,
                target_actor=target_actor,
            )
            if not actions:
                return RuntimeSyncEvent("NO_CHANGE")
            new_transcript=list(self.transcript)
            new_transcript.extend(actions)
            rebuilt=self._rebuild(self.anchor,observed,new_transcript)
            if target_actor is not None:
                rebuilt_public=rebuilt.public_snapshot()
                if rebuilt_public.terminal or int(rebuilt_public.actor)!=int(target_actor):
                    rebuilt.close()
                    raise ObservableTrackerError(
                        "rebuild did not reach required MyTurn actor"
                    )
        except Exception as exc:
            return self._fail(f"observable reconciliation/rebuild failed: {exc}")

        old=self._state
        self._state=rebuilt
        old.close()
        self.transcript=new_transcript
        self.observed=observed
        self.generation+=1
        self._clear_cache()
        return RuntimeSyncEvent("ACTION",action=actions[-1])

    def on_heartbeat(
        self,
        anchor:OpenHoldemHandAnchor,
        observed:ObservedTableSnapshot,
    )->RuntimeSyncEvent:
        # A duplicate chip/card snapshot is deliberately a no-op here.
        # Opponent CHECKs are synchronized only when later observable evidence
        # appears or DLLUpdateOnMyTurn proves that action order reached Hero.
        return self._sync(anchor,observed,target_actor=None)

    def on_my_turn(
        self,
        anchor:OpenHoldemHandAnchor,
        observed:ObservedTableSnapshot,
        compute_decision:Callable[["RuntimeObservableTracker"],Any],
    ):
        if self.anchor is None:
            return None
        event=self._sync(
            anchor,
            observed,
            target_actor=int(self.anchor.hero_logical_seat),
        )
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
