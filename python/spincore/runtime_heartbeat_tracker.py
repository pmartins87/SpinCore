from __future__ import annotations

"""Fail-closed heartbeat/lifecycle tracker for the LT2 OpenHoldem bridge.

This state machine sits above the public-snapshot reconciler.

Responsibilities:
- preserve one hand identity;
- accept duplicate heartbeats as no-ops;
- infer exactly one canonical action from a changed public snapshot;
- append the exact public transcript;
- latch failures until hand reset;
- preserve transcript across NewRound callbacks;
- cache exactly one Hero decision per canonical state so repeated ProcessQuery
  calls never resample/recompute strategy.

The tracker is deliberately strategy-agnostic. The cached decision may be any
runtime object produced by the deployment layer.
"""

from dataclasses import dataclass
from typing import Any, Callable

from spincore.runtime_transition_reconciler import (
    RuntimeReconciliationError,
    infer_single_exact_action,
)
from spincore.solver import Episode, PublicSnapshot, ResolvedExactAction, SolverLibrary


@dataclass(frozen=True)
class RuntimeSyncEvent:
    kind: str  # START, NO_CHANGE, ACTION, FAILED
    action: ResolvedExactAction | None = None
    reason: str | None = None


class RuntimeHeartbeatTracker:
    def __init__(self, solver: SolverLibrary):
        self.solver=solver
        self._state=None
        self.hand_id: str | None=None
        self.hero_seat: int | None=None
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

    def on_handreset(self)->None:
        self.close()
        self.hand_id=None
        self.hero_seat=None
        self.transcript=[]
        self.failed=False
        self.failure_reason=None
        self.generation=0
        self._clear_cache()

    def _clear_cache(self)->None:
        self._cached_generation=None
        self._cached_decision=None

    def _fail(self, reason:str)->RuntimeSyncEvent:
        self.failed=True
        self.failure_reason=str(reason)
        self._clear_cache()
        return RuntimeSyncEvent("FAILED",reason=self.failure_reason)

    def _require_hand(self, hand_id:str)->bool:
        if self.failed:
            return False
        if self._state is None or self.hand_id is None:
            self._fail("no active hand")
            return False
        if str(hand_id)!=self.hand_id:
            self._fail(
                f"hand identity mismatch: active={self.hand_id!r} observed={str(hand_id)!r}"
            )
            return False
        return True

    def start_hand(
        self,
        *,
        hand_id:str,
        episode:Episode,
        holes,
        board,
        hero_seat:int,
        observed_initial:PublicSnapshot,
    )->RuntimeSyncEvent:
        self.on_handreset()
        hid=str(hand_id)
        if not hid:
            return self._fail("empty hand identity")
        hero=int(hero_seat)
        if hero not in (0,1,2):
            return self._fail("invalid Hero seat")
        try:
            state=self.solver.create_with_deal(episode,holes,board)
            canonical=state.public_snapshot()
        except Exception as exc:
            return self._fail(f"hand initialization failed: {exc}")

        self._state=state
        self.hand_id=hid
        self.hero_seat=hero
        if canonical!=observed_initial:
            return self._fail("initial public snapshot does not match canonical hand start")
        return RuntimeSyncEvent("START")

    def on_new_round(self, hand_id:str)->RuntimeSyncEvent:
        if not self._require_hand(hand_id):
            return RuntimeSyncEvent("FAILED",reason=self.failure_reason)
        # Intentionally preserve transcript and decision generation. A betting
        # round callback is lifecycle metadata, not a hand reset.
        return RuntimeSyncEvent("NO_CHANGE")

    def on_heartbeat(self, hand_id:str, observed:PublicSnapshot)->RuntimeSyncEvent:
        if not self._require_hand(hand_id):
            return RuntimeSyncEvent("FAILED",reason=self.failure_reason)

        before=self._state.public_snapshot()
        if observed==before:
            return RuntimeSyncEvent("NO_CHANGE")

        try:
            action=infer_single_exact_action(self._state,observed)
            self._state.apply_exact(action.action_type,action.amount_to)
            if self._state.public_snapshot()!=observed:
                return self._fail("post-apply public snapshot drift")
        except RuntimeReconciliationError as exc:
            return self._fail(f"reconciliation failed: {exc}")
        except Exception as exc:
            return self._fail(f"runtime state update failed: {exc}")

        self.transcript.append(action)
        self.generation+=1
        self._clear_cache()
        return RuntimeSyncEvent("ACTION",action=action)

    def on_my_turn(
        self,
        hand_id:str,
        observed:PublicSnapshot,
        compute_decision:Callable[["RuntimeHeartbeatTracker"],Any],
    ):
        sync=self.on_heartbeat(hand_id,observed)
        if sync.kind=="FAILED":
            return None
        current=self._state.public_snapshot()
        if current.terminal:
            self._fail("MyTurn received on terminal state")
            return None
        if int(current.actor)!=int(self.hero_seat):
            self._fail(
                f"MyTurn actor mismatch: canonical={current.actor} hero={self.hero_seat}"
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

    def process_query(self, hand_id:str):
        if not self._require_hand(hand_id):
            return None
        if self._cached_generation!=self.generation:
            return None
        return self._cached_decision

    def canonical_public_snapshot(self)->PublicSnapshot | None:
        if self.failed or self._state is None:
            return None
        return self._state.public_snapshot()

    def transcript_tuples(self)->tuple[tuple[int,int],...]:
        return tuple((a.action_type,a.amount_to) for a in self.transcript)
