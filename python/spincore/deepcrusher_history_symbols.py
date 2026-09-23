from __future__ import annotations

"""Offline-exact action/history symbols for the frozen DeepCrusher R8 oracle.

The live OpenHoldem implementation derives these symbols from scraper frames,
autoplayer counters and per-street memory.  In the benchmark we have a stronger
source: the complete ordered public action transcript carried by SPNNIV3.
This module therefore reconstructs the same poker facts directly from that
transcript and deliberately bypasses heartbeat-only bookkeeping.

One fidelity caveat is explicit: SPNNIV3 records the poker action (BetTo /
RaiseTo) but not which OpenHoldem button produced it (minimum-raise button vs
f$betsize).  The frozen R8 source only tests didrais together with didbetsize,
so all non-all-in aggressive hero actions are conservatively counted as
"didswag/didbetsize".  The runtime oracle must preserve action-origin metadata
before DC0 can claim general OpenHoldem parity beyond the frozen R8 contract.
"""

from dataclasses import dataclass
import re

from spincore.deepcrusher_state import (
    ACTION_ALL_IN,
    ACTION_BET_TO,
    ACTION_CALL,
    ACTION_CHECK,
    ACTION_FOLD,
    ACTION_RAISE_TO,
    DeepCrusherStateView,
    PublicActionEvent,
    STREET_FLOP,
    STREET_PREFLOP,
    STREET_RIVER,
    STREET_TURN,
)


K_UNDEFINED = -1.0
_EPS = 1.0e-6

_STREET_NAME = {
    "preflop": STREET_PREFLOP,
    "flop": STREET_FLOP,
    "turn": STREET_TURN,
    "river": STREET_RIVER,
}
_STREET_TO_OH = {
    STREET_PREFLOP: 1,
    STREET_FLOP: 2,
    STREET_TURN: 3,
    STREET_RIVER: 4,
}

_DID_RE = re.compile(
    r"^did(chec|call|rais|betsize)(?:round_(preflop|flop|turn|river|previousround))?$",
    re.I,
)
_NBETS_NAMED_RE = re.compile(
    r"^nbetsround_(preflop|flop|turn|river|previousround)$",
    re.I,
)
_NBETS_NUMERIC_RE = re.compile(r"^nbetsround([1-4])$", re.I)
_CALLBITS_NAMED_RE = re.compile(r"^callbits_(preflop|flop|turn|river)$", re.I)
_CALLBITS_NUMERIC_RE = re.compile(r"^callbits([1-4])$", re.I)


@dataclass(frozen=True)
class _AnnotatedEvent:
    event: PublicActionEvent
    highwater_before: float
    highwater_after: float
    aggressive: bool
    allin_call: bool


@dataclass(frozen=True)
class DeepCrusherHistorySymbols:
    view: DeepCrusherStateView
    hero_action_origins: tuple[str, ...] | None = None

    @staticmethod
    def fixed_symbols() -> frozenset[str]:
        names = {
            "prevaction",
            "raischair",
            "lastraiserchair",
            "firstraiserchair",
            "lastcallerchair",
            "firstcallerchair",
            "nopponentscalling",
            "nopponentstruelyraising",
            "StackSize_raischair",
            "BotsActionsOnThisRound",
            "BotsActionsOnThisRoundIncludingChecks",
            "CallsSinceLastPlay",
            "RaisesSinceLastPlay",
            "RaisesBeforeOurFirstAction",
            "Calls",
            "Raises",
            "Bets",
            "RaisesBeforeFlop",
            "RaisesOnFlop",
            "RaisesOnTurn",
            "NumberOfRaisesBeforeFlop",
            "NumberOfRaisesOnFlop",
            "NumberOfRaisesOnTurn",
            "BotCalledBeforeFlop",
            "BotCalledOnFlop",
            "BotCalledOnTurn",
            "BotCalledOnRiver",
            "BotCheckedPreflop",
            "BotCheckedOnFlop",
            "BotCheckedOnTurn",
            "BotCheckedOnRiver",
            "BotRaisedBeforeFlop",
            "BotRaisedOnFlop",
            "BotRaisedOnTurn",
            "BotRaisedOnRiver",
            "OpenPPLHistorySymbolsAlreadyUpdatedThisHeartbeatAfterAutoplayerAction",
            "GameStateChangedSinceLastAutoplayerAction",
        }
        for stem in ("didchec", "didcall", "didrais", "didbetsize"):
            names.add(stem)
            for street in ("preflop", "flop", "turn", "river", "previousround"):
                names.add(f"{stem}round_{street}")
        for street in ("preflop", "flop", "turn", "river", "previousround"):
            names.add(f"nbetsround_{street}")
        for number in range(1, 5):
            names.add(f"nbetsround{number}")
            names.add(f"callbits{number}")
        for street in ("preflop", "flop", "turn", "river"):
            names.add(f"callbits_{street}")
        names.add("lastraised_previousround")
        return frozenset(names)

    @classmethod
    def supports(cls, name: str) -> bool:
        low = str(name).lower()
        return low in {item.lower() for item in cls.fixed_symbols()}

    def _events(self, street: int | None = None) -> tuple[PublicActionEvent, ...]:
        return tuple(
            event
            for event in self.view.history
            if street is None or int(event.street) == int(street)
        )

    def _voluntary(self, street: int | None = None) -> tuple[PublicActionEvent, ...]:
        return tuple(event for event in self._events(street) if not event.forced)

    def _annotated(self, street: int) -> tuple[_AnnotatedEvent, ...]:
        highwater = 0.0
        out: list[_AnnotatedEvent] = []
        for event in self._events(street):
            before = highwater
            resulting = float(event.resulting_commitment_bb)
            aggressive = bool(
                not event.forced
                and int(event.action_type) in (ACTION_BET_TO, ACTION_RAISE_TO, ACTION_ALL_IN)
                and resulting > before + _EPS
            )
            allin_call = bool(
                not event.forced
                and int(event.action_type) == ACTION_ALL_IN
                and resulting <= before + _EPS
            )
            highwater = max(highwater, resulting)
            out.append(
                _AnnotatedEvent(
                    event=event,
                    highwater_before=before,
                    highwater_after=highwater,
                    aggressive=aggressive,
                    allin_call=allin_call,
                )
            )
        return tuple(out)

    def _target_street(self, suffix: str | None) -> int | None:
        if suffix is None:
            return int(self.view.street)
        low = suffix.lower()
        if low == "previousround":
            if self.view.street <= STREET_PREFLOP:
                return None
            return int(self.view.street) - 1
        return _STREET_NAME[low]

    def _hero_events(self, street: int) -> tuple[PublicActionEvent, ...]:
        return tuple(
            event for event in self._voluntary(street)
            if int(event.actor_rel) == 0
        )

    def _hero_origin_pairs(
        self,
        street: int | None = None,
    ) -> tuple[tuple[PublicActionEvent, str | None], ...]:
        all_hero = tuple(
            event for event in self._voluntary()
            if int(event.actor_rel) == 0
        )
        origins = self.hero_action_origins
        if origins is not None and len(origins) != len(all_hero):
            raise RuntimeError(
                "DeepCrusher action-origin/history length mismatch: "
                f"origins={len(origins)} hero_events={len(all_hero)}"
            )
        pairs = tuple(
            (event, None if origins is None else str(origins[index]).lower())
            for index, event in enumerate(all_hero)
        )
        if street is None:
            return pairs
        return tuple(
            (event, origin)
            for event, origin in pairs
            if int(event.street) == int(street)
        )

    def _hero_count(self, kind: str, street: int | None) -> int:
        if street is None:
            return 0
        events = self._hero_events(street)
        if self.hero_action_origins is not None:
            pairs = self._hero_origin_pairs(street)
            wanted = {
                "chec": "check",
                "call": "call",
                "rais": "raise",
                "betsize": "betsize",
            }[kind]
            return sum(origin == wanted for _event, origin in pairs)

        if kind == "chec":
            return sum(int(event.action_type) == ACTION_CHECK for event in events)
        if kind == "call":
            return sum(int(event.action_type) == ACTION_CALL for event in events)
        if kind == "rais":
            # Without action-origin metadata, exact poker actions cannot
            # distinguish the OpenHoldem minimum-raise button from f$betsize.
            return 0
        if kind == "betsize":
            annotated = self._annotated(street)
            return sum(
                item.aggressive and int(item.event.actor_rel) == 0
                and int(item.event.action_type) != ACTION_ALL_IN
                for item in annotated
            )
        raise AssertionError(kind)

    def _last_hero_index(self, street: int) -> int:
        last = -1
        for index, event in enumerate(self._events(street)):
            if not event.forced and int(event.actor_rel) == 0:
                last = index
        return last

    def _opponent_window(self, street: int) -> tuple[_AnnotatedEvent, ...]:
        annotated = self._annotated(street)
        boundary = self._last_hero_index(street)
        return tuple(
            item
            for index, item in enumerate(annotated)
            if index > boundary and int(item.event.actor_rel) != 0
        )

    def _current_callers(self, street: int) -> tuple[_AnnotatedEvent, ...]:
        """Opponent callers visible since hero's last action.

        If a later raise occurs, calls made before that raise no longer count as
        callers of the current wager.  This mirrors the intent of
        CSymbolEngineCallers' current-orbit scan without scraper instability.
        """
        window = self._opponent_window(street)
        if not window:
            return ()
        last_aggression = -1
        for index, item in enumerate(window):
            if item.aggressive:
                last_aggression = index
        relevant = window[last_aggression + 1 :] if last_aggression >= 0 else window
        return tuple(
            item
            for item in relevant
            if int(item.event.action_type) == ACTION_CALL or item.allin_call
        )

    def _aggressions(self, street: int) -> tuple[_AnnotatedEvent, ...]:
        return tuple(item for item in self._annotated(street) if item.aggressive)

    def _last_aggressor(self, street: int) -> int:
        aggressive = self._aggressions(street)
        return int(aggressive[-1].event.actor_rel) if aggressive else -1

    def _current_raischair(self) -> int:
        # OpenHoldem deliberately preserves raischair if a later street has no
        # bet to call. Reconstruct that by walking streets backwards.
        for street in range(int(self.view.street), STREET_PREFLOP - 1, -1):
            chair = self._last_aggressor(street)
            if chair >= 0:
                return chair
        return -1

    def _first_aggressor(self, street: int) -> int:
        aggressive = self._aggressions(street)
        return int(aggressive[0].event.actor_rel) if aggressive else -1

    def _nbetsround(self, street: int | None) -> float:
        if street is None:
            return 0.0
        highest = 0.0
        for event in self._events(street):
            highest = max(highest, float(event.resulting_commitment_bb))
        return highest

    def _callbits(self, street: int) -> int:
        bits = 0
        for item in self._annotated(street):
            event = item.event
            if event.forced or int(event.actor_rel) == 0:
                continue
            if int(event.action_type) == ACTION_CALL or item.allin_call:
                bits |= 1 << int(event.actor_rel)
        return bits

    def _last_hero_action(self) -> PublicActionEvent | None:
        for event in reversed(self._voluntary()):
            if int(event.actor_rel) == 0:
                return event
        return None

    def _prevaction(self) -> int:
        event = self._last_hero_action()
        if event is None:
            return -2

        if self.hero_action_origins is not None:
            pairs = self._hero_origin_pairs()
            if not pairs:
                return -2
            origin = pairs[-1][1]
            values = {
                "fold": -1,
                "check": 0,
                "call": 1,
                "raise": 2,
                "betsize": 3,
                "allin": 4,
            }
            if origin not in values:
                raise RuntimeError(f"unknown DeepCrusher action origin: {origin!r}")
            return values[origin]

        action = int(event.action_type)
        if action == ACTION_FOLD:
            return -1
        if action == ACTION_CHECK:
            return 0
        if action == ACTION_CALL:
            return 1
        if action in (ACTION_BET_TO, ACTION_RAISE_TO):
            return 3
        if action == ACTION_ALL_IN:
            return 4
        raise AssertionError(action)

    def _hero_actions_this_round(self, *, include_checks: bool) -> int:
        events = self._hero_events(int(self.view.street))
        total = 0
        for event in events:
            action = int(event.action_type)
            if action in (ACTION_FOLD, ACTION_ALL_IN):
                # No later live decision should exist after either.
                continue
            if action == ACTION_CHECK and not include_checks:
                continue
            total += 1
        return total

    def _total_opponent_aggressions(self, street: int) -> int:
        return sum(
            item.aggressive and int(item.event.actor_rel) != 0
            for item in self._annotated(street)
        )

    def _bot_acted(self, street: int, action: str) -> bool:
        if action == "called":
            return self._hero_count("call", street) > 0
        if action == "checked":
            return self._hero_count("chec", street) > 0
        if action == "raised":
            return (
                self._hero_count("rais", street)
                + self._hero_count("betsize", street)
            ) > 0
        raise AssertionError(action)

    def resolve(self, name: str) -> float:
        if not self.supports(name):
            raise KeyError(name)

        low = str(name).lower()

        match = _DID_RE.fullmatch(low)
        if match:
            street = self._target_street(match.group(2))
            return float(self._hero_count(match.group(1).lower(), street))

        match = _NBETS_NAMED_RE.fullmatch(low)
        if match:
            return float(self._nbetsround(self._target_street(match.group(1))))
        match = _NBETS_NUMERIC_RE.fullmatch(low)
        if match:
            street = int(match.group(1)) - 1
            return float(self._nbetsround(street))

        match = _CALLBITS_NAMED_RE.fullmatch(low)
        if match:
            return float(self._callbits(_STREET_NAME[match.group(1).lower()]))
        match = _CALLBITS_NUMERIC_RE.fullmatch(low)
        if match:
            return float(self._callbits(int(match.group(1)) - 1))

        if low == "prevaction":
            return float(self._prevaction())

        if low in ("raischair", "lastraiserchair"):
            return float(self._current_raischair())
        if low == "firstraiserchair":
            return float(self._first_aggressor(int(self.view.street)))
        if low == "lastraised_previousround":
            previous = self._target_street("previousround")
            return float(self._last_aggressor(previous)) if previous is not None else K_UNDEFINED

        callers = self._current_callers(int(self.view.street))
        if low == "nopponentscalling":
            return float(len(callers))
        if low == "firstcallerchair":
            return float(callers[0].event.actor_rel) if callers else K_UNDEFINED
        if low == "lastcallerchair":
            return float(callers[-1].event.actor_rel) if callers else K_UNDEFINED
        if low == "nopponentstruelyraising":
            return float(
                sum(item.aggressive for item in self._opponent_window(int(self.view.street)))
            )

        if low == "stacksize_raischair":
            chair = self._current_raischair()
            return float(self.view.stacks_bb[chair]) if chair >= 0 else 0.0

        if low == "botsactionsonthisround":
            return float(self._hero_actions_this_round(include_checks=False))
        if low == "botsactionsonthisroundincludingchecks":
            return float(self._hero_actions_this_round(include_checks=True))

        if low == "callssincelastplay":
            return float(len(callers))
        if low == "raisessincelastplay":
            return float(
                sum(item.aggressive for item in self._opponent_window(int(self.view.street)))
            )
        if low == "raisesbeforeourfirstaction":
            if self._last_hero_index(int(self.view.street)) >= 0:
                return 0.0
            return float(self._total_opponent_aggressions(int(self.view.street)))

        if low == "calls":
            return float(
                sum(
                    1
                    for event in self._voluntary(int(self.view.street))
                    if int(event.actor_rel) != 0 and int(event.action_type) == ACTION_CALL
                )
            )
        if low == "raises":
            return float(self._total_opponent_aggressions(int(self.view.street)))
        if low == "bets":
            if int(self.view.street) == STREET_PREFLOP:
                return 0.0
            return float(self._total_opponent_aggressions(int(self.view.street)))

        if low == "numberofraisesbeforeflop":
            return float(self._total_opponent_aggressions(STREET_PREFLOP))
        if low == "numberofraisesonflop":
            return float(self._total_opponent_aggressions(STREET_FLOP))
        if low == "numberofraisesonturn":
            return float(self._total_opponent_aggressions(STREET_TURN))
        if low == "raisesbeforeflop":
            return float(self._total_opponent_aggressions(STREET_PREFLOP) > 0)
        if low == "raisesonflop":
            return float(self._total_opponent_aggressions(STREET_FLOP) > 0)
        if low == "raisesonturn":
            return float(self._total_opponent_aggressions(STREET_TURN) > 0)

        bot_map = {
            "botcalledbeforeflop": ("called", STREET_PREFLOP),
            "botcalledonflop": ("called", STREET_FLOP),
            "botcalledonturn": ("called", STREET_TURN),
            "botcalledonriver": ("called", STREET_RIVER),
            "botcheckedpreflop": ("checked", STREET_PREFLOP),
            "botcheckedonflop": ("checked", STREET_FLOP),
            "botcheckedonturn": ("checked", STREET_TURN),
            "botcheckedonriver": ("checked", STREET_RIVER),
            "botraisedbeforeflop": ("raised", STREET_PREFLOP),
            "botraisedonflop": ("raised", STREET_FLOP),
            "botraisedonturn": ("raised", STREET_TURN),
            "botraisedonriver": ("raised", STREET_RIVER),
        }
        if low in bot_map:
            action, street = bot_map[low]
            return float(self._bot_acted(street, action))

        if low == "openpplhistorysymbolsalreadyupdatedthisheartbeatafterautoplayeraction":
            # Offline transcript has no asynchronous scraper heartbeat. All
            # history facts above are already computed from the authoritative
            # action list exactly once per decision.
            return 1.0
        if low == "gamestatechangedsincelastautoplayeraction":
            # A new decision state exists, therefore the public game state has
            # advanced since the preceding hero action (or this is first act).
            return 1.0

        raise KeyError(name)

    def __call__(self, name: str) -> float:
        return self.resolve(name)
