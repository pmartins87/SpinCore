from __future__ import annotations

"""Strict primitive-symbol bridge for the frozen DeepCrusher R8 oracle.

This module intentionally implements only symbols whose value is losslessly
present in SPNNIV3 or follows directly from that carrier.  Higher-order
OpenHoldem hand/draw/history symbols remain unsupported until separately
ported and parity-tested.  Unknown symbols fail closed; they are never zero
filled.
"""

from dataclasses import dataclass
from typing import Mapping

from spincore.deepcrusher_card_symbols import (
    DeepCrusherCardSymbols,
    UnknownDeepCrusherCardSymbol,
)
from spincore.deepcrusher_table_symbols import (
    DeepCrusherTableSymbols,
    UnknownDeepCrusherTableSymbol,
)
from spincore.deepcrusher_history_symbols import DeepCrusherHistorySymbols
from spincore.deepcrusher_state import (
    DeepCrusherStateView,
    STREET_PREFLOP,
    STREET_FLOP,
    STREET_TURN,
    STREET_RIVER,
)

STATUS_ACTIVE = 0
STATUS_FOLDED = 1
STATUS_ALL_IN_OR_ABSENT = 2


class UnknownDeepCrusherNativeSymbol(KeyError):
    pass


BENCHMARK_ENVIRONMENT_PROFILE_ID = "GGPoker_NoPT_NoNotes_V1"
OPENHOLDEM_UNDEFINED = -1.0


def frozen_benchmark_environment(native_identifiers) -> dict[str, float]:
    """Freeze non-strategic OpenHoldem environment for the offline benchmark.

    The profile represents a GGPoker table with no PokerTracker connection,
    no named-player chair matches and no manual colour notes.  OpenHoldem's
    log$ symbols evaluate true as a side-effect-friendly expression, while
    unavailable PokerTracker/chair$ lookups evaluate kUndefined (-1).
    prwin/prtie are deliberately NOT supplied here: they are strategic
    card/equity symbols and must be implemented separately.
    """
    out: dict[str, float] = {}
    for raw in native_identifiers:
        name = str(raw)
        low = name.lower()
        if low.startswith("pt_"):
            out[name] = OPENHOLDEM_UNDEFINED
        elif low.startswith("log$"):
            out[name] = 1.0
        elif low.startswith("chair$"):
            out[name] = OPENHOLDEM_UNDEFINED
        elif low.startswith("network$"):
            out[name] = 1.0 if low == "network$ggpoker" else 0.0
        elif low.startswith("colourcode"):
            out[name] = 0.0
    return out


@dataclass(frozen=True)
class DeepCrusherPrimitiveSymbols:
    view: DeepCrusherStateView
    environment: Mapping[str, float] | None = None

    def _is_absent(self, rel: int) -> bool:
        return bool(
            rel != 0
            and self.view.statuses[rel] == STATUS_ALL_IN_OR_ABSENT
            and self.view.stacks_bb[rel] == 0
            and self.view.street_commitments_bb[rel] == 0
            and self.view.total_commitments_bb[rel] == 0
        )

    def _live_opponent_rels(self) -> tuple[int, ...]:
        return tuple(
            rel
            for rel in (1, 2)
            if not self._is_absent(rel) and self.view.statuses[rel] != STATUS_FOLDED
        )

    def _headsup_rel(self) -> int:
        live = self._live_opponent_rels()
        return live[0] if len(live) == 1 else -1

    @staticmethod
    def supported_symbols() -> frozenset[str]:
        primitive = {
            "IsPreflop", "IsFlop", "IsTurn", "IsRiver", "betround",
            "bblind", "sblind",
            "AmountToCall", "DollarsToCall", "PotSize", "pot", "potcommon",
            "StackSize", "balance", "currentbet", "BetSize",
            "dealerchair", "smallblindchair", "bigblindchair", "userchair",
            "headsupchair", "balance_headsupchair", "StackSize_headsupchair",
            "currentbet_headsupchair",
            "InButton", "InSmallBlind", "InBigBlind",
            "nplayersplaying", "nopponentsallin", "OpponentIsAllin",
        }
        return frozenset(
            primitive
            | set(DeepCrusherCardSymbols.fixed_symbols())
            | set(DeepCrusherTableSymbols.fixed_symbols())
            | set(DeepCrusherHistorySymbols.fixed_symbols())
        )

    @classmethod
    def supports(cls, name: str) -> bool:
        folded = {item.lower() for item in cls.supported_symbols()}
        return (
            name.lower() in folded
            or DeepCrusherCardSymbols.supports(name)
            or DeepCrusherTableSymbols.supports(name)
            or DeepCrusherHistorySymbols.supports(name)
        )

    def resolve(self, name: str) -> float:
        if self.environment is not None:
            if name in self.environment:
                return float(self.environment[name])
            folded_env = {key.lower(): value for key, value in self.environment.items()}
            if name.lower() in folded_env:
                return float(folded_env[name.lower()])

        key = name.lower()
        v = self.view
        street_map = {
            STREET_PREFLOP: 1,
            STREET_FLOP: 2,
            STREET_TURN: 3,
            STREET_RIVER: 4,
        }

        if key == "ispreflop":
            return float(v.street == STREET_PREFLOP)
        if key == "isflop":
            return float(v.street == STREET_FLOP)
        if key == "isturn":
            return float(v.street == STREET_TURN)
        if key == "isriver":
            return float(v.street == STREET_RIVER)
        if key == "betround":
            return float(street_map[v.street])

        # SPNNIV3 numeric state is already normalized in big blinds.
        if key == "bblind":
            return 1.0
        if key == "sblind":
            return float(v.small_blind_bb)
        if key in ("amounttocall", "dollarstocall"):
            return float(v.to_call_bb)
        if key in ("potsize", "pot", "potcommon"):
            return float(v.pot_bb)
        if key in ("stacksize", "balance"):
            return float(v.hero_stack_bb)

        # OpenHoldem currentbet is the hero's current-street contribution.
        if key == "currentbet":
            return float(v.street_commitments_bb[0])
        # DeepCrusher's own f$BetSize_BKP documents BetSize as
        # AmountToCall + currentbet/bblind, i.e. the current table high-water
        # wager in normalized BB units.  SPNNIV3 carries that directly.
        if key == "betsize":
            return float(v.current_bet_bb)

        if key == "userchair":
            return 0.0
        if key == "dealerchair":
            return float(v.dealer_rel)
        if key == "smallblindchair":
            return float(v.small_blind_rel)
        if key == "bigblindchair":
            return float(v.big_blind_rel)

        headsup = self._headsup_rel()
        if key == "headsupchair":
            return float(headsup)
        if key in ("balance_headsupchair", "stacksize_headsupchair"):
            return float(v.stacks_bb[headsup]) if headsup >= 0 else 0.0
        if key == "currentbet_headsupchair":
            return float(v.street_commitments_bb[headsup]) if headsup >= 0 else 0.0

        if key == "inbutton":
            return float(v.hero_is_dealer)
        if key == "insmallblind":
            return float(v.hero_is_small_blind)
        if key == "inbigblind":
            return float(v.hero_is_big_blind)

        if key == "nplayersplaying":
            return float(
                sum(
                    1
                    for rel in (0, 1, 2)
                    if not self._is_absent(rel) and v.statuses[rel] != STATUS_FOLDED
                )
            )
        if key == "nopponentsallin":
            return float(
                sum(
                    1
                    for rel in (1, 2)
                    if not self._is_absent(rel)
                    and v.statuses[rel] == STATUS_ALL_IN_OR_ABSENT
                )
            )
        if key == "opponentisallin":
            return float(
                any(
                    not self._is_absent(rel)
                    and v.statuses[rel] == STATUS_ALL_IN_OR_ABSENT
                    for rel in (1, 2)
                )
            )

        try:
            return float(DeepCrusherCardSymbols(v).resolve(name))
        except UnknownDeepCrusherCardSymbol:
            pass

        try:
            return float(DeepCrusherTableSymbols(v).resolve(name))
        except UnknownDeepCrusherTableSymbol:
            pass

        if DeepCrusherHistorySymbols.supports(name):
            try:
                return float(DeepCrusherHistorySymbols(v).resolve(name))
            except KeyError:
                pass

        raise UnknownDeepCrusherNativeSymbol(name)

    def __call__(self, name: str) -> float:
        return self.resolve(name)
