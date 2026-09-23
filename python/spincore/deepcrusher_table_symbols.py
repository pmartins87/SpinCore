from __future__ import annotations

"""Frozen Hold'em/table/topology symbols for the DeepCrusher R8 oracle.

The offline benchmark is a normal 2- or 3-handed no-limit tournament table with
all scenario participants seated and no sit-outs. Chair ids are actor-relative:
hero=0 and opponents occupy the remaining V3 relative slots. For 3H those ids
preserve the physical clockwise seat order; true-HU V3 compacts the only live
opponent into chair 1.
"""

from dataclasses import dataclass
import re

from spincore.deepcrusher_state import (
    DOMAIN_THREE_HANDED,
    DOMAIN_TRUE_HEADS_UP,
    DeepCrusherStateView,
)


K_UNDEFINED = -1.0

# OpenHoldem StdDeck suit constants.
SUIT_CONSTANTS = {
    "hearts": 0.0,
    "diamonds": 1.0,
    "clubs": 2.0,
    "spades": 3.0,
}

# OpenHoldem pokerval category constants.
POKERVAL_CONSTANTS = {
    "hicard": 0x00000000,
    "onepair": 0x01000000,
    "twopair": 0x02000000,
    "threeofakind": 0x04000000,
    "straight": 0x08000000,
    "flush": 0x10000000,
    "fullhouse": 0x20000000,
    "fourofakind": 0x40000000,
    "straightflush": 0x80000000,
    "royalflush": 0x800EDCBA,
}

_CURRENTBET_RE = re.compile(r"^currentbet([0-9])$", re.I)
_BALANCE_RE = re.compile(r"^balance([0-9])$", re.I)


class UnknownDeepCrusherTableSymbol(KeyError):
    pass


@dataclass(frozen=True)
class DeepCrusherTableSymbols:
    view: DeepCrusherStateView

    def _is_absent(self, rel: int) -> bool:
        v = self.view
        return bool(
            rel != 0
            and v.statuses[rel] == 2
            and v.stacks_bb[rel] == 0
            and v.street_commitments_bb[rel] == 0
            and v.total_commitments_bb[rel] == 0
        )

    def _dealt_chairs(self) -> tuple[int, ...]:
        if self.view.domain == DOMAIN_TRUE_HEADS_UP:
            return (0, 1)
        if self.view.domain == DOMAIN_THREE_HANDED:
            return (0, 1, 2)
        raise ValueError(f"unknown domain: {self.view.domain}")

    def _playing_chairs(self) -> tuple[int, ...]:
        return tuple(
            rel for rel in self._dealt_chairs()
            if self.view.statuses[rel] != 1 and not self._is_absent(rel)
        )

    def _allin_chairs(self) -> tuple[int, ...]:
        return tuple(
            rel for rel in self._dealt_chairs()
            if self.view.statuses[rel] == 2 and not self._is_absent(rel)
        )

    @staticmethod
    def _bits(chairs: tuple[int, ...]) -> int:
        bits = 0
        for chair in chairs:
            bits |= 1 << int(chair)
        return bits

    def _dealposition(self) -> int:
        # Normal SpinCore scenarios always contain both blinds.
        if self.view.domain == DOMAIN_TRUE_HEADS_UP:
            # OpenHoldem HU: BB is dealt first, dealer/SB second.
            return 2 if self.view.dealer_rel == 0 else 1
        if self.view.domain == DOMAIN_THREE_HANDED:
            if self.view.small_blind_rel == 0:
                return 1
            if self.view.big_blind_rel == 0:
                return 2
            if self.view.dealer_rel == 0:
                return 3
        return 0

    def _betposition(self) -> int:
        playing = set(self._playing_chairs())
        if 0 not in playing:
            return 0
        if self.view.domain == DOMAIN_TRUE_HEADS_UP:
            # With a nonterminal hero decision both live players still play.
            return self._dealposition()

        dealer = int(self.view.dealer_rel)
        count = 0
        for offset in range(1, 4):
            chair = (dealer + offset) % 3
            if chair in playing:
                count += 1
            if chair == 0:
                return count
        raise AssertionError("hero chair not encountered")

    def _bigstackchair(self) -> int:
        candidates = [rel for rel in self._playing_chairs() if rel != 0]
        if not candidates:
            return -1
        # OpenHoldem Player::stack() is balance + current table bet. Preserve
        # its first-chair tie break by scanning actor-relative ids ascending.
        best = -1
        best_stack = 0.0
        for rel in sorted(candidates):
            amount = (
                float(self.view.stacks_bb[rel])
                + float(self.view.street_commitments_bb[rel])
            )
            if amount > best_stack:
                best_stack = amount
                best = rel
        return best

    @staticmethod
    def fixed_symbols() -> frozenset[str]:
        names = {
            "isnl", "isfl", "ispl", "isomaha", "istournament",
            "ismyturn", "isfinaltable", "sitename$openholdem",
            "buttonchair", "cutoffchair", "mp1chair", "mp2chair", "mp3chair",
            "ep1chair", "ep2chair", "ep3chair", "utgchair",
            "bigstackchair",
            "betposition", "dealposition",
            "nplayersdealt", "nopponentsdealt",
            "nplayersplaying", "nopponentsplaying",
            "nplayersseated", "nopponentsseated",
            "playersdealtbits", "opponentsdealtbits",
            "playersplayingbits", "opponentsplayingbits",
            "playersactivebits", "opponentsactivebits",
            "playersseatedbits", "opponentsseatedbits",
            "playersallinbits", "opponentsallinbits",
            "nplayersallin", "nopponentsallin",
            "ncurrentbets", "ncallbets",
            "balance_bigstackchair", "currentbet_bigstackchair",
            "currentbet_bigblindchair",
            *SUIT_CONSTANTS.keys(),
            *POKERVAL_CONSTANTS.keys(),
        }
        names.update(f"currentbet{i}" for i in range(10))
        names.update(f"balance{i}" for i in range(10))
        return frozenset(names)

    @classmethod
    def supports(cls, name: str) -> bool:
        return str(name).lower() in {x.lower() for x in cls.fixed_symbols()}

    def resolve(self, name: str) -> float:
        low = str(name).lower()
        v = self.view

        if low in SUIT_CONSTANTS:
            return SUIT_CONSTANTS[low]
        if low in POKERVAL_CONSTANTS:
            return float(POKERVAL_CONSTANTS[low])

        frozen = {
            "isnl": 1.0,
            "isfl": 0.0,
            "ispl": 0.0,
            "isomaha": 0.0,
            "istournament": 1.0,
            "ismyturn": 1.0,
            "isfinaltable": 1.0,
            "sitename$openholdem": 1.0,
        }
        if low in frozen:
            return frozen[low]

        if low == "buttonchair":
            return float(v.dealer_rel)
        if low in {
            "cutoffchair", "mp1chair", "mp2chair", "mp3chair",
            "ep1chair", "ep2chair", "ep3chair",
        }:
            # ChairByLogicalPosition returns kUndefined when that logical
            # position would be one of the two blind posters. At 2/3 handed,
            # every non-button logical offset is therefore absent.
            return K_UNDEFINED
        if low == "utgchair":
            return float(v.dealer_rel) if v.domain == DOMAIN_THREE_HANDED else K_UNDEFINED
        if low == "bigstackchair":
            return float(self._bigstackchair())

        if low == "dealposition":
            return float(self._dealposition())
        if low == "betposition":
            return float(self._betposition())

        dealt = self._dealt_chairs()
        playing = self._playing_chairs()
        seated = dealt
        active = dealt  # frozen benchmark has no sit-outs
        allin = self._allin_chairs()

        if low == "nplayersdealt":
            return float(len(dealt))
        if low == "nopponentsdealt":
            return float(len(dealt) - 1)
        if low == "nplayersplaying":
            return float(len(playing))
        if low == "nopponentsplaying":
            return float(sum(1 for x in playing if x != 0))
        if low == "nplayersseated":
            return float(len(seated))
        if low == "nopponentsseated":
            return float(len(seated) - 1)
        if low == "nplayersallin":
            return float(len(allin))
        if low == "nopponentsallin":
            return float(sum(1 for x in allin if x != 0))

        bit_values = {
            "playersdealtbits": self._bits(dealt),
            "opponentsdealtbits": self._bits(tuple(x for x in dealt if x != 0)),
            "playersplayingbits": self._bits(playing),
            "opponentsplayingbits": self._bits(tuple(x for x in playing if x != 0)),
            "playersactivebits": self._bits(active),
            "opponentsactivebits": self._bits(tuple(x for x in active if x != 0)),
            "playersseatedbits": self._bits(seated),
            "opponentsseatedbits": self._bits(tuple(x for x in seated if x != 0)),
            "playersallinbits": self._bits(allin),
            "opponentsallinbits": self._bits(tuple(x for x in allin if x != 0)),
        }
        if low in bit_values:
            return float(bit_values[low])

        # OpenHoldem NL table-limit bet is one BB; current bets are already BB.
        if low == "ncurrentbets":
            return float(v.street_commitments_bb[0])
        if low == "ncallbets":
            return float(v.current_bet_bb)

        match = _CURRENTBET_RE.fullmatch(low)
        if match:
            chair = int(match.group(1))
            return float(v.street_commitments_bb[chair]) if chair < 3 else 0.0
        match = _BALANCE_RE.fullmatch(low)
        if match:
            chair = int(match.group(1))
            return float(v.stacks_bb[chair]) if chair < 3 else 0.0

        if low == "currentbet_bigblindchair":
            chair = int(v.big_blind_rel)
            return float(v.street_commitments_bb[chair])

        if low in ("balance_bigstackchair", "currentbet_bigstackchair"):
            chair = self._bigstackchair()
            if chair < 0:
                return 0.0
            if low == "balance_bigstackchair":
                return float(v.stacks_bb[chair])
            return float(v.street_commitments_bb[chair])

        raise UnknownDeepCrusherTableSymbol(name)

    def __call__(self, name: str) -> float:
        return self.resolve(name)
