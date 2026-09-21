from __future__ import annotations

"""Strict OpenHoldem-symbol snapshot normalization for LT2 runtime integration.

This module is intentionally independent of Windows/OpenHoldem headers. The
future user-DLL scraper fills OpenHoldemRawFrame from GetSymbol()/card symbols;
this adapter then validates and canonicalizes the table state into SpinCore's
three-seat logical world.

Canonical seat convention used by the runtime:
- logical 0 = dealer/button;
- 3H: logical 1 = next dealt chair clockwise (SB), logical 2 = next (BB);
- HU: logical 0 = dealer/SB, logical 1 = opponent/BB, logical 2 = dead.

Only public/scraped facts are normalized here. The authoritative solver remains
responsible for actor/legal-action semantics.
"""

from dataclasses import dataclass
import math
from typing import Iterable

from spincore.legacy_scenario import BLIND_LEVELS
from spincore.solver import Episode


class OpenHoldemAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenHoldemRawFrame:
    hand_id: str
    user_chair: int
    dealer_chair: int
    betround: int
    sblind: float
    bblind: float
    playersdealtbits: int
    playersplayingbits: int
    playersallinbits: int
    balances: tuple[float, ...]          # chairs 0..9
    currentbets: tuple[float, ...]       # chairs 0..9
    pot: float                           # OH: total chips in play incl. player bets
    ncommoncardsknown: int
    hero_cards: tuple[int, int]          # already converted to SpinCore ids 0..51
    board_cards: tuple[int, int, int, int, int]  # -1 for unrevealed


@dataclass(frozen=True)
class OpenHoldemHandAnchor:
    hand_id: str
    episode: Episode
    logical_to_chair: tuple[int, int, int]
    chair_to_logical: tuple[int, ...]    # length 10, -1 when not in hand
    hero_logical_seat: int
    hero_cards: tuple[int, int]


@dataclass(frozen=True)
class ObservedTableSnapshot:
    street: int
    pot: int
    stacks: tuple[int, int, int]
    street_commitments: tuple[int, int, int]
    folded: tuple[bool, bool, bool]
    all_in: tuple[bool, bool, bool]
    visible_board_count: int
    board_cards: tuple[int, int, int, int, int]


def _bits(value:int)->set[int]:
    raw=int(value)
    if raw<0 or raw>0x3ff:
        raise OpenHoldemAdapterError(f"invalid 10-chair bitmask {raw}")
    return {i for i in range(10) if raw & (1<<i)}


def _int_chip(value:float,name:str,*,tol:float=1e-6)->int:
    x=float(value)
    if not math.isfinite(x) or x<0:
        raise OpenHoldemAdapterError(f"{name} must be finite/nonnegative")
    r=int(round(x))
    if abs(x-r)>tol:
        raise OpenHoldemAdapterError(f"{name} must be integral chips, got {x!r}")
    return r


def _validate_chair(chair:int,name:str)->int:
    c=int(chair)
    if c<0 or c>9:
        raise OpenHoldemAdapterError(f"{name} outside 0..9")
    return c


def _clockwise_from(dealer:int,seats:set[int])->list[int]:
    out=[]
    for step in range(10):
        chair=(dealer+step)%10
        if chair in seats:
            out.append(chair)
    return out


def _blind_index(sb:int,bb:int)->int:
    for idx,(x,y) in enumerate(BLIND_LEVELS):
        if int(x)==int(sb) and int(y)==int(bb):
            return idx
    raise OpenHoldemAdapterError(
        f"unsupported blind level {sb}/{bb}; frozen LT2 supports {tuple(BLIND_LEVELS)!r}"
    )


def _validate_cards(hero:tuple[int,int],board:tuple[int,int,int,int,int],visible:int)->None:
    if len(hero)!=2 or len(board)!=5:
        raise OpenHoldemAdapterError("card vector length drift")
    known=[]
    for c in hero:
        if int(c)<0 or int(c)>=52:
            raise OpenHoldemAdapterError("invalid Hero card id")
        known.append(int(c))
    for i,c in enumerate(board):
        c=int(c)
        if i<visible:
            if c<0 or c>=52:
                raise OpenHoldemAdapterError("visible board card missing/invalid")
            known.append(c)
        elif c!=-1:
            raise OpenHoldemAdapterError("unrevealed board slot must be -1")
    if len(known)!=len(set(known)):
        raise OpenHoldemAdapterError("duplicate known cards in OpenHoldem frame")


class OpenHoldemSymbolAdapter:
    def start_hand(self,frame:OpenHoldemRawFrame)->OpenHoldemHandAnchor:
        if len(frame.balances)!=10 or len(frame.currentbets)!=10:
            raise OpenHoldemAdapterError("OpenHoldem frame requires 10 chair balances/currentbets")
        hand_id=str(frame.hand_id)
        if not hand_id:
            raise OpenHoldemAdapterError("empty hand id")
        user=_validate_chair(frame.user_chair,"userchair")
        dealer=_validate_chair(frame.dealer_chair,"dealerchair")
        if int(frame.betround)!=1:
            raise OpenHoldemAdapterError("hand anchor must be captured preflop")
        if int(frame.ncommoncardsknown)!=0:
            raise OpenHoldemAdapterError("hand anchor unexpectedly has community cards")

        dealt=_bits(frame.playersdealtbits)
        playing=_bits(frame.playersplayingbits)
        allin=_bits(frame.playersallinbits)
        if len(dealt) not in (2,3):
            raise OpenHoldemAdapterError("LT2 runtime supports exactly 2 or 3 dealt players")
        if dealer not in dealt:
            raise OpenHoldemAdapterError("dealerchair is not a dealt chair")
        if user not in dealt:
            raise OpenHoldemAdapterError("userchair is not a dealt chair")
        if not playing.issubset(dealt):
            raise OpenHoldemAdapterError("playersplayingbits contains undealt chair")
        if not allin.issubset(dealt):
            raise OpenHoldemAdapterError("playersallinbits contains undealt chair")

        order=_clockwise_from(dealer,dealt)
        if len(order)!=len(dealt) or order[0]!=dealer:
            raise OpenHoldemAdapterError("could not establish clockwise dealt-seat order")

        game_is_hu=len(order)==2
        if game_is_hu:
            logical_to_chair=(order[0],order[1],-1)
            dead_players=(2,)
        else:
            logical_to_chair=(order[0],order[1],order[2])
            dead_players=()

        chair_to_logical=[-1]*10
        for logical,chair in enumerate(logical_to_chair):
            if chair>=0:
                chair_to_logical[chair]=logical

        sb=_int_chip(frame.sblind,"sblind")
        bb=_int_chip(frame.bblind,"bblind")
        if sb<=0 or bb<sb:
            raise OpenHoldemAdapterError("invalid blind amounts")
        blind_index=_blind_index(sb,bb)

        stacks=[0,0,0]
        for logical,chair in enumerate(logical_to_chair):
            if chair<0:
                continue
            bal=_int_chip(frame.balances[chair],f"balance{chair}")
            cur=_int_chip(frame.currentbets[chair],f"currentbet{chair}")
            stacks[logical]=bal+cur

        # The authoritative BettingEngine immediately posts blinds at creation.
        # Requiring the scraped anchor to match those posts catches chair/order
        # mistakes before any transcript is accumulated.
        expected_current=[0,0,0]
        if game_is_hu:
            expected_current[0]=min(stacks[0],sb)
            expected_current[1]=min(stacks[1],bb)
        else:
            expected_current[1]=min(stacks[1],sb)
            expected_current[2]=min(stacks[2],bb)
        actual_current=[
            0 if chair<0 else _int_chip(frame.currentbets[chair],f"currentbet{chair}")
            for chair in logical_to_chair
        ]
        if tuple(actual_current)!=tuple(expected_current):
            raise OpenHoldemAdapterError(
                f"hand-start blind-post mismatch expected={tuple(expected_current)} "
                f"observed={tuple(actual_current)}"
            )

        total=sum(stacks)
        if total<=0:
            raise OpenHoldemAdapterError("nonpositive tournament chip total")

        hero_logical=chair_to_logical[user]
        if hero_logical<0:
            raise OpenHoldemAdapterError("userchair failed logical mapping")

        board=tuple(int(x) for x in frame.board_cards)
        hero=tuple(int(x) for x in frame.hero_cards)
        _validate_cards(hero,board,0)

        episode=Episode(
            total_chips=total,
            game_is_hu=game_is_hu,
            blind_index=blind_index,
            small_blind=sb,
            big_blind=bb,
            stacks=tuple(stacks),
            dealer_id=0,
            dead_players=dead_players,
        )
        return OpenHoldemHandAnchor(
            hand_id=hand_id,
            episode=episode,
            logical_to_chair=logical_to_chair,
            chair_to_logical=tuple(chair_to_logical),
            hero_logical_seat=int(hero_logical),
            hero_cards=hero,
        )

    def normalize(
        self,
        frame:OpenHoldemRawFrame,
        anchor:OpenHoldemHandAnchor,
    )->ObservedTableSnapshot:
        if str(frame.hand_id)!=anchor.hand_id:
            raise OpenHoldemAdapterError("hand id changed without HandReset")
        if len(frame.balances)!=10 or len(frame.currentbets)!=10:
            raise OpenHoldemAdapterError("OpenHoldem frame requires 10 chair balances/currentbets")
        if _validate_chair(frame.user_chair,"userchair") != anchor.logical_to_chair[anchor.hero_logical_seat]:
            raise OpenHoldemAdapterError("userchair changed within hand")
        if _validate_chair(frame.dealer_chair,"dealerchair") != anchor.logical_to_chair[0]:
            raise OpenHoldemAdapterError("dealerchair changed within hand")

        sb=_int_chip(frame.sblind,"sblind")
        bb=_int_chip(frame.bblind,"bblind")
        if (sb,bb)!=(anchor.episode.small_blind,anchor.episode.big_blind):
            raise OpenHoldemAdapterError("blind amounts changed within hand")

        dealt=_bits(frame.playersdealtbits)
        expected_dealt={c for c in anchor.logical_to_chair if c>=0}
        if dealt!=expected_dealt:
            raise OpenHoldemAdapterError("playersdealtbits changed within hand")
        playing=_bits(frame.playersplayingbits)
        allin=_bits(frame.playersallinbits)
        if not playing.issubset(dealt):
            raise OpenHoldemAdapterError("playing chair outside dealt set")
        if not allin.issubset(dealt):
            raise OpenHoldemAdapterError("all-in chair outside dealt set")
        if not allin.issubset(playing):
            raise OpenHoldemAdapterError("all-in chair must remain in playersplayingbits")

        br=int(frame.betround)
        if br<1 or br>4:
            raise OpenHoldemAdapterError("betround outside 1..4")
        street=br-1
        expected_visible=(0,3,4,5)[street]
        visible=int(frame.ncommoncardsknown)
        if visible!=expected_visible:
            raise OpenHoldemAdapterError(
                f"board-count/betround mismatch: street={street} visible={visible}"
            )

        hero=tuple(int(x) for x in frame.hero_cards)
        if hero!=anchor.hero_cards:
            raise OpenHoldemAdapterError("Hero cards changed within hand")
        board=tuple(int(x) for x in frame.board_cards)
        _validate_cards(hero,board,visible)

        stacks=[]
        street_commitments=[]
        folded=[]
        all_in=[]
        for logical,chair in enumerate(anchor.logical_to_chair):
            if chair<0:
                stacks.append(0)
                street_commitments.append(0)
                folded.append(False)
                all_in.append(True)
                continue
            bal=_int_chip(frame.balances[chair],f"balance{chair}")
            cur=_int_chip(frame.currentbets[chair],f"currentbet{chair}")
            stacks.append(bal)
            street_commitments.append(cur)
            is_allin=chair in allin
            is_folded=chair not in playing and not is_allin
            folded.append(bool(is_folded))
            all_in.append(bool(is_allin))

        pot=_int_chip(frame.pot,"pot")
        if pot!=sum(
            anchor.episode.stacks[i]-stacks[i]
            for i in range(3)
            if anchor.logical_to_chair[i]>=0
        ):
            raise OpenHoldemAdapterError(
                "pot does not equal hand-start chips minus current balances"
            )

        return ObservedTableSnapshot(
            street=street,
            pot=pot,
            stacks=tuple(stacks),
            street_commitments=tuple(street_commitments),
            folded=tuple(folded),
            all_in=tuple(all_in),
            visible_board_count=visible,
            board_cards=board,
        )


def canonical_observable_projection(public_snapshot,board_cards)->ObservedTableSnapshot:
    return ObservedTableSnapshot(
        street=int(public_snapshot.street),
        pot=int(public_snapshot.pot),
        stacks=tuple(int(x) for x in public_snapshot.stacks),
        street_commitments=tuple(int(x) for x in public_snapshot.street_commitments),
        folded=tuple(bool(x) for x in public_snapshot.folded),
        all_in=tuple(bool(x) for x in public_snapshot.all_in),
        visible_board_count=int(public_snapshot.visible_board_count),
        board_cards=tuple(int(x) for x in board_cards),
    )
