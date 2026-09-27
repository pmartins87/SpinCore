#!/usr/bin/env python3
from __future__ import annotations

"""Regression guard for OpenHoldem native nouts vs OpenPPL-library NOuts."""

from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))

from spincore.deepcrusher_native_symbols import DeepCrusherPrimitiveSymbols
from spincore.deepcrusher_state import (
    DeepCrusherStateView,
    DOMAIN_THREE_HANDED,
    STREET_FLOP,
)
from spincore.openppl_program import OpenPPLProgram


def make_view()->DeepCrusherStateView:
    # AhKh on Qh Jh 2c. Native OpenHoldem nouts = 18:
    # 9 remaining hearts + 3 non-heart tens + 6 non-heart A/K pairing cards.
    return DeepCrusherStateView(
        domain=DOMAIN_THREE_HANDED,
        street=STREET_FLOP,
        dealer_rel=2,
        small_blind_rel=0,
        big_blind_rel=1,
        live_count=3,
        visible_board=3,
        statuses=(0,0,0),
        ranks=(14,13,12,11,2,0,0),
        same_suit=(0,)*21,
        pot_bb=3.0,
        to_call_bb=0.0,
        current_bet_bb=0.0,
        stacks_bb=(20.0,20.0,20.0),
        street_commitments_bb=(0.0,0.0,0.0),
        total_commitments_bb=(1.0,1.0,1.0),
        small_blind_bb=0.5,
        blind_index=0,
        min_raise_to_bb=1.0,
        max_raise_to_bb=20.0,
        primitive_legal=(False,True,False,True,False,True),
        history=(),
        exact_suits=(0,0,0,0,2,-1,-1),
    )


def main()->int:
    view=make_view()
    provider=DeepCrusherPrimitiveSymbols(view)
    native=float(provider.resolve("nouts"))
    if native!=18.0:
        raise RuntimeError(f"native nouts parity drift: {native} != 18")

    strategy="""##f$test##
WHEN Others RETURN NOuts FORCE
"""
    library="""##NOuts##
WHEN IsFlop RETURN NOutsFlop FORCE
WHEN IsTurn RETURN NOutsTurn FORCE

##NOutsFlop##
WHEN Others RETURN nouts FORCE

##NOutsTurn##
WHEN Others RETURN nouts FORCE
"""
    program=OpenPPLProgram.from_texts(strategy,library_texts=(library,))
    result=program.evaluate("f$test",provider,hand_class=view.hero_hand_class)
    value=float(result.value)
    if value!=native:
        raise RuntimeError(
            f"NOuts library/native collision regression: library={value} native={native}"
        )

    print(f"native_nouts={native:.0f}")
    print(f"library_NOuts={value:.0f}")
    print("DEEPC_RUSHER_NOUTS_NATIVE_COLLISION_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
