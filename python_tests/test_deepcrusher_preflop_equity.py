from __future__ import annotations

import pytest

from spincore.deepcrusher_native_symbols import DeepCrusherPrimitiveSymbols
from spincore.deepcrusher_preflop_equity import (
    SOURCE_TXT_SHA256,
    SOURCE_ZIP_SHA256,
    hero_combo_key,
    range_equity,
)
from spincore.deepcrusher_state import (
    DeepCrusherStateView,
    STREET_PREFLOP,
)
from spincore.openppl_program import OpenPPLProgram


def _aa_view() -> DeepCrusherStateView:
    return DeepCrusherStateView(
        domain=1,
        street=STREET_PREFLOP,
        dealer_rel=0,
        small_blind_rel=0,
        big_blind_rel=1,
        live_count=2,
        visible_board=0,
        statuses=(0, 0, 2),
        ranks=(14, 14, 0, 0, 0, 0, 0),
        same_suit=(0,) * 21,
        pot_bb=1.5,
        to_call_bb=0.5,
        current_bet_bb=1.0,
        stacks_bb=(19.5, 19.0, 0.0),
        street_commitments_bb=(0.5, 1.0, 0.0),
        total_commitments_bb=(0.5, 1.0, 0.0),
        small_blind_bb=0.5,
        blind_index=0,
        min_raise_to_bb=2.0,
        max_raise_to_bb=20.0,
        primitive_legal=(True, False, True, False, True, True),
        history=(),
        exact_suits=(1, 2, -1, -1, -1, -1, -1),  # Ad Ac
    )


def test_compact_preflop_table_is_hash_pinned_and_covers_exact_aa_combo():
    assert SOURCE_ZIP_SHA256 == "52a0a87174b0d7cabd5b16fe43387b0807a6abd036e5a61c1aafbc008ecf50c2"
    assert SOURCE_TXT_SHA256 == "9dd539e2720010684d0006981207489e4f753b1d628f7e0443003b2c7f3e6c9f"
    view = _aa_view()
    assert hero_combo_key(view) == (49, 50)
    assert range_equity(view, 4) == pytest.approx(0.83200857, abs=1e-9)
    assert range_equity(view, 15) == pytest.approx(0.85070519, abs=1e-9)


def test_frozen_r8_multiplex_combined_equity_is_behaviorally_exact():
    program = OpenPPLProgram.from_text(
        """
##f$backup_opp_allin_range##
When Others Return 4 Force

##f$probe##
When Others Return (vs$multiplex$f$backup_opp_allin_range$prwin + vs$multiplex$f$backup_opp_allin_range$prtie/2) Force
"""
    )
    native = DeepCrusherPrimitiveSymbols(_aa_view())
    result = program.evaluate("f$probe", native)
    assert result.value == pytest.approx(0.83200857, abs=1e-9)


def test_multiplex_projection_refuses_unrelated_range_function():
    native = DeepCrusherPrimitiveSymbols(_aa_view())
    with pytest.raises(KeyError):
        native.resolve_versus_multiplex("f$some_other_range", 4, "$prwin")
