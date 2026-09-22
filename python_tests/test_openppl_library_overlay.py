from pathlib import Path

import pytest

from spincore.openppl_program import OpenPPLProgram, OpenPPLProgramError


ROOT = Path(__file__).resolve().parents[1]
LIB1 = ROOT / "fixtures" / "openppl_library_dc0" / "OpenPPL_Library_part1.ohf"
LIB2 = ROOT / "fixtures" / "openppl_library_dc0" / "OpenPPL_Library_part2.ohf"


def _program(strategy: str) -> OpenPPLProgram:
    return OpenPPLProgram.from_texts(
        strategy,
        library_texts=(
            LIB1.read_text(encoding="utf-8"),
            LIB2.read_text(encoding="utf-8"),
        ),
    )


def test_vendored_library_overlay_exposes_r8_standard_symbols():
    p = _program(
        """
##f$probe##
When Others Return HaveTopPair Force
"""
    )
    for name in (
        "HaveTopPair",
        "RaisesBeforeFlop",
        "BotCalledOnFlop",
        "Position",
        "FirstFlopCard",
        "HandIsHeadsup",
    ):
        assert p.has_library_function(name)


def test_library_function_is_compiled_lazily_and_uses_native_leaves():
    p = _program(
        """
##f$probe##
When Others Return HaveTopPair Force
"""
    )
    # HaveTopPair -> rankbitsplayer / BestBoardCard -> rankhicommon.
    value = p.evaluate(
        "f$probe",
        {
            "rankbitsplayer": float(1 << 14),
            "rankhicommon": 14.0,
        },
    )
    assert value.value == 1.0


def test_library_overlay_does_not_change_existing_strategy_function_priority():
    p = _program(
        """
##f$HaveTopPair##
When Others Return 7 Force
##f$probe##
When Others Return f$HaveTopPair Force
"""
    )
    assert p.evaluate("f$probe", {}).value == 7.0


def test_library_unknown_native_leaf_still_fails_closed():
    p = _program(
        """
##f$probe##
When Others Return HaveTopPair Force
"""
    )
    with pytest.raises(Exception):
        p.evaluate("f$probe", {"rankhicommon": 14.0})


def test_explicit_external_value_overrides_stock_library_section():
    p = _program(
        """
##f$probe##
When Others Return Calls Force
"""
    )
    # Offline DeepCrusher history is reconstructed from the authoritative
    # transcript and must override the stock library's heartbeat-memory Calls.
    assert p.evaluate("f$probe", {"Calls": 3.0}).value == 3.0


def test_unrecognized_external_symbol_falls_back_to_stock_library():
    p = _program(
        """
##f$probe##
When Others Return HaveTopPair Force
"""
    )

    def external(name: str) -> float:
        values = {
            "rankbitsplayer": float(1 << 14),
            "rankhicommon": 14.0,
        }
        if name in values:
            return values[name]
        raise KeyError(name)

    assert p.evaluate("f$probe", external).value == 1.0
