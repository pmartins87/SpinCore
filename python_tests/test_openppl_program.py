from __future__ import annotations

import unittest

from spincore.openppl_program import (
    DirectAction,
    OpenPPLProgram,
    OpenPPLSession,
    OpenPPLProgramError,
    ProgramContext,
    ReturnValue,
)


class OpenPPLProgramTests(unittest.TestCase):
    def test_ordered_when_and_others(self):
        p=OpenPPLProgram.from_text("""
##f$x##
When a Return 10 Force
When b Return 20 Force
When Others Return 30 Force
""")
        self.assertEqual(p.evaluate("f$x",{"a":1,"b":1}),ReturnValue(10))
        self.assertEqual(p.evaluate("f$x",{"a":0,"b":1}),ReturnValue(20))
        self.assertEqual(p.evaluate("f$x",{"a":0,"b":0}),ReturnValue(30))

    def test_set_continues_to_next_when(self):
        p=OpenPPLProgram.from_text("""
##f$x##
When a Set user_seen
When user_seen && b Return 7 Force
When Others Return 3 Force
""")
        self.assertEqual(p.evaluate("f$x",{"a":1,"b":1}),ReturnValue(7))
        self.assertEqual(p.evaluate("f$x",{"a":0,"b":1}),ReturnValue(3))

    def test_direct_call_and_fold(self):
        p=OpenPPLProgram.from_text("""
##f$x##
When a Call Force
When Others Fold Force
""")
        self.assertEqual(p.evaluate("f$x",{"a":1}),DirectAction("Call"))
        self.assertEqual(p.evaluate("f$x",{"a":0}),DirectAction("Fold"))

    def test_open_ended_when_backpatches_to_next_open_ender(self):
        p=OpenPPLProgram.from_text("""
##f$x##
When parent
When child_a Return 1 Force
When child_b Return 2 Force
When Others
When fallback Return 3 Force
When Others Return 4 Force
""")
        self.assertEqual(
            p.evaluate("f$x",{"parent":1,"child_a":0,"child_b":1,"fallback":0}),
            ReturnValue(2),
        )
        # parent false skips child_a/child_b and enters next open-ended branch.
        self.assertEqual(
            p.evaluate("f$x",{"parent":0,"child_a":1,"child_b":1,"fallback":1}),
            ReturnValue(3),
        )
        self.assertEqual(
            p.evaluate("f$x",{"parent":0,"child_a":1,"child_b":1,"fallback":0}),
            ReturnValue(4),
        )

    def test_helper_function_resolution(self):
        p=OpenPPLProgram.from_text("""
##f$helper##
x > 5
##f$main##
When f$helper Return 9 Force
When Others Return 0 Force
""")
        self.assertEqual(p.evaluate("f$main",{"x":6}),ReturnValue(9))
        self.assertEqual(p.evaluate("f$main",{"x":5}),ReturnValue(0))

    def test_user_variables_default_false_per_context(self):
        p=OpenPPLProgram.from_text("""
##f$main##
When trigger Set user_flag
When user_flag Return 1 Force
When Others Return 0 Force
""")
        self.assertEqual(p.evaluate("f$main",{"trigger":1}),ReturnValue(1))
        self.assertEqual(p.evaluate("f$main",{"trigger":0}),ReturnValue(0))

    def test_plain_ternary_function(self):
        p=OpenPPLProgram.from_text("""
##f$x##
a ? { b ? 10 : 20 } : 30
""")
        self.assertEqual(p.evaluate("f$x",{"a":1,"b":1}),ReturnValue(10))
        self.assertEqual(p.evaluate("f$x",{"a":1,"b":0}),ReturnValue(20))
        self.assertEqual(p.evaluate("f$x",{"a":0,"b":1}),ReturnValue(30))


    def test_multiline_when_condition_is_joined(self):
        p=OpenPPLProgram.from_text("""
##f$x##
When a
&& b
&& c Return 11 Force
When Others Return 4 Force
""")
        self.assertEqual(p.evaluate("f$x",{"a":1,"b":1,"c":1}),ReturnValue(11))
        self.assertEqual(p.evaluate("f$x",{"a":1,"b":1,"c":0}),ReturnValue(4))

    def test_direct_bet_action_preserves_name(self):
        p=OpenPPLProgram.from_text("""
##f$x##
When a BetHalfPot Force
When b BetMax Force
When Others BetThirdPot Force
""")
        self.assertEqual(p.evaluate("f$x",{"a":1,"b":0}),DirectAction("BetHalfPot"))
        self.assertEqual(p.evaluate("f$x",{"a":0,"b":1}),DirectAction("BetMax"))
        self.assertEqual(p.evaluate("f$x",{"a":0,"b":0}),DirectAction("BetThirdPot"))


    def test_hand_list_membership(self):
        p=OpenPPLProgram.from_text("""
##list_open##
AA AKs AQo
##f$x##
When list_open Return 1 Force
When Others Return 0 Force
""")
        self.assertEqual(p.evaluate("f$x",{},hand_class="AA"),ReturnValue(1))
        self.assertEqual(p.evaluate("f$x",{},hand_class="AKs"),ReturnValue(1))
        self.assertEqual(p.evaluate("f$x",{},hand_class="AQo"),ReturnValue(1))
        self.assertEqual(p.evaluate("f$x",{},hand_class="AKo"),ReturnValue(0))
        self.assertEqual(p.evaluate("f$x",{},hand_class="72o"),ReturnValue(0))

    def test_hand_list_requires_hand_class(self):
        p=OpenPPLProgram.from_text("""
##list_open##
AA
##f$x##
When list_open Return 1 Force
When Others Return 0 Force
""")
        with self.assertRaises(OpenPPLProgramError):
            p.evaluate("f$x",{})


    def test_user_variables_persist_for_hand_session_and_clear_on_reset(self):
        p=OpenPPLProgram.from_text("""
##f$mark##
When trigger Set user_seen
When Others Return 0 Force
##f$read##
When user_seen Return 1 Force
When Others Return 0 Force
""")
        session=OpenPPLSession(p)
        self.assertEqual(session.evaluate("f$read",{"trigger":0}),ReturnValue(0))
        self.assertEqual(session.evaluate("f$mark",{"trigger":1}),ReturnValue(0))
        self.assertEqual(session.evaluate("f$read",{"trigger":0}),ReturnValue(1))
        session.reset_hand()
        self.assertEqual(session.evaluate("f$read",{"trigger":0}),ReturnValue(0))


    def test_openholdem_memory_symbols_persist_across_hands_until_connection_reset(self):
        p=OpenPPLProgram.from_text("""
##f$store##
When go Set me_st_ES_stackvalue
When Others Return 0 Force
##f$recall##
me_re_ES
""")
        session=OpenPPLSession(p)
        self.assertEqual(session.evaluate("f$store",{"go":1,"stackvalue":7.5}),ReturnValue(0))
        self.assertEqual(session.evaluate("f$recall",{}),ReturnValue(7.5))
        session.reset_hand()
        self.assertEqual(session.evaluate("f$recall",{}),ReturnValue(7.5))
        session.reset_connection()
        self.assertEqual(session.evaluate("f$recall",{}),ReturnValue(0))

    def test_memory_store_increment_add_sub_and_expression_side_effect(self):
        p=OpenPPLProgram.from_text("""
##f$ops##
me_st_X_2 + me_inc_X + me_add_X_3 + me_sub_X_1 + me_re_X
""")
        session=OpenPPLSession(p)
        self.assertEqual(session.evaluate("f$ops",{}),ReturnValue(5))
        self.assertEqual(session.memory_symbols["x"],5)


    def test_parameterized_raise_actions(self):
        p=OpenPPLProgram.from_text("""
##f$x##
When a RaiseTo (AmountToCall + 4) Force
When b RaiseBy 50% Force
When Others Fold Force
""")
        self.assertEqual(
            p.evaluate("f$x",{"a":1,"b":0,"AmountToCall":2}),
            DirectAction("RaiseTo",6,"bb_expression"),
        )
        self.assertEqual(
            p.evaluate("f$x",{"a":0,"b":1,"AmountToCall":2}),
            DirectAction("RaiseBy",0.5,"pot_fraction"),
        )

    def test_when_with_tabs(self):
        p=OpenPPLProgram.from_text("""
##f$x##
When\tmyturnbits\t=\t0b00100\tReturn\ttrue\tForce
When Others Return false Force
""")
        self.assertEqual(p.evaluate("f$x",{"myturnbits":4}),ReturnValue(1))
        self.assertEqual(p.evaluate("f$x",{"myturnbits":2}),ReturnValue(0))

    def test_eof_without_action_fails_closed(self):
        p=OpenPPLProgram.from_text("""
##f$x##
When a Return 1 Force
""")
        with self.assertRaises(OpenPPLProgramError):
            p.evaluate("f$x",{"a":0})


    def test_reserved_initialization_callback_may_fall_off_end_after_set(self):
        p=OpenPPLProgram.from_text("""
##f$ini_function_on_my_turn##
When trigger Set me_st_X_7
""")
        session=OpenPPLSession(p)
        self.assertEqual(
            session.run_initialization(
                "f$ini_function_on_my_turn",
                {"trigger":1},
            ),
            0,
        )
        self.assertEqual(session.memory_symbols["x"], 7)

    def test_normal_strategy_callback_still_fails_closed_at_eof(self):
        p=OpenPPLProgram.from_text("""
##f$main##
When trigger Set user_seen
""")
        session=OpenPPLSession(p)
        with self.assertRaises(OpenPPLProgramError):
            session.evaluate("f$main", {"trigger":1})


if __name__=="__main__":
    unittest.main()

