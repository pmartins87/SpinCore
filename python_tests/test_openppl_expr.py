from __future__ import annotations

import unittest

from spincore.openppl_expr import (
    OpenPPLExpressionError,
    UnknownOpenPPLSymbol,
    evaluate_expression,
    oh_is_equal,
)


class OpenPPLExpressionTests(unittest.TestCase):
    def test_openholdem_priority_table(self):
        self.assertEqual(evaluate_expression("2*3 + 4*5", {}), 26.0)
        self.assertEqual(evaluate_expression("1 + 2 << 2", {}), 12.0)
        self.assertEqual(evaluate_expression("1 | 2 & 4", {}), 1.0)
        self.assertEqual(evaluate_expression("1 || 0 && missing", {}), 1.0)

    def test_openppl_percentage_is_binary_multiplicative_operator(self):
        self.assertAlmostEqual(evaluate_expression("25% 80", {}), 20.0)
        self.assertAlmostEqual(
            evaluate_expression(
                "AmountToCall + 75% PotSize",
                {"AmountToCall": 2.0, "PotSize": 12.0},
            ),
            11.0,
        )

    def test_comparisons_match_openholdem_epsilon(self):
        self.assertEqual(evaluate_expression("1 = 1.0000005", {}), 1.0)
        self.assertEqual(evaluate_expression("1 < 1.0000005", {}), 0.0)
        self.assertEqual(evaluate_expression("1 <= 1.0000005", {}), 1.0)
        self.assertEqual(evaluate_expression("1 != 1.0000005", {}), 1.0)
        self.assertTrue(oh_is_equal(1.0, 1.0 + 5e-7))

    def test_word_and_symbol_operators_are_equivalent(self):
        symbols = {"a": 1.0, "b": 0.0}
        self.assertEqual(evaluate_expression("a AND NOT b", symbols), 1.0)
        self.assertEqual(evaluate_expression("a && !b", symbols), 1.0)
        self.assertEqual(evaluate_expression("7 MOD 4", {}), 3.0)
        self.assertEqual(evaluate_expression("BITCOUNT 0xf", {}), 4.0)
        self.assertEqual(evaluate_expression("BITNOT 0", {}), float(0xFFFFFFFF))

    def test_r8_style_expression(self):
        symbols = {
            "AmountToCall": 2.0,
            "bblind": 20.0,
            "potcommon": 80.0,
            "f$CF7_LowSPR": 1.0,
            "f$calc_call_by_maths": 0.0,
            "IsRiver": 0.0,
            "f$CF9_PremiumDraw": 1.0,
        }
        self.assertEqual(
            evaluate_expression(
                "potcommon > 0 && (AmountToCall*bblind) <= (0.75*potcommon)",
                symbols,
            ),
            1.0,
        )
        self.assertEqual(
            evaluate_expression(
                "!IsRiver && f$CF9_PremiumDraw && "
                "(f$CF7_LowSPR || f$calc_call_by_maths)",
                symbols,
            ),
            1.0,
        )

    def test_ternary_is_short_circuit(self):
        self.assertEqual(evaluate_expression("true ? 7 : missing", {}), 7.0)
        self.assertEqual(evaluate_expression("false ? missing : 9", {}), 9.0)

    def test_case_insensitive_symbols(self):
        self.assertEqual(
            evaluate_expression(
                "amounttocall + BBLInd",
                {"AmountToCall": 2.0, "bblind": 3.0},
            ),
            5.0,
        )

    def test_unknown_is_never_silent_zero(self):
        with self.assertRaises(UnknownOpenPPLSymbol):
            evaluate_expression("Known + Missing", {"Known": 1.0})

    def test_division_by_zero_fails_closed(self):
        with self.assertRaises(OpenPPLExpressionError):
            evaluate_expression("1 / 0", {})


    def test_binary_integer_literals(self):
        self.assertEqual(evaluate_expression("0b11110", {}), 30.0)
        self.assertEqual(evaluate_expression("myturnbits = 0b00100", {"myturnbits": 4}), 1.0)


if __name__ == "__main__":
    unittest.main()
