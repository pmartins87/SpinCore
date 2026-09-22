from __future__ import annotations

"""Portable OpenPPL/OpenHoldem expression evaluator for DeepCrusher DC0.

The operator table and floating-point comparisons intentionally mirror the
OpenHoldem source bundled with the project. This is a strict evaluator:
unknown symbols raise instead of silently becoming zero.

This module evaluates expressions only. Ordered WHEN/SET/RETURN sections and
poker-native symbol semantics are separate layers.
"""

from dataclasses import dataclass
import math
from typing import Callable, Iterable

EPSILON = 1.0e-6


class OpenPPLExpressionError(ValueError):
    pass


class UnknownOpenPPLSymbol(OpenPPLExpressionError):
    pass


def oh_is_equal(a: float, b: float) -> bool:
    return (a > b - EPSILON) and (a < b + EPSILON)


def oh_is_smaller(a: float, b: float) -> bool:
    return a <= b - EPSILON


def oh_is_greater(a: float, b: float) -> bool:
    return a >= b + EPSILON


def oh_is_smaller_or_equal(a: float, b: float) -> bool:
    return a < b + EPSILON


def oh_is_greater_or_equal(a: float, b: float) -> bool:
    return a > b - EPSILON


def oh_round(value: float) -> int:
    return int(float(value) + 0.5)


def oh_is_approximately_equal(a: float, b: float) -> bool:
    return oh_round(a) == oh_round(b)


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    pos: int


_MULTI = ("&&", "||", "^^", "<<", ">>", "<=", ">=", "!=", "==", "~~", "**")
_SINGLE = set("+-*/%<>=!~^&|\x60?:()[]{}")
_WORD_OPS = {
    "AND": "&&",
    "OR": "||",
    "XOR": "^^",
    "NOT": "!",
    "BITAND": "&",
    "BITOR": "|",
    "BITXOR": "^",
    "BITNOT": "~",
    "BITCOUNT": "\x60",
    "MOD": "MOD",
    "LN": "LN",
}
_CONSTANTS = {
    "TRUE": 1.0,
    "FALSE": 0.0,
    "ACE": 14.0,
    "KING": 13.0,
    "QUEEN": 12.0,
    "JACK": 11.0,
    "TEN": 10.0,
    "NINE": 9.0,
    "EIGHT": 8.0,
    "SEVEN": 7.0,
    "SIX": 6.0,
    "FIVE": 5.0,
    "FOUR": 4.0,
    "THREE": 3.0,
    "TWO": 2.0,
}


def tokenize(expression: str) -> tuple[Token, ...]:
    out: list[Token] = []
    i = 0
    n = len(expression)
    while i < n:
        c = expression[i]
        if c.isspace():
            i += 1
            continue
        if expression.startswith("//", i):
            break

        matched = False
        for op in _MULTI:
            if expression.startswith(op, i):
                out.append(Token("OP", op, i))
                i += len(op)
                matched = True
                break
        if matched:
            continue

        if c in _SINGLE:
            out.append(Token("OP", c, i))
            i += 1
            continue

        if c.isdigit() or (c == "." and i + 1 < n and expression[i + 1].isdigit()):
            start = i
            if expression.startswith(("0x", "0X"), i):
                i += 2
                hex_start = i
                while i < n and expression[i] in "0123456789abcdefABCDEF":
                    i += 1
                if i == hex_start:
                    raise OpenPPLExpressionError(f"bad hex literal at {start}")
            elif expression.startswith(("0b", "0B"), i):
                i += 2
                binary_start = i
                while i < n and expression[i] in "01":
                    i += 1
                if i == binary_start:
                    raise OpenPPLExpressionError(f"bad binary literal at {start}")
            else:
                saw_dot = False
                while i < n and (
                    expression[i].isdigit()
                    or (expression[i] == "." and not saw_dot)
                ):
                    saw_dot = saw_dot or expression[i] == "."
                    i += 1
                if i < n and expression[i] in "eE":
                    j = i + 1
                    if j < n and expression[j] in "+-":
                        j += 1
                    digit = j
                    while j < n and expression[j].isdigit():
                        j += 1
                    if j == digit:
                        raise OpenPPLExpressionError(f"bad exponent at {i}")
                    i = j
            out.append(Token("NUMBER", expression[start:i], start))
            continue

        if c.isalpha() or c in "_$":
            start = i
            i += 1
            while i < n and (
                expression[i].isalnum() or expression[i] in "_$"
            ):
                i += 1
            word = expression[start:i]
            upper = word.upper()
            if upper in _WORD_OPS:
                out.append(Token("OP", _WORD_OPS[upper], start))
            else:
                out.append(Token("IDENT", word, start))
            continue

        raise OpenPPLExpressionError(
            f"unexpected character {c!r} at position {i} in {expression!r}"
        )

    out.append(Token("EOF", "", n))
    return tuple(out)


# Priorities mirror OpenHoldem TokenizerConstants.cpp.
_PRECEDENCE = {
    "?": 1,
    "||": 2,
    "^^": 3,
    "&&": 4,
    "|": 5,
    "^": 6,
    "&": 7,
    "=": 8,
    "==": 8,
    "!=": 8,
    "~~": 8,
    "<": 9,
    "<=": 9,
    ">": 9,
    ">=": 9,
    "<<": 10,
    ">>": 10,
    "+": 11,
    "-": 11,
    "*": 12,
    "/": 12,
    "MOD": 12,
    "%": 12,
    "**": 14,
}
_UNARY = {"+", "-", "!", "~", "\x60", "LN"}
_UNARY_PRECEDENCE = 13


class Expr:
    def eval(self, resolve: Callable[[str], float]) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class Number(Expr):
    value: float

    def eval(self, resolve: Callable[[str], float]) -> float:
        return float(self.value)


@dataclass(frozen=True)
class Identifier(Expr):
    name: str

    def eval(self, resolve: Callable[[str], float]) -> float:
        upper = self.name.upper()
        if upper in _CONSTANTS:
            return float(_CONSTANTS[upper])
        try:
            return float(resolve(self.name))
        except UnknownOpenPPLSymbol:
            raise
        except (KeyError, LookupError) as exc:
            raise UnknownOpenPPLSymbol(self.name) from exc


@dataclass(frozen=True)
class Unary(Expr):
    op: str
    value: Expr

    def eval(self, resolve: Callable[[str], float]) -> float:
        x = float(self.value.eval(resolve))
        if self.op == "+":
            return x
        if self.op == "-":
            return -x
        if self.op == "!":
            return 0.0 if bool(x) else 1.0
        if self.op == "~":
            return float((~(int(x) & 0xFFFFFFFF)) & 0xFFFFFFFF)
        if self.op == "\x60":
            return float((int(x) & 0xFFFFFFFF).bit_count())
        if self.op == "LN":
            return float(math.log(x))
        raise AssertionError(self.op)


@dataclass(frozen=True)
class Binary(Expr):
    op: str
    left: Expr
    right: Expr

    def eval(self, resolve: Callable[[str], float]) -> float:
        a = float(self.left.eval(resolve))

        if self.op == "&&":
            if not bool(a):
                return 0.0
            return 1.0 if bool(self.right.eval(resolve)) else 0.0
        if self.op == "||":
            if bool(a):
                return 1.0
            return 1.0 if bool(self.right.eval(resolve)) else 0.0

        b = float(self.right.eval(resolve))
        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            return a * b
        if self.op == "/":
            if b == 0.0:
                raise OpenPPLExpressionError("division by zero")
            return a / b
        if self.op == "MOD":
            ib = int(b) & 0xFFFFFFFF
            if ib == 0:
                raise OpenPPLExpressionError("modulo by zero")
            return float((int(a) & 0xFFFFFFFF) % ib)
        if self.op == "**":
            return float(math.pow(a, b))
        if self.op == "%":
            return a * b * 0.01
        if self.op in ("=", "=="):
            return 1.0 if oh_is_equal(a, b) else 0.0
        if self.op == "~~":
            return 1.0 if oh_is_approximately_equal(a, b) else 0.0
        if self.op == "!=":
            return 1.0 if a != b else 0.0
        if self.op == "<":
            return 1.0 if oh_is_smaller(a, b) else 0.0
        if self.op == "<=":
            return 1.0 if oh_is_smaller_or_equal(a, b) else 0.0
        if self.op == ">":
            return 1.0 if oh_is_greater(a, b) else 0.0
        if self.op == ">=":
            return 1.0 if oh_is_greater_or_equal(a, b) else 0.0
        if self.op == "^^":
            return 1.0 if bool(a) != bool(b) else 0.0

        ia = int(a) & 0xFFFFFFFF
        ib = int(b) & 0xFFFFFFFF
        if self.op == "&":
            return float(ia & ib)
        if self.op == "|":
            return float(ia | ib)
        if self.op == "^":
            return float(ia ^ ib)
        if self.op == "<<":
            return float((ia << ib) & 0xFFFFFFFF)
        if self.op == ">>":
            return float(ia >> ib)
        raise AssertionError(self.op)


@dataclass(frozen=True)
class Conditional(Expr):
    condition: Expr
    yes: Expr
    no: Expr

    def eval(self, resolve: Callable[[str], float]) -> float:
        if bool(self.condition.eval(resolve)):
            return float(self.yes.eval(resolve))
        return float(self.no.eval(resolve))


class Parser:
    def __init__(self, tokens: Iterable[Token]):
        self.tokens = tuple(tokens)
        self.index = 0

    def peek(self) -> Token:
        return self.tokens[self.index]

    def take(self) -> Token:
        token = self.peek()
        self.index += 1
        return token

    def parse(self) -> Expr:
        value = self.parse_expression(0)
        if self.peek().kind != "EOF":
            token = self.peek()
            raise OpenPPLExpressionError(
                f"unexpected token {token.text!r} at {token.pos}"
            )
        return value

    def parse_expression(self, min_precedence: int) -> Expr:
        left = self.parse_prefix()

        while True:
            token = self.peek()
            if token.kind != "OP" or token.text not in _PRECEDENCE:
                break
            precedence = _PRECEDENCE[token.text]
            if precedence < min_precedence:
                break
            op = self.take().text
            if op == "?":
                yes = self.parse_expression(0)
                colon = self.take()
                if colon.kind != "OP" or colon.text != ":":
                    raise OpenPPLExpressionError(
                        f"expected ':' in conditional at {colon.pos}"
                    )
                no = self.parse_expression(precedence)
                left = Conditional(left, yes, no)
                continue

            right = self.parse_expression(precedence + 1)
            left = Binary(op, left, right)

        return left

    def parse_prefix(self) -> Expr:
        token = self.take()
        if token.kind == "NUMBER":
            if token.text.lower().startswith("0x"):
                return Number(float(int(token.text, 16)))
            if token.text.lower().startswith("0b"):
                return Number(float(int(token.text, 2)))
            return Number(float(token.text))
        if token.kind == "IDENT":
            return Identifier(token.text)
        if token.kind == "OP" and token.text in _UNARY:
            return Unary(token.text, self.parse_expression(_UNARY_PRECEDENCE))
        if token.kind == "OP" and token.text in ("(", "[", "{"):
            close = {"(": ")", "[": "]", "{": "}"}[token.text]
            value = self.parse_expression(0)
            closing = self.take()
            if closing.kind != "OP" or closing.text != close:
                raise OpenPPLExpressionError(
                    f"expected {close!r}, got {closing.text!r} at {closing.pos}"
                )
            return value
        raise OpenPPLExpressionError(
            f"expected expression at {token.pos}, got {token.text!r}"
        )


def compile_expression(expression: str) -> Expr:
    return Parser(tokenize(expression)).parse()


def evaluate_expression(
    expression: str,
    symbols: dict[str, float] | Callable[[str], float],
) -> float:
    if callable(symbols):
        resolver = symbols
    else:
        exact = dict(symbols)
        folded = {key.lower(): value for key, value in exact.items()}

        def resolver(name: str) -> float:
            if name in exact:
                return float(exact[name])
            key = name.lower()
            if key in folded:
                return float(folded[key])
            raise UnknownOpenPPLSymbol(name)

    return float(compile_expression(expression).eval(resolver))
