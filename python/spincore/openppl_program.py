from __future__ import annotations

"""Structural OpenPPL function evaluator for the DeepCrusher R8 DC0 oracle.

It reproduces the parse-tree behavior of OpenHoldem's ordered WHEN sequences:
an open-ended WHEN points its then-branch at the next WHEN and is back-patched
so its else-branch points at the next open-ended WHEN. SET user_* is a special
action: it sets the user variable and then evaluation continues through the
current WHEN node's else branch.

This module deliberately does not claim native poker-symbol coverage. Unknown
external symbols still fail closed through openppl_expr.UnknownOpenPPLSymbol.
"""

from dataclasses import dataclass
import re
from typing import Callable, Mapping

from spincore.openppl_expr import (
    Expr,
    UnknownOpenPPLSymbol,
    compile_expression,
)

_SECTION = re.compile(r"^##([^#\r\n]+)##\s*$", re.M)
_WHEN = re.compile(r"^\s*When\s+(.*?)\s*$", re.I)
_RETURN = re.compile(r"^(.*?)\s+Return\s+(.+?)\s+Force\s*$", re.I)
_SET = re.compile(r"^(.*?)\s+Set\s+([A-Za-z_][A-Za-z0-9_$]*)\s*$", re.I)
_DIRECT = re.compile(
    r"^(.*?)\s+(Call|Fold|Check|BetMax|BetPot|BetHalfPot|BetThirdPot|"
    r"BetTwoThirdPot|BetThreeFourthPot|BetMin|RaiseMin)\s+Force\s*$",
    re.I,
)
_OTHERS = re.compile(r"^Others$", re.I)
_HAND_CLASS = re.compile(r"^(?:[AKQJT98765432]{2}|[AKQJT98765432]{2}[so])$")


class OpenPPLProgramError(ValueError):
    pass


@dataclass(frozen=True)
class ReturnValue:
    value: float


@dataclass(frozen=True)
class DirectAction:
    name: str


@dataclass(frozen=True)
class SetUserVariable:
    name: str


Action = ReturnValue | DirectAction | SetUserVariable


@dataclass
class WhenNode:
    condition: Expr
    action_kind: str  # "return", "direct", "set", "open"
    action_expr: Expr | None = None
    action_name: str | None = None
    then_index: int | None = None
    else_index: int | None = None
    source_line: int = 0


@dataclass(frozen=True)
class CompiledFunction:
    name: str
    expression: Expr | None
    whens: tuple[WhenNode, ...]

    @property
    def is_when_function(self) -> bool:
        return bool(self.whens)


def split_sections(text: str) -> dict[str, str]:
    matches = list(_SECTION.finditer(text))
    out: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        out[name] = text[start:end]
    return out


def _strip_comments(body: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_block = False
    for line_number, raw in enumerate(body.splitlines(), 1):
        line = raw
        # Minimal C-style block-comment handling, because formulas use both //
        # and /* ... */ in historical sources.
        clean = []
        i = 0
        while i < len(line):
            if in_block:
                end = line.find("*/", i)
                if end < 0:
                    i = len(line)
                    continue
                in_block = False
                i = end + 2
                continue
            if line.startswith("/*", i):
                in_block = True
                i += 2
                continue
            if line.startswith("//", i):
                break
            clean.append(line[i])
            i += 1
        value = "".join(clean).strip()
        if value:
            rows.append((line_number, value))
    return rows


def _logical_rows(rows: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Join OpenPPL physical-line continuations into logical statements.

    R8 v22 contains a small number of WHEN conditions split over multiple
    physical lines (usually a leading WHEN followed by lines beginning with
    &&).  OpenHoldem parses those as one expression.  Preserve the source line
    of the first physical row for diagnostics.
    """
    if not rows:
        return []

    has_when = any(text.lower().startswith("when ") for _, text in rows)
    if not has_when:
        # Plain expression functions may also span lines.
        return [(rows[0][0], " ".join(text for _, text in rows))]

    out: list[tuple[int, str]] = []
    for line_number, text in rows:
        if text.lower().startswith("when "):
            out.append((line_number, text))
            continue
        if not out:
            raise OpenPPLProgramError(
                f"line {line_number}: continuation before first WHEN: {text!r}"
            )
        prev_line, prev_text = out[-1]
        out[-1] = (prev_line, prev_text + " " + text)
    return out


def parse_hand_list(body: str) -> frozenset[str]:
    """Parse one OpenPPL list section into canonical 169 hand classes."""
    hands: set[str] = set()
    for _, row in _strip_comments(body):
        for token in row.split():
            if not _HAND_CLASS.fullmatch(token):
                raise OpenPPLProgramError(f"unsupported hand-list token: {token!r}")
            hands.add(token)
    return frozenset(hands)


def _compile_condition(text: str) -> Expr:
    if _OTHERS.fullmatch(text.strip()):
        return compile_expression("true")
    return compile_expression(text.strip())


def _parse_when(line_number: int, text: str) -> WhenNode:
    match = _WHEN.match(text)
    if not match:
        raise OpenPPLProgramError(f"line {line_number}: expected WHEN")
    tail = match.group(1).strip()

    m = _RETURN.match(tail)
    if m:
        return WhenNode(
            condition=_compile_condition(m.group(1)),
            action_kind="return",
            action_expr=compile_expression(m.group(2).strip()),
            source_line=line_number,
        )

    m = _SET.match(tail)
    if m:
        name = m.group(2)
        if not name.lower().startswith("user"):
            raise OpenPPLProgramError(
                f"line {line_number}: only OpenPPL user-variable SET is handled "
                f"by this layer, got {name!r}"
            )
        return WhenNode(
            condition=_compile_condition(m.group(1)),
            action_kind="set",
            action_name=name,
            source_line=line_number,
        )

    m = _DIRECT.match(tail)
    if m:
        return WhenNode(
            condition=_compile_condition(m.group(1)),
            action_kind="direct",
            action_name=m.group(2),
            source_line=line_number,
        )

    # An open-ended WHEN has condition only; its then branch is the next WHEN.
    return WhenNode(
        condition=_compile_condition(tail),
        action_kind="open",
        source_line=line_number,
    )


def compile_function(name: str, body: str) -> CompiledFunction:
    rows = _logical_rows(_strip_comments(body))
    when_rows = [(line_no, text) for line_no, text in rows if text.lower().startswith("when ")]

    if not when_rows:
        expression_text = " ".join(text for _, text in rows).strip()
        if not expression_text:
            expression_text = "0"
        return CompiledFunction(
            name=str(name),
            expression=compile_expression(expression_text),
            whens=(),
        )

    if len(when_rows) != len(rows):
        offenders = [
            f"{line_no}:{text}"
            for line_no, text in rows
            if not text.lower().startswith("when ")
        ]
        raise OpenPPLProgramError(
            f"{name}: mixed WHEN/non-WHEN executable lines are unsupported: "
            + "; ".join(offenders[:5])
        )

    nodes = [_parse_when(line_no, text) for line_no, text in rows]
    n = len(nodes)

    # Initial parser links: open-ended -> second/then = next WHEN;
    # action WHEN -> third/else = next WHEN.
    for index, node in enumerate(nodes):
        nxt = index + 1 if index + 1 < n else None
        if node.action_kind == "open":
            node.then_index = nxt
        else:
            node.else_index = nxt

    # OpenHoldem BackPatchOpenEndedWhenConditionSequence:
    # each open-ended WHEN becomes the else target of the previous open-ended
    # WHEN encountered while following the initial chain.
    last_open: int | None = None
    current: int | None = 0 if nodes else None
    visited: set[int] = set()
    while current is not None:
        if current in visited:
            raise OpenPPLProgramError(f"{name}: cyclic WHEN chain")
        visited.add(current)
        node = nodes[current]
        if node.action_kind == "open":
            if last_open is not None:
                nodes[last_open].else_index = current
            last_open = current
            current = node.then_index
        else:
            current = node.else_index

    # End-of-function is represented by None. OpenHoldem also patches the final
    # still-open node's else to EOF; None already has that meaning here.
    return CompiledFunction(name=str(name), expression=None, whens=tuple(nodes))


class ProgramContext:
    """One OpenPPL decision-heartbeat context.

    user_* variables default false and are local to the heartbeat. External
    native symbols and persistent me_* memory are supplied by the caller.
    """

    def __init__(
        self,
        program: "OpenPPLProgram",
        external: Mapping[str, float] | Callable[[str], float],
        *,
        hand_class: str | None = None,
    ):
        self.program = program
        self.external = external
        self.hand_class = hand_class
        self.user_variables: set[str] = set()
        self.cache: dict[str, float] = {}
        self.in_progress: set[str] = set()

    def _external_value(self, name: str) -> float:
        if callable(self.external):
            return float(self.external(name))
        if name in self.external:
            return float(self.external[name])
        folded = {key.lower(): value for key, value in self.external.items()}
        if name.lower() in folded:
            return float(folded[name.lower()])
        raise UnknownOpenPPLSymbol(name)

    def resolve(self, name: str) -> float:
        low = name.lower()
        if low == "others":
            return 1.0
        if low.startswith("user") and not low.startswith("userchair"):
            return 1.0 if low in self.user_variables else 0.0
        if self.program.has_hand_list(name):
            if self.hand_class is None:
                raise OpenPPLProgramError(
                    f"hand class required to evaluate OpenPPL list {name!r}"
                )
            return 1.0 if self.hand_class in self.program.hand_list(name) else 0.0
        if low.startswith("f$") and self.program.has_function(name):
            return float(self.evaluate_function(name))
        return self._external_value(name)

    def set_user(self, name: str) -> None:
        self.user_variables.add(name.lower())

    def evaluate_function(self, name: str) -> float:
        canonical = self.program.canonical_name(name)
        # OpenHoldem formula values are cached within a heartbeat. Preserve that
        # for pure helper f$ functions. Functions containing SET are deliberately
        # not cached by this foundation layer until runtime parity fixtures prove
        # the exact cache/side-effect order for the primary callbacks.
        fn = self.program.functions[canonical]
        cacheable = not any(node.action_kind == "set" for node in fn.whens)
        if cacheable and canonical in self.cache:
            return self.cache[canonical]
        if canonical in self.in_progress:
            raise OpenPPLProgramError(f"recursive function cycle at {canonical}")
        self.in_progress.add(canonical)
        try:
            result = self.program._evaluate_compiled(fn, self)
        finally:
            self.in_progress.remove(canonical)
        if isinstance(result, DirectAction):
            raise OpenPPLProgramError(
                f"{canonical} returned direct action {result.name}; numerical "
                "evaluation requested"
            )
        value = float(result.value)
        if cacheable:
            self.cache[canonical] = value
        return value


class OpenPPLProgram:
    def __init__(
        self,
        functions: dict[str, CompiledFunction],
        hand_lists: Mapping[str, frozenset[str]] | None = None,
    ):
        self.functions = dict(functions)
        self._folded = {name.lower(): name for name in self.functions}
        self.hand_lists = dict(hand_lists or {})
        self._lists_folded = {name.lower(): name for name in self.hand_lists}

    @classmethod
    def from_text(cls, text: str) -> "OpenPPLProgram":
        sections = split_sections(text)
        functions = {
            name: compile_function(name, body)
            for name, body in sections.items()
            if name.lower().startswith("f$")
        }
        hand_lists = {
            name: parse_hand_list(body)
            for name, body in sections.items()
            if name.lower().startswith("list")
        }
        return cls(functions, hand_lists)

    def has_function(self, name: str) -> bool:
        return name in self.functions or name.lower() in self._folded

    def has_hand_list(self, name: str) -> bool:
        return name in self.hand_lists or name.lower() in self._lists_folded

    def hand_list(self, name: str) -> frozenset[str]:
        if name in self.hand_lists:
            return self.hand_lists[name]
        try:
            return self.hand_lists[self._lists_folded[name.lower()]]
        except KeyError as exc:
            raise UnknownOpenPPLSymbol(name) from exc

    def canonical_name(self, name: str) -> str:
        if name in self.functions:
            return name
        try:
            return self._folded[name.lower()]
        except KeyError as exc:
            raise UnknownOpenPPLSymbol(name) from exc

    def _evaluate_compiled(
        self,
        fn: CompiledFunction,
        ctx: ProgramContext,
    ) -> ReturnValue | DirectAction:
        if not fn.is_when_function:
            assert fn.expression is not None
            return ReturnValue(float(fn.expression.eval(ctx.resolve)))

        index: int | None = 0
        while index is not None:
            node = fn.whens[index]
            condition = float(node.condition.eval(ctx.resolve))
            if bool(condition):
                if node.action_kind == "open":
                    index = node.then_index
                    continue
                if node.action_kind == "set":
                    assert node.action_name is not None
                    ctx.set_user(node.action_name)
                    # OpenHoldem's EvaluateTernaryExpression explicitly falls
                    # through to the third sibling after SET.
                    index = node.else_index
                    continue
                if node.action_kind == "return":
                    assert node.action_expr is not None
                    return ReturnValue(float(node.action_expr.eval(ctx.resolve)))
                if node.action_kind == "direct":
                    assert node.action_name is not None
                    return DirectAction(node.action_name)
                raise AssertionError(node.action_kind)
            index = node.else_index

        # CParseTreeTerminalNodeEndOfFunction yields the standard empty-formula
        # value. For DC0 we fail closed instead of guessing its context-specific
        # Fold/zero interpretation.
        raise OpenPPLProgramError(f"{fn.name}: reached end of function without action")

    def evaluate(
        self,
        name: str,
        external: Mapping[str, float] | Callable[[str], float],
        *,
        hand_class: str | None = None,
    ) -> ReturnValue | DirectAction:
        ctx = ProgramContext(self, external, hand_class=hand_class)
        return self._evaluate_compiled(self.functions[self.canonical_name(name)], ctx)
