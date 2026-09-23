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
    r"^(.*?)\s+(Call|Fold|Check|Allin|BetMax|BetPot|BetHalfPot|BetThirdPot|"
    r"BetTwoThirdPot|BetThreeFourthPot|BetMin|RaiseMin|RaiseMax)\s+Force\s*$",
    re.I,
)
_PARAM_DIRECT = re.compile(
    r"^(.*?)\s+(RaiseTo|RaiseBy)\s+(.+?)\s+Force\s*$",
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
    amount: float | None = None
    amount_kind: str | None = None


@dataclass(frozen=True)
class SetUserVariable:
    name: str


Action = ReturnValue | DirectAction | SetUserVariable


# Numeric encodings from the pinned OpenPPL library. Helper functions that
# return actions are legal numerical subexpressions in OpenHoldem (for example
# f$preflop -> RETURN f$BestBetsize, where f$BestBetsize ultimately returns
# BetHalfPot or RaiseBy 50%). The primary callback still exposes DirectAction
# structurally when the action is written directly in that callback.
_OPENPPL_FIXED_ACTION_VALUE = {
    "fold": -1000001.0,
    "betfourthpot": -1000003.0,
    "raisefourthpot": -1000003.0,
    "betthirdpot": -1000004.0,
    "raisethirdpot": -1000004.0,
    "bethalfpot": -1000005.0,
    "raisehalfpot": -1000005.0,
    "bettwothirdpot": -1000006.0,
    "raisetwothirdpot": -1000006.0,
    "betthreefourthpot": -1000007.0,
    "raisethreefourthpot": -1000007.0,
    "betpot": -1000008.0,
    "raisepot": -1000008.0,
    "betmax": -1000009.0,
    "raisemax": -1000009.0,
    "allin": -1000009.0,
    "call": -1000010.0,
    "bet": -1000012.0,
    "raise": -1000012.0,
    "betmin": -1000012.0,
    "raisemin": -1000012.0,
    "check": 0.0,
}


@dataclass
class WhenNode:
    condition: Expr
    action_kind: str  # "return", "direct", "set", "open"
    action_expr: Expr | None = None
    action_name: str | None = None
    action_amount_kind: str | None = None
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


def _is_when_row(text: str) -> bool:
    return bool(re.match(r"^when\b", text, flags=re.I))


def _logical_rows(rows: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Join OpenPPL physical-line continuations into logical statements.

    R8 v22 contains a small number of WHEN conditions split over multiple
    physical lines (usually a leading WHEN followed by lines beginning with
    &&).  OpenHoldem parses those as one expression.  Preserve the source line
    of the first physical row for diagnostics.
    """
    if not rows:
        return []

    has_when = any(_is_when_row(text) for _, text in rows)
    if not has_when:
        # Plain expression functions may also span lines.
        return [(rows[0][0], " ".join(text for _, text in rows))]

    out: list[tuple[int, str]] = []
    for line_number, text in rows:
        if _is_when_row(text):
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


def _compile_action_amount(text: str) -> tuple[Expr, str]:
    value = text.strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)%", value)
    if m:
        # OpenPPL RaiseBy N% is a percentage-of-pot action, not N/100 BB.
        return compile_expression(str(float(m.group(1)) / 100.0)), "pot_fraction"
    return compile_expression(value), "bb_expression"


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
        low = name.lower()
        if not (low.startswith("user") or low.startswith("me_")):
            raise OpenPPLProgramError(
                f"line {line_number}: unsupported OpenPPL SET target {name!r}"
            )
        return WhenNode(
            condition=_compile_condition(m.group(1)),
            action_kind="set",
            action_name=name,
            source_line=line_number,
        )

    m = _PARAM_DIRECT.match(tail)
    if m:
        action_expr, amount_kind = _compile_action_amount(m.group(3))
        return WhenNode(
            condition=_compile_condition(m.group(1)),
            action_kind="direct",
            action_expr=action_expr,
            action_name=m.group(2),
            action_amount_kind=amount_kind,
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
    when_rows = [(line_no, text) for line_no, text in rows if _is_when_row(text)]

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
            if not _is_when_row(text)
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

    user_* variables default false and, like OpenHoldem, may be shared across
    all decision heartbeats of one hand. External native symbols and persistent
    me_* memory are supplied by the caller.
    """

    def __init__(
        self,
        program: "OpenPPLProgram",
        external: Mapping[str, float] | Callable[[str], float],
        *,
        hand_class: str | None = None,
        user_variables: set[str] | None = None,
        memory_symbols: dict[str, float] | None = None,
    ):
        self.program = program
        self.external = external
        self.hand_class = hand_class
        self.user_variables = user_variables if user_variables is not None else set()
        self.memory_symbols = memory_symbols if memory_symbols is not None else {}
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
        if low.startswith("me_"):
            return self.evaluate_memory_symbol(name)
        if self.program.has_hand_list(name):
            if self.hand_class is None:
                raise OpenPPLProgramError(
                    f"hand class required to evaluate OpenPPL list {name!r}"
                )
            return 1.0 if self.hand_class in self.program.hand_list(name) else 0.0
        if low.startswith("f$") and self.program.has_function(name):
            return float(self.evaluate_function(name))
        if low.startswith("vs$multiplex$"):
            return float(self.evaluate_versus_multiplex(name))

        # The offline benchmark may provide an exact transcript-derived value
        # for standard OpenPPL symbols whose stock library implementation
        # depends on live heartbeat/autoplayer memory. Prefer such an explicit
        # provider; if it does not recognize the symbol, fall back to the
        # pinned OpenPPL library.
        try:
            return self._external_value(name)
        except (UnknownOpenPPLSymbol, KeyError, LookupError):
            pass

        if self.program.has_library_function(name):
            return float(self.evaluate_library_function(name))
        raise UnknownOpenPPLSymbol(name)

    def set_user(self, name: str) -> None:
        self.user_variables.add(name.lower())

    def _memory_parts(self, command: str) -> tuple[str, str | None]:
        low = command.lower()
        prefixes = ("me_st_", "me_add_", "me_sub_")
        for prefix in prefixes:
            if low.startswith(prefix):
                tail = command[len(prefix):]
                if "_" not in tail:
                    raise OpenPPLProgramError(f"memory store command missing RHS: {command!r}")
                left, rhs = tail.split("_", 1)
                if not left or not rhs:
                    raise OpenPPLProgramError(f"invalid memory command: {command!r}")
                return left.lower(), rhs
        if low.startswith("me_inc_"):
            left = command[len("me_inc_"):]
            if not left:
                raise OpenPPLProgramError(f"invalid memory increment: {command!r}")
            return left.lower(), None
        if low.startswith("me_re_"):
            left = command[len("me_re_"):]
            if not left:
                raise OpenPPLProgramError(f"invalid memory recall: {command!r}")
            return left.lower(), None
        raise OpenPPLProgramError(f"unsupported memory symbol: {command!r}")

    def _memory_rhs_value(self, rhs: str) -> float:
        if rhs and rhs[0].isdigit():
            return float(rhs.replace("_", "."))
        return float(self.resolve(rhs))

    def evaluate_memory_symbol(self, command: str) -> float:
        low = command.lower()
        left, rhs = self._memory_parts(command)
        if low.startswith("me_re_"):
            return float(self.memory_symbols.get(left, 0.0))
        if low.startswith("me_inc_"):
            self.memory_symbols[left] = float(self.memory_symbols.get(left, 0.0)) + 1.0
            return 0.0
        assert rhs is not None
        value = self._memory_rhs_value(rhs)
        if low.startswith("me_st_"):
            self.memory_symbols[left] = value
        elif low.startswith("me_add_"):
            self.memory_symbols[left] = float(self.memory_symbols.get(left, 0.0)) + value
        elif low.startswith("me_sub_"):
            self.memory_symbols[left] = float(self.memory_symbols.get(left, 0.0)) - value
        else:
            raise OpenPPLProgramError(f"unsupported memory command: {command!r}")
        return 0.0

    def set_symbol(self, name: str) -> None:
        low = name.lower()
        if low.startswith("user") and not low.startswith("userchair"):
            self.set_user(name)
            return
        if low.startswith("me_"):
            self.evaluate_memory_symbol(name)
            return
        raise OpenPPLProgramError(f"unsupported SET target: {name!r}")

    def direct_action_numeric(self, action: DirectAction) -> float:
        """Evaluate an OpenPPL action node as its numeric decision value.

        This mirrors CParseTreeTerminalNodeBetsizeAction and the pinned action
        constants so action-returning sizing helpers can be nested inside
        RETURN expressions exactly as in real OpenHoldem.
        """
        low = str(action.name).lower()
        if low in _OPENPPL_FIXED_ACTION_VALUE:
            return float(_OPENPPL_FIXED_ACTION_VALUE[low])

        if low == "raiseto":
            if action.amount is None or action.amount_kind != "bb_expression":
                raise OpenPPLProgramError("RaiseTo requires BB expression")
            return float(action.amount)

        if low == "raiseby":
            if action.amount is None:
                raise OpenPPLProgramError("RaiseBy requires amount")
            bblind = float(self.resolve("bblind"))
            if bblind <= 0:
                raise OpenPPLProgramError("RaiseBy requires positive bblind")
            ncallbets = (
                float(self.resolve("currentbet")) / bblind
                + float(self.resolve("AmountToCall"))
            )
            if action.amount_kind == "bb_expression":
                return ncallbets + float(action.amount)
            if action.amount_kind == "pot_fraction":
                pot_after_call_bb = (
                    float(self.resolve("PotSize")) / bblind
                    + float(self.resolve("AmountToCall"))
                )
                return ncallbets + float(action.amount) * pot_after_call_bb
            raise OpenPPLProgramError(
                f"unsupported RaiseBy amount kind: {action.amount_kind!r}"
            )

        raise OpenPPLProgramError(
            f"no numeric OpenPPL encoding for direct action {action.name!r}"
        )

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
            value = self.direct_action_numeric(result)
        else:
            value = float(result.value)
        if cacheable:
            self.cache[canonical] = value
        return value

    def evaluate_versus_multiplex(self, name: str) -> float:
        low = str(name).lower()
        prefix = "vs$multiplex$"
        postfix = None
        for candidate in ("$prwin", "$prtie", "$prlos"):
            if low.endswith(candidate):
                postfix = candidate
                break
        if postfix is None or not low.startswith(prefix):
            raise UnknownOpenPPLSymbol(name)

        infix = str(name)[len(prefix): len(str(name)) - len(postfix)]
        if not infix:
            raise UnknownOpenPPLSymbol(name)

        resolver = getattr(self.external, "resolve_versus_multiplex", None)
        if resolver is None:
            raise UnknownOpenPPLSymbol(name)

        cache_key = "versus::" + low
        if cache_key in self.cache:
            return self.cache[cache_key]

        if infix.lower().startswith("f$") and self.program.has_function(infix):
            list_value = float(self.evaluate_function(infix))
        elif self.program.has_library_function(infix):
            list_value = float(self.evaluate_library_function(infix))
        else:
            list_value = float(self._external_value(infix))
        list_id = int(list_value + 0.5)
        value = float(resolver(infix, list_id, postfix))
        self.cache[cache_key] = value
        return value

    def evaluate_library_function(self, name: str) -> float:
        canonical = self.program.canonical_library_name(name)
        cache_key = "library::" + canonical.lower()
        fn = self.program.compiled_library_function(canonical)
        cacheable = not any(node.action_kind == "set" for node in fn.whens)
        if cacheable and cache_key in self.cache:
            return self.cache[cache_key]
        if cache_key in self.in_progress:
            raise OpenPPLProgramError(
                f"recursive library function cycle at {canonical}"
            )
        self.in_progress.add(cache_key)
        try:
            result = self.program._evaluate_compiled(fn, self)
        finally:
            self.in_progress.remove(cache_key)
        if isinstance(result, DirectAction):
            value = self.direct_action_numeric(result)
        else:
            value = float(result.value)
        if cacheable:
            self.cache[cache_key] = value
        return value


class OpenPPLSession:
    """Stateful OpenPPL hand session.

    OpenHoldem keeps user_* variables until the next hand reset.  A session
    therefore owns that set and reuses it across decision heartbeats; reset_hand
    is the exact lifecycle boundary for these variables.
    """

    def __init__(self, program: "OpenPPLProgram"):
        self.program = program
        self.user_variables: set[str] = set()
        self.memory_symbols: dict[str, float] = {}

    def reset_connection(self) -> None:
        self.user_variables.clear()
        self.memory_symbols.clear()

    def reset_hand(self) -> None:
        self.user_variables.clear()

    def evaluate(
        self,
        name: str,
        external: Mapping[str, float] | Callable[[str], float],
        *,
        hand_class: str | None = None,
    ) -> ReturnValue | DirectAction:
        ctx = ProgramContext(
            self.program,
            external,
            hand_class=hand_class,
            user_variables=self.user_variables,
            memory_symbols=self.memory_symbols,
        )
        return self.program._evaluate_compiled(
            self.program.functions[self.program.canonical_name(name)],
            ctx,
        )

    def run_initialization(
        self,
        name: str,
        external: Mapping[str, float] | Callable[[str], float],
        *,
        hand_class: str | None = None,
    ) -> float:
        """Execute one reserved OpenHoldem f$ini_function_* callback.

        These callbacks are allowed to contain only memory/user-variable side
        effects and to fall off the end of the function, which OpenHoldem
        evaluates as zero.
        """
        canonical = self.program.canonical_name(name)
        if not canonical.lower().startswith("f$ini_function_"):
            raise OpenPPLProgramError(
                f"run_initialization requires reserved ini callback, got {name!r}"
            )
        ctx = ProgramContext(
            self.program,
            external,
            hand_class=hand_class,
            user_variables=self.user_variables,
            memory_symbols=self.memory_symbols,
        )
        result = self.program._evaluate_compiled(
            self.program.functions[canonical],
            ctx,
            eof_zero=True,
        )
        if isinstance(result, DirectAction):
            raise OpenPPLProgramError(
                f"initialization callback {canonical} returned poker action {result.name}"
            )
        return float(result.value)


class OpenPPLProgram:
    def __init__(
        self,
        functions: dict[str, CompiledFunction],
        hand_lists: Mapping[str, frozenset[str]] | None = None,
        library_sections: Mapping[str, str] | None = None,
    ):
        self.functions = dict(functions)
        self._folded = {name.lower(): name for name in self.functions}
        self.hand_lists = dict(hand_lists or {})
        self._lists_folded = {name.lower(): name for name in self.hand_lists}
        self.library_sections = dict(library_sections or {})
        self._library_folded = {
            name.lower(): name for name in self.library_sections
        }
        self._compiled_library: dict[str, CompiledFunction] = {}

    @classmethod
    def from_text(cls, text: str) -> "OpenPPLProgram":
        return cls.from_texts(text)

    @classmethod
    def from_texts(
        cls,
        strategy_text: str,
        *,
        library_texts: tuple[str, ...] | list[str] = (),
    ) -> "OpenPPLProgram":
        sections = split_sections(strategy_text)
        functions: dict[str, CompiledFunction] = {}
        for name, body in sections.items():
            if not name.lower().startswith("f$"):
                continue
            try:
                functions[name] = compile_function(name, body)
            except Exception as exc:
                raise OpenPPLProgramError(f"{name}: {exc}") from exc
        hand_lists = {
            name: parse_hand_list(body)
            for name, body in sections.items()
            if name.lower().startswith("list")
        }

        library_sections: dict[str, str] = {}
        library_folded: dict[str, str] = {}
        for library_text in library_texts:
            for name, body in split_sections(library_text).items():
                low = name.lower()
                if low == "openppl_license_text":
                    continue
                if low in library_folded:
                    raise OpenPPLProgramError(
                        f"duplicate OpenPPL library section {name!r}"
                    )
                library_folded[low] = name
                library_sections[name] = body

        return cls(functions, hand_lists, library_sections)

    def has_function(self, name: str) -> bool:
        return name in self.functions or name.lower() in self._folded

    def has_hand_list(self, name: str) -> bool:
        return name in self.hand_lists or name.lower() in self._lists_folded

    def has_library_function(self, name: str) -> bool:
        return name in self.library_sections or name.lower() in self._library_folded

    def canonical_library_name(self, name: str) -> str:
        if name in self.library_sections:
            return name
        try:
            return self._library_folded[name.lower()]
        except KeyError as exc:
            raise UnknownOpenPPLSymbol(name) from exc

    def compiled_library_function(self, name: str) -> CompiledFunction:
        canonical = self.canonical_library_name(name)
        cached = self._compiled_library.get(canonical)
        if cached is not None:
            return cached
        try:
            compiled = compile_function(canonical, self.library_sections[canonical])
        except Exception as exc:
            raise OpenPPLProgramError(
                f"OpenPPL library section {canonical}: {exc}"
            ) from exc
        self._compiled_library[canonical] = compiled
        return compiled

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
        *,
        eof_zero: bool = True,
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
                    ctx.set_symbol(node.action_name)
                    # OpenHoldem's EvaluateTernaryExpression explicitly falls
                    # through to the third sibling after SET.
                    index = node.else_index
                    continue
                if node.action_kind == "return":
                    assert node.action_expr is not None
                    return ReturnValue(float(node.action_expr.eval(ctx.resolve)))
                if node.action_kind == "direct":
                    assert node.action_name is not None
                    amount = None
                    if node.action_expr is not None:
                        amount = float(node.action_expr.eval(ctx.resolve))
                    return DirectAction(node.action_name, amount, node.action_amount_kind)
                raise AssertionError(node.action_kind)
            index = node.else_index

        # OpenHoldem evaluates the explicit
        # empty_expression__false__zero__when_others_fold_force terminal at the
        # end of every function. Its numeric value is zero (false/fold); main
        # callbacks later translate zero through normal check/fold semantics.
        if eof_zero:
            return ReturnValue(0.0)
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
