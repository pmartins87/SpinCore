from __future__ import annotations

"""Canonical DC0 contract for the frozen DeepCrusher R8 v22 oracle.

This module is intentionally strict.  A source parser or a strategy imitation is
not enough for a canonical SpinCore-vs-DeepCrusher claim.  DC0 is admitted only
when the portable decision oracle has explicit semantics for every dependency it
can reach and passes source/runtime parity fixtures.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
from typing import Iterable

from spincore.deepcrusher_benchmark import (
    DEEPC_RUSHER_BASELINE_BRANCH,
    DEEPC_RUSHER_OPERATIONAL_SHA256,
    DEEPC_RUSHER_OPERATIONAL_SOURCE,
)

DC0_SCHEMA = "SPINCORE_DEEPCRUSHER_R8_DC0_CONTRACT_V1"
PRIMARY_CALLBACKS = ("f$preflop", "f$flop", "f$turn", "f$river")

# These are environment-dependent by construction and cannot be silently
# replaced by zero.  Any benchmark profile must state how they are provided.
_ENVIRONMENT_PATTERNS = (
    re.compile(r"^pt_", re.I),
    re.compile(r"^colourcode", re.I),
    re.compile(r"^network\$", re.I),
    re.compile(r"^chair\$", re.I),
    re.compile(r"^log\$", re.I),
)

# Keywords/operators/constants are not external symbol-provider dependencies.
_LANGUAGE_WORDS = {
    "when", "others", "return", "force", "set",
    "true", "false", "and", "or", "not",
    "ace", "king", "queen", "jack", "ten", "nine", "eight", "seven",
    "six", "five", "four", "three", "two",
    "preflop", "flop", "turn", "river",
    "first", "middle", "last", "none",
    "call", "check", "fold", "bet", "raises",
    "allin", "betmin", "betfourthpot", "betthirdpot", "bethalfpot",
    "bettwothirdpot", "betthreefourthpot", "betpot", "betmax",
    "raisemin", "raiseto", "raiseby", "raisefourthpot", "raisethirdpot",
    "raisehalfpot", "raisetwothirdpot", "raisethreefourthpot", "raisepot",
    "raisemax",
}

_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_$])([A-Za-z_][A-Za-z0-9_$]*)(?![A-Za-z0-9_$])")
_FREF = re.compile(r"(?<![A-Za-z0-9_$])(f\$[A-Za-z0-9_]+)(?![A-Za-z0-9_$])")
_SECTION = re.compile(r"^##([^#\r\n]+)##\s*$", re.M)


@dataclass(frozen=True)
class R8SourceClosure:
    sections: int
    function_sections: int
    list_sections: int
    when_lines: int
    reachable_functions: tuple[str, ...]
    missing_function_dependencies: tuple[str, ...]
    native_identifiers: tuple[str, ...]
    environment_identifiers: tuple[str, ...]


@dataclass(frozen=True)
class OracleCapability:
    expression_language: bool
    when_return_force: bool
    set_user_variables: bool
    persistent_memory_symbols: bool
    hand_ranges_and_lists: bool
    native_state_symbols: bool
    exact_bet_sizing: bool
    exact_allin_fallback: bool
    topology_semantics: bool
    environment_profile_frozen: bool
    runtime_parity_fixtures: bool

    def blockers(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, value in self.__dict__.items()
            if not bool(value)
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_r8_operational_source(path: Path) -> None:
    path = Path(path)
    if path.name != DEEPC_RUSHER_OPERATIONAL_SOURCE:
        raise ValueError(
            f"wrong R8 operational filename: {path.name!r}; "
            f"expected {DEEPC_RUSHER_OPERATIONAL_SOURCE!r}"
        )
    actual = sha256_file(path)
    if actual.lower() != DEEPC_RUSHER_OPERATIONAL_SHA256.lower():
        raise ValueError(
            f"R8 operational SHA mismatch: {actual}; "
            f"expected {DEEPC_RUSHER_OPERATIONAL_SHA256}"
        )


def split_sections(text: str) -> dict[str, str]:
    matches = list(_SECTION.finditer(text))
    out: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        if name in out:
            raise ValueError(f"duplicate section: {name}")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        out[name] = text[start:end]
    return out


def _executable_text(body: str) -> str:
    lines = []
    for raw in body.splitlines():
        line = raw.split("//", 1)[0]
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


def reachable_closure(
    sections: dict[str, str],
    roots: Iterable[str] = PRIMARY_CALLBACKS,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    seen: set[str] = set()
    missing: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        body = sections.get(name)
        if body is None:
            missing.add(name)
            continue
        for match in _FREF.finditer(_executable_text(body)):
            child = match.group(1)
            if child not in sections:
                missing.add(child)
            elif child not in seen:
                stack.append(child)
    return tuple(sorted(seen - missing)), tuple(sorted(missing))


def native_identifier_inventory(
    sections: dict[str, str],
    reachable: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    identifiers: set[str] = set()
    for name in reachable:
        body = _executable_text(sections[name])
        identifiers.update(match.group(1) for match in _IDENTIFIER.finditer(body))

    internal = set(sections)
    native = {
        token
        for token in identifiers
        if token.lower() not in _LANGUAGE_WORDS
        and token not in internal
        and not token.lower().startswith("f$")
        and not token.lower().startswith("list")
        and not token.lower().startswith("hand$")
        and not token.lower().startswith("user_")
        and not token.lower().startswith(("me_st_", "me_re_", "me_inc_", "me_add_", "me_sub_"))
    }
    env = {
        token
        for token in native
        if any(pattern.search(token) for pattern in _ENVIRONMENT_PATTERNS)
    }
    return tuple(sorted(native, key=str.lower)), tuple(sorted(env, key=str.lower))


def inspect_r8_source(path: Path) -> R8SourceClosure:
    path = Path(path)
    verify_r8_operational_source(path)
    text = path.read_text(encoding="ascii")
    sections = split_sections(text)
    reachable, missing = reachable_closure(sections)
    native, env = native_identifier_inventory(sections, reachable)
    return R8SourceClosure(
        sections=len(sections),
        function_sections=sum(1 for name in sections if name.startswith("f$")),
        list_sections=sum(1 for name in sections if name.lower().startswith("list")),
        when_lines=sum(
            1
            for line in text.splitlines()
            if line.lstrip().lower().startswith("when ")
        ),
        reachable_functions=reachable,
        missing_function_dependencies=missing,
        native_identifiers=native,
        environment_identifiers=env,
    )


def contract_summary() -> dict[str, object]:
    return {
        "schema": DC0_SCHEMA,
        "deepcrusher_branch": DEEPC_RUSHER_BASELINE_BRANCH,
        "operational_source": DEEPC_RUSHER_OPERATIONAL_SOURCE,
        "operational_sha256": DEEPC_RUSHER_OPERATIONAL_SHA256,
        "primary_callbacks": list(PRIMARY_CALLBACKS),
        "strict_unknown_symbol_policy": "ERROR_NEVER_ZERO_FILL",
        "required_parity": {
            "action_family": "exact",
            "bet_raise_amount_to": "exact",
            "user_memory_side_effects": "exact",
            "native_hu_vs_3h_origin_topology": "exact",
        },
    }
