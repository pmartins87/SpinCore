from __future__ import annotations

"""Source-level OpenPPL inventory used by the DeepCrusher offline oracle.

This is intentionally a parser/inventory layer, not yet a claim of full
OpenHoldem semantic parity.  It gives the benchmark a deterministic view of the
frozen DeepCrusher sections, hand lists and transitive f$ dependencies so the
portable decision oracle can be implemented and validated in bounded pieces.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

SECTION_RE = re.compile(r"^##([^#\r\n]+)##\s*$", re.MULTILINE)
F_REF_RE = re.compile(r"\bf\$[A-Za-z0-9_]+")
HAND_REF_RE = re.compile(r"\bhand\$[A-Za-z0-9_]+")
USER_REF_RE = re.compile(r"\buser_[A-Za-z0-9_$]+")
ME_REF_RE = re.compile(r"\bme_(?:st|re)_[A-Za-z0-9_$]+")
LIST_NAME_RE = re.compile(r"^list(?:\$|_)?", re.IGNORECASE)

PRIMARY_ROOTS = ("f$preflop", "f$flop", "f$turn", "f$river")


@dataclass(frozen=True)
class OpenPPLSection:
    name: str
    body: str
    start_line: int

    @property
    def is_function(self) -> bool:
        return self.name.startswith("f$")

    @property
    def is_list(self) -> bool:
        return bool(LIST_NAME_RE.match(self.name))

    @property
    def f_dependencies(self) -> tuple[str, ...]:
        return tuple(sorted(set(F_REF_RE.findall(_without_comments(self.body)))) )

    @property
    def hand_dependencies(self) -> tuple[str, ...]:
        return tuple(sorted(set(HAND_REF_RE.findall(_without_comments(self.body)))))

    @property
    def user_variables(self) -> tuple[str, ...]:
        return tuple(sorted(set(USER_REF_RE.findall(_without_comments(self.body)))))

    @property
    def memory_symbols(self) -> tuple[str, ...]:
        return tuple(sorted(set(ME_REF_RE.findall(_without_comments(self.body)))))


@dataclass(frozen=True)
class OpenPPLSource:
    path: Path
    sections: tuple[OpenPPLSection, ...]

    def by_name(self) -> dict[str, OpenPPLSection]:
        return {section.name: section for section in self.sections}

    def require(self, name: str) -> OpenPPLSection:
        try:
            return self.by_name()[name]
        except KeyError as exc:
            raise KeyError(f"OpenPPL section not found: {name}") from exc

    @property
    def functions(self) -> tuple[OpenPPLSection, ...]:
        return tuple(section for section in self.sections if section.is_function)

    @property
    def lists(self) -> tuple[OpenPPLSection, ...]:
        return tuple(section for section in self.sections if section.is_list)

    def transitive_function_closure(self, roots: Iterable[str] = PRIMARY_ROOTS) -> tuple[str, ...]:
        by_name = self.by_name()
        pending = list(roots)
        seen: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            section = by_name.get(name)
            if section is None:
                continue
            for dep in section.f_dependencies:
                if dep not in seen and dep in by_name:
                    pending.append(dep)
        return tuple(sorted(seen))

    def external_f_symbols(self, roots: Iterable[str] = PRIMARY_ROOTS) -> tuple[str, ...]:
        by_name = self.by_name()
        closure = self.transitive_function_closure(roots)
        external: set[str] = set()
        for name in closure:
            section = by_name.get(name)
            if section is None:
                external.add(name)
                continue
            for dep in section.f_dependencies:
                if dep not in by_name:
                    external.add(dep)
        return tuple(sorted(external))

    def referenced_hand_ranges(self, roots: Iterable[str] = PRIMARY_ROOTS) -> tuple[str, ...]:
        by_name = self.by_name()
        out: set[str] = set()
        for name in self.transitive_function_closure(roots):
            section = by_name.get(name)
            if section:
                out.update(section.hand_dependencies)
        return tuple(sorted(out))

    def referenced_user_variables(self, roots: Iterable[str] = PRIMARY_ROOTS) -> tuple[str, ...]:
        by_name = self.by_name()
        out: set[str] = set()
        for name in self.transitive_function_closure(roots):
            section = by_name.get(name)
            if section:
                out.update(section.user_variables)
        return tuple(sorted(out))

    def inventory(self) -> dict[str, object]:
        by_name = self.by_name()
        missing_roots = [name for name in PRIMARY_ROOTS if name not in by_name]
        closure = self.transitive_function_closure()
        return {
            "path": str(self.path.resolve()),
            "sections": len(self.sections),
            "function_sections": len(self.functions),
            "list_sections": len(self.lists),
            "primary_roots": list(PRIMARY_ROOTS),
            "missing_primary_roots": missing_roots,
            "reachable_function_sections": len([name for name in closure if name in by_name]),
            "external_f_symbols": list(self.external_f_symbols()),
            "hand_ranges": list(self.referenced_hand_ranges()),
            "user_variables": list(self.referenced_user_variables()),
        }


def _without_comments(text: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines())


def parse_openppl_source(path: str | Path) -> OpenPPLSource:
    target = Path(path)
    text = target.read_text(encoding="utf-8", errors="strict")
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        raise ValueError("no ##section## markers found in OpenPPL source")
    sections: list[OpenPPLSection] = []
    line_starts = [0]
    for match in re.finditer("\n", text):
        line_starts.append(match.end())

    def line_number(offset: int) -> int:
        import bisect
        return bisect.bisect_right(line_starts, offset)

    for index, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(
            OpenPPLSection(
                name=match.group(1).strip(),
                body=text[body_start:body_end],
                start_line=line_number(match.start()),
            )
        )
    source = OpenPPLSource(target, tuple(sections))
    missing = [root for root in PRIMARY_ROOTS if root not in source.by_name()]
    if missing:
        raise ValueError(f"OpenPPL source missing primary roots: {missing}")
    return source
