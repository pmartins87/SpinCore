from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

USER_RE = re.compile(r"\buser_[A-Za-z0-9_]+\b", re.IGNORECASE)
SET_RE = re.compile(r"\bSet\s+(user_[A-Za-z0-9_]+)\b", re.IGNORECASE)
SECTION_RE = re.compile(r"^##([^#]+)##\s*$")


def annotated_lines(text: str) -> list[tuple[int, str, int, str]]:
    out: list[tuple[int, str, int, str]] = []
    section = ""
    section_line = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        m = SECTION_RE.match(line.strip())
        if m:
            section = m.group(1).strip()
            section_line = 0
            out.append((lineno, section, section_line, line))
            continue
        section_line += 1
        out.append((lineno, section, section_line, line))
    return out


def previous_executable_rules(
    rows: list[tuple[int, str, int, str]],
    *,
    target_line: int,
    section: str,
    limit: int = 14,
) -> list[dict[str, object]]:
    candidates = []
    for lineno, row_section, section_line, line in rows:
        if lineno >= target_line:
            break
        clean = line.strip()
        if (
            row_section == section
            and clean
            and not clean.startswith("//")
            and clean.lower().startswith("when ")
        ):
            candidates.append({
                "line": lineno,
                "section_line": section_line,
                "text": clean,
            })
    return candidates[-limit:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    text = args.source.read_text(encoding="utf-8", errors="replace")
    rows = annotated_lines(text)

    setters: dict[str, list[dict[str, object]]] = {}
    for lineno, section, section_line, line in rows:
        for m in SET_RE.finditer(line):
            key = m.group(1).lower()
            setters.setdefault(key, []).append(
                {
                    "line": lineno,
                    "section": section,
                    "section_line": section_line,
                    "text": line.strip(),
                }
            )

    consumers: list[dict[str, object]] = []
    for lineno, section, section_line, line in rows:
        if "bettwothirdpot" not in line.lower():
            continue
        users = sorted({x.lower() for x in USER_RE.findall(line)})
        consumers.append(
            {
                "line": lineno,
                "section": section,
                "section_line": section_line,
                "text": line.strip(),
                "previous_executable_rules": previous_executable_rules(
                    rows,
                    target_line=lineno,
                    section=section,
                ),
                "user_dependencies": [
                    {
                        "name": u,
                        "setter_count": len(setters.get(u, [])),
                        "setters": setters.get(u, []),
                    }
                    for u in users
                ],
            }
        )

    hotspot_ranges = {
        "f$BetsizeTurnHeadsup": (40, 55),
        "f$BetsizeRiverHeadsup": (20, 35),
    }
    sizing_hotspots: dict[str, list[dict[str, object]]] = {}
    for hotspot_section, (lo, hi) in hotspot_ranges.items():
        sizing_hotspots[hotspot_section] = [
            {
                "line": lineno,
                "section_line": section_line,
                "text": line.strip(),
            }
            for lineno, section, section_line, line in rows
            if section == hotspot_section
            and lo <= section_line <= hi
            and line.strip()
            and not line.strip().startswith("//")
        ]

    competing_flags = [
        "user_turn50",
        "user_turn75",
        "user_river50",
        "user_river75",
        "user_turnprobeopportunity",
        "user_river66",
        "user_rivercbet60",
    ]
    competing_flag_setters = {
        name: setters.get(name, [])
        for name in competing_flags
    }

    payload = {
        "schema": "SPINCORE_DEEPCRUSHER_067_SOURCE_AUDIT_V5",
        "source": str(args.source),
        "bet_two_third_consumers": consumers,
        "consumer_count": len(consumers),
        "sizing_hotspots": sizing_hotspots,
        "competing_flag_setters": competing_flag_setters,
        "user_dependencies_with_no_setter": sorted(
            {
                d["name"]
                for c in consumers
                for d in c["user_dependencies"]
                if d["setter_count"] == 0
            }
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.write_text(rendered + "\n", encoding="utf-8")

    if not consumers:
        raise SystemExit("no BetTwoThirdPot consumers found in frozen source")
    print("DEEPC_RUSHER_067_SOURCE_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
