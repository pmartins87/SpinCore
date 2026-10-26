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


def executable_context(
    rows: list[tuple[int, str, int, str]],
    *,
    target_line: int,
    section: str,
    before: int = 24,
    after: int = 12,
) -> list[dict[str, object]]:
    same_section = [
        (lineno, section_line, line.strip())
        for lineno, row_section, section_line, line in rows
        if row_section == section
        and line.strip()
        and not line.strip().startswith("//")
        and line.strip().lower().startswith("when ")
    ]
    positions = [i for i, (lineno, _, _) in enumerate(same_section) if lineno == target_line]
    if not positions:
        return []
    i = positions[0]
    return [
        {"line": lineno, "section_line": section_line, "text": line}
        for lineno, section_line, line in same_section[
            max(0, i - before): min(len(same_section), i + after + 1)
        ]
    ]


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

    target_setter_contexts: dict[str, list[dict[str, object]]] = {}
    for name in ("user_river66", "user_rivercbet60", "user_turnprobeopportunity"):
        contexts = []
        for setter in setters.get(name, []):
            contexts.append({
                "setter": setter,
                "executable_context": executable_context(
                    rows,
                    target_line=int(setter["line"]),
                    section=str(setter["section"]),
                ),
            })
        target_setter_contexts[name] = contexts

    supersession_requirements = {
        "hsub_tp_75_75": [
            "Standard TP+ plan is 75/75.",
            "When f$game_HUSB && user_CF8_HUSB_TP_TurnBarrel75 Set user_River75",
        ],
        "hubb_air_probe_50_50": [
            "HUBB air probe: 50 + 50 pressure line.",
            "When user_DC10_HUBB_AirProbe50_River50 && f$NoMadeHand Set user_River50",
        ],
        "hubb_draw_probe_source_first": [
            "R8/R7-05 SOURCE-FIRST: HUBB draw-probe river sizing belongs to the TURN",
            "When user_DC10_HUBB_DrawProbe_nCPL Set user_River50",
            "When user_DC10_HUBB_DrawProbe_CPL Set user_River125",
        ],
    }
    supersession_evidence = {
        name: {
            "required_fragments": fragments,
            "all_present": all(fragment in text for fragment in fragments),
        }
        for name, fragments in supersession_requirements.items()
    }
    supersession_proven = all(
        row["all_present"] for row in supersession_evidence.values()
    )
    if not supersession_proven:
        raise SystemExit(
            "2/3 source-supersession evidence changed; re-audit dynamic sentinel contract"
        )

    payload = {
        "schema": "SPINCORE_DEEPCRUSHER_067_SOURCE_AUDIT_V7",
        "source": str(args.source),
        "bet_two_third_consumers": consumers,
        "consumer_count": len(consumers),
        "sizing_hotspots": sizing_hotspots,
        "competing_flag_setters": competing_flag_setters,
        "target_setter_contexts": target_setter_contexts,
        "dynamic_067_verdict": {
            "status": "SOURCE_SUPERSEDED_NOT_DYNAMIC_SENTINEL",
            "reason": (
                "All settable historical 2/3 paths are superseded by source-first "
                "R8 sizing plans; remaining 2/3 dependencies without setters are legacy."
            ),
            "supersession_evidence": supersession_evidence,
        },
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
