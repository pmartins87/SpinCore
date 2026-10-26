from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

USER_RE = re.compile(r"\buser_[A-Za-z0-9_]+\b", re.IGNORECASE)
SET_RE = re.compile(r"\bSet\s+(user_[A-Za-z0-9_]+)\b", re.IGNORECASE)
SECTION_RE = re.compile(r"^##([^#]+)##\s*$")


def annotated_lines(text: str) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    section = ""
    for lineno, line in enumerate(text.splitlines(), 1):
        m = SECTION_RE.match(line.strip())
        if m:
            section = m.group(1).strip()
        out.append((lineno, section, line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    text = args.source.read_text(encoding="utf-8", errors="replace")
    rows = annotated_lines(text)

    setters: dict[str, list[dict[str, object]]] = {}
    for lineno, section, line in rows:
        for m in SET_RE.finditer(line):
            key = m.group(1).lower()
            setters.setdefault(key, []).append(
                {"line": lineno, "section": section, "text": line.strip()}
            )

    consumers: list[dict[str, object]] = []
    for lineno, section, line in rows:
        if "bettwothirdpot" not in line.lower():
            continue
        users = sorted({x.lower() for x in USER_RE.findall(line)})
        consumers.append(
            {
                "line": lineno,
                "section": section,
                "text": line.strip(),
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

    payload = {
        "schema": "SPINCORE_DEEPCRUSHER_067_SOURCE_AUDIT_V2",
        "source": str(args.source),
        "bet_two_third_consumers": consumers,
        "consumer_count": len(consumers),
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
