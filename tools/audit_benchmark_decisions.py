#!/usr/bin/env python3
from __future__ import annotations

"""Audit SpinCore decision traces from an offline DeepCrusher benchmark JSONL."""

import argparse
from collections import Counter
from dataclasses import fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.deepcrusher_benchmark import DecisionTrace  # noqa: E402
from spincore.decision_sanity import flag_to_dict, sanity_flags  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--max-examples-per-code", type=int, default=20)
    return p.parse_args()


def _trace_from_dict(payload: dict) -> DecisionTrace:
    allowed = {x.name for x in fields(DecisionTrace)}
    missing = allowed - set(payload)
    if missing:
        raise ValueError(f"trace missing fields: {sorted(missing)}")
    data = {k: payload[k] for k in allowed}
    data["lineup"] = tuple(data["lineup"])
    data["stacks"] = tuple(data["stacks"])
    data["street_commitments"] = tuple(data["street_commitments"])
    data["total_commitments"] = tuple(data["total_commitments"])
    if data["hole_cards"] is not None:
        data["hole_cards"] = tuple(data["hole_cards"])
    data["board"] = tuple(data["board"])
    return DecisionTrace(**data)


def main() -> int:
    args = parse_args()
    counts = Counter()
    severity = Counter()
    examples: dict[str, list[dict]] = {}
    decisions = 0
    spincore_decisions = 0

    with args.traces.open("r", encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            decisions += 1
            payload = json.loads(raw)
            trace = _trace_from_dict(payload)
            if trace.policy_id == "SPINCORE":
                spincore_decisions += 1
            for flag in sanity_flags(trace):
                counts[flag.code] += 1
                severity[flag.severity] += 1
                rows = examples.setdefault(flag.code, [])
                if len(rows) < args.max_examples_per_code:
                    item = flag_to_dict(flag)
                    item["trace_line"] = line_no
                    rows.append(item)

    report = {
        "schema": "SPINCORE_EXTERNAL_DECISION_SANITY_AUDIT_V1",
        "source": str(args.traces.resolve()),
        "decisions_total": decisions,
        "spincore_decisions": spincore_decisions,
        "flag_counts": dict(sorted(counts.items())),
        "severity_counts": dict(sorted(severity.items())),
        "examples": examples,
        "interpretation": (
            "Heuristic review queue only. A flag is not automatically a poker error; "
            "critical deterministic-looking anomalies and repeated patterns require "
            "hand-level inspection with betting context."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== SpinCore external decision sanity audit ===")
    print(f"decisions_total={decisions}")
    print(f"spincore_decisions={spincore_decisions}")
    for code, count in sorted(counts.items()):
        print(f"{code}={count}")
    print(f"report={args.report.resolve()}")
    print("SPINCORE_EXTERNAL_DECISION_SANITY_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
