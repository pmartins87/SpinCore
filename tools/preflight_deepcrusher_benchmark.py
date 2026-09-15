#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.deepcrusher_benchmark import (  # noqa: E402
    DEEPC_RUSHER_OPERATIONAL_SOURCE,
    balanced_three_handed_lineups,
    benchmark_contract,
    validate_three_handed_balance,
    verify_frozen_deepcrusher_source,
)

SECTION_RE = re.compile(r"^##([^#\r\n]+)##\s*$", re.MULTILINE)
FUNC_REF_RE = re.compile(r"\bf\$[A-Za-z0-9_]+")
HAND_REF_RE = re.compile(r"\bhand\$[A-Za-z0-9_]+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preflight the frozen DeepCrusher source for the offline benchmark")
    p.add_argument("--deepcrusher-source", type=Path, required=True)
    p.add_argument("--json", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    verified = verify_frozen_deepcrusher_source(args.deepcrusher_source, operational=True)
    text = args.deepcrusher_source.read_text(encoding="utf-8", errors="strict")
    sections = SECTION_RE.findall(text)
    section_set = set(sections)
    required_roots = {"f$preflop", "f$flop", "f$turn", "f$river"}
    missing = sorted(required_roots - section_set)
    if missing:
        raise SystemExit(f"frozen source is missing primary OpenPPL roots: {missing}")

    functions_defined = sorted(s for s in section_set if s.startswith("f$"))
    functions_referenced = sorted(set(FUNC_REF_RE.findall(text)))
    hand_ranges_referenced = sorted(set(HAND_REF_RE.findall(text)))
    unresolved_function_names = sorted(set(functions_referenced) - set(functions_defined))
    # Unresolved f$ names are not necessarily errors: OpenHoldem supplies native
    # f$ symbols too.  The list is an oracle-implementation inventory.

    lineups = balanced_three_handed_lineups()
    validate_three_handed_balance(lineups)
    report = {
        "schema": "SPINCORE_DEEPC_RUSHER_BENCHMARK_PREFLIGHT_V1",
        "verified_source": verified,
        "expected_filename": DEEPC_RUSHER_OPERATIONAL_SOURCE,
        "bytes": args.deepcrusher_source.stat().st_size,
        "line_count": text.count("\n") + 1,
        "section_count": len(sections),
        "function_sections": len(functions_defined),
        "function_references_unique": len(functions_referenced),
        "hand_range_references_unique": len(hand_ranges_referenced),
        "external_or_native_f_references_unique": len(unresolved_function_names),
        "primary_roots": sorted(required_roots),
        "balanced_3h_lineups": [list(row.seats) for row in lineups],
        "benchmark_contract": benchmark_contract(),
        "oracle_status": "NOT_YET_CANONICAL — source inventory only; exact OpenPPL decision parity still required",
        "external_or_native_f_reference_sample": unresolved_function_names[:100],
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== DeepCrusher benchmark preflight ===")
    print(f"source={verified['path']}")
    print(f"sha256={verified['sha256']}")
    print(f"bytes={report['bytes']} lines={report['line_count']}")
    print(f"sections={report['section_count']} f_sections={report['function_sections']}")
    print(f"unique_f_refs={report['function_references_unique']} unique_hand_ranges={report['hand_range_references_unique']}")
    print(f"external_or_native_f_refs={report['external_or_native_f_references_unique']}")
    print("fairness=HU paired seat swap; 3H six-game AAB/ABB exact seat balance")
    print("DEEPC_RUSHER_BENCHMARK_PREFLIGHT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
