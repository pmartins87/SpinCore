#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from spincore.deepcrusher_openppl import parse_openppl_source  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preflight the frozen DeepCrusher source for the offline benchmark")
    p.add_argument("--deepcrusher-source", type=Path, required=True)
    p.add_argument("--json", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    verified = verify_frozen_deepcrusher_source(args.deepcrusher_source, operational=True)
    source = parse_openppl_source(args.deepcrusher_source)
    text = args.deepcrusher_source.read_text(encoding="utf-8", errors="strict")
    inventory = source.inventory()

    lineups = balanced_three_handed_lineups()
    validate_three_handed_balance(lineups)
    report = {
        "schema": "SPINCORE_DEEPC_RUSHER_BENCHMARK_PREFLIGHT_V2",
        "verified_source": verified,
        "expected_filename": DEEPC_RUSHER_OPERATIONAL_SOURCE,
        "bytes": args.deepcrusher_source.stat().st_size,
        "line_count": text.count("\n") + 1,
        "openppl_inventory": inventory,
        "balanced_3h_lineups": [list(row.seats) for row in lineups],
        "benchmark_contract": benchmark_contract(),
        "oracle_status": "NOT_YET_CANONICAL — source/dependency inventory only; exact OpenPPL decision parity still required",
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== DeepCrusher benchmark preflight ===")
    print(f"source={verified['path']}")
    print(f"sha256={verified['sha256']}")
    print(f"bytes={report['bytes']} lines={report['line_count']}")
    print(
        "sections=" + str(inventory["sections"])
        + " f_sections=" + str(inventory["function_sections"])
        + " reachable_f_sections=" + str(inventory["reachable_function_sections"])
        + " list_sections=" + str(inventory["list_sections"])
    )
    print(
        "external_f_symbols=" + str(len(inventory["external_f_symbols"]))
        + " hand_ranges=" + str(len(inventory["hand_ranges"]))
        + " user_variables=" + str(len(inventory["user_variables"]))
    )
    print("fairness=HU paired seat swap; 3H six-game AAB/ABB exact seat balance")
    print("DEEPC_RUSHER_BENCHMARK_PREFLIGHT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
