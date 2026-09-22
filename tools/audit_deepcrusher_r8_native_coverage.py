#!/usr/bin/env python3
from __future__ import annotations

"""Measure portable native-symbol coverage for the frozen DeepCrusher R8 oracle.

This audit is descriptive until --require-complete is supplied.  It lets DC0
advance category by category without pretending that unknown OpenHoldem symbols
are zero.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.deepcrusher_native_symbols import (  # noqa: E402
    BENCHMARK_ENVIRONMENT_PROFILE_ID,
    DeepCrusherPrimitiveSymbols,
    frozen_benchmark_environment,
)
from spincore.deepcrusher_r8_dc0 import inspect_r8_source  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--deepcrusher-source", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--require-complete", action="store_true")
    return p.parse_args()


def _classify(name: str) -> str:
    low = name.lower()
    if low.startswith("pt_"):
        return "pokertracker"
    if low in {"prwin", "prtie"}:
        return "equity"
    if low.startswith(("network$", "chair$", "log$")) or low in {"colourcode_dealerchair", "colourcode_headsupchair"}:
        return "environment"
    if low.startswith(("have", "is")) or any(
        token in low
        for token in (
            "rank", "suit", "flush", "straight", "pair", "card", "pokerval",
            "overcards", "outs", "wheel", "uncoordinated",
        )
    ):
        return "cards_and_hand"
    if any(
        token in low
        for token in (
            "bot", "raise", "call", "bet", "action", "previousround",
            "lastraised", "prevaction",
        )
    ):
        return "action_history"
    if any(token in low for token in ("chair", "position", "blind")):
        return "position_topology"
    if any(token in low for token in ("stack", "balance", "pot", "currentbet", "amounttocall", "dollar")):
        return "chips_and_stacks"
    return "other"


def main() -> int:
    args = parse_args()
    closure = inspect_r8_source(args.deepcrusher_source)
    native = list(closure.native_identifiers)
    environment_profile = frozen_benchmark_environment(native)
    environment_folded = {name.lower() for name in environment_profile}
    supported = sorted(
        (name for name in native if DeepCrusherPrimitiveSymbols.supports(name)),
        key=str.lower,
    )
    supported_folded = {name.lower() for name in supported}
    environment_supported = sorted(
        (name for name in native if name.lower() in environment_folded),
        key=str.lower,
    )
    unresolved = sorted(
        (
            name for name in native
            if name.lower() not in supported_folded
            and name.lower() not in environment_folded
        ),
        key=str.lower,
    )

    groups: dict[str, list[str]] = {}
    for name in unresolved:
        groups.setdefault(_classify(name), []).append(name)

    payload = {
        "schema": "SPINCORE_DEEPCRUSHER_R8_NATIVE_COVERAGE_V1",
        "native_identifiers_total": len(native),
        "primitive_supported_count": len(supported),
        "primitive_supported": supported,
        "environment_profile_id": BENCHMARK_ENVIRONMENT_PROFILE_ID,
        "environment_profile_supported_count": len(environment_supported),
        "environment_profile_supported": environment_supported,
        "environment_profile_values": dict(sorted(environment_profile.items(), key=lambda kv: kv[0].lower())),
        "environment_identifiers_count": len(closure.environment_identifiers),
        "environment_identifiers": list(closure.environment_identifiers),
        "unresolved_count": len(unresolved),
        "unresolved_by_category": {
            key: {"count": len(values), "symbols": values}
            for key, values in sorted(groups.items())
        },
        "unknown_symbol_policy": "ERROR_NEVER_ZERO_FILL",
        "complete": not unresolved,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== DeepCrusher R8 native symbol coverage ===")
    print(f"native_total={len(native)}")
    print(f"primitive_supported={len(supported)}")
    print(f"environment_profile={BENCHMARK_ENVIRONMENT_PROFILE_ID}")
    print(f"environment_profile_supported={len(environment_supported)}")
    print(f"unresolved={len(unresolved)}")
    for key, values in sorted(groups.items()):
        print(f"unresolved_{key}={len(values)}")
    print("coverage_complete=" + str(not unresolved))
    print(f"report={args.report}")

    if args.require_complete and unresolved:
        print("DEEPC_RUSHER_R8_NATIVE_COVERAGE_BLOCKED")
        return 2
    print("DEEPC_RUSHER_R8_NATIVE_COVERAGE_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
