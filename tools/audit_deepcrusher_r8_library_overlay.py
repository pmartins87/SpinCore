#!/usr/bin/env python3
from __future__ import annotations

"""Audit how much of frozen DeepCrusher R8 is supplied by the pinned OpenPPL library."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.deepcrusher_native_symbols import (  # noqa: E402
    DeepCrusherPrimitiveSymbols,
    frozen_benchmark_environment,
)
from spincore.deepcrusher_r8_dc0 import inspect_r8_source  # noqa: E402
from spincore.openppl_program import split_sections  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deepcrusher-source", type=Path, required=True)
    p.add_argument("--library", type=Path, action="append", required=True)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    closure = inspect_r8_source(args.deepcrusher_source)

    library_names: dict[str, str] = {}
    duplicate_nonlicense: list[str] = []
    library_sections_total = 0
    for path in args.library:
        sections = split_sections(path.read_text(encoding="utf-8"))
        library_sections_total += len(sections)
        for name in sections:
            low = name.lower()
            if low == "openppl_license_text":
                continue
            if low in library_names:
                duplicate_nonlicense.append(name)
            else:
                library_names[low] = name

    if duplicate_nonlicense:
        raise RuntimeError(
            "duplicate non-license OpenPPL library sections: "
            + ", ".join(sorted(set(duplicate_nonlicense), key=str.lower))
        )

    native = list(closure.native_identifiers)
    primitive_direct = {
        name.lower()
        for name in native
        if DeepCrusherPrimitiveSymbols.supports(name)
    }
    environment = {x.lower() for x in frozen_benchmark_environment(native)}
    library_direct = {
        name.lower()
        for name in native
        if name.lower() in library_names
    }

    resolved_direct = primitive_direct | environment | library_direct
    remaining = sorted(
        (name for name in native if name.lower() not in resolved_direct),
        key=str.lower,
    )

    payload = {
        "schema": "SPINCORE_DEEPCRUSHER_R8_OPENPPL_LIBRARY_OVERLAY_V1",
        "source_native_identifiers": len(native),
        "primitive_direct_count": len(primitive_direct),
        "environment_direct_count": sum(1 for x in native if x.lower() in environment),
        "openppl_library_direct_count": sum(1 for x in native if x.lower() in library_direct),
        "remaining_direct_count": len(remaining),
        "remaining_direct": remaining,
        "library_files": [str(x.resolve()) for x in args.library],
        "library_sections_total_including_repeated_license": library_sections_total,
        "library_unique_nonlicense_sections": len(library_names),
        "duplicate_nonlicense_sections": [],
        "important_scope_note": (
            "Direct-source coverage only. Library functions can themselves depend on "
            "additional native OpenHoldem leaves; those transitive leaves remain part "
            "of DC0 until runtime/static closure proves them implemented."
        ),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== DeepCrusher R8 OpenPPL library overlay ===")
    print(f"source_native={len(native)}")
    print(f"primitive_direct={payload['primitive_direct_count']}")
    print(f"environment_direct={payload['environment_direct_count']}")
    print(f"openppl_library_direct={payload['openppl_library_direct_count']}")
    print(f"remaining_direct={len(remaining)}")
    print("remaining_direct_symbols=" + ",".join(remaining))
    print(f"report={args.report.resolve()}")
    print("DEEPC_RUSHER_R8_OPENPPL_LIBRARY_OVERLAY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
