#!/usr/bin/env python3
from __future__ import annotations

"""Compile the complete frozen DeepCrusher R8 v22 OpenPPL source.

This is a structural/source gate only.  PASS means the pinned 1.27 MB formula,
including every f$ function and every list_* range, is accepted by SpinCore's
portable OpenPPL parser.  It does not by itself authorize canonical DC1/DC2;
native OpenHoldem symbols, persistent memory, exact sizing and runtime parity
remain separate DC0 gates.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.deepcrusher_benchmark import verify_frozen_deepcrusher_source  # noqa: E402
from spincore.deepcrusher_r8_dc0 import inspect_r8_source  # noqa: E402
from spincore.openppl_program import OpenPPLProgram  # noqa: E402

EXPECTED_SECTIONS = 1267
EXPECTED_FUNCTIONS = 721
EXPECTED_LISTS = 545


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--deepcrusher-source", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    verified = verify_frozen_deepcrusher_source(args.deepcrusher_source, operational=True)
    closure = inspect_r8_source(args.deepcrusher_source)
    text = args.deepcrusher_source.read_text(encoding="ascii", errors="strict")

    program = OpenPPLProgram.from_text(text)

    errors: list[str] = []
    if closure.sections != EXPECTED_SECTIONS:
        errors.append(f"sections={closure.sections} expected={EXPECTED_SECTIONS}")
    if len(program.functions) != EXPECTED_FUNCTIONS:
        errors.append(f"functions={len(program.functions)} expected={EXPECTED_FUNCTIONS}")
    if len(program.hand_lists) != EXPECTED_LISTS:
        errors.append(f"lists={len(program.hand_lists)} expected={EXPECTED_LISTS}")
    for root in ("f$preflop", "f$flop", "f$turn", "f$river"):
        if not program.has_function(root):
            errors.append(f"missing_primary_root={root}")

    list_token_count = sum(len(hands) for hands in program.hand_lists.values())
    verdict = "PASS" if not errors else "FAIL"
    payload = {
        "schema": "SPINCORE_DEEPCRUSHER_R8_OPENPPL_COMPILE_V1",
        "verdict": verdict,
        "verified_source": verified,
        "sections": closure.sections,
        "function_sections": closure.function_sections,
        "compiled_functions": len(program.functions),
        "list_sections": closure.list_sections,
        "compiled_lists": len(program.hand_lists),
        "unique_memberships_across_lists": list_token_count,
        "reachable_function_count": len(closure.reachable_functions),
        "missing_function_dependencies": list(closure.missing_function_dependencies),
        "errors": errors,
        "scope": "STRUCTURAL_SOURCE_COMPILE_ONLY_NOT_CANONICAL_DC0",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== DeepCrusher R8 v22 full OpenPPL compile audit ===")
    print(f"source={verified['path']}")
    print(f"sha256={verified['sha256']}")
    print(f"sections={closure.sections}")
    print(f"compiled_functions={len(program.functions)}")
    print(f"compiled_lists={len(program.hand_lists)}")
    print(f"unique_memberships_across_lists={list_token_count}")
    print(f"missing_reachable_f_functions={len(closure.missing_function_dependencies)}")
    print("errors=" + ("NONE" if not errors else " | ".join(errors)))
    print(f"VERDICT={verdict}")
    if verdict == "PASS":
        print("DEEPC_RUSHER_R8_OPENPPL_COMPILE_PASS")
        return 0
    print("DEEPC_RUSHER_R8_OPENPPL_COMPILE_FAIL")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
