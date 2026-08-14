from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from spincore.flop184_descendants import audit_descendant, write_exact_class_mapping  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Audit deterministic suit-invariant descendants of the recovered Solver-V2 184Flops.json. "
            "No descendant is selected as production strategy by this tool."
        )
    )
    ap.add_argument("mapping", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--write-canonical-map", type=Path, default=None)
    ap.add_argument("--write-majority-map", type=Path, default=None)
    args = ap.parse_args()

    report = {
        "schema": "SPINCORE_R7_5_LEGACY184_DESCENDANT_COMPARISON_V1",
        "canonical_input": audit_descendant(args.mapping, mode="canonical_input"),
        "majority_min_change": audit_descendant(args.mapping, mode="majority_min_change"),
        "production_selection": None,
    }

    if args.write_canonical_map is not None:
        write_exact_class_mapping(args.mapping, args.write_canonical_map, mode="canonical_input")
    if args.write_majority_map is not None:
        write_exact_class_mapping(args.mapping, args.write_majority_map, mode="majority_min_change")

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
