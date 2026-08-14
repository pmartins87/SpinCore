from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from spincore.flop_abstraction import (  # noqa: E402
    audit_legacy_184_mapping,
    audit_legacy_184_summary,
    reference_counts,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "R7.5 deterministic flop-abstraction audit: establishes the exact 22,100-flop "
            "reference, 1,755 suit-isomorphic classes, the 53-class control taxonomy, and "
            "optionally audits recovered Solver-V2 184 artifacts."
        )
    )
    ap.add_argument("--legacy-184-summary", type=Path, default=None)
    ap.add_argument("--legacy-184-mapping", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    report: dict[str, object] = {
        "schema": "SPINCORE_R7_5_FLOP_ABSTRACTION_AUDIT_V1",
        "reference": reference_counts(),
        "legacy_184_summary": None,
        "legacy_184_mapping": None,
    }

    if args.legacy_184_summary is not None:
        report["legacy_184_summary"] = audit_legacy_184_summary(args.legacy_184_summary)
    if args.legacy_184_mapping is not None:
        report["legacy_184_mapping"] = audit_legacy_184_mapping(args.legacy_184_mapping)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
