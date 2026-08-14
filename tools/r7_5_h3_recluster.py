from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from spincore.flop184_recluster import build_h3_payload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the precommitted deterministic R7.5 H3 184-medoid flop mapping."
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = build_h3_payload()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
