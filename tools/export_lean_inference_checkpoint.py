#!/usr/bin/env python3
from __future__ import annotations

"""Export a small inference-only SpinCore checkpoint from a full training checkpoint.

The full LT1/LT2 checkpoints contain multi-GB reservoirs and optimizer state. The
strategy evaluator only needs the finalized AveragePolicy weights plus the schema
fields checked by LeanFunctionalAgent. Loading full checkpoints independently in
many evaluation workers would multiply memory use unnecessarily, so this tool
loads the source once and writes a compact policy-only artifact.
"""

import argparse
import hashlib
from pathlib import Path
import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    source = args.source.resolve(strict=True)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("schema") != "SPINCORE_LEAN_FUNCTIONAL_TRAINING_V1":
        raise SystemExit("wrong source checkpoint schema")
    if not bool(payload.get("finalized")):
        raise SystemExit("source checkpoint is not finalized")

    domains = payload.get("domains") or {}
    for domain in ("THREE_HANDED", "TRUE_HEADS_UP"):
        if domain not in domains or "policy" not in domains[domain]:
            raise SystemExit(f"source checkpoint missing policy for {domain}")

    compact = {
        "schema": payload["schema"],
        "representation": payload["representation"],
        "action_candidate": payload["action_candidate"],
        "finalized": True,
        "source_checkpoint": str(source),
        "source_sha256": sha256(source),
        "completed_iteration": int(payload.get("completed_iteration", -1)),
        "domains": {
            domain: {"policy": domains[domain]["policy"]}
            for domain in ("THREE_HANDED", "TRUE_HEADS_UP")
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, args.out)
    print(f"INFERENCE_CHECKPOINT_EXPORTED source_iteration={compact['completed_iteration']}")
    print(f"source_sha256={compact['source_sha256']}")
    print(f"out={args.out.resolve()}")
    print(f"out_sha256={sha256(args.out.resolve())}")
    print(f"out_bytes={args.out.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
