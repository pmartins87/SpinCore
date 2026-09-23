#!/usr/bin/env python3
from __future__ import annotations

"""Export a compact research hybrid inference bundle for benchmark workers.

THREE_HANDED uses finalized AveragePolicy from the training checkpoint.
TRUE_HEADS_UP uses the matched ENS8 current-behavior sidecar. The output is
inference-only: no reservoirs, optimizers or training RNG state.
"""

import argparse
import hashlib
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_hybrid_deployment_agent import (
    GENERIC_INFERENCE_SCHEMA,
    MODE_AVERAGE_POLICY,
    MODE_HU_ENS8,
    REPRESENTATION,
)

CHECKPOINT_SCHEMA = "SPINCORE_LEAN_FUNCTIONAL_TRAINING_V1"
ENSEMBLE_SCHEMAS = {
    "SPINCORE_LT2_HU_ENS8_CURRENT_STATE_V1",
    "SPINCORE_LT3_HU_ENS8_CURRENT_STATE_V1",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ensemble", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    ensemble = args.ensemble.resolve(strict=True)
    cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ens = torch.load(ensemble, map_location="cpu", weights_only=False)

    if cp.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("wrong training checkpoint schema")
    if cp.get("representation") != REPRESENTATION:
        raise SystemExit("checkpoint representation drift")
    if cp.get("action_candidate") != FIRST_RELEASE_ACTION_SPEC.candidate_id:
        raise SystemExit("checkpoint action candidate drift")
    if not bool(cp.get("finalized")):
        raise SystemExit("checkpoint must be finalized")

    iteration = int(cp.get("completed_iteration", -1))
    if iteration < 0:
        raise SystemExit("checkpoint missing completed_iteration")
    if ens.get("schema") not in ENSEMBLE_SCHEMAS:
        raise SystemExit(f"unsupported HU ensemble schema: {ens.get('schema')!r}")
    if int(ens.get("completed_iteration", -2)) != iteration:
        raise SystemExit("checkpoint/ensemble iteration mismatch")
    members = list(ens.get("members") or [])
    if int(ens.get("ensemble_size", -1)) != 8 or len(members) != 8:
        raise SystemExit("hybrid benchmark requires exactly 8 HU members")

    d3 = (cp.get("domains") or {}).get("THREE_HANDED") or {}
    if "policy" not in d3:
        raise SystemExit("checkpoint missing THREE_HANDED AveragePolicy")

    bundle = {
        "schema": GENERIC_INFERENCE_SCHEMA,
        "representation": REPRESENTATION,
        "action_candidate": FIRST_RELEASE_ACTION_SPEC.candidate_id,
        "completed_iteration": iteration,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "source_ensemble": str(ensemble),
        "source_ensemble_sha256": sha256(ensemble),
        "candidate": (
            f"THREE_HANDED AveragePolicy + TRUE_HEADS_UP current ENS8@{iteration}"
        ),
        "research_only": True,
        "domains": {
            "THREE_HANDED": {
                "mode": MODE_AVERAGE_POLICY,
                "policy": d3["policy"],
            },
            "TRUE_HEADS_UP": {
                "mode": MODE_HU_ENS8,
                "ensemble_size": 8,
                "member_steps": int(ens.get("member_steps", 400)),
                "semantics": ens.get("semantics"),
                "members": members,
                "member_meta": ens.get("member_meta"),
            },
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.out)
    print(f"SPINCORE_HYBRID_INFERENCE_EXPORTED iteration={iteration}")
    print(f"source_checkpoint_sha256={bundle['source_checkpoint_sha256']}")
    print(f"source_ensemble_sha256={bundle['source_ensemble_sha256']}")
    print(f"out={args.out.resolve()}")
    print(f"out_sha256={sha256(args.out.resolve())}")
    print(f"out_bytes={args.out.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
