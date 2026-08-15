from __future__ import annotations

import argparse
import json
from pathlib import Path

from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_final_policy import extract_final_v3_policy_light
from spincore.r7_5_representation_v3_stage_contract import DOMAINS, TRAINING_SEEDS

SCHEMA = "SPINCORE_R7_5_3C_PHASE2_LIGHT_POLICY_INVENTORY_V1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frozen Phase2 final AveragePolicy artifacts")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-execution-sha", required=True)
    args = parser.parse_args()

    checkpoints = sorted(args.input_root.rglob("checkpoint.pt"))
    if len(checkpoints) != 8:
        raise RuntimeError(f"expected exactly 8 final Phase2 checkpoints, found {len(checkpoints)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for checkpoint in checkpoints:
        temporary = args.output_root / "_extracting.pt"
        metadata = extract_final_v3_policy_light(
            checkpoint,
            temporary,
            expected_training_execution_sha=str(args.training_execution_sha),
        )
        key = (
            str(metadata["representation"]),
            str(metadata["domain"]),
            int(metadata["training_seed"]),
        )
        if key in seen:
            raise RuntimeError(f"duplicate Phase2 final policy cell: {key}")
        seen.add(key)
        filename = f"{key[0]}__{key[1]}__{key[2]}.pt"
        destination = args.output_root / filename
        temporary.replace(destination)
        rows.append({**metadata, "file": filename})

    expected = {
        (representation, domain, int(seed))
        for representation in (H2_FINAL, H3_FINAL)
        for domain in DOMAINS
        for seed in TRAINING_SEEDS
    }
    if seen != expected:
        raise RuntimeError(
            f"Phase2 light-policy inventory mismatch: missing={sorted(expected-seen)} extra={sorted(seen-expected)}"
        )
    payload = {
        "schema": SCHEMA,
        "training_execution_sha": str(args.training_execution_sha),
        "expected_cells": 8,
        "observed_cells": len(rows),
        "cells": sorted(rows, key=lambda row: (row["representation"], row["domain"], row["training_seed"])),
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    inventory = args.output_root / "inventory.json"
    inventory.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
