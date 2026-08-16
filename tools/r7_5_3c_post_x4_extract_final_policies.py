from __future__ import annotations

"""Extract final SPNNIV3 AveragePolicy artifacts from the frozen x4 remediation.

The original Phase-2 extractor is intentionally strict about the original
64-roots/iteration budget (192 roots/seed).  The x4 remediation preserves the
same model/action/training semantics but has 256 roots/iteration (768 roots per
seed), so reusing the old extractor would correctly fail its historical root
identity check.

This tool is a narrow x4 adapter.  It emits the *same* light-policy schema used
by the frozen strategic evaluator and also derives a training-inventory artifact
with the same schema expected by the frozen aggregate.  It changes no strategic
metric, winner rule or gate.
"""

import argparse
import json
from pathlib import Path

import torch

from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from spincore.r7_5_representation_v3_final_policy import LIGHT_SCHEMA
from spincore.r7_5_representation_v3_phase2_eval import validate_training_final_report
from spincore.r7_5_representation_v3_stage import FINAL_REPORT_SCHEMA
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    DOMAINS,
    ITERATIONS,
    MODEL_FINGERPRINTS,
    MODEL_PARAMETER_COUNTS,
    POLICY_STEPS,
    TRAINING_SEEDS,
    validate_phase2_v3_contract,
)

SCHEMA = "SPINCORE_R7_5_3C_POST_X4_LIGHT_POLICY_INVENTORY_V1"
TRAINING_INVENTORY_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_TRAINING_INVENTORY_V1"
COVERAGE_MULTIPLIER = 4
ROOTS_PER_ITERATION = 256
ROOTS_PER_SEED = ROOTS_PER_ITERATION * ITERATIONS
ROOTS_PER_CHUNK = 64
CHUNKS_PER_ITERATION = 4


def _load_checkpoint(path: Path) -> dict:
    rng = torch.get_rng_state().clone()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        torch.set_rng_state(rng)
    if not torch.equal(rng, torch.get_rng_state()):
        raise RuntimeError("loading x4 final checkpoint changed global Torch RNG")
    return payload


def _validate_x4_final(payload: dict, *, repo_root: Path, training_sha: str) -> tuple[dict, dict]:
    failures: list[str] = []
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        failures.append("CHECKPOINT_SCHEMA")
    if payload.get("execution_sha") != training_sha:
        failures.append("TRAINING_EXECUTION_SHA")
    representation = str(payload.get("representation"))
    domain = str(payload.get("domain"))
    seed = int(payload.get("seed", -1))
    if representation not in (H2_FINAL, H3_FINAL):
        failures.append("REPRESENTATION")
    if domain not in DOMAINS:
        failures.append("DOMAIN")
    if seed not in TRAINING_SEEDS:
        failures.append("TRAINING_SEED")
    if payload.get("action_candidate") != ACTION_CANDIDATE:
        failures.append("ACTION_CANDIDATE")
    if representation in MODEL_FINGERPRINTS and payload.get("architecture_fingerprint_sha256") != MODEL_FINGERPRINTS[representation]:
        failures.append("ARCHITECTURE_FINGERPRINT")
    if bool(payload.get("production_training_authorized")) or bool(payload.get("ready_for_tables")):
        failures.append("ILLEGAL_PRODUCTION_OR_TABLE_AUTHORIZATION")

    if representation in (H2_FINAL, H3_FINAL) and domain in DOMAINS and seed in TRAINING_SEEDS:
        contract = validate_phase2_v3_contract(
            repo_root,
            representation=representation,
            domain=domain,
            training_seed=seed,
        )
        if dict(payload.get("config") or {}) != dict(contract["live_model"]["config"]):
            failures.append("MODEL_CONFIG")

    progress = dict(payload.get("progress") or {})
    if progress.get("phase") != "post_policy_fit":
        failures.append("NOT_POST_POLICY_FIT")
    if int(progress.get("iteration", -1)) != ITERATIONS:
        failures.append("PROGRESS_ITERATION")
    if int(progress.get("policy_optimizer_step", -1)) != POLICY_STEPS:
        failures.append("POLICY_OPTIMIZER_STEPS")

    extra = dict(payload.get("extra") or {})
    final_report = dict(extra.get("final_report") or {})
    state = dict(extra.get("stage_state") or {})
    if final_report.get("schema") != FINAL_REPORT_SCHEMA:
        failures.append("FINAL_REPORT_SCHEMA")
    if final_report.get("representation") != representation:
        failures.append("FINAL_REPORT_REPRESENTATION")
    if final_report.get("domain") != domain:
        failures.append("FINAL_REPORT_DOMAIN")
    if int(final_report.get("training_seed", -1)) != seed:
        failures.append("FINAL_REPORT_TRAINING_SEED")
    if int(final_report.get("iterations", -1)) != ITERATIONS:
        failures.append("FINAL_REPORT_ITERATIONS")
    if int(final_report.get("roots", -1)) != ROOTS_PER_SEED:
        failures.append("FINAL_REPORT_X4_ROOTS")
    if int(final_report.get("average_policy_optimizer_steps", -1)) != POLICY_STEPS:
        failures.append("FINAL_REPORT_POLICY_STEPS")

    if int(state.get("chance_coverage_multiplier", -1)) != COVERAGE_MULTIPLIER:
        failures.append("STATE_CHANCE_MULTIPLIER")
    if int(state.get("effective_roots_per_iteration", -1)) != ROOTS_PER_ITERATION:
        failures.append("STATE_ROOTS_PER_ITERATION")
    rows = list(final_report.get("iteration_reports") or [])
    if len(rows) != ITERATIONS:
        failures.append("ITERATION_REPORT_COUNT")
    else:
        for expected_iteration, row in enumerate(rows, start=1):
            if int(row.get("iteration", -1)) != expected_iteration:
                failures.append(f"ITERATION_{expected_iteration}_IDENTITY")
            if int(row.get("roots_added", -1)) != ROOTS_PER_ITERATION:
                failures.append(f"ITERATION_{expected_iteration}_ROOTS")
            chunks = list(row.get("chance_coverage_chunks") or [])
            if len(chunks) != CHUNKS_PER_ITERATION:
                failures.append(f"ITERATION_{expected_iteration}_CHUNK_COUNT")
            elif any(int(chunk.get("roots", -1)) != ROOTS_PER_CHUNK for chunk in chunks):
                failures.append(f"ITERATION_{expected_iteration}_CHUNK_ROOTS")

    gate = validate_training_final_report(final_report) if final_report else {
        "gate_pass": False,
        "failures": ["MISSING_FINAL_REPORT"],
    }
    if failures:
        raise RuntimeError(
            f"invalid x4 final checkpoint {representation}|{domain}|{seed}: {failures}"
        )
    return final_report, gate


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract final light policies from frozen R7.5.3C x4 checkpoints")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--training-execution-sha", required=True)
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    training_sha = str(args.training_execution_sha).strip()
    if not training_sha:
        raise SystemExit("--training-execution-sha is required")
    checkpoints = sorted(args.input_root.rglob("checkpoint.pt"))
    if len(checkpoints) != 8:
        raise RuntimeError(f"expected exactly 8 x4 final checkpoints, found {len(checkpoints)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    training_rows: list[dict] = []
    seen: set[tuple[str, str, int]] = set()

    for checkpoint in checkpoints:
        payload = _load_checkpoint(checkpoint)
        representation = str(payload.get("representation"))
        domain = str(payload.get("domain"))
        seed = int(payload.get("seed", -1))
        key = (representation, domain, seed)
        if key in seen:
            raise RuntimeError(f"duplicate x4 final policy cell: {key}")
        final_report, gate = _validate_x4_final(
            payload,
            repo_root=repo_root,
            training_sha=training_sha,
        )
        seen.add(key)
        filename = f"{representation}__{domain}__{seed}.pt"
        light = {
            "schema": LIGHT_SCHEMA,
            "representation": representation,
            "domain": domain,
            "training_seed": seed,
            "action_candidate": ACTION_CANDIDATE,
            "training_execution_sha": training_sha,
            "architecture_fingerprint_sha256": MODEL_FINGERPRINTS[representation],
            "config": dict(payload["config"]),
            "parameter_count": MODEL_PARAMETER_COUNTS[representation],
            "policy_state_dict": payload["policy"],
            "final_report": final_report,
            "production_training_authorized": False,
            "ready_for_tables": False,
        }
        torch.save(light, args.output_root / filename)
        rows.append({
            key: value for key, value in light.items() if key != "policy_state_dict"
        } | {"file": filename, "x4_roots_per_seed": ROOTS_PER_SEED})
        training_rows.append({
            "representation": representation,
            "domain": domain,
            "training_seed": seed,
            "final_report_gate": gate,
            "x4_roots_per_seed": ROOTS_PER_SEED,
            "source_checkpoint": str(checkpoint),
        })

    expected = {
        (representation, domain, int(seed))
        for representation in (H2_FINAL, H3_FINAL)
        for domain in DOMAINS
        for seed in TRAINING_SEEDS
    }
    if seen != expected:
        raise RuntimeError(f"x4 light-policy inventory mismatch: missing={sorted(expected-seen)} extra={sorted(seen-expected)}")

    quality_pass = all(bool(row["final_report_gate"].get("gate_pass")) for row in training_rows)
    light_inventory = {
        "schema": SCHEMA,
        "training_execution_sha": training_sha,
        "chance_coverage_multiplier": COVERAGE_MULTIPLIER,
        "roots_per_iteration": ROOTS_PER_ITERATION,
        "roots_per_seed": ROOTS_PER_SEED,
        "expected_cells": 8,
        "observed_cells": len(rows),
        "cells": sorted(rows, key=lambda row: (row["representation"], row["domain"], row["training_seed"])),
        "representation_winner": None,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    (args.output_root / "inventory.json").write_text(
        json.dumps(light_inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Compatibility inventory for the already-frozen strategic aggregate.  Its
    # validator intentionally checks only provenance, exact 8-cell integrity and
    # the informational training-quality flag; all hard gates are independently
    # recomputed from the final reports embedded in the heldout cells.
    training_inventory = {
        "schema": TRAINING_INVENTORY_SCHEMA,
        "execution_sha": training_sha,
        "expected_cells": 8,
        "observed_cells": len(training_rows),
        "integrity_complete": True,
        "training_quality_pass": quality_pass,
        "training_quality_failures": [
            {
                "representation": row["representation"],
                "domain": row["domain"],
                "training_seed": row["training_seed"],
                "failures": list(row["final_report_gate"].get("failures") or []),
            }
            for row in training_rows
            if not row["final_report_gate"].get("gate_pass")
        ],
        "derived_from_x4_final_checkpoints": True,
        "chance_coverage_multiplier": COVERAGE_MULTIPLIER,
        "roots_per_iteration": ROOTS_PER_ITERATION,
        "roots_per_seed": ROOTS_PER_SEED,
        "cells": sorted(training_rows, key=lambda row: (row["representation"], row["domain"], row["training_seed"])),
        "representation_winner": None,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    (args.output_root / "R7_5_3C_PHASE2_TRAINING_INVENTORY.json").write_text(
        json.dumps(training_inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "training_execution_sha": training_sha,
        "observed_cells": len(rows),
        "training_quality_pass": quality_pass,
        "x4_roots_per_seed": ROOTS_PER_SEED,
        "representation_winner": None,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
