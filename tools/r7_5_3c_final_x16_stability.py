from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_final_policy import (
    extract_final_v3_policy_light,
    load_finalized_v3_policy_light,
)
from spincore.r7_5_representation_v3_phase2_eval import (
    cross_seed_policy_stability,
    validate_training_final_report,
)
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact
from spincore.r7_5_representation_v3_stage_contract import (
    CROSS_SEED_MEAN_TV_MAX,
    CROSS_SEED_P95_TV_MAX,
    DOMAINS,
    EVALUATION_SEEDS,
    TORCH_THREADS,
    TRAINING_SEEDS,
)

SCHEMA = "SPINCORE_R7_5_3C_FINAL_CHANCE_COVERAGE_X16_STABILITY_RESULT_V1"
FREEZE_SCHEMA = "SPINCORE_R7_5_3C_FINAL_CONTINGENCY_X16_FREEZE_V1"
HELDOUT_SHA = "dfe5f83742495a457e92b29f97db5d3b631bca22"
REPRESENTATIONS = (H2_FINAL, H3_FINAL)
POLICY_COUNT = 1024
EXPECTED_ROOTS_PER_ITERATION = 1024
EXPECTED_TOTAL_ROOTS = 3072


def _find_heldout(root: Path, domain: str, evaluation_seed: int) -> Path:
    matches = []
    for path in root.rglob("states.json.gz"):
        try:
            payload = load_heldout_v3_artifact(
                path,
                expected_domain=domain,
                expected_evaluation_seed=int(evaluation_seed),
                expected_count=2048,
            )
        except Exception:
            continue
        if payload:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"heldout identity mismatch for {domain}/{evaluation_seed}: {matches}")
    return matches[0]


def _extract_policies(input_root: Path, output_root: Path, training_sha: str):
    checkpoints = sorted(input_root.rglob("checkpoint.pt"))
    finals = []
    for checkpoint in checkpoints:
        report = checkpoint.parent / "report.json"
        if not report.exists():
            continue
        payload = json.loads(report.read_text(encoding="utf-8"))
        if bool(payload.get("finalized")):
            finals.append(checkpoint)
    if len(finals) != 8:
        raise RuntimeError(f"expected exactly 8 x16 final checkpoints, found {len(finals)}")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = {}
    for index, checkpoint in enumerate(finals):
        temporary = output_root / f"extracting_{index}.pt"
        metadata = extract_final_v3_policy_light(
            checkpoint,
            temporary,
            expected_training_execution_sha=training_sha,
        )
        key = (
            str(metadata["representation"]),
            str(metadata["domain"]),
            int(metadata["training_seed"]),
        )
        if key in rows:
            raise RuntimeError(f"duplicate x16 final policy {key}")
        destination = output_root / f"{key[0]}__{key[1]}__{key[2]}.pt"
        temporary.replace(destination)
        rows[key] = destination
    expected = {
        (rep, domain, int(seed))
        for rep in REPRESENTATIONS
        for domain in DOMAINS
        for seed in TRAINING_SEEDS
    }
    if set(rows) != expected:
        raise RuntimeError(f"x16 final policy inventory mismatch missing={sorted(expected-set(rows))} extra={sorted(set(rows)-expected)}")
    return rows


def _validate_training_shape(policy) -> dict:
    gate = validate_training_final_report(policy.final_report)
    report = policy.final_report
    shape_failures = []
    if int(report.get("roots", -1)) != EXPECTED_TOTAL_ROOTS:
        shape_failures.append("TOTAL_ROOTS")
    iteration_reports = list(report.get("iteration_reports") or [])
    if len(iteration_reports) != 3:
        shape_failures.append("ITERATION_REPORT_COUNT")
    else:
        for expected_iteration, row in enumerate(iteration_reports, start=1):
            if int(row.get("iteration", -1)) != expected_iteration:
                shape_failures.append(f"ITERATION_{expected_iteration}_IDENTITY")
            if int(row.get("roots_added", -1)) != EXPECTED_ROOTS_PER_ITERATION:
                shape_failures.append(f"ITERATION_{expected_iteration}_ROOTS")
            chunks = list(row.get("chance_coverage_chunks") or [])
            if len(chunks) != 16:
                shape_failures.append(f"ITERATION_{expected_iteration}_CHUNK_COUNT")
            elif any(int(chunk.get("roots", -1)) != 64 for chunk in chunks):
                shape_failures.append(f"ITERATION_{expected_iteration}_CHUNK_ROOTS")
    return {
        **gate,
        "expected_roots_per_iteration": EXPECTED_ROOTS_PER_ITERATION,
        "expected_total_roots": EXPECTED_TOTAL_ROOTS,
        "shape_failures": shape_failures,
        "gate_pass": bool(gate["gate_pass"] and not shape_failures),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hard stability readout for the final R7.5.3C x16 chance-coverage contingency")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--training-execution-sha", required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise RuntimeError("wrong final x16 freeze schema")
    if freeze.get("status") != "FROZEN_BEFORE_X16_OUTPUTS":
        raise RuntimeError("final x16 freeze is not authoritative")
    remediation = dict(freeze.get("remediation") or {})
    if int(remediation.get("roots_per_iteration_effective", -1)) != EXPECTED_ROOTS_PER_ITERATION:
        raise RuntimeError("x16 freeze roots-per-iteration drift")
    if int(remediation.get("roots_per_seed", -1)) != EXPECTED_TOTAL_ROOTS:
        raise RuntimeError("x16 freeze roots-per-seed drift")
    if list(map(int, remediation.get("training_seeds") or [])) != list(map(int, TRAINING_SEEDS)):
        raise RuntimeError("x16 freeze training-seed drift")

    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("torch thread contract drift")

    light_root = args.out.parent / "light_policies"
    policy_paths = _extract_policies(args.input_root, light_root, str(args.training_execution_sha))
    policies = {}
    training_gates = []
    for key, path in sorted(policy_paths.items()):
        rep, domain, seed = key
        policy = load_finalized_v3_policy_light(
            path,
            repo_root=args.repo_root,
            expected_training_execution_sha=str(args.training_execution_sha),
            expected_representation=rep,
            expected_domain=domain,
            expected_training_seed=int(seed),
        )
        policies[key] = policy
        training_gates.append({
            "representation": rep,
            "domain": domain,
            "training_seed": int(seed),
            **_validate_training_shape(policy),
        })

    seed_a, seed_b = map(int, TRAINING_SEEDS)
    stability_rows = []
    for rep in REPRESENTATIONS:
        for domain in DOMAINS:
            for evaluation_seed in EVALUATION_SEEDS:
                heldout_path = _find_heldout(args.heldout_root, domain, int(evaluation_seed))
                descriptors = load_heldout_v3_artifact(
                    heldout_path,
                    expected_domain=domain,
                    expected_evaluation_seed=int(evaluation_seed),
                    expected_count=2048,
                )[:POLICY_COUNT]
                indices = [int(item.state_index) for item in descriptors]
                left = policies[(rep, domain, seed_a)].batch_probabilities(
                    [item.observation_v3 for item in descriptors],
                    [item.legal_slots for item in descriptors],
                )
                right = policies[(rep, domain, seed_b)].batch_probabilities(
                    [item.observation_v3 for item in descriptors],
                    [item.legal_slots for item in descriptors],
                )
                metric = cross_seed_policy_stability(left, right)
                stability_rows.append({
                    "representation": rep,
                    "domain": domain,
                    "evaluation_seed": int(evaluation_seed),
                    "training_seed_pair": [seed_a, seed_b],
                    "heldout_state_indices": indices,
                    "metric": metric,
                })

    training_pass = all(bool(row["gate_pass"]) for row in training_gates)
    stability_pass = all(bool(row["metric"]["gate_pass"]) for row in stability_rows)
    all_pass = bool(training_pass and stability_pass)
    result = {
        "schema": SCHEMA,
        "status": "STABILITY_PASS" if all_pass else "STABILITY_BLOCKED_FINAL",
        "purpose": "Final winner-independent hard stability readmission after x16 independent chance coverage. This result cannot select H2/H3 by itself.",
        "training_execution_sha": str(args.training_execution_sha),
        "chance_coverage": {
            "multiplier": 16,
            "roots_per_iteration": EXPECTED_ROOTS_PER_ITERATION,
            "iterations": 3,
            "roots_per_seed": EXPECTED_TOTAL_ROOTS,
            "independent_training_seeds": list(map(int, TRAINING_SEEDS)),
            "production_deck_seed_semantics_preserved": True,
        },
        "frozen_hard_gates": {
            "cross_seed_mean_tv_max": CROSS_SEED_MEAN_TV_MAX,
            "cross_seed_p95_tv_max": CROSS_SEED_P95_TV_MAX,
            "all_local_training_gates_required": True,
        },
        "training_gates": training_gates,
        "cross_seed_stability": stability_rows,
        "summary": {
            "training_cells": len(training_gates),
            "training_gate_pass_count": sum(bool(row["gate_pass"]) for row in training_gates),
            "cross_seed_rows": len(stability_rows),
            "cross_seed_gate_pass_count": sum(bool(row["metric"]["gate_pass"]) for row in stability_rows),
            "all_local_training_gates_pass": training_pass,
            "all_cross_seed_gates_pass": stability_pass,
            "stability_readmission_pass": all_pass,
        },
        "next_action_if_pass": "Persist/certify the exact result, then run the complete already-frozen H2/H3 strategic Phase-2 evaluation on these x16 policies and apply the existing selection cascade.",
        "next_action_if_blocked": "Close R7.5.3 FAIL/BLOCKED. No further R7.5.3 remediation is permitted; require an explicit architecture/fallback decision before downstream production work.",
        "representation_winner": None,
        "selection_rule_changed": False,
        "changes_frozen_thresholds": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], **result["summary"]}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
