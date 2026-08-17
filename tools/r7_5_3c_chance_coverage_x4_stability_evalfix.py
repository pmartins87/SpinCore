from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from spincore.r7_5_representation_v3_stage import FINAL_REPORT_SCHEMA
from spincore.r7_5_representation_v3_final_policy import LIGHT_SCHEMA, load_finalized_v3_policy_light
from spincore.r7_5_representation_v3_phase2_eval import cross_seed_policy_stability, validate_training_final_report
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    CROSS_SEED_MEAN_TV_MAX,
    CROSS_SEED_P95_TV_MAX,
    DOMAINS,
    EVALUATION_SEEDS,
    ITERATIONS,
    MODEL_FINGERPRINTS,
    MODEL_PARAMETER_COUNTS,
    POLICY_STEPS,
    TORCH_THREADS,
    TRAINING_SEEDS,
)

SCHEMA = "SPINCORE_R7_5_3C_CHANCE_COVERAGE_X4_STABILITY_RESULT_V1"
SOURCE_DIAGNOSTIC_SHA256 = "6655f5fcea3788a15e6a0671b00ad7d98b15e0f87f8dcdadc2ded57de5d46304"
REPRESENTATIONS = (H2_FINAL, H3_FINAL)
POLICY_COUNT = 1024
EXPECTED_ROOTS_PER_ITERATION = 256
EXPECTED_TOTAL_ROOTS = 768


def _find_heldout(root: Path, domain: str, evaluation_seed: int) -> Path:
    matches = []
    for path in root.rglob("states.json.gz"):
        try:
            payload = load_heldout_v3_artifact(path, expected_domain=domain, expected_evaluation_seed=int(evaluation_seed), expected_count=2048)
        except Exception:
            continue
        if payload:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"heldout identity mismatch for {domain}/{evaluation_seed}: {matches}")
    return matches[0]


def _extract_x4_policy_light(checkpoint_path: Path, output_path: Path, *, expected_training_execution_sha: str) -> dict:
    torch_rng = torch.get_rng_state().clone()
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    finally:
        torch.set_rng_state(torch_rng)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("not a final Phase2 V3 checkpoint")
    if payload.get("execution_sha") != str(expected_training_execution_sha):
        raise ValueError("training execution SHA mismatch during x4 light extraction")
    representation = str(payload.get("representation"))
    domain = str(payload.get("domain"))
    training_seed = int(payload.get("seed", -1))
    if representation not in REPRESENTATIONS:
        raise ValueError("unknown final representation")
    if domain not in DOMAINS:
        raise ValueError("unexpected final domain")
    if training_seed not in TRAINING_SEEDS:
        raise ValueError("unexpected final training seed")
    if payload.get("action_candidate") != ACTION_CANDIDATE:
        raise ValueError("final checkpoint action candidate mismatch")
    if payload.get("architecture_fingerprint_sha256") != MODEL_FINGERPRINTS[representation]:
        raise ValueError("final checkpoint architecture fingerprint mismatch")
    if bool(payload.get("production_training_authorized")) or bool(payload.get("ready_for_tables")):
        raise ValueError("final checkpoint illegally authorizes production/table use")
    progress = dict(payload.get("progress") or {})
    if progress.get("phase") != "post_policy_fit":
        raise ValueError("checkpoint is not finalized after AveragePolicy fit")
    if int(progress.get("iteration", -1)) != ITERATIONS:
        raise ValueError("final checkpoint iteration mismatch")
    if int(progress.get("policy_optimizer_step", -1)) != POLICY_STEPS:
        raise ValueError("final checkpoint policy optimizer step mismatch")
    extra = dict(payload.get("extra") or {})
    final_report = dict(extra.get("final_report") or {})
    if final_report.get("schema") != FINAL_REPORT_SCHEMA:
        raise ValueError("final checkpoint missing final report")
    if final_report.get("representation") != representation or final_report.get("domain") != domain:
        raise ValueError("final report identity mismatch")
    if int(final_report.get("training_seed", -1)) != training_seed:
        raise ValueError("final report training seed mismatch")
    if int(final_report.get("iterations", -1)) != ITERATIONS:
        raise ValueError("final report iteration count mismatch")
    if int(final_report.get("roots", -1)) != EXPECTED_TOTAL_ROOTS:
        raise ValueError("x4 final report root count mismatch")
    if int(final_report.get("average_policy_optimizer_steps", -1)) != POLICY_STEPS:
        raise ValueError("final report policy optimizer step mismatch")
    iteration_reports = list(final_report.get("iteration_reports") or [])
    if len(iteration_reports) != ITERATIONS:
        raise ValueError("x4 final report iteration report count mismatch")
    for expected_iteration, row in enumerate(iteration_reports, start=1):
        if int(row.get("iteration", -1)) != expected_iteration:
            raise ValueError("x4 final report iteration identity mismatch")
        if int(row.get("roots_added", -1)) != EXPECTED_ROOTS_PER_ITERATION:
            raise ValueError("x4 final report roots-per-iteration mismatch")
    light = {
        "schema": LIGHT_SCHEMA,
        "representation": representation,
        "domain": domain,
        "training_seed": training_seed,
        "action_candidate": ACTION_CANDIDATE,
        "training_execution_sha": str(expected_training_execution_sha),
        "architecture_fingerprint_sha256": MODEL_FINGERPRINTS[representation],
        "config": dict(payload["config"]),
        "parameter_count": MODEL_PARAMETER_COUNTS[representation],
        "policy_state_dict": payload["policy"],
        "final_report": final_report,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(light, output_path)
    return {k: v for k, v in light.items() if k != "policy_state_dict"}


def _extract_policies(input_root: Path, output_root: Path, training_sha: str):
    checkpoints = sorted(input_root.rglob("checkpoint.pt"))
    if len(checkpoints) != 8:
        raise RuntimeError(f"expected exactly 8 x4 final checkpoints, found {len(checkpoints)}")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = {}
    for index, checkpoint in enumerate(checkpoints):
        temporary = output_root / f"extracting_{index}.pt"
        metadata = _extract_x4_policy_light(checkpoint, temporary, expected_training_execution_sha=training_sha)
        key = (str(metadata["representation"]), str(metadata["domain"]), int(metadata["training_seed"]))
        if key in rows:
            raise RuntimeError(f"duplicate x4 final policy {key}")
        destination = output_root / f"{key[0]}__{key[1]}__{key[2]}.pt"
        temporary.replace(destination)
        rows[key] = destination
    expected = {(rep, domain, int(seed)) for rep in REPRESENTATIONS for domain in DOMAINS for seed in TRAINING_SEEDS}
    if set(rows) != expected:
        raise RuntimeError(f"x4 final policy inventory mismatch missing={sorted(expected-set(rows))} extra={sorted(set(rows)-expected)}")
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
    return {**gate, "expected_roots_per_iteration": EXPECTED_ROOTS_PER_ITERATION, "expected_total_roots": EXPECTED_TOTAL_ROOTS, "shape_failures": shape_failures, "gate_pass": bool(gate["gate_pass"] and not shape_failures)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--training-execution-sha", required=True)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    evidence = json.loads(args.source_evidence.read_text(encoding="utf-8"))
    if evidence.get("schema") != "SPINCORE_R7_5_3C_SAMPLING_SPLIT_EVIDENCE_V1" or evidence.get("classification") != "DECK_CHANCE_DOMINANT":
        raise RuntimeError("wrong source diagnostic evidence")
    if evidence.get("source_run", {}).get("result_file_sha256") != SOURCE_DIAGNOSTIC_SHA256:
        raise RuntimeError("source diagnostic result hash drift")
    if CROSS_SEED_MEAN_TV_MAX != 0.15 or CROSS_SEED_P95_TV_MAX != 0.35:
        raise RuntimeError("hard cross-seed gate drift")
    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("torch thread contract drift")

    light_root = args.out.parent / "light_policies"
    policy_paths = _extract_policies(args.input_root, light_root, str(args.training_execution_sha))
    policies = {}
    training_gates = []
    for key, path in sorted(policy_paths.items()):
        rep, domain, seed = key
        policy = load_finalized_v3_policy_light(path, repo_root=args.repo_root, expected_training_execution_sha=str(args.training_execution_sha), expected_representation=rep, expected_domain=domain, expected_training_seed=int(seed))
        policies[key] = policy
        training_gates.append({"representation": rep, "domain": domain, "training_seed": int(seed), **_validate_training_shape(policy)})

    seed_a, seed_b = map(int, TRAINING_SEEDS)
    stability_rows = []
    for rep in REPRESENTATIONS:
        for domain in DOMAINS:
            for evaluation_seed in EVALUATION_SEEDS:
                heldout_path = _find_heldout(args.heldout_root, domain, int(evaluation_seed))
                descriptors = load_heldout_v3_artifact(heldout_path, expected_domain=domain, expected_evaluation_seed=int(evaluation_seed), expected_count=2048)[:POLICY_COUNT]
                left = policies[(rep, domain, seed_a)].batch_probabilities([x.observation_v3 for x in descriptors], [x.legal_slots for x in descriptors])
                right = policies[(rep, domain, seed_b)].batch_probabilities([x.observation_v3 for x in descriptors], [x.legal_slots for x in descriptors])
                stability_rows.append({"representation": rep, "domain": domain, "evaluation_seed": int(evaluation_seed), "training_seed_pair": [seed_a, seed_b], "heldout_state_indices": [int(x.state_index) for x in descriptors], "metric": cross_seed_policy_stability(left, right)})

    training_pass = all(bool(row["gate_pass"]) for row in training_gates)
    stability_pass = all(bool(row["metric"]["gate_pass"]) for row in stability_rows)
    all_pass = bool(training_pass and stability_pass)
    result = {
        "schema": SCHEMA,
        "status": "STABILITY_PASS" if all_pass else "STABILITY_BLOCKED",
        "purpose": "Winner-independent hard stability readmission after x4 independent chance coverage. Mechanical extraction compatibility fix only; no representation selection.",
        "training_execution_sha": str(args.training_execution_sha),
        "source_diagnostic_result_sha256": SOURCE_DIAGNOSTIC_SHA256,
        "chance_coverage": {"multiplier": 4, "roots_per_iteration": 256, "iterations": 3, "roots_per_seed": 768, "independent_training_seeds": list(map(int, TRAINING_SEEDS)), "production_deck_seed_semantics_preserved": True},
        "frozen_hard_gates": {"cross_seed_mean_tv_max": CROSS_SEED_MEAN_TV_MAX, "cross_seed_p95_tv_max": CROSS_SEED_P95_TV_MAX, "all_local_training_gates_required": True},
        "training_gates": training_gates,
        "cross_seed_stability": stability_rows,
        "summary": {"training_cells": len(training_gates), "training_gate_pass_count": sum(bool(r["gate_pass"]) for r in training_gates), "cross_seed_rows": len(stability_rows), "cross_seed_gate_pass_count": sum(bool(r["metric"]["gate_pass"]) for r in stability_rows), "all_local_training_gates_pass": training_pass, "all_cross_seed_gates_pass": stability_pass, "stability_readmission_pass": all_pass},
        "next_action_if_pass": "Freeze and run the complete original H2/H3 strategic Phase 2 evaluation on these x4-trained policies before any representation winner is admitted.",
        "next_action_if_blocked": "Remain BLOCKED; do not relax gates. Apply at most one final winner-independent remediation under the precommitted rule before mandatory decision.",
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
