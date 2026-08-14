from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_eval_artifacts import (
    CANDIDATE_CELL_SCHEMA,
    EXPECTED_EXECUTION_SHA,
    CandidateCellEvidence,
    expected_candidate_crossplay_samples,
    load_dense_cell_cache,
    save_candidate_cell_evidence,
    summarize_omission,
)
from spincore.r7_5_final_policy import load_finalized_action_policy
from spincore.r7_5_referee_crossplay import candidate_seats, score_candidate_from_crossplay_reference
from spincore.r7_5_referee_omission import score_candidate_from_omission_cache
from spincore.solver import SolverLibrary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--dense-cache", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--domain", required=True, choices=["TRUE_HEADS_UP", "THREE_HANDED"])
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--evaluation-seed", required=True, type=int)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--evidence-out", required=True)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()

    if args.execution_sha != EXPECTED_EXECUTION_SHA:
        raise ValueError("candidate evaluation worker execution SHA differs from frozen training run")
    torch.set_num_threads(2)
    repo = Path(args.repo_root)
    solver = SolverLibrary(args.solver)
    specs = postflop_candidate_specs(repo)
    if args.candidate not in specs or not specs[args.candidate].eligible_to_win:
        raise ValueError("candidate evaluation worker accepts only production-eligible PF0-PF4 candidates")
    dense_spec = specs["PF_DENSE_REFERENCE"]
    candidate_spec = specs[args.candidate]
    cache = load_dense_cell_cache(args.dense_cache, exact_counts=True)
    if (
        cache.domain != args.domain
        or cache.training_seed != args.training_seed
        or cache.evaluation_seed != args.evaluation_seed
    ):
        raise ValueError("candidate worker dense-cache cell identity mismatch")
    dense = load_finalized_action_policy(
        args.dense_checkpoint,
        repo_root=repo,
        expected_execution_sha=args.execution_sha,
        expected_candidate_id="PF_DENSE_REFERENCE",
        expected_domain=args.domain,
        expected_training_seed=args.training_seed,
    )
    candidate = load_finalized_action_policy(
        args.candidate_checkpoint,
        repo_root=repo,
        expected_execution_sha=args.execution_sha,
        expected_candidate_id=args.candidate,
        expected_domain=args.domain,
        expected_training_seed=args.training_seed,
    )

    omission_results = score_candidate_from_omission_cache(
        solver=solver,
        descriptors=cache.descriptors,
        references=cache.q_references,
        dense_action_spec=dense_spec,
        candidate_action_spec=candidate_spec,
    )
    omission_samples = tuple(float(row.omission) for row in omission_results)
    crossplay_samples: list[float] = []
    for seat in candidate_seats(args.domain):
        crossplay_samples.extend(
            score_candidate_from_crossplay_reference(
                solver=solver,
                references=cache.crossplay_references,
                dense_action_spec=dense_spec,
                dense_policy=dense,
                candidate_action_spec=candidate_spec,
                candidate_policy=candidate,
                domain=args.domain,
                training_seed=args.training_seed,
                evaluation_seed=args.evaluation_seed,
                candidate_seat=seat,
            )
        )
    if len(crossplay_samples) != expected_candidate_crossplay_samples(args.domain):
        raise RuntimeError("candidate crossplay sample count drift")
    crossplay_mean = float(sum(crossplay_samples) / len(crossplay_samples))
    evidence = CandidateCellEvidence(
        schema=CANDIDATE_CELL_SCHEMA,
        execution_sha=args.execution_sha,
        candidate_id=args.candidate,
        domain=args.domain,
        training_seed=args.training_seed,
        evaluation_seed=args.evaluation_seed,
        omission_samples=omission_samples,
        crossplay_samples=tuple(crossplay_samples),
        omission_summary=summarize_omission(omission_samples, cache.diagnostics),
        crossplay_mean=crossplay_mean,
    )
    save_candidate_cell_evidence(args.evidence_out, evidence, exact_counts=True)
    report = {
        "schema": "SPINCORE_R7_5_4A_CANDIDATE_CELL_REPORT_V1",
        "execution_sha": args.execution_sha,
        "candidate_id": args.candidate,
        "domain": args.domain,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "omission_count": len(omission_samples),
        "omission_mean": evidence.omission_summary["overall"]["mean"],
        "omission_p95": evidence.omission_summary["overall"]["p95"],
        "crossplay_count": len(crossplay_samples),
        "crossplay_mean": crossplay_mean,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
