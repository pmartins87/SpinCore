from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_action_cross_seed import build_cross_seed_common_corpus, cross_seed_policy_stability
from spincore.r7_5_action_stage_contract import POSTFLOP_TRAINING_SEEDS
from spincore.r7_5_final_policy import load_finalized_action_policy
from spincore.solver import SolverLibrary

ROOT_LEVEL = 320


def _validate_sha(value: str) -> str:
    sha = str(value)
    if len(sha) != 40 or any(ch not in "0123456789abcdef" for ch in sha):
        raise ValueError("320 cross-seed worker requires a lowercase 40-hex execution SHA")
    return sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--domain", required=True, choices=["TRUE_HEADS_UP", "THREE_HANDED"])
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()

    execution_sha = _validate_sha(args.execution_sha)
    if len(args.checkpoint) != len(POSTFLOP_TRAINING_SEEDS):
        raise ValueError("cross-seed worker requires exactly three finalized checkpoints")
    torch.set_num_threads(2)
    repo = Path(args.repo_root)
    solver = SolverLibrary(args.solver)
    specs = postflop_candidate_specs(repo)
    if args.candidate not in specs:
        raise ValueError("unknown postflop action candidate")
    policies = [
        load_finalized_action_policy(
            path,
            repo_root=repo,
            expected_execution_sha=execution_sha,
            expected_root_level=ROOT_LEVEL,
            expected_candidate_id=args.candidate,
            expected_domain=args.domain,
        )
        for path in args.checkpoint
    ]
    by_seed = {policy.training_seed: policy for policy in policies}
    if set(by_seed) != set(POSTFLOP_TRAINING_SEEDS):
        raise ValueError("cross-seed checkpoint seed set differs from frozen training seeds")
    corpus = build_cross_seed_common_corpus(
        solver=solver,
        dense_action_spec=specs["PF_DENSE_REFERENCE"],
        domain=args.domain,
    )
    report = cross_seed_policy_stability(
        solver=solver,
        descriptors=corpus,
        dense_action_spec=specs["PF_DENSE_REFERENCE"],
        candidate_action_spec=specs[args.candidate],
        policies_by_seed=by_seed,
        candidate_id=args.candidate,
        domain=args.domain,
    )
    report["execution_sha"] = execution_sha
    report["root_level"] = ROOT_LEVEL
    report["seed_reports"] = [by_seed[seed].final_report for seed in POSTFLOP_TRAINING_SEEDS]
    report["common_corpus_generator"] = "PF_DENSE_REFERENCE_UNIFORM_POLICY_INDEPENDENT_OF_TRAINED_POLICIES"
    report["production_training_authorized"] = False
    report["ready_for_tables"] = False
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("candidate_id", "domain", "mean_tv", "p95_tv", "gate_pass")}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
