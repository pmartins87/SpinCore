from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_action_stage_contract import EXACT_OPPONENT_LEVELS
from spincore.r7_5_eval_artifacts import (
    CROSSPLAY_HANDS,
    DENSE_CACHE_SCHEMA,
    OMISSION_COUNT_PER_CELL,
    DenseCellCache,
    MemoizedPolicy,
    save_dense_cell_cache,
    state_diagnostic,
)
from spincore.r7_5_final_policy import load_finalized_action_policy
from spincore.r7_5_referee_crossplay import build_dense_crossplay_reference
from spincore.r7_5_referee_omission import build_dense_omission_cache
from spincore.r7_5_referee_states import generate_heldout_referee_states, replay_heldout_referee_state
from spincore.solver import SolverLibrary

ROOT_LEVEL = 320


def _validate_sha(value: str) -> str:
    sha = str(value)
    if len(sha) != 40 or any(ch not in "0123456789abcdef" for ch in sha):
        raise ValueError("320 dense worker requires a lowercase 40-hex execution SHA")
    return sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--domain", required=True, choices=["TRUE_HEADS_UP", "THREE_HANDED"])
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--evaluation-seed", required=True, type=int)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--cache-out", required=True)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()

    execution_sha = _validate_sha(args.execution_sha)
    torch.set_num_threads(2)
    repo = Path(args.repo_root)
    solver = SolverLibrary(args.solver)
    specs = postflop_candidate_specs(repo)
    dense_spec = specs["PF_DENSE_REFERENCE"]
    dense = load_finalized_action_policy(
        args.dense_checkpoint,
        repo_root=repo,
        expected_execution_sha=execution_sha,
        expected_root_level=ROOT_LEVEL,
        expected_candidate_id="PF_DENSE_REFERENCE",
        expected_domain=args.domain,
        expected_training_seed=args.training_seed,
    )
    q_policy = MemoizedPolicy(dense)
    descriptors = generate_heldout_referee_states(
        solver=solver,
        action_spec=dense_spec,
        policy=q_policy,
        domain=args.domain,
        training_seed=args.training_seed,
        evaluation_seed=args.evaluation_seed,
        count=OMISSION_COUNT_PER_CELL,
    )
    q_references = build_dense_omission_cache(
        solver=solver,
        descriptors=descriptors,
        dense_action_spec=dense_spec,
        dense_policy=q_policy,
        exact_opponent_levels=EXACT_OPPONENT_LEVELS,
    )
    diagnostics = []
    for descriptor in descriptors:
        state = replay_heldout_referee_state(
            solver=solver,
            action_spec=dense_spec,
            descriptor=descriptor,
        )
        try:
            diagnostics.append(state_diagnostic(state))
        finally:
            state.close()
    crossplay = build_dense_crossplay_reference(
        solver=solver,
        dense_action_spec=dense_spec,
        dense_policy=dense,
        domain=args.domain,
        training_seed=args.training_seed,
        evaluation_seed=args.evaluation_seed,
        hand_count=CROSSPLAY_HANDS[args.domain],
    )
    cache = DenseCellCache(
        schema=DENSE_CACHE_SCHEMA,
        execution_sha=execution_sha,
        domain=args.domain,
        training_seed=args.training_seed,
        evaluation_seed=args.evaluation_seed,
        descriptors=tuple(descriptors),
        q_references=tuple(q_references),
        diagnostics=tuple(diagnostics),
        crossplay_references=tuple(crossplay),
    )
    save_dense_cell_cache(
        args.cache_out,
        cache,
        exact_counts=True,
        expected_execution_sha=execution_sha,
    )
    report = {
        "schema": "SPINCORE_R7_5_4A_DENSE_CELL_REPORT_V1",
        "root_level": ROOT_LEVEL,
        "execution_sha": execution_sha,
        "domain": args.domain,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "heldout_states": len(descriptors),
        "q_references": len(q_references),
        "crossplay_reference_hands": len(crossplay),
        "memoized_dense_policy_hits": q_policy.hits,
        "memoized_dense_policy_misses": q_policy.misses,
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
