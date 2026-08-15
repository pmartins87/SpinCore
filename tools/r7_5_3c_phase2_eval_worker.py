from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Sequence

import torch

from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_crossplay import (
    live_candidate_seats,
    mirrored_h3_vs_h2_scores,
    run_v3_crossplay_hands,
    uniform_pf0_policy,
)
from spincore.r7_5_representation_v3_final_policy import load_finalized_v3_policy_light
from spincore.r7_5_representation_v3_local_deviation import evaluate_local_deviation_heldout
from spincore.r7_5_representation_v3_phase2_eval import (
    validate_sentinel_vectors,
    validate_training_final_report,
)
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact
from spincore.r7_5_representation_v3_referee_states import replay_heldout_v3_state
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    EVALUATION_SEEDS,
    TORCH_THREADS,
    TRAINING_SEEDS,
)
from spincore.solver import SolverLibrary
from spincore_nn.models_v3_final import collate_v3_observations

HELDOUT_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_HELDOUT_EVAL_CELL_V1"
COMMONREF_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_COMMONREF_CELL_V1"
PAIRWISE_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_PAIRWISE_CELL_V1"
HELDOUT_COUNT = 2048
POLICY_AND_LOCALDEV_COUNT = 1024
SENTINEL_COUNT = 256


def _write_json_gz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as handle:
            handle.write(encoded)


def _heldout_generator_sha(path: Path) -> str:
    with gzip.open(path, "rb") as handle:
        payload = json.loads(handle.read().decode("utf-8"))
    return str(payload.get("generator_execution_sha") or "")


def _load_policy(
    path: Path,
    *,
    repo_root: Path,
    training_sha: str,
    representation: str,
    domain: str,
    training_seed: int,
):
    return load_finalized_v3_policy_light(
        path,
        repo_root=repo_root,
        expected_training_execution_sha=training_sha,
        expected_representation=representation,
        expected_domain=domain,
        expected_training_seed=training_seed,
    )


def _sentinel_logits(policy, descriptors: Sequence) -> tuple[tuple[float, ...], ...]:
    observations = [item.observation_v3 for item in descriptors]
    legal_sets = [item.legal_slots for item in descriptors]
    batch = collate_v3_observations(
        observations,
        [legal_mask(row) for row in legal_sets],
        with_semantics=policy.with_semantics,
        device="cpu",
    )
    policy.model.eval()
    with torch.no_grad():
        logits = policy.model(batch).masked_fill(~batch["legal"], -1e9).detach().cpu().tolist()
    return tuple(tuple(float(value) for value in row) for row in logits)


def _heldout_mode(args, solver, spec) -> dict:
    if _heldout_generator_sha(args.heldout_file) != args.heldout_execution_sha:
        raise RuntimeError("heldout generator execution SHA mismatch")
    descriptors = load_heldout_v3_artifact(
        args.heldout_file,
        expected_domain=args.domain,
        expected_evaluation_seed=args.evaluation_seed,
        expected_count=HELDOUT_COUNT,
    )
    policy = _load_policy(
        args.policy,
        repo_root=args.repo_root,
        training_sha=args.training_execution_sha,
        representation=args.representation,
        domain=args.domain,
        training_seed=args.training_seed,
    )
    final_report_gate = validate_training_final_report(policy.final_report)

    sentinel_descriptors = descriptors[:SENTINEL_COUNT]
    for descriptor in sentinel_descriptors:
        state = replay_heldout_v3_state(solver=solver, action_spec=spec, descriptor=descriptor)
        state.close()
    sentinel_probabilities = policy.batch_probabilities(
        [item.observation_v3 for item in sentinel_descriptors],
        [item.legal_slots for item in sentinel_descriptors],
    )
    sentinel_logits = _sentinel_logits(policy, sentinel_descriptors)
    sentinel_gate = validate_sentinel_vectors(
        probabilities=sentinel_probabilities,
        legal_sets=[item.legal_slots for item in sentinel_descriptors],
        logits=sentinel_logits,
    )
    sentinel_gate["spnniv3_replay_byte_identity_count"] = SENTINEL_COUNT

    audit_descriptors = descriptors[:POLICY_AND_LOCALDEV_COUNT]
    policy_rows = policy.batch_probabilities(
        [item.observation_v3 for item in audit_descriptors],
        [item.legal_slots for item in audit_descriptors],
    )
    local = evaluate_local_deviation_heldout(
        solver=solver,
        descriptors=audit_descriptors,
        action_spec=spec,
        candidate_policy=policy,
        exact_opponent_levels=2,
    )
    if tuple(item.state_index for item in local) != tuple(item.state_index for item in audit_descriptors):
        raise RuntimeError("local-deviation heldout identity drift")
    return {
        "schema": HELDOUT_SCHEMA,
        "evaluator_execution_sha": args.evaluator_execution_sha,
        "training_execution_sha": args.training_execution_sha,
        "heldout_execution_sha": args.heldout_execution_sha,
        "representation": args.representation,
        "domain": args.domain,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "policy_state_indices": [int(item.state_index) for item in audit_descriptors],
        "policy_rows": [list(row) for row in policy_rows],
        "local_deviation_gains": [float(item.gain) for item in local],
        "final_report": policy.final_report,
        "final_report_gate": final_report_gate,
        "sentinel_gate": sentinel_gate,
        "rng_namespace": "SpinCore|R7.5.3C|PHASE2|REFV1",
        "candidate_or_training_seed_in_referee_rng_key": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def _commonref_mode(args, solver, spec) -> dict:
    policy = _load_policy(
        args.policy,
        repo_root=args.repo_root,
        training_sha=args.training_execution_sha,
        representation=args.representation,
        domain=args.domain,
        training_seed=args.training_seed,
    )
    hand_count = 20000 if args.domain == "TRUE_HEADS_UP" else 10000
    uniform = uniform_pf0_policy
    reference = run_v3_crossplay_hands(
        solver=solver,
        action_spec=spec,
        domain=args.domain,
        evaluation_seed=args.evaluation_seed,
        hand_count=hand_count,
        seat_policies=(uniform, uniform, uniform),
        rng_scope="commonref",
    )
    seats = {}
    for seat in live_candidate_seats(args.domain):
        seat_policies = [uniform, uniform, uniform]
        seat_policies[int(seat)] = policy
        tested = run_v3_crossplay_hands(
            solver=solver,
            action_spec=spec,
            domain=args.domain,
            evaluation_seed=args.evaluation_seed,
            hand_count=hand_count,
            seat_policies=tuple(seat_policies),
            rng_scope="commonref",
        )
        scores = tuple(
            float(test[int(seat)] - ref[int(seat)])
            for ref, test in zip(reference, tested)
        )
        if len(scores) != hand_count:
            raise RuntimeError("common-reference score count drift")
        seats[str(seat)] = [float(value) for value in scores]
    return {
        "schema": COMMONREF_SCHEMA,
        "evaluator_execution_sha": args.evaluator_execution_sha,
        "training_execution_sha": args.training_execution_sha,
        "representation": args.representation,
        "domain": args.domain,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "hands_per_candidate_seat": hand_count,
        "seats": seats,
        "selection_role": "DIAGNOSTIC_ONLY",
        "rng_namespace": "SpinCore|R7.5.3C|PHASE2|REFV1",
        "candidate_or_training_seed_in_referee_rng_key": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def _pairwise_mode(args, solver, spec) -> dict:
    h2 = _load_policy(
        args.h2_policy,
        repo_root=args.repo_root,
        training_sha=args.training_execution_sha,
        representation=H2_FINAL,
        domain=args.domain,
        training_seed=args.h2_training_seed,
    )
    h3 = _load_policy(
        args.h3_policy,
        repo_root=args.repo_root,
        training_sha=args.training_execution_sha,
        representation=H3_FINAL,
        domain=args.domain,
        training_seed=args.h3_training_seed,
    )
    hand_count = 20000 if args.domain == "TRUE_HEADS_UP" else 10000
    seats = {}
    for seat in live_candidate_seats(args.domain):
        scores = mirrored_h3_vs_h2_scores(
            solver=solver,
            action_spec=spec,
            h2_policy=h2,
            h3_policy=h3,
            domain=args.domain,
            evaluation_seed=args.evaluation_seed,
            candidate_seat=seat,
            hand_count=hand_count,
        )
        if len(scores) != hand_count:
            raise RuntimeError("pairwise score count drift")
        seats[str(seat)] = [float(value) for value in scores]
    return {
        "schema": PAIRWISE_SCHEMA,
        "evaluator_execution_sha": args.evaluator_execution_sha,
        "training_execution_sha": args.training_execution_sha,
        "domain": args.domain,
        "evaluation_seed": args.evaluation_seed,
        "h2_training_seed": args.h2_training_seed,
        "h3_training_seed": args.h3_training_seed,
        "hands_per_candidate_seat": hand_count,
        "seats": seats,
        "score_perspective": "H3",
        "rng_namespace": "SpinCore|R7.5.3C|PHASE2|REFV1",
        "candidate_or_training_seed_in_referee_rng_key": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen R7.5.3C Phase2 strategic evaluation worker")
    parser.add_argument("--mode", choices=("heldout", "commonref", "pairwise"), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--domain", choices=("TRUE_HEADS_UP", "THREE_HANDED"), required=True)
    parser.add_argument("--evaluation-seed", type=int, choices=EVALUATION_SEEDS, required=True)
    parser.add_argument("--training-execution-sha", required=True)
    parser.add_argument("--evaluator-execution-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--representation", choices=(H2_FINAL, H3_FINAL))
    parser.add_argument("--training-seed", type=int, choices=TRAINING_SEEDS)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--heldout-file", type=Path)
    parser.add_argument("--heldout-execution-sha")
    parser.add_argument("--h2-training-seed", type=int, choices=TRAINING_SEEDS)
    parser.add_argument("--h3-training-seed", type=int, choices=TRAINING_SEEDS)
    parser.add_argument("--h2-policy", type=Path)
    parser.add_argument("--h3-policy", type=Path)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()

    if not args.training_execution_sha.strip() or not args.evaluator_execution_sha.strip():
        raise SystemExit("immutable execution SHAs are required")
    if args.mode in ("heldout", "commonref"):
        if args.representation is None or args.training_seed is None or args.policy is None:
            raise SystemExit("heldout/commonref requires representation, training-seed and policy")
    if args.mode == "heldout" and (args.heldout_file is None or not args.heldout_execution_sha):
        raise SystemExit("heldout mode requires heldout artifact and execution SHA")
    if args.mode == "pairwise":
        if None in (args.h2_training_seed, args.h3_training_seed) or args.h2_policy is None or args.h3_policy is None:
            raise SystemExit("pairwise mode requires both frozen seed policies")

    torch.set_num_threads(TORCH_THREADS)
    solver = SolverLibrary(args.solver)
    spec = postflop_candidate_specs(args.repo_root)[ACTION_CANDIDATE]
    if args.mode == "heldout":
        payload = _heldout_mode(args, solver, spec)
    elif args.mode == "commonref":
        payload = _commonref_mode(args, solver, spec)
    else:
        payload = _pairwise_mode(args, solver, spec)
    _write_json_gz(args.out, payload)
    print(json.dumps({key: payload[key] for key in payload if key not in ("policy_rows", "local_deviation_gains", "seats", "final_report")}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
