from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_representation_v3_referee_artifacts import save_heldout_v3_artifact
from spincore.r7_5_representation_v3_referee_states import (
    generate_heldout_v3_states,
    replay_heldout_v3_state,
)
from spincore.r7_5_representation_v3_stage_contract import ACTION_CANDIDATE, EVALUATION_SEEDS
from spincore.solver import SolverLibrary
from spincore_nn.codec_v3 import decode_spnniv3

SCHEMA = "SPINCORE_R7_5_3C_HELDOUT_V3_GENERATOR_REPORT_V1"
DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")
COUNT = 2048


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--evaluation-seed", type=int, choices=EVALUATION_SEEDS, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--out-artifact", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    evaluation = json.loads(
        (root / "validation" / "R7_5_3C_PHASE2_EVALUATION_FREEZE_20260815.json").read_text()
    )
    if evaluation.get("schema") != "SPINCORE_R7_5_3C_PHASE2_EVALUATION_FREEZE_V1":
        raise RuntimeError("heldout generator evaluation freeze mismatch")
    if evaluation.get("status") != "FROZEN_BEFORE_PHASE2_STRATEGIC_OUTPUTS":
        raise RuntimeError("heldout generator evaluation freeze is not immutable")
    if evaluation.get("strategic_output_seen_before_freeze") is not False:
        raise RuntimeError("heldout generator freeze was not pre-output")
    state_cfg = evaluation["state_generation"]
    if state_cfg.get("candidate_independent") is not True or state_cfg.get("training_seed_independent") is not True:
        raise RuntimeError("heldout generation independence contract drift")
    if int(state_cfg.get("heldout_states_per_domain_evaluation_seed", -1)) != COUNT:
        raise RuntimeError("heldout state count drift")
    if tuple(evaluation.get("evaluation_seeds") or ()) != EVALUATION_SEEDS:
        raise RuntimeError("heldout evaluation seed drift")
    if evaluation.get("action_candidate") != ACTION_CANDIDATE:
        raise RuntimeError("heldout action candidate drift")

    torch.set_num_threads(2)
    solver = SolverLibrary(args.solver)
    spec = postflop_candidate_specs(root)[ACTION_CANDIDATE]
    started = time.perf_counter()
    states = generate_heldout_v3_states(
        solver=solver,
        action_spec=spec,
        domain=str(args.domain),
        evaluation_seed=int(args.evaluation_seed),
        count=COUNT,
    )
    generation_seconds = time.perf_counter() - started

    streets = {0: 0, 1: 0, 2: 0, 3: 0}
    history_lengths: list[int] = []
    for descriptor in states:
        decoded = decode_spnniv3(descriptor.observation_v3)
        streets[int(decoded.categorical[1])] += 1
        history_lengths.append(int(decoded.history_len))
    if any(streets[street] < 1 for street in range(4)):
        raise RuntimeError(f"heldout corpus missed a street: {streets}")

    replay_started = time.perf_counter()
    for descriptor in states:
        state = replay_heldout_v3_state(
            solver=solver,
            action_spec=spec,
            descriptor=descriptor,
        )
        state.close()
    replay_seconds = time.perf_counter() - replay_started

    metadata = save_heldout_v3_artifact(
        args.out_artifact,
        states,
        generator_execution_sha=str(args.execution_sha),
    )
    artifact_sha256 = _sha256(args.out_artifact)
    report = {
        "schema": SCHEMA,
        "execution_sha": str(args.execution_sha),
        "domain": str(args.domain),
        "evaluation_seed": int(args.evaluation_seed),
        "count": len(states),
        "candidate_independent": True,
        "training_seed_independent": True,
        "all_states_replayed_byte_identically": True,
        "street_counts": {str(key): int(value) for key, value in streets.items()},
        "history_length": {
            "min": min(history_lengths),
            "mean": sum(history_lengths) / len(history_lengths),
            "max": max(history_lengths),
        },
        "generation_seconds": float(generation_seconds),
        "full_replay_seconds": float(replay_seconds),
        "state_payload_sha256": metadata["state_payload_sha256"],
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": int(args.out_artifact.stat().st_size),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
