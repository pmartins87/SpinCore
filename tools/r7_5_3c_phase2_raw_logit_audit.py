from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import torch

from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_representation_v3_final_policy import load_finalized_v3_policy_light
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact
from spincore_nn.models_v3_final import collate_v3_observations

SCHEMA = "SPINCORE_R7_5_3C_PHASE2_RAW_LOGIT_SENTINEL_AUDIT_V1"
COUNT = 256


def _heldout_generator_sha(path: Path) -> str:
    with gzip.open(path, "rb") as handle:
        return str(json.loads(handle.read().decode("utf-8")).get("generator_execution_sha") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent raw-logit validity guard for Phase2")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--training-execution-sha", required=True)
    parser.add_argument("--heldout-execution-sha", required=True)
    parser.add_argument("--audit-execution-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if _heldout_generator_sha(args.heldout) != args.heldout_execution_sha:
        raise RuntimeError("heldout execution SHA mismatch")
    states = load_heldout_v3_artifact(
        args.heldout,
        expected_domain=args.domain,
        expected_evaluation_seed=args.evaluation_seed,
        expected_count=2048,
    )[:COUNT]
    policy = load_finalized_v3_policy_light(
        args.policy,
        repo_root=args.repo_root,
        expected_training_execution_sha=args.training_execution_sha,
        expected_representation=args.representation,
        expected_domain=args.domain,
        expected_training_seed=args.training_seed,
    )
    batch = collate_v3_observations(
        [item.observation_v3 for item in states],
        [legal_mask(item.legal_slots) for item in states],
        with_semantics=policy.with_semantics,
        device="cpu",
    )
    policy.model.eval()
    with torch.no_grad():
        raw = policy.model(batch).detach().cpu()
    if raw.ndim != 2 or raw.shape[0] != COUNT:
        raise RuntimeError(f"unexpected raw-logit shape {tuple(raw.shape)}")
    finite = torch.isfinite(raw)
    failures = []
    for row_index in range(COUNT):
        bad = torch.nonzero(~finite[row_index], as_tuple=False).flatten().tolist()
        if bad:
            failures.append({"state_index": int(states[row_index].state_index), "nonfinite_slots": [int(x) for x in bad]})
    max_abs = float(torch.abs(raw[finite]).max()) if bool(finite.any()) else math.inf
    report = {
        "schema": SCHEMA,
        "audit_execution_sha": args.audit_execution_sha,
        "training_execution_sha": args.training_execution_sha,
        "heldout_execution_sha": args.heldout_execution_sha,
        "representation": args.representation,
        "domain": args.domain,
        "training_seed": args.training_seed,
        "evaluation_seed": args.evaluation_seed,
        "sentinel_count": COUNT,
        "all_raw_logits_finite": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "max_abs_finite_raw_logit": max_abs,
        "hard_validity_guard_pass": not failures,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
