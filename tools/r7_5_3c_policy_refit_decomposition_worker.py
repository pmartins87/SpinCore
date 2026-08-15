from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch

from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL, make_representation_v3_bundle
from spincore.r7_5_representation_v3_checkpoint import load_representation_v3_checkpoint
from spincore.r7_5_representation_v3_final_policy import FinalizedV3Policy
from spincore.r7_5_representation_v3_fit import audit_v3_policy_model
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    BATCH_SIZE,
    EVALUATION_SEEDS,
    LEARNING_RATE,
    MODEL_FINGERPRINTS,
    POLICY_STEPS,
    POLICY_TV_MAX,
    TORCH_THREADS,
    TRAINING_SEEDS,
)
from spincore_nn.models_v3_final import collate_v3_observations
from spincore_nn.training import train_step

SCHEMA = "SPINCORE_R7_5_3C_POLICY_REFIT_DECOMPOSITION_CELL_V1"
TRAINING_SHA = "9b0ccc207135c3adaec76ea87de8ec21f7415957"
HELDOUT_SHA = "dfe5f83742495a457e92b29f97db5d3b631bca22"
POLICY_COUNT = 1024
AUDIT_SEED = 0x53A81EED


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_gz(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def find_heldout(root: Path, domain: str, evaluation_seed: int) -> Path:
    matches = []
    for path in root.rglob("states.json.gz"):
        raw = read_gz(path)
        if str(raw.get("domain")) == str(domain) and int(raw.get("evaluation_seed", -1)) == int(evaluation_seed):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"heldout identity expected exactly one match for {domain}/{evaluation_seed}, got {matches}")
    return matches[0]


def train_policy_on_frozen_memory(bundle, memory, representation: str) -> list[float]:
    losses: list[float] = []
    with_semantics = representation == H3_FINAL
    if not memory.items:
        raise RuntimeError("empty frozen strategy memory")
    for _ in range(POLICY_STEPS):
        samples = memory.sample(min(BATCH_SIZE, len(memory.items)), bundle.batch_rng)
        batch = collate_v3_observations(
            [sample.observation for sample in samples],
            [sample.legal for sample in samples],
            with_semantics=with_semantics,
            device="cpu",
        )
        target = torch.tensor([sample.target for sample in samples], dtype=torch.float32)
        weights = torch.tensor([sample.weight for sample in samples], dtype=torch.float32)
        losses.append(float(train_step(bundle.policy, bundle.pol_opt, batch, target, weights, "strategy")))
    return losses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--source-checkpoint", type=Path, required=True)
    ap.add_argument("--heldout-root", type=Path, required=True)
    ap.add_argument("--representation", choices=(H2_FINAL, H3_FINAL), required=True)
    ap.add_argument("--domain", choices=("TRUE_HEADS_UP", "THREE_HANDED"), required=True)
    ap.add_argument("--source-training-seed", type=int, choices=TRAINING_SEEDS, required=True)
    ap.add_argument("--learner-seed", type=int, choices=TRAINING_SEEDS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("torch thread contract drift")

    source_bundle, progress, _spec, extra = load_representation_v3_checkpoint(
        args.source_checkpoint,
        repo_root=args.repo_root,
        expected_domain=args.domain,
        expected_representation=args.representation,
        expected_seed=args.source_training_seed,
        expected_action_candidate=ACTION_CANDIDATE,
        expected_execution_sha=TRAINING_SHA,
        expected_architecture_fingerprint_sha256=MODEL_FINGERPRINTS[args.representation],
        device="cpu",
    )
    if progress.phase != "post_policy_fit" or int(progress.iteration) != 3:
        raise RuntimeError("source checkpoint is not a frozen finalized Phase2 cell")
    final_report = dict(extra.get("final_report") or {})
    if int(final_report.get("average_policy_optimizer_steps", -1)) != POLICY_STEPS:
        raise RuntimeError("source final AveragePolicy step count drift")

    memory = source_bundle.pol_mem
    fresh = make_representation_v3_bundle(
        args.representation,
        int(args.learner_seed),
        device="cpu",
        reservoir_capacity=int(memory.capacity),
        lr=LEARNING_RATE,
    )
    losses = train_policy_on_frozen_memory(fresh, memory, args.representation)
    if len(losses) != POLICY_STEPS:
        raise RuntimeError("diagnostic policy refit step-count drift")

    audit_tv = audit_v3_policy_model(
        fresh.policy,
        memory.items,
        representation=args.representation,
        sample_size=min(2048, len(memory.items)),
        seed=AUDIT_SEED,
    )
    policy = FinalizedV3Policy(
        representation=args.representation,
        domain=args.domain,
        training_seed=int(args.learner_seed),
        model=fresh.policy,
        final_report={},
        training_execution_sha=TRAINING_SHA,
        artifact_path="DIAGNOSTIC_REFIT_IN_MEMORY",
    )

    evaluations = []
    for evaluation_seed in EVALUATION_SEEDS:
        heldout_path = find_heldout(args.heldout_root, args.domain, int(evaluation_seed))
        descriptors = load_heldout_v3_artifact(
            heldout_path,
            expected_domain=args.domain,
            expected_evaluation_seed=int(evaluation_seed),
            expected_count=2048,
        )[:POLICY_COUNT]
        rows = policy.batch_probabilities(
            [item.observation_v3 for item in descriptors],
            [item.legal_slots for item in descriptors],
        )
        evaluations.append({
            "evaluation_seed": int(evaluation_seed),
            "policy_state_indices": list(range(POLICY_COUNT)),
            "policy_rows": [list(map(float, row)) for row in rows],
        })

    result = {
        "schema": SCHEMA,
        "representation": args.representation,
        "domain": args.domain,
        "source_training_seed": int(args.source_training_seed),
        "learner_seed": int(args.learner_seed),
        "source_training_execution_sha": TRAINING_SHA,
        "heldout_execution_sha": HELDOUT_SHA,
        "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
        "source_strategy_memory": {
            "items": len(memory.items),
            "seen": int(memory.seen),
            "capacity": int(memory.capacity),
        },
        "refit": {
            "policy_steps": POLICY_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "learner_seed_semantics": "fresh SPNNIV3 bundle seed; therefore fixed policy initialization namespace and fresh batch_rng namespace, while source strategy memory is held immutable",
            "final_loss": float(losses[-1]),
            "source_memory_weighted_mean_tv": float(audit_tv),
            "source_memory_policy_gate_threshold_reference_only": POLICY_TV_MAX,
        },
        "evaluations": evaluations,
        "diagnostic_only": True,
        "representation_winner": None,
        "changes_frozen_thresholds": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    print(json.dumps({
        "representation": args.representation,
        "domain": args.domain,
        "source_training_seed": args.source_training_seed,
        "learner_seed": args.learner_seed,
        "memory_items": len(memory.items),
        "memory_seen": int(memory.seen),
        "refit_policy_tv": float(audit_tv),
        "final_loss": float(losses[-1]),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
