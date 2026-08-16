from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch

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

SCHEMA = "SPINCORE_R7_5_3C_SAMPLING_SPLIT_FINAL_CELL_V1"
HELDOUT_SHA = "dfe5f83742495a457e92b29f97db5d3b631bca22"
FIXED_LEARNING_SEED = 1801739323
FINAL_POLICY_LEARNER_SEED = 1538612375
FINAL_POLICY_SEED_NAMESPACE = "SpinCore|R7.5.3C|UPSTREAM-FACTORIAL|FINAL-POLICY-LEARNER"
POLICY_COUNT = 1024
AUDIT_SEED = 0x6417A11D
REPRESENTATIONS = (H2_FINAL, H3_FINAL)


def _derived_fixed_seed() -> int:
    raw = hashlib.sha256(FINAL_POLICY_SEED_NAMESPACE.encode("utf-8")).digest()[:4]
    return int.from_bytes(raw, "big") & 0x7FFFFFFF


def _read_gz(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def _find_heldout(root: Path, domain: str, evaluation_seed: int) -> Path:
    matches = []
    for path in root.rglob("states.json.gz"):
        raw = _read_gz(path)
        if str(raw.get("domain")) == str(domain) and int(raw.get("evaluation_seed", -1)) == int(evaluation_seed):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"heldout identity mismatch for {domain}/{evaluation_seed}: {matches}")
    return matches[0]


def _train_fixed_policy(bundle, memory, representation: str) -> list[float]:
    if not memory.items:
        raise RuntimeError("empty sampling-split strategy memory")
    with_semantics = representation == H3_FINAL
    losses = []
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
    ap = argparse.ArgumentParser(description="Fixed final-policy readout for deck-vs-traversal split")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--source-checkpoint", type=Path, required=True)
    ap.add_argument("--heldout-root", type=Path, required=True)
    ap.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    ap.add_argument("--domain", choices=("TRUE_HEADS_UP", "THREE_HANDED"), required=True)
    ap.add_argument("--deck-seed", type=int, choices=TRAINING_SEEDS, required=True)
    ap.add_argument("--traversal-seed", type=int, choices=TRAINING_SEEDS, required=True)
    ap.add_argument("--execution-sha", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if _derived_fixed_seed() != FINAL_POLICY_LEARNER_SEED:
        raise RuntimeError("fixed final-policy learner seed derivation drift")
    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("torch thread contract drift")

    source_bundle, progress, _spec, extra = load_representation_v3_checkpoint(
        args.source_checkpoint,
        repo_root=args.repo_root,
        expected_domain=args.domain,
        expected_representation=args.representation,
        expected_seed=FIXED_LEARNING_SEED,
        expected_action_candidate=ACTION_CANDIDATE,
        expected_execution_sha=args.execution_sha,
        expected_architecture_fingerprint_sha256=MODEL_FINGERPRINTS[args.representation],
        device="cpu",
    )
    if progress.phase != "post_advantage_fit" or int(progress.iteration) != 3:
        raise RuntimeError("source checkpoint must be completed iteration-3 pre-policy-fit")
    state = dict(extra.get("stage_state") or {})
    if int(state.get("diagnostic_fixed_learning_seed", -1)) != FIXED_LEARNING_SEED:
        raise RuntimeError("source fixed learning-seed mismatch")
    if int(state.get("diagnostic_deck_seed", -1)) != int(args.deck_seed):
        raise RuntimeError("source diagnostic deck-seed mismatch")
    if int(state.get("diagnostic_traversal_seed", -1)) != int(args.traversal_seed):
        raise RuntimeError("source diagnostic traversal-seed mismatch")

    memory = source_bundle.pol_mem
    fresh = make_representation_v3_bundle(
        args.representation,
        FINAL_POLICY_LEARNER_SEED,
        device="cpu",
        reservoir_capacity=int(memory.capacity),
        lr=LEARNING_RATE,
    )
    losses = _train_fixed_policy(fresh, memory, args.representation)
    if len(losses) != POLICY_STEPS:
        raise RuntimeError("fixed final-policy fit step-count drift")
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
        training_seed=FINAL_POLICY_LEARNER_SEED,
        model=fresh.policy,
        final_report={},
        training_execution_sha=args.execution_sha,
        artifact_path="SAMPLING_SPLIT_FIXED_READOUT",
    )

    evaluations = []
    for evaluation_seed in EVALUATION_SEEDS:
        path = _find_heldout(args.heldout_root, args.domain, int(evaluation_seed))
        descriptors = load_heldout_v3_artifact(
            path,
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
        "deck_seed": int(args.deck_seed),
        "traversal_seed": int(args.traversal_seed),
        "fixed_learning_seed": FIXED_LEARNING_SEED,
        "execution_sha": args.execution_sha,
        "heldout_execution_sha": HELDOUT_SHA,
        "strategy_memory": {"items": len(memory.items), "seen": int(memory.seen), "capacity": int(memory.capacity)},
        "fixed_final_policy": {
            "learner_seed": FINAL_POLICY_LEARNER_SEED,
            "learner_seed_namespace": FINAL_POLICY_SEED_NAMESPACE,
            "steps": POLICY_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "final_loss": float(losses[-1]),
            "strategy_memory_weighted_mean_tv": float(audit_tv),
            "policy_gate_threshold_reference_only": POLICY_TV_MAX,
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
        "deck_seed": args.deck_seed,
        "traversal_seed": args.traversal_seed,
        "strategy_memory_items": len(memory.items),
        "strategy_memory_seen": int(memory.seen),
        "fixed_policy_tv": float(audit_tv),
        "final_loss": float(losses[-1]),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
