from __future__ import annotations

"""Evaluation-only recovery for completed R7.5.3D Phase 2A artifacts.

The source execution completed both seed trajectories and all twelve policy fits,
then failed before heldout metrics because variable-length `legal_slots` tuples
were passed directly to the V3 collator, which requires ten-slot masks.

This tool consumes the completed source artifacts, applies the canonical
`legal_mask` conversion already used by FinalizedV3Policy.batch_probabilities,
and recomputes only the parent heldout evaluation/result JSON.  It never invokes
traversal, reservoir replay, Advantage fitting, or AveragePolicy fitting.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import r7_5_3d_v1plus_phase2a_strategy_capacity as base
from spincore.r7_5_action_cfr import legal_mask, validate_policy
from spincore_nn.models_v3_final import collate_v3_observations

SOURCE_EXECUTION_SHA = "4bfa55d69029cd69536fa6dbfcadd162719cb887"
RECOVERY_SCHEMA = "SPINCORE_R7_5_3D_V1PLUS_PHASE2A_EVAL_RECOVERY_V1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _probabilities_fixed(model, descriptors) -> list[list[float]]:
    if not descriptors:
        return []
    legal_sets = [tuple(int(x) for x in item.legal_slots) for item in descriptors]
    masks = [legal_mask(row) for row in legal_sets]
    if any(len(mask) != 10 for mask in masks):
        raise RuntimeError("Phase2A recovery legal-mask width drift")
    batch = collate_v3_observations(
        [item.observation_v3 for item in descriptors],
        masks,
        with_semantics=False,
        device="cpu",
    )
    model.eval()
    with torch.no_grad():
        logits = model(batch).masked_fill(~batch["legal"], -1e9)
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
    out = []
    for raw, legal in zip(probs, legal_sets):
        row = validate_policy(tuple(float(x) for x in raw), legal)
        out.append([float(x) for x in row])
    return out


def _require_completed_source(output_root: Path) -> dict:
    inventory = {"seed_results": [], "policy_artifacts": []}
    for seed in map(int, base.TRAINING_SEEDS):
        seed_root = output_root / f"seed_{seed}"
        result_path = seed_root / "seed_result.json"
        if not result_path.is_file():
            raise RuntimeError(f"missing completed Phase2A seed result: {result_path}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") != "SEED_COMPLETE":
            raise RuntimeError(f"Phase2A seed not complete: {seed}")
        if payload.get("execution_sha") != SOURCE_EXECUTION_SHA:
            raise RuntimeError(
                f"Phase2A source execution mismatch for seed {seed}: "
                f"{payload.get('execution_sha')!r} != {SOURCE_EXECUTION_SHA}"
            )
        if not bool(payload.get("all_advantage_gates_pass")):
            raise RuntimeError(f"Phase2A source local Advantage gate failed for seed {seed}")
        inventory["seed_results"].append({
            "training_seed": seed,
            "path": str(result_path),
            "sha256": sha256_file(result_path),
        })
        for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
            for arm in base.CAPACITIES:
                artifact = seed_root / "policies" / f"{mode}__{arm}.pt"
                meta = seed_root / "policies" / f"{mode}__{arm}.json"
                if not artifact.is_file() or not meta.is_file():
                    raise RuntimeError(f"missing completed Phase2A policy artifact: {seed}/{mode}/{arm}")
                mp = json.loads(meta.read_text(encoding="utf-8"))
                if mp.get("status") != "POLICY_FIT_COMPLETE":
                    raise RuntimeError(f"incomplete Phase2A policy metadata: {meta}")
                if int(mp.get("training_seed", -1)) != seed or mp.get("learner_mode") != mode or mp.get("arm") != arm:
                    raise RuntimeError(f"Phase2A policy metadata identity mismatch: {meta}")
                if int(mp.get("capacity", -1)) != int(base.CAPACITIES[arm]):
                    raise RuntimeError(f"Phase2A policy capacity mismatch: {meta}")
                if int(mp.get("authoritative_policy_audit_seed", -1)) != (seed ^ 0x71A5BEEF):
                    raise RuntimeError(f"Phase2A policy audit-seed mismatch: {meta}")
                inventory["policy_artifacts"].append({
                    "training_seed": seed,
                    "learner_mode": mode,
                    "arm": arm,
                    "artifact": str(artifact),
                    "artifact_sha256": sha256_file(artifact),
                    "metadata": str(meta),
                    "metadata_sha256": sha256_file(meta),
                })
    if len(inventory["policy_artifacts"]) != 12:
        raise RuntimeError("Phase2A recovery expected exactly twelve completed policy artifacts")
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluation-only recovery for completed Phase2A artifacts")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--heldout-root", type=Path, default=Path("heldout_v3_bundle"))
    parser.add_argument("--output-root", type=Path, default=Path("ryzen_v1plus_phase2a"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    heldout_root = args.heldout_root.resolve()
    output_root = args.output_root.resolve()
    out = args.out.resolve() if args.out is not None else output_root / "R7_5_3D_V1PLUS_PHASE2A_RESULT.json"

    torch.set_num_threads(base.TORCH_THREADS)
    if torch.get_num_threads() != base.TORCH_THREADS:
        raise RuntimeError("Phase2A recovery torch-thread contract drift")

    inventory = _require_completed_source(output_root)

    # Mechanical correction only: reuse the frozen parent evaluator with the
    # canonical legal-set -> ten-slot-mask conversion.
    base._probabilities = _probabilities_fixed

    class Args:
        pass

    frozen = Args()
    frozen.repo_root = repo_root
    frozen.heldout_root = heldout_root
    frozen.output_root = output_root
    frozen.execution_sha = SOURCE_EXECUTION_SHA

    result = base._evaluate_parent(frozen)
    result["evaluation_recovery"] = {
        "schema": RECOVERY_SCHEMA,
        "source_execution_sha": SOURCE_EXECUTION_SHA,
        "correction": "CANONICAL_VARIABLE_LENGTH_LEGAL_SET_TO_TEN_SLOT_MASK_BEFORE_V3_COLLATION",
        "canonical_helper": "spincore.r7_5_action_cfr.legal_mask",
        "probability_validation": "spincore.r7_5_action_cfr.validate_policy",
        "training_replayed": False,
        "reservoir_replayed": False,
        "policy_refit": False,
        "completed_source_inventory": inventory,
    }
    result["production_training_authorized"] = False
    result["ready_for_tables"] = False
    base._atomic_json(result, out)

    print(json.dumps({
        "status": result["status"],
        "source_execution_sha": SOURCE_EXECUTION_SHA,
        "common_mean_tv": result["pooled_mean_tv"]["COMMON_LEARNER"],
        "native_mean_tv": result["pooled_mean_tv"]["NATIVE_LEARNER"],
        "absolute_improvement_100k_to_800k": result["decision"]["common_100k_to_800k_absolute_improvement"],
        "relative_improvement_100k_to_800k": result["decision"]["common_100k_to_800k_relative_improvement"],
        "result": str(out),
        "result_sha256": sha256_file(out),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
