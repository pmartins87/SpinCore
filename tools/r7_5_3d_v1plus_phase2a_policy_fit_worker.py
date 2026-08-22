from __future__ import annotations

"""Isolated AveragePolicy fit worker for the frozen R7.5.3D Phase 2A ablation.

One process owns one Strategy-memory capacity arm and fits its COMMON_LEARNER and
NATIVE_LEARNER readouts sequentially.  Capacity arms are independent after the
frozen Strategy stream has been generated, so separate processes improve Ryzen
utilization without changing traversal, reservoirs, samples, seeds, model
architecture, optimizer budget, or heldout evaluation.
"""

import argparse
import json
from pathlib import Path
import random

import torch

import r7_5_3d_v1plus_phase2a_strategy_capacity as base
from spincore_nn.reservoir import UniformReservoir

CONTEXT_SCHEMA = "SPINCORE_R7_5_3D_PHASE2A_POLICY_FIT_CONTEXT_V1"


def _valid_existing(meta: Path, artifact: Path, *, training_seed: int, arm: str) -> dict | None:
    if not (meta.is_file() and artifact.is_file()):
        return None
    try:
        saved = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    if (
        saved.get("status") == "POLICY_FIT_COMPLETE"
        and int(saved.get("training_seed", -1)) == int(training_seed)
        and saved.get("arm") == arm
        and int(saved.get("capacity", -1)) == int(base.CAPACITIES[arm])
        and int(saved.get("authoritative_policy_audit_seed", -1)) == (int(training_seed) ^ 0x71A5BEEF)
    ):
        return saved
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5.3D Phase2A isolated policy-fit arm worker")
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, choices=base.TRAINING_SEEDS, required=True)
    parser.add_argument("--arm", choices=tuple(base.CAPACITIES), required=True)
    args = parser.parse_args()

    torch.set_num_threads(base.TORCH_THREADS)
    if torch.get_num_threads() != base.TORCH_THREADS:
        raise RuntimeError("Phase2A policy-fit worker torch-thread contract drift")

    payload = torch.load(args.context, map_location="cpu", weights_only=False)
    if payload.get("schema") != CONTEXT_SCHEMA:
        raise RuntimeError("Phase2A policy-fit context schema mismatch")
    if int(payload.get("training_seed", -1)) != int(args.training_seed):
        raise RuntimeError("Phase2A policy-fit context training-seed mismatch")
    if payload.get("arm") != str(args.arm):
        raise RuntimeError("Phase2A policy-fit context arm mismatch")
    if int(payload.get("capacity", -1)) != int(base.CAPACITIES[str(args.arm)]):
        raise RuntimeError("Phase2A policy-fit context capacity mismatch")

    memory = UniformReservoir.from_state_dict(payload["memory_state"])
    if int(memory.capacity) != int(base.CAPACITIES[str(args.arm)]):
        raise RuntimeError("Phase2A reconstructed Strategy reservoir capacity mismatch")
    native_state = payload["native_batch_rng_state"]

    policy_root = args.seed_root.resolve() / "policies"
    policy_root.mkdir(parents=True, exist_ok=True)
    audit_seed = int(args.training_seed) ^ 0x71A5BEEF
    rows = {}

    # Sequential inside one arm keeps peak memory modest; the three capacity
    # arms (and the two training seeds) are parallelized at process level.
    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        key = f"{mode}__{args.arm}"
        artifact = policy_root / f"{key}.pt"
        meta = policy_root / f"{key}.json"
        existing = _valid_existing(
            meta,
            artifact,
            training_seed=int(args.training_seed),
            arm=str(args.arm),
        )
        if existing is not None:
            rows[key] = existing
            print(f"[Phase2A policy resume] seed={args.training_seed} {key}", flush=True)
            continue

        if mode == "COMMON_LEARNER":
            init_seed = base.COMMON_POLICY_INIT_SEED
            rng = random.Random(base.COMMON_BATCH_SEED)
        else:
            init_seed = (int(args.training_seed) ^ 0x5DEECE66D) & 0x7FFFFFFF
            rng = random.Random()
            rng.setstate(native_state)

        print(
            f"[Phase2A parallel policy fit] seed={args.training_seed} {key} "
            f"retained={len(memory.items)} threads={torch.get_num_threads()}",
            flush=True,
        )
        model, fit = base._fit_policy(
            memory,
            init_seed=init_seed,
            rng=rng,
            audit_seed=audit_seed,
        )
        model_payload = {
            "schema": base.SEED_SCHEMA,
            "status": "POLICY_FIT_COMPLETE",
            "representation": base.REPRESENTATION,
            "domain": base.DOMAIN,
            "training_seed": int(args.training_seed),
            "learner_mode": mode,
            "arm": str(args.arm),
            "capacity": int(base.CAPACITIES[str(args.arm)]),
            "authoritative_policy_audit_seed": int(audit_seed),
            "parallel_fit_process": True,
            "model_state": model.state_dict(),
            "fit": fit,
        }
        base._atomic_torch_save(model_payload, artifact)
        saved = {
            "schema": base.SEED_SCHEMA,
            "status": "POLICY_FIT_COMPLETE",
            "training_seed": int(args.training_seed),
            "learner_mode": mode,
            "arm": str(args.arm),
            "capacity": int(base.CAPACITIES[str(args.arm)]),
            "authoritative_policy_audit_seed": int(audit_seed),
            "parallel_fit_process": True,
            "artifact": str(artifact),
            "fit": fit,
        }
        base._atomic_json(saved, meta)
        rows[key] = saved

    print(json.dumps({
        "status": "ARM_POLICY_FITS_COMPLETE",
        "training_seed": int(args.training_seed),
        "arm": str(args.arm),
        "capacity": int(base.CAPACITIES[str(args.arm)]),
        "rows": sorted(rows),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
