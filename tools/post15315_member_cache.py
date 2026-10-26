#!/usr/bin/env python3
from __future__ import annotations

"""Durable exact collection cache for post-15315 rebuild members.

The target-Ryzen performance gate validated four process-parallel collection
workers while canonical fits remain non-fanned and use eight threads. This
module is the bridge between those two phases: workers collect only the sample
streams they consume, persist them atomically, and the member builder later
revalidates exact identity/order before fitting.

PROJECT_CONTRACT_IDS:
TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,
VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,VALID-038,
SAFE-001,SAFE-010,SAFE-011,PERF-001,PERF-002,CKPT-001,ART-001
"""

import os
from pathlib import Path
from typing import Any

import torch

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import post15315_collection as collect
from spincore.solver import SolverLibrary

SCHEMA = "SPINCORE_POST15315_REBUILD_COLLECTION_CACHE_V1"
FINAL_ITERATION = 15315


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save(payload, tmp)
    os.replace(tmp, path)


def cache_path(out_dir: Path, member: int) -> Path:
    return Path(out_dir) / f"member_{int(member):02d}_collection_cache.pt"


def collect_member_cache(
    *,
    member: int,
    protocol: dict,
    protocol_path: Path,
    teacher_models,
    teacher_sha256: str,
    solver: SolverLibrary,
    out_dir: Path,
    collection_threads: int,
) -> Path:
    member = int(member)
    rows = [
        dict(x)
        for x in protocol["rebuild_plan"]["member_seeds"]
        if int(x["member"]) == member
    ]
    if len(rows) != 1:
        raise RuntimeError(f"member {member} is not uniquely preregistered")
    seeds = rows[0]
    avg = protocol["rebuild_plan"]["average_policy"]
    sp = protocol["rebuild_plan"]["specialist"]

    torch.set_num_threads(int(collection_threads))

    ordinary_train, ordinary_holdout, ordinary_stats = collect.collect_fresh_split(
        solver,
        teacher_models,
        int(avg["ordinary_teacher_episodes_per_rebuild"]),
        master_seed=int(seeds["base_collection_seed"]),
        sample_iteration=FINAL_ITERATION,
        progress_prefix=f"POST15315_MEMBER_{member:02d}_ORDINARY",
    )
    aug_train, aug_holdout, aug_stats = collect.collect_strong_split(
        solver,
        teacher_models,
        int(avg["strong_augmentation_teacher_episodes_per_rebuild"]),
        master_seed=int(seeds["augment_collection_seed"]),
        sample_iteration=FINAL_ITERATION,
        progress_prefix=f"POST15315_MEMBER_{member:02d}_AUGMENT",
    )
    specialist_train, specialist_holdout, specialist_stats = collect.collect_strong_split(
        solver,
        teacher_models,
        int(sp["teacher_episodes_per_rebuild"]),
        master_seed=int(seeds["specialist_collection_seed"]),
        sample_iteration=FINAL_ITERATION,
        progress_prefix=f"POST15315_MEMBER_{member:02d}_SPECIALIST",
    )

    protocol_sha = distill.sha256(Path(protocol_path).resolve(strict=True))
    payload = {
        "schema": SCHEMA,
        "member": member,
        "sample_iteration": FINAL_ITERATION,
        "teacher_sha256": str(teacher_sha256),
        "protocol_sha256": protocol_sha,
        "member_seeds": seeds,
        "collection_threads": int(collection_threads),
        "ordinary": {
            "stats": collect.canonical_collection_stats(ordinary_stats),
            "train_digest": collect.sample_digest(ordinary_train),
            "holdout_digest": collect.sample_digest(ordinary_holdout),
            "train": ordinary_train,
            "holdout": ordinary_holdout,
        },
        "augmentation": {
            "stats": collect.canonical_collection_stats(aug_stats),
            "train_digest": collect.sample_digest(aug_train),
            "holdout_digest": collect.sample_digest(aug_holdout),
            "train": aug_train,
            "holdout": aug_holdout,
        },
        "specialist": {
            "stats": collect.canonical_collection_stats(specialist_stats),
            "train_digest": collect.sample_digest(specialist_train),
            "holdout_digest": collect.sample_digest(specialist_holdout),
            "train": specialist_train,
            "holdout": specialist_holdout,
        },
    }
    path = cache_path(out_dir, member)
    _atomic_torch_save(payload, path)
    load_member_cache(
        path,
        expected_member=member,
        expected_teacher_sha256=teacher_sha256,
        expected_protocol_sha256=protocol_sha,
        expected_member_seeds=seeds,
        expected_collection_threads=int(collection_threads),
    )
    return path


def load_member_cache(
    path: Path,
    *,
    expected_member: int,
    expected_teacher_sha256: str,
    expected_protocol_sha256: str,
    expected_member_seeds: dict,
    expected_collection_threads: int,
) -> dict[str, Any]:
    p = torch.load(Path(path).resolve(strict=True), map_location="cpu", weights_only=False)
    if p.get("schema") != SCHEMA:
        raise RuntimeError(f"wrong collection-cache schema: {p.get('schema')!r}")
    if int(p.get("member", -1)) != int(expected_member):
        raise RuntimeError("collection-cache member mismatch")
    if int(p.get("sample_iteration", -1)) != FINAL_ITERATION:
        raise RuntimeError("collection-cache lineage mismatch")
    if str(p.get("teacher_sha256")) != str(expected_teacher_sha256):
        raise RuntimeError("collection-cache teacher SHA mismatch")
    if str(p.get("protocol_sha256")) != str(expected_protocol_sha256):
        raise RuntimeError("collection-cache protocol SHA mismatch")
    if dict(p.get("member_seeds") or {}) != dict(expected_member_seeds):
        raise RuntimeError("collection-cache member seeds mismatch")
    if int(p.get("collection_threads", -1)) != int(expected_collection_threads):
        raise RuntimeError("collection-cache thread-profile mismatch")

    for key in ("ordinary", "augmentation", "specialist"):
        row = dict(p.get(key) or {})
        train = list(row.get("train") or [])
        holdout = list(row.get("holdout") or [])
        if collect.sample_digest(train) != str(row.get("train_digest")):
            raise RuntimeError(f"{key} collection-cache train digest mismatch")
        if collect.sample_digest(holdout) != str(row.get("holdout_digest")):
            raise RuntimeError(f"{key} collection-cache holdout digest mismatch")
        if int(row.get("stats", {}).get("sample_iteration", -1)) != FINAL_ITERATION:
            raise RuntimeError(f"{key} collection-cache stats lineage mismatch")
    return p


if __name__ == "__main__":
    raise SystemExit("Import this helper from guarded post-15315 tools.")
