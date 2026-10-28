#!/usr/bin/env python3
from __future__ import annotations

"""HU-only V1+semantic ENS8 migration and online continuation from frozen 10105.

The source HU V1 ENS8 and checkpoint stay read-only.  First, every source member
is embedded mechanically into the semantic architecture by appending exact-zero
semantic columns and parity is checked on deterministic HU reservoir states.
Then a fresh1600 semantic ENS8 is fit from the accumulated HU Advantage
reservoir. Only after that migration succeeds are new HU roots collected.

PROJECT_CONTRACT_IDS:
TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,VALID-025,VALID-034,VALID-035,
SAFE-001,SAFE-010,CKPT-001,ART-001,PERF-001,PERF-002
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import lt3_3h_semantic_parallel_fit as semantic_fit

from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_functional_training import (
    load_checkpoint,
    _root_deck_seed,
    _root_policy_seed,
)
from spincore.lean_parallel import RootJob
from spincore.semantic_parallel import SemanticParallelRootExecutor
from spincore.solver import SolverLibrary
from spincore_nn.action_models import (
    collate_action_observations,
    make_advantage_action_model,
)
from spincore_nn.semantic_v1 import (
    SEM_DIM,
    expand_v1_state_with_zero_semantics,
    make_semantic_advantage_model,
    semantic_vector_from_observation,
)

SCHEMA = "SPINCORE_HU_SEMANTIC_ENS8_CONTINUATION_RESULT_V1"
RESUME_SCHEMA = "SPINCORE_HU_SEMANTIC_ENS8_CONTINUATION_RESUME_V1"
ENSEMBLE_SCHEMA = "SPINCORE_HU_SEMANTIC_ENS8_V1"
SOURCE_ENSEMBLE_SCHEMA = "SPINCORE_LT3_HU_ENS8_CURRENT_STATE_V1"
DOMAIN_HU = "TRUE_HEADS_UP"

SOURCE_ITERATION = 10105
ROOTS_PER_UPDATE = 273
UPDATES_IN_BLOCK = 469
NEW_ROOTS = ROOTS_PER_UPDATE * UPDATES_IN_BLOCK
MEMBER_STEPS = 1600
ENSEMBLE_SIZE = 8
PARITY_SAMPLE_COUNT = 20000
PARITY_SAMPLE_SEED = 153170105
PROTECTED_COUNT = 50000
PROTECTED_SEED = 153170106
MAX_RAW_ABS_ERROR = 2e-6
MAX_POLICY_ABS_ERROR = 2e-6
CHECKPOINT_EVERY = 25
SNAPSHOT_EVERY = 50

EXPECTED_CHECKPOINT_SHA = "f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_SOURCE_ENSEMBLE_SHA = "8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def states_finite(states) -> bool:
    for state in states:
        for value in state.values():
            if not torch.isfinite(value.detach()).all().item():
                return False
    return True


def load_source_ensemble(path: Path):
    payload = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if payload.get("schema") != SOURCE_ENSEMBLE_SCHEMA:
        raise RuntimeError("wrong source HU ensemble schema")
    if int(payload.get("completed_iteration", -1)) != SOURCE_ITERATION:
        raise RuntimeError("source HU ensemble iteration mismatch")
    if int(payload.get("ensemble_size", -1)) != ENSEMBLE_SIZE:
        raise RuntimeError("source HU ensemble size mismatch")
    states = tuple(payload.get("members") or [])
    meta = list(payload.get("member_meta") or [])
    if len(states) != ENSEMBLE_SIZE or len(meta) != ENSEMBLE_SIZE:
        raise RuntimeError("source HU ensemble members/meta mismatch")
    for index, row in enumerate(meta):
        if int(row.get("member", index)) != index:
            raise RuntimeError("source HU member metadata ordering drift")
        if "init_seed" not in row or "batch_seed" not in row:
            raise RuntimeError("source HU member metadata lacks fit seeds")
    if not states_finite(states):
        raise RuntimeError("source HU ensemble contains non-finite tensors")
    return payload, states, meta


def policy_from_raw(raw, legal):
    return np.asarray(
        lean_regret_matching_policy(
            [float(x) for x in raw],
            tuple(int(a) for a in legal),
        ),
        dtype=np.float64,
    )


def migration_parity(source_states, semantic_states, samples, *, batch_size: int = 2048):
    v1_models = []
    sem_models = []
    for index, (v1_state, sem_state) in enumerate(zip(source_states, semantic_states)):
        _, v1 = make_advantage_action_model(
            "C0_V1_FROZEN_CONTROL",
            device="cpu",
            seed=index,
        )
        v1.load_state_dict(v1_state)
        v1.eval()
        sem = make_semantic_advantage_model(
            index,
            legacy_exact_zero_mode=True,
        )
        sem.load_state_dict(sem_state)
        sem.eval()
        v1_models.append(v1)
        sem_models.append(sem)

    max_raw = 0.0
    max_policy = 0.0
    argmax_mismatches = 0
    member_max_raw = [0.0] * ENSEMBLE_SIZE

    with torch.no_grad():
        for start in range(0, len(samples), int(batch_size)):
            chunk = samples[start : start + int(batch_size)]
            observations = [s.observation for s in chunk]
            legal_masks = [tuple(bool(x) for x in s.legal) for s in chunk]
            batch = collate_action_observations(
                "C0_V1_FROZEN_CONTROL",
                observations,
                legal_masks,
                device="cpu",
            )
            sem_batch = dict(batch)
            sem_batch["semantic"] = torch.tensor(
                np.asarray(
                    [semantic_vector_from_observation(obs) for obs in observations],
                    dtype=np.float32,
                ),
                dtype=torch.float32,
            )

            raw_v1_members = torch.stack([m(batch) for m in v1_models], dim=0)
            raw_sem_members = torch.stack([m(sem_batch) for m in sem_models], dim=0)
            diff = torch.abs(raw_v1_members - raw_sem_members)
            max_raw = max(max_raw, float(diff.max().item()))
            for member in range(ENSEMBLE_SIZE):
                member_max_raw[member] = max(
                    member_max_raw[member],
                    float(diff[member].max().item()),
                )

            raw_v1 = raw_v1_members.mean(dim=0).cpu().numpy()
            raw_sem = raw_sem_members.mean(dim=0).cpu().numpy()
            for row, sample in enumerate(chunk):
                legal = tuple(i for i, flag in enumerate(sample.legal) if bool(flag))
                p1 = policy_from_raw(raw_v1[row], legal)
                ps = policy_from_raw(raw_sem[row], legal)
                max_policy = max(max_policy, float(np.max(np.abs(p1 - ps))))
                if int(np.argmax(p1)) != int(np.argmax(ps)):
                    argmax_mismatches += 1

    return {
        "sample_count": len(samples),
        "max_raw_abs_error": max_raw,
        "max_policy_abs_error": max_policy,
        "argmax_mismatches": argmax_mismatches,
        "member_max_raw_abs_error": member_max_raw,
        "pass": bool(
            max_raw <= MAX_RAW_ABS_ERROR
            and max_policy <= MAX_POLICY_ABS_ERROR
            and argmax_mismatches == 0
        ),
    }


def precompute_semantics(memory):
    rows = np.empty((len(memory.items), SEM_DIM), dtype=np.float32)
    started = time.perf_counter()
    for index, sample in enumerate(memory.items):
        rows[index] = semantic_vector_from_observation(sample.observation)
        if (index + 1) % 200000 == 0:
            print(
                f"HU_SEMANTIC_PRECOMPUTE {index + 1}/{len(memory.items)}",
                flush=True,
            )

    def observer(index, sample):
        rows[int(index)] = semantic_vector_from_observation(sample.observation)

    memory.set_write_observer(observer)
    return rows, float(time.perf_counter() - started)


def fit_semantic_ensemble(fitter):
    rows = fitter.fit()
    if len(rows) != ENSEMBLE_SIZE:
        raise RuntimeError("semantic HU fit returned wrong member count")
    rows.sort(key=lambda row: int(row["member"]))
    states = tuple(row["state"] for row in rows)
    losses = [float(row["loss_last"]) for row in rows]
    if not all(math.isfinite(x) for x in losses):
        raise RuntimeError("semantic HU fit produced non-finite loss")
    if not states_finite(states):
        raise RuntimeError("semantic HU fit produced non-finite tensor")
    meta = [
        {
            "member": int(row["member"]),
            "init_seed": int(row["init_seed"]),
            "batch_seed": int(row["batch_seed"]),
            "steps": int(row["steps"]),
            "fit_seconds": float(row["fit_seconds"]),
            "loss_last": float(row["loss_last"]),
            "maxrss_kib": int(row["maxrss_kib"]),
            "parallel_wall_seconds": float(row["parallel_wall_seconds"]),
            "concurrency": int(row["concurrency"]),
            "threads_per_member": int(row["threads_per_member"]),
        }
        for row in rows
    ]
    return states, meta


def save_ensemble(
    path: Path,
    *,
    source_checkpoint: Path,
    source_ensemble: Path,
    completed_updates: int,
    states,
    member_meta,
    migration_parity_report: dict,
) -> None:
    atomic_torch(
        path,
        {
            "schema": ENSEMBLE_SCHEMA,
            "research_lane": "HU_SEMANTIC_ENS8_TEACHER_10105_PLUS469_V1",
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "source_v1_ensemble": str(source_ensemble.resolve()),
            "source_v1_ensemble_sha256": EXPECTED_SOURCE_ENSEMBLE_SHA,
            "source_iteration": SOURCE_ITERATION,
            "completed_updates": int(completed_updates),
            "virtual_seed_iteration": SOURCE_ITERATION + int(completed_updates),
            "hu_only_roots": int(completed_updates) * ROOTS_PER_UPDATE,
            "ensemble_size": ENSEMBLE_SIZE,
            "member_steps": MEMBER_STEPS,
            "semantic_dim": SEM_DIM,
            "members": states,
            "member_meta": list(member_meta),
            "behavior_contract": (
                "MEAN_EIGHT_RAW_V1_SEMANTIC_ADVANTAGES_THEN_CANONICAL_LEAN_REGRET_MATCHING"
            ),
            "migration_parity": dict(migration_parity_report),
            "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
        },
    )


def save_resume(
    path: Path,
    *,
    sampler,
    hu,
    completed_updates: int,
    states,
    member_meta,
    rows,
    migration_parity_report,
) -> None:
    atomic_torch(
        path,
        {
            "schema": RESUME_SCHEMA,
            "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "source_v1_ensemble_sha256": EXPECTED_SOURCE_ENSEMBLE_SHA,
            "source_iteration": SOURCE_ITERATION,
            "target_updates": UPDATES_IN_BLOCK,
            "roots_per_update": ROOTS_PER_UPDATE,
            "member_steps": MEMBER_STEPS,
            "completed_updates": int(completed_updates),
            "virtual_seed_iteration": SOURCE_ITERATION + int(completed_updates),
            "sampler_rng_state": sampler.rng.bit_generator.state,
            "hu_adv_mem": hu.bundle.adv_mem.state_dict(),
            "hu_counters": dict(hu.bundle.counters),
            "semantic_states": states,
            "member_meta": list(member_meta),
            "migration_parity": dict(migration_parity_report),
            "recent_rows": list(rows[-20:]),
            "three_handed_training_performed": False,
            "average_policy_training_performed": False,
            "strategy_reservoir_collection_performed": False,
            "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
        },
    )


def restore_resume(path: Path, *, sampler, hu):
    from spincore_nn.reservoir import UniformReservoir

    p = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if p.get("schema") != RESUME_SCHEMA:
        raise RuntimeError("wrong HU semantic resume schema")
    if str(p.get("source_checkpoint_sha256")) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("HU semantic resume source checkpoint mismatch")
    if str(p.get("source_v1_ensemble_sha256")) != EXPECTED_SOURCE_ENSEMBLE_SHA:
        raise RuntimeError("HU semantic resume source ensemble mismatch")
    if int(p.get("target_updates", -1)) != UPDATES_IN_BLOCK:
        raise RuntimeError("HU semantic resume target drift")
    if int(p.get("roots_per_update", -1)) != ROOTS_PER_UPDATE:
        raise RuntimeError("HU semantic resume roots/update drift")
    if int(p.get("member_steps", -1)) != MEMBER_STEPS:
        raise RuntimeError("HU semantic resume fit-step drift")
    if bool(p.get("three_handed_training_performed")):
        raise RuntimeError("HU semantic resume records 3H training")
    if bool(p.get("average_policy_training_performed")):
        raise RuntimeError("HU semantic resume records AveragePolicy training")
    if bool(p.get("strategy_reservoir_collection_performed")):
        raise RuntimeError("HU semantic resume records strategy collection")

    completed = int(p.get("completed_updates", -1))
    if not 0 <= completed <= UPDATES_IN_BLOCK:
        raise RuntimeError("HU semantic resume completed update out of range")
    parity = dict(p.get("migration_parity") or {})
    if not bool(parity.get("pass")):
        raise RuntimeError("HU semantic resume lacks passing migration parity")

    hu.bundle.adv_mem = UniformReservoir.from_state_dict(p["hu_adv_mem"])
    hu.session.bundle = hu.bundle
    hu.session.collector.advantage_memory = hu.bundle.adv_mem
    hu.bundle.counters.clear()
    hu.bundle.counters.update(dict(p["hu_counters"]))
    sampler.rng.bit_generator.state = p["sampler_rng_state"]

    states = tuple(p.get("semantic_states") or [])
    if len(states) != ENSEMBLE_SIZE or not states_finite(states):
        raise RuntimeError("HU semantic resume ensemble invalid")
    meta = list(p.get("member_meta") or [])
    if len(meta) != ENSEMBLE_SIZE:
        raise RuntimeError("HU semantic resume metadata invalid")
    return completed, states, meta, list(p.get("recent_rows") or []), parity


def make_jobs(*, sampler, seed: int, virtual_iteration: int):
    jobs = []
    blind_counts = {}
    for local_root in range(ROOTS_PER_UPDATE):
        episode = sampler.sample_episode(force_domain=DOMAIN_HU)
        if not bool(episode.game_is_hu):
            raise RuntimeError("HU-only sampler produced non-HU episode")
        key = f"{episode.small_blind}/{episode.big_blind}"
        blind_counts[key] = blind_counts.get(key, 0) + 1
        jobs.append(
            RootJob(
                root_index=int(local_root),
                episode=episode,
                deck_seed=_root_deck_seed(seed, DOMAIN_HU, virtual_iteration, local_root),
                policy_seed=_root_policy_seed(seed, DOMAIN_HU, virtual_iteration, local_root),
            )
        )
    return jobs, blind_counts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source-ensemble", type=Path, required=True)
    p.add_argument("--solver", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=31)
    p.add_argument("--fit-concurrency", type=int, default=4)
    p.add_argument("--threads-per-member", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    source_ensemble = args.source_ensemble.resolve(strict=True)
    solver_path = args.solver.resolve(strict=True)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("frozen 10105 checkpoint SHA mismatch")
    if sha256(source_ensemble) != EXPECTED_SOURCE_ENSEMBLE_SHA:
        raise RuntimeError("frozen 10105 HU V1 ensemble SHA mismatch")
    if int(args.fit_concurrency) * int(args.threads_per_member) > 32:
        raise RuntimeError("semantic HU fit profile oversubscribes 32 logical threads")

    source_hashes_before = {
        "checkpoint": sha256(checkpoint),
        "ensemble": sha256(source_ensemble),
    }

    solver = SolverLibrary(solver_path)
    seed, config, completed_source, sampler, runtimes, _history, finalized = load_checkpoint(
        checkpoint, solver=solver
    )
    if int(completed_source) != SOURCE_ITERATION or not bool(finalized):
        raise RuntimeError("expected finalized source checkpoint @10105")
    if int(config.roots_per_iteration) != 600:
        raise RuntimeError("source roots/iteration drift")
    if int(round(int(config.roots_per_iteration) * float(config.heads_up_prob))) != ROOTS_PER_UPDATE:
        raise RuntimeError("source HU root-share drift")

    hu = runtimes[DOMAIN_HU]
    memory = hu.bundle.adv_mem
    if len(memory.items) != int(memory.capacity):
        raise RuntimeError("HU Advantage reservoir must be saturated")
    if len(memory.items) <= PROTECTED_COUNT + PARITY_SAMPLE_COUNT:
        raise RuntimeError("HU Advantage reservoir unexpectedly small")

    _source_payload, source_states, source_member_meta = load_source_ensemble(source_ensemble)

    resume_path = out / "resume_state.pt"
    report_path = out / "hu_semantic_ens8_continuation_10105_plus469.json"
    initial_path = out / "hu_semantic_ensemble_initial_10105.pt"
    final_path = out / "hu_semantic_ensemble_final_plus469.pt"
    milestones = out / "milestones"
    packed_dir = out / "work" / "packed_hu_semantic_adv"

    completed_updates = 0
    states = None
    member_meta = None
    rows = []
    parity_report = None

    print("HU_SEMANTIC_PRECOMPUTE_BEGIN", flush=True)
    semantic_rows, semantic_seconds = precompute_semantics(memory)
    print(f"HU_SEMANTIC_PRECOMPUTE_PASS seconds={semantic_seconds:.3f}", flush=True)

    protected_rng = random.Random(PROTECTED_SEED)
    protected = set(protected_rng.sample(range(len(memory.items)), PROTECTED_COUNT))
    train_pool = [i for i in range(len(memory.items)) if i not in protected]
    if len(train_pool) != len(memory.items) - PROTECTED_COUNT:
        raise RuntimeError("HU semantic train-pool size drift")

    if resume_path.exists():
        completed_updates, states, member_meta, rows, parity_report = restore_resume(
            resume_path,
            sampler=sampler,
            hu=hu,
        )
        print(
            f"HU_SEMANTIC_CONTINUATION_RESUME completed_updates={completed_updates} "
            f"virtual_seed_iteration={SOURCE_ITERATION + completed_updates}",
            flush=True,
        )
        # restore_resume replaced the memory object, so rebuild semantic rows for it.
        memory = hu.bundle.adv_mem
        print("HU_SEMANTIC_RESUME_PRECOMPUTE_BEGIN", flush=True)
        semantic_rows, resume_sem_seconds = precompute_semantics(memory)
        print(
            f"HU_SEMANTIC_RESUME_PRECOMPUTE_PASS seconds={resume_sem_seconds:.3f}",
            flush=True,
        )
    else:
        zero_states = tuple(
            expand_v1_state_with_zero_semantics(state) for state in source_states
        )
        if not states_finite(zero_states):
            raise RuntimeError("zero-column HU semantic migration produced non-finite state")

        parity_rng = random.Random(PARITY_SAMPLE_SEED)
        parity_indices = parity_rng.sample(range(len(memory.items)), PARITY_SAMPLE_COUNT)
        parity_samples = [memory.items[i] for i in parity_indices]
        print("HU_SEMANTIC_MIGRATION_PARITY_BEGIN", flush=True)
        parity_report = migration_parity(
            source_states,
            zero_states,
            parity_samples,
        )
        print(
            "HU_SEMANTIC_MIGRATION_PARITY_RESULT "
            + json.dumps(parity_report, sort_keys=True),
            flush=True,
        )
        if not bool(parity_report["pass"]):
            raise RuntimeError("HU semantic zero-column migration parity failed")
        print("HU_SEMANTIC_MIGRATION_PARITY_PASS", flush=True)

    packed, packed_meta = semantic_fit.PackedSemanticAdvantageReservoir.build(
        memory,
        semantic_rows,
        train_pool,
        packed_dir,
    )
    packed.bind_authoritative(
        memory,
        semantic_rows,
        semantic_fn=lambda sample: semantic_vector_from_observation(sample.observation),
    )
    fit_contract = semantic_fit.make_fit_contract(
        source_member_meta,
        learning_rate=float(config.learning_rate),
        batch_size=int(config.batch_size),
        member_steps=MEMBER_STEPS,
    )
    fitter = semantic_fit.ParallelSemanticEnsembleFitter(
        manifest_path=packed.manifest_path,
        contract=fit_contract,
        concurrency=int(args.fit_concurrency),
        threads_per_member=int(args.threads_per_member),
    )

    if states is None:
        print("HU_SEMANTIC_INITIAL_FIT_BEGIN", flush=True)
        states, member_meta = fit_semantic_ensemble(fitter)
        print(
            "HU_SEMANTIC_INITIAL_FIT_PASS "
            + json.dumps(
                {
                    "loss_last": [round(float(x["loss_last"]), 8) for x in member_meta],
                    "member_steps": MEMBER_STEPS,
                    "train_count": len(train_pool),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        save_ensemble(
            initial_path,
            source_checkpoint=checkpoint,
            source_ensemble=source_ensemble,
            completed_updates=0,
            states=states,
            member_meta=member_meta,
            migration_parity_report=parity_report,
        )
        save_resume(
            resume_path,
            sampler=sampler,
            hu=hu,
            completed_updates=0,
            states=states,
            member_meta=member_meta,
            rows=rows,
            migration_parity_report=parity_report,
        )

    if completed_updates == UPDATES_IN_BLOCK:
        if not final_path.exists():
            save_ensemble(
                final_path,
                source_checkpoint=checkpoint,
                source_ensemble=source_ensemble,
                completed_updates=completed_updates,
                states=states,
                member_meta=member_meta,
                migration_parity_report=parity_report,
            )
        print("HU_SEMANTIC_CONTINUATION_ALREADY_COMPLETE", flush=True)
        fitter.close()
        packed.close()
        return 0

    root_executor = SemanticParallelRootExecutor(solver_path, int(args.workers))
    started = time.perf_counter()
    try:
        for update in range(completed_updates + 1, UPDATES_IN_BLOCK + 1):
            virtual_iteration = SOURCE_ITERATION + update
            jobs, blind_counts = make_jobs(
                sampler=sampler,
                seed=int(seed),
                virtual_iteration=int(virtual_iteration),
            )
            seen_before = int(memory.seen)
            root_stats = root_executor.collect(
                domain=DOMAIN_HU,
                bundle=hu.bundle,
                iteration=int(virtual_iteration),
                exact_opponent_levels=int(config.exact_opponent_levels),
                jobs=jobs,
                ensemble_model_states=tuple(states),
            )
            if int(root_stats.get("roots", -1)) != ROOTS_PER_UPDATE:
                raise RuntimeError("HU semantic root-count mismatch")
            if int(root_stats.get("ensemble_size", -1)) != ENSEMBLE_SIZE:
                raise RuntimeError("HU semantic root ensemble-size mismatch")
            if int(memory.seen) <= seen_before:
                raise RuntimeError("HU semantic roots added no Advantage samples")

            states, member_meta = fit_semantic_ensemble(fitter)
            completed_updates = update
            row = {
                "update": int(update),
                "virtual_seed_iteration": int(virtual_iteration),
                "hu_roots": ROOTS_PER_UPDATE,
                "adv_seen_before": seen_before,
                "adv_seen_after": int(memory.seen),
                "root_stats": dict(root_stats),
                "blind_counts": blind_counts,
                "member_loss_last": [
                    float(x["loss_last"]) for x in member_meta
                ],
                "fit_parallel_wall_seconds": float(
                    member_meta[0]["parallel_wall_seconds"]
                ),
            }
            rows.append(row)
            rows = rows[-20:]

            if update % CHECKPOINT_EVERY == 0 or update == UPDATES_IN_BLOCK:
                packed.flush()
                save_resume(
                    resume_path,
                    sampler=sampler,
                    hu=hu,
                    completed_updates=completed_updates,
                    states=states,
                    member_meta=member_meta,
                    rows=rows,
                    migration_parity_report=parity_report,
                )
                print(
                    f"HU_SEMANTIC_RESUME_PASS update={update} "
                    f"hu_roots_total={update * ROOTS_PER_UPDATE}",
                    flush=True,
                )

            if update % SNAPSHOT_EVERY == 0 or update == UPDATES_IN_BLOCK:
                snapshot = milestones / f"hu_semantic_ensemble_update_{update:04d}.pt"
                save_ensemble(
                    snapshot,
                    source_checkpoint=checkpoint,
                    source_ensemble=source_ensemble,
                    completed_updates=completed_updates,
                    states=states,
                    member_meta=member_meta,
                    migration_parity_report=parity_report,
                )
                print(
                    f"HU_SEMANTIC_MILESTONE update={update} "
                    f"hu_roots_total={update * ROOTS_PER_UPDATE}",
                    flush=True,
                )

            atomic_json(
                report_path,
                {
                    "schema": SCHEMA,
                    "status": "RUNNING",
                    "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
                    "source_v1_ensemble_sha256": EXPECTED_SOURCE_ENSEMBLE_SHA,
                    "source_iteration": SOURCE_ITERATION,
                    "semantic_dim": SEM_DIM,
                    "member_steps": MEMBER_STEPS,
                    "target_updates": UPDATES_IN_BLOCK,
                    "completed_updates": completed_updates,
                    "roots_per_update": ROOTS_PER_UPDATE,
                    "new_hu_roots_so_far": completed_updates * ROOTS_PER_UPDATE,
                    "migration_parity": parity_report,
                    "protected_diagnostic_count": PROTECTED_COUNT,
                    "packed_reservoir": packed_meta,
                    "recent_rows": rows,
                    "three_handed_training_performed": False,
                    "average_policy_training_performed": False,
                    "strategy_reservoir_collection_performed": False,
                    "sealed_holdout_touched": False,
                    "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
                },
            )
            print(
                "HU_SEMANTIC_UPDATE "
                + json.dumps(
                    {
                        "update": update,
                        "target": UPDATES_IN_BLOCK,
                        "virtual_seed_iteration": virtual_iteration,
                        "hu_roots_total": update * ROOTS_PER_UPDATE,
                        "adv_seen": int(memory.seen),
                        "fit_seconds": float(
                            member_meta[0]["parallel_wall_seconds"]
                        ),
                        "member_loss_last": [
                            round(float(x["loss_last"]), 6) for x in member_meta
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        root_executor.close()
        fitter.close()
        packed.flush()
        packed.close()

    source_hashes_after = {
        "checkpoint": sha256(checkpoint),
        "ensemble": sha256(source_ensemble),
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("frozen HU source artifacts changed during semantic continuation")

    if completed_updates != UPDATES_IN_BLOCK:
        raise RuntimeError("HU semantic continuation ended before frozen target")

    save_ensemble(
        final_path,
        source_checkpoint=checkpoint,
        source_ensemble=source_ensemble,
        completed_updates=completed_updates,
        states=states,
        member_meta=member_meta,
        migration_parity_report=parity_report,
    )
    final = {
        "schema": SCHEMA,
        "status": "PASS",
        "research_lane": "HU_SEMANTIC_ENS8_TEACHER_10105_PLUS469_V1",
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "source_v1_ensemble_sha256": EXPECTED_SOURCE_ENSEMBLE_SHA,
        "source_iteration": SOURCE_ITERATION,
        "completed_updates": completed_updates,
        "virtual_seed_iteration_end": SOURCE_ITERATION + completed_updates,
        "roots_per_update": ROOTS_PER_UPDATE,
        "new_hu_roots": completed_updates * ROOTS_PER_UPDATE,
        "expected_new_hu_roots": NEW_ROOTS,
        "semantic_dim": SEM_DIM,
        "ensemble_size": ENSEMBLE_SIZE,
        "member_steps": MEMBER_STEPS,
        "migration_parity": parity_report,
        "protected_diagnostic_count": PROTECTED_COUNT,
        "initial_semantic_ensemble": str(initial_path.resolve()),
        "initial_semantic_ensemble_sha256": sha256(initial_path),
        "final_semantic_ensemble": str(final_path.resolve()),
        "final_semantic_ensemble_sha256": sha256(final_path),
        "resume_state": str(resume_path.resolve()),
        "resume_state_sha256": sha256(resume_path),
        "source_unchanged": True,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "three_handed_training_performed": False,
        "average_policy_training_performed": False,
        "strategy_reservoir_collection_performed": False,
        "sealed_holdout_touched": False,
        "wall_seconds_online_block": float(time.perf_counter() - started),
        "recent_rows": rows,
        "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
        "interpretation": (
            "PASS means HU migrated mechanically to the validated V1+30-semantic "
            "ENS8 architecture and accumulated the preregistered HU-only online block "
            "without integrity failure. It is not a strength/promotion result."
        ),
    }
    atomic_json(report_path, final)

    print("HU_SEMANTIC_ENS8_CONTINUATION_10105_PLUS469_PASS", flush=True)
    print(f"new_hu_roots={final['new_hu_roots']}", flush=True)
    print(f"migration_max_raw_abs_error={parity_report['max_raw_abs_error']}", flush=True)
    print(f"migration_max_policy_abs_error={parity_report['max_policy_abs_error']}", flush=True)
    print(f"migration_argmax_mismatches={parity_report['argmax_mismatches']}", flush=True)
    print(f"final_semantic_ensemble_sha256={final['final_semantic_ensemble_sha256']}", flush=True)
    print(f"resume_state_sha256={final['resume_state_sha256']}", flush=True)
    print(f"report={report_path}", flush=True)
    print("STOP: HU semantic training endpoint only; no strength or promotion.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
