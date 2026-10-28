#!/usr/bin/env python3
from __future__ import annotations

"""Generic resumable HU Semantic ENS8 long-block continuation.

This runner never performs the original V1->semantic migration. It starts only
from a completed HU Semantic ENS8 endpoint, restores the authoritative HU
Advantage reservoir + sampler RNG + ensemble state, and advances one frozen
cumulative-update interval. The same code can therefore carry successive
three-month training blocks without changing the learning algorithm.

PROJECT_CONTRACT_IDS:
TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,VALID-025,VALID-034,VALID-035,
SAFE-001,SAFE-010,CKPT-001,ART-001,PERF-001,PERF-002
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import run_hu_semantic_ens8_continuation_10105_plus469 as base

from spincore.lean_functional_training import load_checkpoint
from spincore.semantic_parallel import SemanticParallelRootExecutor
from spincore.solver import SolverLibrary
from spincore_nn.reservoir import UniformReservoir
from spincore_nn.semantic_v1 import semantic_vector_from_observation

SCHEMA = "SPINCORE_HU_SEMANTIC_ENS8_LONG_BLOCK_RESULT_V1"
RESUME_SCHEMA = "SPINCORE_HU_SEMANTIC_ENS8_LONG_BLOCK_RESUME_V1"
ENSEMBLE_SCHEMA = "SPINCORE_HU_SEMANTIC_ENS8_LONG_ENSEMBLE_V1"

SOURCE_ITERATION = base.SOURCE_ITERATION
ROOTS_PER_UPDATE = base.ROOTS_PER_UPDATE
MEMBER_STEPS = base.MEMBER_STEPS
ENSEMBLE_SIZE = base.ENSEMBLE_SIZE
PROTECTED_COUNT = base.PROTECTED_COUNT
PROTECTED_SEED = base.PROTECTED_SEED
CHECKPOINT_EVERY = 25
SNAPSHOT_EVERY = 50


def state_dicts_equal(left, right) -> bool:
    if set(left) != set(right):
        return False
    return all(torch.equal(left[k].cpu(), right[k].cpu()) for k in left)


def ensemble_states_equal(left, right) -> bool:
    if len(left) != len(right):
        return False
    return all(state_dicts_equal(a, b) for a, b in zip(left, right))


def load_source_report(path: Path):
    report = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    schema = report.get("schema")
    if schema == base.SCHEMA:
        if report.get("status") != "PASS":
            raise RuntimeError("source HU semantic migration block is not PASS")
        completed = int(report.get("completed_updates", -1))
    elif schema == SCHEMA:
        if report.get("status") != "PASS":
            raise RuntimeError("source HU semantic long block is not PASS")
        completed = int(report.get("cumulative_completed_updates", -1))
    else:
        raise RuntimeError(f"unsupported source HU semantic report schema: {schema!r}")
    return report, completed


def load_source_endpoint(
    *,
    source_resume: Path,
    source_ensemble: Path,
    source_report: Path,
    expected_start: int,
):
    report, report_completed = load_source_report(source_report)
    if report_completed != int(expected_start):
        raise RuntimeError(
            f"source HU report endpoint {report_completed} != expected {expected_start}"
        )

    resume_sha = base.sha256(source_resume)
    ensemble_sha = base.sha256(source_ensemble)
    if str(report.get("resume_state_sha256")) != resume_sha:
        raise RuntimeError("source HU resume hash does not match PASS report")
    if str(report.get("final_semantic_ensemble_sha256")) != ensemble_sha:
        raise RuntimeError("source HU ensemble hash does not match PASS report")

    rp = torch.load(source_resume.resolve(strict=True), map_location="cpu", weights_only=False)
    ep = torch.load(source_ensemble.resolve(strict=True), map_location="cpu", weights_only=False)

    if rp.get("schema") == base.RESUME_SCHEMA:
        completed = int(rp.get("completed_updates", -1))
        states = tuple(rp.get("semantic_states") or [])
        meta = list(rp.get("member_meta") or [])
        parity = dict(rp.get("migration_parity") or {})
    elif rp.get("schema") == RESUME_SCHEMA:
        completed = int(rp.get("cumulative_completed_updates", -1))
        states = tuple(rp.get("semantic_states") or [])
        meta = list(rp.get("member_seed_contract") or [])
        parity = dict(rp.get("migration_parity") or {})
    else:
        raise RuntimeError("unsupported source HU semantic resume schema")

    if ep.get("schema") == base.ENSEMBLE_SCHEMA:
        ensemble_completed = int(ep.get("completed_updates", -1))
        ensemble_states = tuple(ep.get("members") or [])
    elif ep.get("schema") == ENSEMBLE_SCHEMA:
        ensemble_completed = int(ep.get("cumulative_completed_updates", -1))
        ensemble_states = tuple(ep.get("members") or [])
    else:
        raise RuntimeError("unsupported source HU semantic ensemble schema")

    if completed != int(expected_start) or ensemble_completed != int(expected_start):
        raise RuntimeError("source HU resume/ensemble cumulative endpoint mismatch")
    if len(states) != ENSEMBLE_SIZE or len(ensemble_states) != ENSEMBLE_SIZE:
        raise RuntimeError("source HU semantic ensemble size drift")
    if len(meta) != ENSEMBLE_SIZE:
        raise RuntimeError("source HU semantic member metadata drift")
    if not base.states_finite(states) or not base.states_finite(ensemble_states):
        raise RuntimeError("source HU semantic endpoint contains non-finite tensors")
    if not ensemble_states_equal(states, ensemble_states):
        raise RuntimeError("source HU resume/final-ensemble state mismatch")
    if not bool(parity.get("pass")):
        raise RuntimeError("source HU semantic lineage lacks passing migration parity")
    if bool(rp.get("three_handed_training_performed")):
        raise RuntimeError("source HU semantic resume records 3H training")
    if bool(rp.get("average_policy_training_performed")):
        raise RuntimeError("source HU semantic resume records AveragePolicy training")
    if bool(rp.get("strategy_reservoir_collection_performed")):
        raise RuntimeError("source HU semantic resume records strategy collection")

    return {
        "report": report,
        "report_sha256": base.sha256(source_report),
        "resume": rp,
        "resume_sha256": resume_sha,
        "ensemble": ep,
        "ensemble_sha256": ensemble_sha,
        "completed": completed,
        "states": states,
        "member_meta": meta,
        "migration_parity": parity,
    }


def restore_source_runtime(source, *, sampler, hu):
    rp = source["resume"]
    hu.bundle.adv_mem = UniformReservoir.from_state_dict(rp["hu_adv_mem"])
    hu.session.bundle = hu.bundle
    hu.session.collector.advantage_memory = hu.bundle.adv_mem
    hu.bundle.counters.clear()
    hu.bundle.counters.update(dict(rp["hu_counters"]))
    sampler.rng.bit_generator.state = rp["sampler_rng_state"]


def save_ensemble(
    path: Path,
    *,
    source,
    start_updates: int,
    target_updates: int,
    states,
    member_meta,
    migration_parity,
):
    base.atomic_torch(
        path,
        {
            "schema": ENSEMBLE_SCHEMA,
            "research_lane": "HU_SEMANTIC_ENS8_LONG_BLOCK_V1",
            "source_checkpoint_sha256": base.EXPECTED_CHECKPOINT_SHA,
            "source_endpoint_resume_sha256": source["resume_sha256"],
            "source_endpoint_ensemble_sha256": source["ensemble_sha256"],
            "source_endpoint_report_sha256": source["report_sha256"],
            "source_cumulative_updates": int(start_updates),
            "cumulative_completed_updates": int(target_updates),
            "virtual_seed_iteration": SOURCE_ITERATION + int(target_updates),
            "cumulative_hu_roots_since_10105": int(target_updates) * ROOTS_PER_UPDATE,
            "ensemble_size": ENSEMBLE_SIZE,
            "member_steps": MEMBER_STEPS,
            "semantic_dim": base.SEM_DIM,
            "members": states,
            "member_seed_contract": list(member_meta),
            "behavior_contract": (
                "MEAN_EIGHT_RAW_V1_SEMANTIC_ADVANTAGES_THEN_CANONICAL_LEAN_REGRET_MATCHING"
            ),
            "migration_parity": dict(migration_parity),
            "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
        },
    )


def save_resume(
    path: Path,
    *,
    source,
    sampler,
    hu,
    start_updates: int,
    target_updates: int,
    completed_updates: int,
    states,
    member_meta,
    migration_parity,
    rows,
):
    base.atomic_torch(
        path,
        {
            "schema": RESUME_SCHEMA,
            "source_checkpoint_sha256": base.EXPECTED_CHECKPOINT_SHA,
            "source_endpoint_resume_sha256": source["resume_sha256"],
            "source_endpoint_ensemble_sha256": source["ensemble_sha256"],
            "source_endpoint_report_sha256": source["report_sha256"],
            "source_cumulative_updates": int(start_updates),
            "target_cumulative_updates": int(target_updates),
            "cumulative_completed_updates": int(completed_updates),
            "virtual_seed_iteration": SOURCE_ITERATION + int(completed_updates),
            "roots_per_update": ROOTS_PER_UPDATE,
            "member_steps": MEMBER_STEPS,
            "sampler_rng_state": sampler.rng.bit_generator.state,
            "hu_adv_mem": hu.bundle.adv_mem.state_dict(),
            "hu_counters": dict(hu.bundle.counters),
            "semantic_states": states,
            "member_seed_contract": list(member_meta),
            "migration_parity": dict(migration_parity),
            "recent_rows": list(rows[-20:]),
            "three_handed_training_performed": False,
            "average_policy_training_performed": False,
            "strategy_reservoir_collection_performed": False,
            "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
        },
    )


def restore_local_resume(path: Path, *, source, sampler, hu, start_updates: int, target_updates: int):
    p = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if p.get("schema") != RESUME_SCHEMA:
        raise RuntimeError("wrong local HU semantic long-resume schema")
    if str(p.get("source_endpoint_resume_sha256")) != source["resume_sha256"]:
        raise RuntimeError("local HU semantic resume source-resume binding mismatch")
    if str(p.get("source_endpoint_ensemble_sha256")) != source["ensemble_sha256"]:
        raise RuntimeError("local HU semantic resume source-ensemble binding mismatch")
    if str(p.get("source_endpoint_report_sha256")) != source["report_sha256"]:
        raise RuntimeError("local HU semantic resume source-report binding mismatch")
    if int(p.get("source_cumulative_updates", -1)) != int(start_updates):
        raise RuntimeError("local HU semantic resume start drift")
    if int(p.get("target_cumulative_updates", -1)) != int(target_updates):
        raise RuntimeError("local HU semantic resume target drift")
    if int(p.get("roots_per_update", -1)) != ROOTS_PER_UPDATE:
        raise RuntimeError("local HU semantic resume roots/update drift")
    if int(p.get("member_steps", -1)) != MEMBER_STEPS:
        raise RuntimeError("local HU semantic resume member-step drift")
    if bool(p.get("three_handed_training_performed")):
        raise RuntimeError("local HU semantic resume records 3H training")
    if bool(p.get("average_policy_training_performed")):
        raise RuntimeError("local HU semantic resume records AveragePolicy training")
    if bool(p.get("strategy_reservoir_collection_performed")):
        raise RuntimeError("local HU semantic resume records strategy collection")

    completed = int(p.get("cumulative_completed_updates", -1))
    if not int(start_updates) <= completed <= int(target_updates):
        raise RuntimeError("local HU semantic resume completed endpoint out of range")

    hu.bundle.adv_mem = UniformReservoir.from_state_dict(p["hu_adv_mem"])
    hu.session.bundle = hu.bundle
    hu.session.collector.advantage_memory = hu.bundle.adv_mem
    hu.bundle.counters.clear()
    hu.bundle.counters.update(dict(p["hu_counters"]))
    sampler.rng.bit_generator.state = p["sampler_rng_state"]

    states = tuple(p.get("semantic_states") or [])
    meta = list(p.get("member_seed_contract") or [])
    if len(states) != ENSEMBLE_SIZE or len(meta) != ENSEMBLE_SIZE:
        raise RuntimeError("local HU semantic resume ensemble/meta drift")
    if not base.states_finite(states):
        raise RuntimeError("local HU semantic resume non-finite state")
    parity = dict(p.get("migration_parity") or {})
    if not bool(parity.get("pass")):
        raise RuntimeError("local HU semantic resume lacks migration parity lineage")
    return completed, states, meta, parity, list(p.get("recent_rows") or [])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source-resume", type=Path, required=True)
    p.add_argument("--source-ensemble", type=Path, required=True)
    p.add_argument("--source-report", type=Path, required=True)
    p.add_argument("--start-cumulative-updates", type=int, required=True)
    p.add_argument("--target-cumulative-updates", type=int, required=True)
    p.add_argument("--solver", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=31)
    p.add_argument("--fit-concurrency", type=int, default=4)
    p.add_argument("--threads-per-member", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    start_updates = int(args.start_cumulative_updates)
    target_updates = int(args.target_cumulative_updates)
    if start_updates < 0 or target_updates <= start_updates:
        raise RuntimeError("invalid HU semantic cumulative-update interval")
    if int(args.fit_concurrency) * int(args.threads_per_member) > 32:
        raise RuntimeError("semantic HU fit profile oversubscribes 32 logical threads")

    checkpoint = args.checkpoint.resolve(strict=True)
    if base.sha256(checkpoint) != base.EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("frozen 10105 checkpoint SHA mismatch")

    source_resume = args.source_resume.resolve(strict=True)
    source_ensemble = args.source_ensemble.resolve(strict=True)
    source_report = args.source_report.resolve(strict=True)
    source = load_source_endpoint(
        source_resume=source_resume,
        source_ensemble=source_ensemble,
        source_report=source_report,
        expected_start=start_updates,
    )

    source_hashes_before = {
        "checkpoint": base.sha256(checkpoint),
        "resume": base.sha256(source_resume),
        "ensemble": base.sha256(source_ensemble),
        "report": base.sha256(source_report),
    }

    solver_path = args.solver.resolve(strict=True)
    solver = SolverLibrary(solver_path)
    seed, config, completed_source, sampler, runtimes, _history, finalized = load_checkpoint(
        checkpoint, solver=solver
    )
    if int(completed_source) != SOURCE_ITERATION or not bool(finalized):
        raise RuntimeError("expected finalized frozen source checkpoint @10105")
    if int(round(int(config.roots_per_iteration) * float(config.heads_up_prob))) != ROOTS_PER_UPDATE:
        raise RuntimeError("source HU root-share drift")

    hu = runtimes[base.DOMAIN_HU]
    restore_source_runtime(source, sampler=sampler, hu=hu)
    memory = hu.bundle.adv_mem
    if len(memory.items) != int(memory.capacity):
        raise RuntimeError("HU Advantage reservoir must remain saturated")

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    resume_path = out / "resume_state.pt"
    report_path = out / "hu_semantic_ens8_long_block.json"
    final_path = out / f"hu_semantic_ensemble_final_cum{target_updates}.pt"
    milestones = out / "milestones"
    packed_dir = out / "work" / "packed_hu_semantic_adv"

    completed_updates = start_updates
    states = tuple(source["states"])
    member_meta = list(source["member_meta"])
    migration_parity = dict(source["migration_parity"])
    rows = []

    if resume_path.exists():
        completed_updates, states, member_meta, migration_parity, rows = restore_local_resume(
            resume_path,
            source=source,
            sampler=sampler,
            hu=hu,
            start_updates=start_updates,
            target_updates=target_updates,
        )
        memory = hu.bundle.adv_mem
        print(
            f"HU_SEMANTIC_LONG_RESUME cumulative_updates={completed_updates} "
            f"virtual_seed_iteration={SOURCE_ITERATION + completed_updates}",
            flush=True,
        )

    print("HU_SEMANTIC_LONG_PRECOMPUTE_BEGIN", flush=True)
    semantic_rows, semantic_seconds = base.precompute_semantics(memory)
    print(
        f"HU_SEMANTIC_LONG_PRECOMPUTE_PASS seconds={semantic_seconds:.3f}",
        flush=True,
    )

    protected_rng = random.Random(PROTECTED_SEED)
    protected = set(protected_rng.sample(range(len(memory.items)), PROTECTED_COUNT))
    train_pool = [i for i in range(len(memory.items)) if i not in protected]
    if len(train_pool) != len(memory.items) - PROTECTED_COUNT:
        raise RuntimeError("HU semantic long train-pool size drift")

    packed, packed_meta = base.semantic_fit.PackedSemanticAdvantageReservoir.build(
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
    fit_contract = base.semantic_fit.make_fit_contract(
        member_meta,
        learning_rate=float(config.learning_rate),
        batch_size=int(config.batch_size),
        member_steps=MEMBER_STEPS,
    )
    fitter = base.semantic_fit.ParallelSemanticEnsembleFitter(
        manifest_path=packed.manifest_path,
        contract=fit_contract,
        concurrency=int(args.fit_concurrency),
        threads_per_member=int(args.threads_per_member),
    )
    root_executor = SemanticParallelRootExecutor(solver_path, int(args.workers))

    started = time.perf_counter()
    try:
        for cumulative_update in range(completed_updates + 1, target_updates + 1):
            virtual_iteration = SOURCE_ITERATION + cumulative_update
            jobs, blind_counts = base.make_jobs(
                sampler=sampler,
                seed=int(seed),
                virtual_iteration=int(virtual_iteration),
            )
            seen_before = int(memory.seen)
            root_stats = root_executor.collect(
                domain=base.DOMAIN_HU,
                bundle=hu.bundle,
                iteration=int(virtual_iteration),
                exact_opponent_levels=int(config.exact_opponent_levels),
                jobs=jobs,
                ensemble_model_states=tuple(states),
            )
            if int(root_stats.get("roots", -1)) != ROOTS_PER_UPDATE:
                raise RuntimeError("HU semantic long root-count mismatch")
            if int(root_stats.get("ensemble_size", -1)) != ENSEMBLE_SIZE:
                raise RuntimeError("HU semantic long root ensemble-size mismatch")
            if int(memory.seen) <= seen_before:
                raise RuntimeError("HU semantic long roots added no Advantage samples")

            states, member_meta = base.fit_semantic_ensemble(fitter)
            completed_updates = cumulative_update
            block_updates_done = completed_updates - start_updates
            row = {
                "cumulative_update": int(completed_updates),
                "block_update": int(block_updates_done),
                "virtual_seed_iteration": int(virtual_iteration),
                "hu_roots": ROOTS_PER_UPDATE,
                "adv_seen_before": seen_before,
                "adv_seen_after": int(memory.seen),
                "root_stats": dict(root_stats),
                "blind_counts": blind_counts,
                "member_loss_last": [float(x["loss_last"]) for x in member_meta],
                "fit_parallel_wall_seconds": float(member_meta[0]["parallel_wall_seconds"]),
            }
            rows.append(row)
            rows = rows[-20:]

            if block_updates_done % CHECKPOINT_EVERY == 0 or completed_updates == target_updates:
                packed.flush()
                save_resume(
                    resume_path,
                    source=source,
                    sampler=sampler,
                    hu=hu,
                    start_updates=start_updates,
                    target_updates=target_updates,
                    completed_updates=completed_updates,
                    states=states,
                    member_meta=member_meta,
                    migration_parity=migration_parity,
                    rows=rows,
                )
                print(
                    f"HU_SEMANTIC_LONG_RESUME_PASS cumulative_update={completed_updates} "
                    f"block_roots={block_updates_done * ROOTS_PER_UPDATE}",
                    flush=True,
                )

            if block_updates_done % SNAPSHOT_EVERY == 0 or completed_updates == target_updates:
                snap = milestones / f"hu_semantic_ensemble_cum_{completed_updates:05d}.pt"
                save_ensemble(
                    snap,
                    source=source,
                    start_updates=start_updates,
                    target_updates=completed_updates,
                    states=states,
                    member_meta=member_meta,
                    migration_parity=migration_parity,
                )
                print(
                    f"HU_SEMANTIC_LONG_MILESTONE cumulative_update={completed_updates} "
                    f"block_roots={block_updates_done * ROOTS_PER_UPDATE}",
                    flush=True,
                )

            base.atomic_json(
                report_path,
                {
                    "schema": SCHEMA,
                    "status": "RUNNING",
                    "source_checkpoint_sha256": base.EXPECTED_CHECKPOINT_SHA,
                    "source_cumulative_updates": start_updates,
                    "target_cumulative_updates": target_updates,
                    "cumulative_completed_updates": completed_updates,
                    "block_completed_updates": block_updates_done,
                    "roots_per_update": ROOTS_PER_UPDATE,
                    "block_new_hu_roots_so_far": block_updates_done * ROOTS_PER_UPDATE,
                    "cumulative_hu_roots_since_10105": completed_updates * ROOTS_PER_UPDATE,
                    "semantic_dim": base.SEM_DIM,
                    "member_steps": MEMBER_STEPS,
                    "protected_diagnostic_count": PROTECTED_COUNT,
                    "packed_reservoir": packed_meta,
                    "recent_rows": rows,
                    "source_endpoint_resume_sha256": source["resume_sha256"],
                    "source_endpoint_ensemble_sha256": source["ensemble_sha256"],
                    "source_endpoint_report_sha256": source["report_sha256"],
                    "three_handed_training_performed": False,
                    "average_policy_training_performed": False,
                    "strategy_reservoir_collection_performed": False,
                    "sealed_holdout_touched": False,
                    "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
                },
            )
            print(
                "HU_SEMANTIC_LONG_UPDATE "
                + json.dumps(
                    {
                        "cumulative_update": completed_updates,
                        "target_cumulative_updates": target_updates,
                        "block_update": block_updates_done,
                        "virtual_seed_iteration": virtual_iteration,
                        "block_roots": block_updates_done * ROOTS_PER_UPDATE,
                        "adv_seen": int(memory.seen),
                        "fit_seconds": float(member_meta[0]["parallel_wall_seconds"]),
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

    if completed_updates != target_updates:
        raise RuntimeError("HU semantic long block ended before frozen target")

    source_hashes_after = {
        "checkpoint": base.sha256(checkpoint),
        "resume": base.sha256(source_resume),
        "ensemble": base.sha256(source_ensemble),
        "report": base.sha256(source_report),
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("frozen HU source endpoint changed during long continuation")

    save_ensemble(
        final_path,
        source=source,
        start_updates=start_updates,
        target_updates=target_updates,
        states=states,
        member_meta=member_meta,
        migration_parity=migration_parity,
    )
    save_resume(
        resume_path,
        source=source,
        sampler=sampler,
        hu=hu,
        start_updates=start_updates,
        target_updates=target_updates,
        completed_updates=target_updates,
        states=states,
        member_meta=member_meta,
        migration_parity=migration_parity,
        rows=rows,
    )

    block_updates = target_updates - start_updates
    final = {
        "schema": SCHEMA,
        "status": "PASS",
        "research_lane": "HU_SEMANTIC_ENS8_LONG_BLOCK_V1",
        "source_checkpoint_sha256": base.EXPECTED_CHECKPOINT_SHA,
        "source_cumulative_updates": start_updates,
        "target_cumulative_updates": target_updates,
        "cumulative_completed_updates": target_updates,
        "block_additional_updates": block_updates,
        "roots_per_update": ROOTS_PER_UPDATE,
        "block_new_hu_roots": block_updates * ROOTS_PER_UPDATE,
        "cumulative_hu_roots_since_10105": target_updates * ROOTS_PER_UPDATE,
        "virtual_seed_iteration_end": SOURCE_ITERATION + target_updates,
        "semantic_dim": base.SEM_DIM,
        "ensemble_size": ENSEMBLE_SIZE,
        "member_steps": MEMBER_STEPS,
        "migration_parity": migration_parity,
        "protected_diagnostic_count": PROTECTED_COUNT,
        "source_endpoint_resume_sha256": source["resume_sha256"],
        "source_endpoint_ensemble_sha256": source["ensemble_sha256"],
        "source_endpoint_report_sha256": source["report_sha256"],
        "final_semantic_ensemble": str(final_path.resolve()),
        "final_semantic_ensemble_sha256": base.sha256(final_path),
        "resume_state": str(resume_path.resolve()),
        "resume_state_sha256": base.sha256(resume_path),
        "source_unchanged": True,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "three_handed_training_performed": False,
        "average_policy_training_performed": False,
        "strategy_reservoir_collection_performed": False,
        "sealed_holdout_touched": False,
        "wall_seconds_this_invocation": float(time.perf_counter() - started),
        "recent_rows": rows,
        "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
        "interpretation": (
            "PASS means one frozen HU Semantic ENS8 long block completed from the "
            "bound prior endpoint with the same direct-teacher architecture and no "
            "cross-domain training. It is not a strength/promotion result."
        ),
    }
    base.atomic_json(report_path, final)

    print("HU_SEMANTIC_ENS8_LONG_BLOCK_PASS", flush=True)
    print(f"source_cumulative_updates={start_updates}", flush=True)
    print(f"target_cumulative_updates={target_updates}", flush=True)
    print(f"block_new_hu_roots={final['block_new_hu_roots']}", flush=True)
    print(f"cumulative_hu_roots_since_10105={final['cumulative_hu_roots_since_10105']}", flush=True)
    print(f"final_semantic_ensemble_sha256={final['final_semantic_ensemble_sha256']}", flush=True)
    print(f"resume_state_sha256={final['resume_state_sha256']}", flush=True)
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
