#!/usr/bin/env python3
from __future__ import annotations

"""HU-only ENS8 teacher continuation from frozen 10105.

This runner deliberately leaves THREE_HANDED untouched. It resumes the frozen
10105 TRUE_HEADS_UP Advantage reservoir and ENS8 behavior, adds deterministic
HU roots only, and refits the same fresh400 eight-member ensemble after every
HU update. No AveragePolicy/strategy-reservoir training and no strength test are
performed.

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
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import lt3_hu_ens8_parallel_fit as par
import lt3_parallel_continuation_core as core

from spincore.lean_functional_training import (
    load_checkpoint,
    _root_deck_seed,
    _root_policy_seed,
)
from spincore.lean_parallel import ParallelRootExecutor, RootJob
from spincore_nn.reservoir import UniformReservoir
from spincore.solver import SolverLibrary

SCHEMA = "SPINCORE_HU_ENS8_CONTINUATION_RESULT_V1"
RESUME_SCHEMA = "SPINCORE_HU_ENS8_CONTINUATION_RESUME_V1"
ENSEMBLE_SCHEMA = "SPINCORE_LT3_HU_ENS8_CURRENT_STATE_V1"

DOMAIN_HU = "TRUE_HEADS_UP"
SOURCE_ITERATION = 10105
ROOTS_PER_UPDATE = 273
UPDATES_IN_BLOCK = 469
NEW_ROOTS = ROOTS_PER_UPDATE * UPDATES_IN_BLOCK
CHECKPOINT_EVERY = 50
ENSEMBLE_SNAPSHOT_EVERY = 100
ENSEMBLE_SIZE = 8
MEMBER_STEPS = 400

EXPECTED_CHECKPOINT_SHA = "f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_ENSEMBLE_SHA = "8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"


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


def save_ensemble(
    path: Path,
    *,
    source_checkpoint: Path,
    source_ensemble: Path,
    completed_updates: int,
    states,
    member_meta,
) -> None:
    virtual_iteration = SOURCE_ITERATION + int(completed_updates)
    atomic_torch(
        path,
        {
            "schema": ENSEMBLE_SCHEMA,
            "research_lane": "HU_ENS8_TEACHER_ONLY_10105_PLUS469_V1",
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "source_ensemble_state": str(source_ensemble.resolve()),
            "source_ensemble_sha256": EXPECTED_ENSEMBLE_SHA,
            "source_iteration": SOURCE_ITERATION,
            "completed_iteration": virtual_iteration,
            "hu_only_updates": int(completed_updates),
            "hu_only_roots": int(completed_updates) * ROOTS_PER_UPDATE,
            "virtual_seed_iteration_note": (
                "completed_iteration is used only for deterministic HU root seed continuity; "
                "THREE_HANDED was not trained"
            ),
            "ensemble_size": ENSEMBLE_SIZE,
            "member_steps": MEMBER_STEPS,
            "member_meta": list(member_meta),
            "members": states,
            "semantics": "mean raw Advantage outputs then unchanged lean_regret_matching_policy",
            "fit_seed_contract": "fixed ENS8_A init/batch seeds reused every HU update",
            "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
        },
    )


def save_resume(
    path: Path,
    *,
    source_checkpoint: Path,
    source_ensemble: Path,
    sampler,
    hu,
    completed_updates: int,
    states,
    member_meta,
    rows,
) -> None:
    atomic_torch(
        path,
        {
            "schema": RESUME_SCHEMA,
            "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
            "source_ensemble_sha256": EXPECTED_ENSEMBLE_SHA,
            "source_iteration": SOURCE_ITERATION,
            "target_updates": UPDATES_IN_BLOCK,
            "roots_per_update": ROOTS_PER_UPDATE,
            "completed_updates": int(completed_updates),
            "virtual_seed_iteration": SOURCE_ITERATION + int(completed_updates),
            "sampler_rng_state": sampler.rng.bit_generator.state,
            "hu_adv_mem": hu.bundle.adv_mem.state_dict(),
            "hu_counters": dict(hu.bundle.counters),
            "hu_states": states,
            "member_meta": list(member_meta),
            "recent_rows": list(rows[-20:]),
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_ensemble": str(source_ensemble.resolve()),
            "three_handed_training_performed": False,
            "strategy_reservoir_collection_performed": False,
            "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
        },
    )


def restore_resume(path: Path, *, sampler, hu, source_checkpoint: Path, source_ensemble: Path):
    p = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if p.get("schema") != RESUME_SCHEMA:
        raise RuntimeError("wrong HU continuation resume schema")
    if str(p.get("source_checkpoint_sha256")) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("HU resume source checkpoint mismatch")
    if str(p.get("source_ensemble_sha256")) != EXPECTED_ENSEMBLE_SHA:
        raise RuntimeError("HU resume source ensemble mismatch")
    if int(p.get("source_iteration", -1)) != SOURCE_ITERATION:
        raise RuntimeError("HU resume source iteration drift")
    if int(p.get("target_updates", -1)) != UPDATES_IN_BLOCK:
        raise RuntimeError("HU resume target drift")
    if int(p.get("roots_per_update", -1)) != ROOTS_PER_UPDATE:
        raise RuntimeError("HU resume roots/update drift")
    if bool(p.get("three_handed_training_performed")):
        raise RuntimeError("HU resume unexpectedly records 3H training")
    if bool(p.get("strategy_reservoir_collection_performed")):
        raise RuntimeError("HU resume unexpectedly records strategy collection")

    completed = int(p.get("completed_updates", -1))
    if not 0 < completed <= UPDATES_IN_BLOCK:
        raise RuntimeError("HU resume completed update out of range")

    hu.bundle.adv_mem = UniformReservoir.from_state_dict(p["hu_adv_mem"])
    hu.session.bundle = hu.bundle
    hu.session.collector.advantage_memory = hu.bundle.adv_mem
    hu.bundle.counters.clear()
    hu.bundle.counters.update(dict(p["hu_counters"]))
    sampler.rng.bit_generator.state = p["sampler_rng_state"]

    states = tuple(p.get("hu_states") or [])
    if len(states) != ENSEMBLE_SIZE or not states_finite(states):
        raise RuntimeError("HU resume ensemble invalid")
    par.install_ensemble_behavior(hu, states)

    member_meta = list(p.get("member_meta") or [])
    rows = list(p.get("recent_rows") or [])
    return completed, states, member_meta, rows


def make_jobs(*, sampler, seed: int, virtual_iteration: int):
    jobs = []
    blind_counts = {}
    for local_root in range(ROOTS_PER_UPDATE):
        episode = sampler.sample_episode(force_domain=DOMAIN_HU)
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
    p.add_argument("--parallel-concurrency", type=int, default=4)
    p.add_argument("--parallel-member-threads", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(int(args.parallel_member_threads))

    checkpoint = args.checkpoint.resolve(strict=True)
    source_ensemble = args.source_ensemble.resolve(strict=True)
    solver_path = args.solver.resolve(strict=True)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("frozen 10105 checkpoint SHA mismatch")
    if sha256(source_ensemble) != EXPECTED_ENSEMBLE_SHA:
        raise RuntimeError("frozen 10105 HU ensemble SHA mismatch")

    source_hashes_before = {
        "checkpoint": sha256(checkpoint),
        "ensemble": sha256(source_ensemble),
    }

    solver = SolverLibrary(solver_path)
    seed, config, completed_source, sampler, runtimes, _history, finalized = load_checkpoint(
        checkpoint, solver=solver
    )
    if completed_source != SOURCE_ITERATION or not finalized:
        raise RuntimeError("expected finalized source checkpoint @10105")
    if int(config.roots_per_iteration) != 600:
        raise RuntimeError("source roots/iteration drift")
    if abs(float(config.heads_up_prob) - 0.4548) > 1e-12:
        raise RuntimeError("source HU probability drift")
    if int(config.hu_advantage_steps or 0) != MEMBER_STEPS:
        raise RuntimeError("source HU fresh400 contract drift")
    expected_hu_roots = int(round(int(config.roots_per_iteration) * float(config.heads_up_prob)))
    if expected_hu_roots != ROOTS_PER_UPDATE:
        raise RuntimeError(
            f"historical HU roots/update drift: expected {ROOTS_PER_UPDATE}, got {expected_hu_roots}"
        )

    hu = runtimes[DOMAIN_HU]
    states, source_member_meta, source_schema = core.load_ensemble_behavior(
        hu, source_ensemble, SOURCE_ITERATION
    )
    if source_schema != ENSEMBLE_SCHEMA:
        raise RuntimeError("unexpected source HU ensemble schema")
    if not states_finite(states):
        raise RuntimeError("source HU ensemble contains non-finite tensors")

    resume_path = out / "resume_state.pt"
    final_ensemble_path = out / "hu_ensemble_state_plus469.pt"
    report_path = out / "hu_ens8_continuation_10105_plus469.json"
    snapshots_dir = out / "milestones"
    work_dir = out / "work" / "packed_hu_adv"

    completed_updates = 0
    member_meta = list(source_member_meta)
    rows = []
    if resume_path.exists():
        completed_updates, states, member_meta, rows = restore_resume(
            resume_path,
            sampler=sampler,
            hu=hu,
            source_checkpoint=checkpoint,
            source_ensemble=source_ensemble,
        )
        print(
            f"HU_ENS8_CONTINUATION_RESUME completed_updates={completed_updates} "
            f"virtual_seed_iteration={SOURCE_ITERATION + completed_updates}",
            flush=True,
        )

    if completed_updates == UPDATES_IN_BLOCK:
        if not final_ensemble_path.exists() or not report_path.exists():
            raise RuntimeError("completed HU resume exists without durable final artifacts")
        print("HU_ENS8_CONTINUATION_ALREADY_COMPLETE", flush=True)
        return 0

    if int(args.parallel_concurrency) * int(args.parallel_member_threads) > 32:
        raise RuntimeError("HU fit profile oversubscribes 32 logical threads")

    mirror, packed_meta = par.PackedAdvantageReservoir.build(hu.bundle.adv_mem, work_dir)
    mirror.bind_authoritative(hu.bundle.adv_mem)
    fit_contract = par.make_fit_contract(hu, config)
    fitter = par.ParallelEnsembleFitter(
        manifest_path=mirror.manifest_path,
        contract=fit_contract,
        concurrency=int(args.parallel_concurrency),
        threads_per_member=int(args.parallel_member_threads),
    )
    root_executor = ParallelRootExecutor(solver_path, int(args.workers))

    started = time.perf_counter()
    try:
        for update in range(completed_updates + 1, UPDATES_IN_BLOCK + 1):
            virtual_iteration = SOURCE_ITERATION + update
            jobs, blind_counts = make_jobs(
                sampler=sampler,
                seed=int(seed),
                virtual_iteration=int(virtual_iteration),
            )
            seen_before = int(hu.bundle.adv_mem.seen)
            root_stats = root_executor.collect(
                domain=DOMAIN_HU,
                bundle=hu.bundle,
                iteration=int(virtual_iteration),
                exact_opponent_levels=int(config.exact_opponent_levels),
                jobs=jobs,
                hu_preflop_board_average_k=1,
                ensemble_model_states=states,
            )
            if int(root_stats.get("roots", -1)) != ROOTS_PER_UPDATE:
                raise RuntimeError("HU root-count mismatch")
            if int(root_stats.get("ensemble_size", -1)) != ENSEMBLE_SIZE:
                raise RuntimeError("HU root behavior ensemble-size mismatch")
            if int(hu.bundle.adv_mem.seen) <= seen_before:
                raise RuntimeError("HU Advantage reservoir did not receive new samples")

            states, member_meta, fit_seconds, last_optimizer = fitter.fit()
            losses = [float(x["loss_last"]) for x in member_meta]
            if len(states) != ENSEMBLE_SIZE or len(losses) != ENSEMBLE_SIZE:
                raise RuntimeError("HU fresh400 fit returned wrong ensemble size")
            if not all(math.isfinite(x) for x in losses):
                raise RuntimeError("HU fresh400 fit produced non-finite loss")
            if not states_finite(states):
                raise RuntimeError("HU fresh400 fit produced non-finite model tensor")
            par.install_parallel_result(
                hu, states, last_optimizer, config=config
            )
            par.install_ensemble_behavior(hu, states)

            completed_updates = update
            row = {
                "update": int(update),
                "virtual_seed_iteration": int(virtual_iteration),
                "hu_roots": ROOTS_PER_UPDATE,
                "adv_seen_before": seen_before,
                "adv_seen_after": int(hu.bundle.adv_mem.seen),
                "root_stats": dict(root_stats),
                "blind_counts": blind_counts,
                "fit_seconds": float(fit_seconds),
                "member_loss_last": losses,
            }
            rows.append(row)
            rows = rows[-20:]

            checkpoint_due = (
                update % CHECKPOINT_EVERY == 0 or update == UPDATES_IN_BLOCK
            )
            snapshot_due = (
                update % ENSEMBLE_SNAPSHOT_EVERY == 0 or update == UPDATES_IN_BLOCK
            )
            if checkpoint_due:
                mirror.flush()
                save_resume(
                    resume_path,
                    source_checkpoint=checkpoint,
                    source_ensemble=source_ensemble,
                    sampler=sampler,
                    hu=hu,
                    completed_updates=completed_updates,
                    states=states,
                    member_meta=member_meta,
                    rows=rows,
                )
                print(
                    f"HU_ENS8_CONTINUATION_RESUME_PASS update={update}",
                    flush=True,
                )
            if snapshot_due:
                snap = snapshots_dir / f"hu_ensemble_state_update_{update:04d}.pt"
                save_ensemble(
                    snap,
                    source_checkpoint=checkpoint,
                    source_ensemble=source_ensemble,
                    completed_updates=completed_updates,
                    states=states,
                    member_meta=member_meta,
                )
                print(
                    f"HU_ENS8_CONTINUATION_MILESTONE update={update} "
                    f"hu_roots_total={update * ROOTS_PER_UPDATE}",
                    flush=True,
                )

            atomic_json(
                report_path,
                {
                    "schema": SCHEMA,
                    "status": "RUNNING",
                    "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
                    "source_ensemble_sha256": EXPECTED_ENSEMBLE_SHA,
                    "source_iteration": SOURCE_ITERATION,
                    "target_updates": UPDATES_IN_BLOCK,
                    "completed_updates": completed_updates,
                    "roots_per_update": ROOTS_PER_UPDATE,
                    "new_hu_roots_so_far": completed_updates * ROOTS_PER_UPDATE,
                    "three_handed_training_performed": False,
                    "average_policy_training_performed": False,
                    "strategy_reservoir_collection_performed": False,
                    "sealed_holdout_touched": False,
                    "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
                    "packed_reservoir": packed_meta,
                    "recent_rows": rows,
                },
            )
            print(
                "HU_ENS8_CONTINUATION_UPDATE "
                + json.dumps(
                    {
                        "update": update,
                        "target": UPDATES_IN_BLOCK,
                        "virtual_seed_iteration": virtual_iteration,
                        "hu_roots_total": update * ROOTS_PER_UPDATE,
                        "adv_seen": int(hu.bundle.adv_mem.seen),
                        "fit_seconds": float(fit_seconds),
                        "member_loss_last": [round(x, 6) for x in losses],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        root_executor.close()
        fitter.close()
        mirror.flush()
        mirror.close()

    source_hashes_after = {
        "checkpoint": sha256(checkpoint),
        "ensemble": sha256(source_ensemble),
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("frozen 10105 HU source artifacts changed during continuation")

    save_ensemble(
        final_ensemble_path,
        source_checkpoint=checkpoint,
        source_ensemble=source_ensemble,
        completed_updates=completed_updates,
        states=states,
        member_meta=member_meta,
    )
    final_resume_sha = sha256(resume_path)
    final_ensemble_sha = sha256(final_ensemble_path)
    final = {
        "schema": SCHEMA,
        "status": "PASS",
        "research_lane": "HU_ENS8_TEACHER_ONLY_10105_PLUS469_V1",
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "source_ensemble_sha256": EXPECTED_ENSEMBLE_SHA,
        "source_iteration": SOURCE_ITERATION,
        "completed_updates": completed_updates,
        "virtual_seed_iteration_end": SOURCE_ITERATION + completed_updates,
        "roots_per_update": ROOTS_PER_UPDATE,
        "new_hu_roots": completed_updates * ROOTS_PER_UPDATE,
        "expected_new_hu_roots": NEW_ROOTS,
        "ensemble_size": ENSEMBLE_SIZE,
        "member_fit_steps": MEMBER_STEPS,
        "three_handed_training_performed": False,
        "average_policy_training_performed": False,
        "strategy_reservoir_collection_performed": False,
        "sealed_holdout_touched": False,
        "source_unchanged": True,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "final_ensemble": str(final_ensemble_path.resolve()),
        "final_ensemble_sha256": final_ensemble_sha,
        "resume_state": str(resume_path.resolve()),
        "resume_state_sha256": final_resume_sha,
        "recent_rows": rows,
        "wall_seconds": float(time.perf_counter() - started),
        "production_status": "RESEARCH_ONLY_NOT_PROMOTED",
        "interpretation": (
            "PASS means the frozen HU ENS8 fresh400 teacher architecture accumulated "
            "one additional HU-only root-balanced block without integrity failure. "
            "It is not a strength or promotion result."
        ),
    }
    atomic_json(report_path, final)

    if completed_updates != UPDATES_IN_BLOCK:
        raise RuntimeError("HU continuation ended before the frozen block target")
    if int(final["new_hu_roots"]) != NEW_ROOTS:
        raise RuntimeError("HU continuation root budget drift")

    print("HU_ENS8_CONTINUATION_10105_PLUS469_PASS", flush=True)
    print(f"new_hu_roots={final['new_hu_roots']}", flush=True)
    print(f"final_ensemble_sha256={final_ensemble_sha}", flush=True)
    print(f"resume_state_sha256={final_resume_sha}", flush=True)
    print(f"report={report_path}", flush=True)
    print("STOP: training endpoint only; no strength or promotion.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
