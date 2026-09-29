#!/usr/bin/env python3
from __future__ import annotations

"""Ryzen parity/throughput gate for migrating the active 3H semantic long run.

PROJECT_CONTRACT_IDS:
GOV-023,PERF-001,PERF-002,PERF-010,PERF-011,PERF-012,PERF-013,PERF-014,
PERF-015,PERF-016,PERF-017,PERF-019,PERF-020,PERF-022,PERF-024,
TRAIN-020,MODEL-020,RNG-001,RNG-002,RNG-003,CKPT-004,ART-015

This is a bounded gate, not training.  It reads the latest durable long-run
resume, benchmarks exact-parity semantic member fitting on the same reservoir,
and writes a machine-readable recommendation.  It never mutates the resume.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import run_3h_semantic_long_10115_10315 as longrun
import run_3h_semantic_online_pilot_10105 as pilot
import lt3_3h_semantic_parallel_fit as parallel_fit
from spincore_nn.reservoir import UniformReservoir

SCHEMA = "SPINCORE_3H_SEMANTIC_PARALLEL_FIT_GATE_V1"
QUICK_STEPS = 200
PRODUCTION_STEPS = 1600
CANONICAL_THREADS_PER_MEMBER = 8
PROFILES = ((2, 8), (4, 8))
MIN_PRODUCTION_FIT_SPEEDUP = 1.25
MIN_PROJECTED_NET_SAVED_SECONDS = 1800.0
MIN_MEM_AVAILABLE_GIB = 4.0
MAX_SWAP_USED_GIB = 1.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _meminfo():
    rows = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, rest = line.split(":", 1)
            rows[key] = int(rest.strip().split()[0])
    except Exception:
        return None
    return {
        "mem_total_kib": rows.get("MemTotal", 0),
        "mem_available_kib": rows.get("MemAvailable", 0),
        "swap_total_kib": rows.get("SwapTotal", 0),
        "swap_free_kib": rows.get("SwapFree", 0),
    }


def _cpu_stat():
    try:
        parts = Path("/proc/stat").read_text().splitlines()[0].split()
        if not parts or parts[0] != "cpu":
            return None
        vals = [int(x) for x in parts[1:]]
        total = sum(vals)
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return total, idle
    except Exception:
        return None


class ResourceMonitor:
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.mem_available = []
        self.swap_used = []
        self.cpu_busy = []
        self._last_cpu = None

    def _sample(self):
        mem = _meminfo()
        if mem is not None:
            self.mem_available.append(int(mem["mem_available_kib"]))
            self.swap_used.append(
                int(mem["swap_total_kib"]) - int(mem["swap_free_kib"])
            )
        cur = _cpu_stat()
        if cur is not None and self._last_cpu is not None:
            dt = cur[0] - self._last_cpu[0]
            di = cur[1] - self._last_cpu[1]
            if dt > 0:
                self.cpu_busy.append(100.0 * (1.0 - di / dt))
        self._last_cpu = cur

    def _run(self):
        while not self.stop_event.wait(1.0):
            self._sample()
        self._sample()

    def start(self):
        self._sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)

    def report(self):
        return {
            "min_mem_available_gib": (
                min(self.mem_available) / (1024.0 * 1024.0)
                if self.mem_available
                else None
            ),
            "max_swap_used_gib": (
                max(self.swap_used) / (1024.0 * 1024.0)
                if self.swap_used
                else None
            ),
            "cpu_busy_mean_percent": (
                statistics.fmean(self.cpu_busy) if self.cpu_busy else None
            ),
            "cpu_busy_p95_percent": (
                sorted(self.cpu_busy)[
                    min(
                        len(self.cpu_busy) - 1,
                        max(0, math.ceil(0.95 * len(self.cpu_busy)) - 1),
                    )
                ]
                if self.cpu_busy
                else None
            ),
        }


def _member_summaries(rows):
    return [
        {
            "member": int(row["member"]),
            "init_seed": int(row["init_seed"]),
            "batch_seed": int(row["batch_seed"]),
            "steps": int(row["steps"]),
            "loss_last": float(row["loss_last"]),
            "fit_seconds": float(row["fit_seconds"]),
            "state_sha256": parallel_fit.state_digest(row["state"]),
            "pid": int(row["pid"]),
            "maxrss_kib": int(row["maxrss_kib"]),
        }
        for row in rows
    ]


def _exact_rows(reference, candidate):
    try:
        parallel_fit.assert_fit_rows_exact(reference, candidate)
        return True, None
    except AssertionError as exc:
        return False, str(exc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--resume", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    probe_path = args.probe.resolve(strict=True)
    resume_path = args.resume.resolve(strict=True)
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    source_sha = sha256(checkpoint)
    if source_sha != pilot.EXPECTED_SHA:
        raise RuntimeError("source 10105 checkpoint SHA mismatch")

    raw_cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(raw_cp.get("config") or {})
    lr = float(cfg["learning_rate"])
    batch_size = int(cfg["batch_size"])

    resume_sha_before = sha256(resume_path)
    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    if resume.get("schema") != longrun.RESUME_SCHEMA:
        raise RuntimeError("wrong semantic long resume schema")
    if str(resume.get("source_checkpoint_sha256")) != pilot.EXPECTED_SHA:
        raise RuntimeError("semantic resume source checkpoint mismatch")
    completed = int(resume.get("completed_iteration", -1))
    if not longrun.START_ITERATION < completed < longrun.TARGET_ITERATION:
        raise RuntimeError(
            f"benchmark requires an in-progress durable resume, got {completed}"
        )

    member_meta = list(resume.get("member_seed_contract") or [])
    if len(member_meta) != 8:
        raise RuntimeError("semantic resume member metadata drift")
    memory = UniformReservoir.from_state_dict(resume["adv_mem"])
    if len(memory.items) != 2_000_000:
        raise RuntimeError("expected saturated 2M semantic Advantage reservoir")

    probe = torch.load(probe_path, map_location="cpu", weights_only=False)
    if probe.get("schema") != pilot.PROBE_SCHEMA:
        raise RuntimeError("controlled-split probe schema mismatch")
    split_rng = random.Random(int(probe["holdout_seed"]))
    protected = set(
        split_rng.sample(
            range(len(memory.items)),
            int(probe["holdout_size"]),
        )
    )
    if len(protected) != 50_000:
        raise RuntimeError("protected split size drift")
    train_pool = [i for i in range(len(memory.items)) if i not in protected]
    if len(train_pool) != 1_950_000:
        raise RuntimeError("semantic train-pool size drift")

    monitor = ResourceMonitor().start()
    gate_started = time.perf_counter()
    packed = None
    try:
        print("SEMANTIC_PARALLEL_GATE_PRECOMPUTE_BEGIN", flush=True)
        sem_rows, precompute_seconds = pilot.precompute_semantics(memory)
        print(
            f"SEMANTIC_PARALLEL_GATE_PRECOMPUTE_PASS seconds={precompute_seconds:.3f}",
            flush=True,
        )

        pack_dir = work_dir / "packed"
        packed, pack_meta = parallel_fit.PackedSemanticAdvantageReservoir.build(
            memory,
            sem_rows,
            train_pool,
            pack_dir,
        )
        print(
            "SEMANTIC_PARALLEL_GATE_PACK_PASS "
            + json.dumps(
                {
                    "bytes": int(pack_meta["bytes"]),
                    "build_seconds": float(pack_meta["build_seconds"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        quick_contract = parallel_fit.make_fit_contract(
            member_meta,
            learning_rate=lr,
            batch_size=batch_size,
            member_steps=QUICK_STEPS,
        )
        quick_ref_started = time.perf_counter()
        quick_reference = parallel_fit.fit_members_sequential_reference(
            memory,
            sem_rows,
            train_pool,
            quick_contract,
            threads=CANONICAL_THREADS_PER_MEMBER,
        )
        quick_reference_wall = float(time.perf_counter() - quick_ref_started)

        profile_rows = []
        parity_profiles = []
        for concurrency, threads in PROFILES:
            with parallel_fit.ParallelSemanticEnsembleFitter(
                manifest_path=packed.manifest_path,
                contract=quick_contract,
                concurrency=concurrency,
                threads_per_member=threads,
            ) as fitter:
                started = time.perf_counter()
                candidate = fitter.fit()
                wall = float(time.perf_counter() - started)
                exact, error = _exact_rows(quick_reference, candidate)
                row = {
                    "concurrency": concurrency,
                    "threads_per_member": threads,
                    "declared_threads": concurrency * threads,
                    "quick_steps": QUICK_STEPS,
                    "wall_seconds": wall,
                    "sequential_reference_wall_seconds": quick_reference_wall,
                    "speedup_vs_sequential": (
                        quick_reference_wall / wall if wall > 0 else None
                    ),
                    "exact_parity": exact,
                    "parity_error": error,
                    "pool_startup_seconds": float(fitter.startup_seconds),
                    "workers": fitter.worker_pings,
                    "members": _member_summaries(candidate),
                }
            profile_rows.append(row)
            print(
                "SEMANTIC_PARALLEL_GATE_PROFILE "
                + json.dumps(
                    {
                        "concurrency": concurrency,
                        "threads_per_member": threads,
                        "wall_seconds": wall,
                        "speedup": row["speedup_vs_sequential"],
                        "exact_parity": exact,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if exact:
                parity_profiles.append(row)

        if not parity_profiles:
            selected = None
            full = None
            parity_pass = False
        else:
            selected = min(parity_profiles, key=lambda row: row["wall_seconds"])
            parity_pass = True

            # Canonical production reference: call the exact current serial
            # pilot.fit_ensemble implementation at 8 threads and 1600 steps.
            torch.set_num_threads(CANONICAL_THREADS_PER_MEMBER)
            print("SEMANTIC_PARALLEL_GATE_FULL_SERIAL_BEGIN", flush=True)
            _models, serial_states, serial_meta, serial_wall = pilot.fit_ensemble(
                memory,
                sem_rows,
                train_pool,
                member_meta,
                lr=lr,
                batch_size=batch_size,
                iteration=completed + 1,
            )
            serial_rows = []
            for member, state, meta in zip(range(8), serial_states, serial_meta):
                serial_rows.append(
                    {
                        "member": member,
                        "init_seed": int(meta["init_seed"]),
                        "batch_seed": int(meta["batch_seed"]),
                        "steps": int(meta["steps"]),
                        "loss_last": float(meta["loss_last"]),
                        "fit_seconds": float(meta["fit_seconds"]),
                        "state": state,
                        "pid": int(os.getpid()),
                        "maxrss_kib": 0,
                    }
                )
            print(
                f"SEMANTIC_PARALLEL_GATE_FULL_SERIAL_PASS seconds={serial_wall:.3f}",
                flush=True,
            )

            production_contract = parallel_fit.make_fit_contract(
                member_meta,
                learning_rate=lr,
                batch_size=batch_size,
                member_steps=PRODUCTION_STEPS,
            )
            print(
                "SEMANTIC_PARALLEL_GATE_FULL_PARALLEL_BEGIN "
                f"profile={selected['concurrency']}x{selected['threads_per_member']}",
                flush=True,
            )
            with parallel_fit.ParallelSemanticEnsembleFitter(
                manifest_path=packed.manifest_path,
                contract=production_contract,
                concurrency=int(selected["concurrency"]),
                threads_per_member=int(selected["threads_per_member"]),
            ) as fitter:
                started = time.perf_counter()
                candidate = fitter.fit()
                candidate_wall = float(time.perf_counter() - started)
                tensor_exact = all(
                    parallel_fit.tensor_states_equal(
                        serial_rows[i]["state"], candidate[i]["state"]
                    )
                    for i in range(8)
                )
                final_loss_exact = all(
                    float(serial_rows[i]["loss_last"])
                    == float(candidate[i]["loss_last"])
                    for i in range(8)
                )
                meta_exact = all(
                    int(serial_rows[i][key]) == int(candidate[i][key])
                    for i in range(8)
                    for key in ("init_seed", "batch_seed", "steps")
                )
                full = {
                    "selected_profile": {
                        "concurrency": int(selected["concurrency"]),
                        "threads_per_member": int(selected["threads_per_member"]),
                        "declared_threads": int(selected["declared_threads"]),
                    },
                    "serial_fit_wall_seconds": float(serial_wall),
                    "parallel_fit_wall_seconds": candidate_wall,
                    "fit_speedup": (
                        float(serial_wall) / candidate_wall
                        if candidate_wall > 0
                        else None
                    ),
                    "tensor_exact": tensor_exact,
                    "final_loss_exact": final_loss_exact,
                    "member_metadata_exact": meta_exact,
                    "serial_members": _member_summaries(serial_rows),
                    "parallel_members": _member_summaries(candidate),
                    "pool_startup_seconds": float(fitter.startup_seconds),
                    "workers": fitter.worker_pings,
                }
            print(
                "SEMANTIC_PARALLEL_GATE_FULL_PARALLEL_PASS "
                + json.dumps(
                    {
                        "wall_seconds": candidate_wall,
                        "speedup": full["fit_speedup"],
                        "tensor_exact": tensor_exact,
                        "final_loss_exact": final_loss_exact,
                        "member_metadata_exact": meta_exact,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        online_rows = list(resume.get("online_rows") or [])
        tail = online_rows[-min(10, len(online_rows)) :]
        historical_tree = [
            float(row["tree"]["seconds"])
            for row in tail
            if row.get("tree") and row["tree"].get("seconds") is not None
        ]
        historical_fit = [
            float(row["fit_seconds"])
            for row in tail
            if row.get("fit_seconds") is not None
        ]
        avg_tree = (
            statistics.fmean(historical_tree) if historical_tree else None
        )
        avg_historical_fit = (
            statistics.fmean(historical_fit) if historical_fit else None
        )
        remaining = int(longrun.TARGET_ITERATION - completed)

        if full is not None and avg_tree is not None:
            serial_per_iteration = avg_tree + float(
                full["serial_fit_wall_seconds"]
            )
            parallel_per_iteration = avg_tree + float(
                full["parallel_fit_wall_seconds"]
            )
            projected_serial = remaining * serial_per_iteration
            projected_parallel = (
                float(pack_meta["build_seconds"])
                + float(full["pool_startup_seconds"])
                + remaining * parallel_per_iteration
            )
            projected_saved = projected_serial - projected_parallel
            projected_speedup = (
                projected_serial / projected_parallel
                if projected_parallel > 0
                else None
            )
        else:
            serial_per_iteration = None
            parallel_per_iteration = None
            projected_serial = None
            projected_parallel = None
            projected_saved = None
            projected_speedup = None

        resources = monitor.report()
        resume_sha_after = sha256(resume_path)
        resume_unchanged = resume_sha_after == resume_sha_before

        criteria = {
            "quick_profile_exact_parity_exists": bool(parity_pass),
            "production_tensor_exact": bool(
                full is not None and full["tensor_exact"]
            ),
            "production_final_loss_exact": bool(
                full is not None and full["final_loss_exact"]
            ),
            "production_member_metadata_exact": bool(
                full is not None and full["member_metadata_exact"]
            ),
            "production_fit_speedup_at_least_1_25x": bool(
                full is not None
                and float(full["fit_speedup"] or 0.0)
                >= MIN_PRODUCTION_FIT_SPEEDUP
            ),
            "projected_net_remaining_savings_at_least_30m": bool(
                projected_saved is not None
                and projected_saved >= MIN_PROJECTED_NET_SAVED_SECONDS
            ),
            "memory_headroom_at_least_4gib": bool(
                resources["min_mem_available_gib"] is not None
                and resources["min_mem_available_gib"]
                >= MIN_MEM_AVAILABLE_GIB
            ),
            "swap_used_no_more_than_1gib": bool(
                resources["max_swap_used_gib"] is not None
                and resources["max_swap_used_gib"] <= MAX_SWAP_USED_GIB
            ),
            "resume_artifact_unchanged": bool(resume_unchanged),
        }
        passed = all(criteria.values())

        report = {
            "schema": SCHEMA,
            "status": "PASS" if passed else "FAIL",
            "source_checkpoint_sha256": source_sha,
            "resume_path": str(resume_path),
            "resume_sha256_before": resume_sha_before,
            "resume_sha256_after": resume_sha_after,
            "completed_iteration_at_gate": completed,
            "target_iteration": int(longrun.TARGET_ITERATION),
            "remaining_iterations_at_gate": remaining,
            "learning_rate": lr,
            "batch_size": batch_size,
            "ensemble_size": 8,
            "production_member_steps": PRODUCTION_STEPS,
            "canonical_threads_per_member": CANONICAL_THREADS_PER_MEMBER,
            "quick_steps": QUICK_STEPS,
            "quick_reference_wall_seconds": quick_reference_wall,
            "profiles": profile_rows,
            "selected_profile": (
                {
                    "concurrency": int(selected["concurrency"]),
                    "threads_per_member": int(selected["threads_per_member"]),
                    "declared_threads": int(selected["declared_threads"]),
                }
                if selected is not None
                else None
            ),
            "production_parity_and_throughput": full,
            "historical_tail_iterations": len(tail),
            "historical_tree_seconds_mean": avg_tree,
            "historical_serial_fit_seconds_mean": avg_historical_fit,
            "projected_serial_seconds_remaining": projected_serial,
            "projected_parallel_seconds_remaining_including_pack_startup": projected_parallel,
            "projected_net_seconds_saved": projected_saved,
            "projected_end_to_end_speedup": projected_speedup,
            "pack": pack_meta,
            "precompute_seconds": precompute_seconds,
            "resources": resources,
            "criteria": criteria,
            "gate_wall_seconds": float(time.perf_counter() - gate_started),
            "interpretation": (
                "PASS authorizes updating the parallel-resume stage manifest to "
                "READY for the exact selected Ryzen profile. It does not itself "
                "interrupt or resume the long trainer."
            ),
        }
        atomic_json(report, args.report.resolve())
        print(
            "SEMANTIC_PARALLEL_FIT_GATE_PASS"
            if passed
            else "SEMANTIC_PARALLEL_FIT_GATE_FAIL",
            flush=True,
        )
        print("report=" + str(args.report.resolve()), flush=True)
        return 0 if passed else 5
    finally:
        monitor.close()
        if packed is not None:
            packed.close()


if __name__ == "__main__":
    raise SystemExit(main())
