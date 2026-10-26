#!/usr/bin/env python3
from __future__ import annotations

"""Target-Ryzen performance/parity gate for the frozen post-15315 K=8 rebuild.

PROJECT_CONTRACT_IDS:
PERF-001,PERF-002,TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,
VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,
VALID-038,SAFE-001,SAFE-010,SAFE-011,CKPT-001,ART-001

This gate changes no scientific budget or seed.  It validates only an execution
optimization already required by the frozen 2026-10-01 preregistration:
parallelize independent teacher-target collection streams without loading the
multi-GB 10105 checkpoint into collection workers, while leaving the canonical
8-thread fit path unchanged.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import resource
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_fresh_semantic_strategy_independent500_10105 as confirm
import post15315_collection as collect
from audit_3h_average_policy_semantic_continuation_10105 import semantic_vector_from_obs
from spincore.solver import SolverLibrary
from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.training import train_step

SCHEMA = "SPINCORE_POST15315_MULTIREBUILD_PERFORMANCE_GATE_V1"
PROTOCOL_SCHEMA = "SPINCORE_POST15315_MULTIREBUILD_PREREG_V1"
LONG_SCHEMA = "SPINCORE_3H_SEMANTIC_LONG_CONTINUATION_10315_15315_V1"
ENSEMBLE_SCHEMA = "SPINCORE_3H_SEMANTIC_LONG_ENSEMBLE_V1"
TARGET_ITERATION = 15315
EXPECTED_TEACHER_SHA256 = "ac49b77c8ec3204e07f6d44d3718c68880e60675c5521f538438a3fa3ceae746"

# Exact reduced member-shaped workload: 1/40 of every collection budget.
SCALE = 40
REDUCED_ORDINARY_EPISODES = 200
REDUCED_AUGMENT_EPISODES = 4500
REDUCED_SPECIALIST_EPISODES = 7500
THREAD_PROFILES = (1, 2, 4, 8)
PARALLEL_WORKERS = 4
MIN_SPEEDUP = 1.25
MIN_PROJECTED_SAVINGS_SECONDS = 1800.0
MIN_MEM_AVAILABLE_GIB = 4.0
MAX_SWAP_USED_GIB = 1.0
MINI_FIT_STEPS = 25

_MODELS = None
_SOLVER = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _set_threads(n: int) -> None:
    n = int(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["SPINCORE_TORCH_THREADS"] = str(n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _load_teacher(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != ENSEMBLE_SCHEMA:
        raise RuntimeError("wrong teacher schema")
    if int(payload.get("completed_iteration", -1)) != TARGET_ITERATION:
        raise RuntimeError("wrong teacher iteration")
    states = list(payload.get("members") or [])
    if len(states) != 8:
        raise RuntimeError("expected eight teacher members")
    return [distill.load_semantic_advantage(x) for x in states]


def _collection_init(teacher_path: str, solver_path: str, threads: int) -> None:
    global _MODELS, _SOLVER
    _set_threads(int(threads))
    _MODELS = _load_teacher(teacher_path)
    _SOLVER = SolverLibrary(Path(solver_path).resolve(strict=True))


def _sample_key_digest(train, hold) -> dict:
    return {
        "train_count": len(train),
        "holdout_count": len(hold),
        "train_digest": collect.sample_digest(train),
        "holdout_digest": collect.sample_digest(hold),
    }


def _strong_unique(train, hold):
    unique = collect.unique_strong_from_split(train, hold)
    return {
        **_sample_key_digest(train, hold),
        "unique_count": len(unique),
        "unique_digest": collect.sample_digest(unique),
    }


def _member_collection(task):
    member, seeds = task
    started = time.perf_counter()

    ordinary_train, ordinary_hold, ordinary_stats = collect.collect_fresh_split(
        _SOLVER,
        _MODELS,
        REDUCED_ORDINARY_EPISODES,
        master_seed=int(seeds["base_collection_seed"]),
        sample_iteration=TARGET_ITERATION,
        progress_prefix=None,
    )
    aug_train, aug_hold, aug_stats = collect.collect_strong_split(
        _SOLVER,
        _MODELS,
        REDUCED_AUGMENT_EPISODES,
        master_seed=int(seeds["augment_collection_seed"]),
        sample_iteration=TARGET_ITERATION,
        progress_prefix=None,
    )
    sp_train, sp_hold, sp_stats = collect.collect_strong_split(
        _SOLVER,
        _MODELS,
        REDUCED_SPECIALIST_EPISODES,
        master_seed=int(seeds["specialist_collection_seed"]),
        sample_iteration=TARGET_ITERATION,
        progress_prefix=None,
    )

    return {
        "member": int(member),
        "ordinary": {
            "stats": collect.canonical_collection_stats(ordinary_stats),
            **_sample_key_digest(ordinary_train, ordinary_hold),
        },
        "augmentation": {
            "stats": collect.canonical_collection_stats(aug_stats),
            **_strong_unique(aug_train, aug_hold),
        },
        "specialist": {
            "stats": collect.canonical_collection_stats(sp_stats),
            **_strong_unique(sp_train, sp_hold),
        },
        "wall_seconds": float(time.perf_counter() - started),
        "maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "pid": int(os.getpid()),
    }


def _serial_members(teacher: str, solver: str, tasks, threads: int):
    _collection_init(teacher, solver, threads)
    started = time.perf_counter()
    rows = [_member_collection(task) for task in tasks]
    return {
        "rows": rows,
        "wall_seconds": float(time.perf_counter() - started),
        "maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _isolated_serial(teacher: str, solver: str, tasks, threads: int):
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        return list(
            pool.map(
                _serial_wrapper,
                [(teacher, solver, tasks, int(threads))],
                chunksize=1,
            )
        )[0]


def _serial_wrapper(args):
    return _serial_members(*args)


def _parallel_members(teacher: str, solver: str, tasks, threads: int):
    ctx = mp.get_context("spawn")
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=min(PARALLEL_WORKERS, len(tasks)),
        mp_context=ctx,
        initializer=_collection_init,
        initargs=(teacher, solver, int(threads)),
    ) as pool:
        rows = list(pool.map(_member_collection, tasks, chunksize=1))
    rows.sort(key=lambda x: int(x["member"]))
    return {
        "rows": rows,
        "wall_seconds": float(time.perf_counter() - started),
        "worker_maxrss_kib": max(int(x["maxrss_kib"]) for x in rows),
    }


def _rows_exact(a, b) -> bool:
    if len(a) != len(b):
        return False
    keys = ("member", "ordinary", "augmentation", "specialist")
    return all(
        all(x[k] == y[k] for k in keys)
        for x, y in zip(
            sorted(a, key=lambda r: int(r["member"])),
            sorted(b, key=lambda r: int(r["member"])),
        )
    )


def _tensor_digest(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str(arr.dtype).encode("ascii"))
        h.update(str(tuple(arr.shape)).encode("ascii"))
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _fit_parity_worker(task):
    checkpoint, teacher_path, solver_path, member_seed, fit_seed = task
    _set_threads(8)

    cp = Path(checkpoint).resolve(strict=True)
    payload = torch.load(cp, map_location="cpu", weights_only=False)
    d3 = (payload.get("domains") or {}).get(distill.DOMAIN) or {}
    cfg = dict(payload.get("config") or {})
    models = _load_teacher(teacher_path)
    solver = SolverLibrary(Path(solver_path).resolve(strict=True))

    train, hold, stats = collect.collect_fresh_split(
        solver,
        models,
        REDUCED_ORDINARY_EPISODES,
        master_seed=int(member_seed),
        sample_iteration=TARGET_ITERATION,
        progress_prefix=None,
    )
    if not train:
        raise RuntimeError("mini fit has no train samples")

    _, _, model, opt = confirm.initial_policy_pair(d3, cfg)
    rng = random.Random(int(fit_seed))
    bs = int(cfg["batch_size"])
    losses = []
    started = time.perf_counter()
    for _ in range(MINI_FIT_STEPS):
        idx = rng.sample(range(len(train)), min(bs, len(train)))
        ss = [train[i] for i in idx]
        b, t, w = vectorized_batch(ss, "cpu")
        sb = dict(b)
        sb["semantic"] = torch.tensor(
            np.asarray(
                [semantic_vector_from_obs(s.observation) for s in ss],
                dtype=np.float32,
            ),
            dtype=torch.float32,
        )
        losses.append(float(train_step(model, opt, sb, t, w, "strategy")))

    return {
        "collection_stats": collect.canonical_collection_stats(stats),
        "train_digest": collect.sample_digest(train),
        "holdout_digest": collect.sample_digest(hold),
        "losses": losses,
        "loss_digest": hashlib.sha256(
            json.dumps(losses, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        ).hexdigest(),
        "model_tensor_digest": _tensor_digest(model),
        "wall_seconds": float(time.perf_counter() - started),
        "maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _isolated_fit(task):
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        return list(pool.map(_fit_parity_worker, [task], chunksize=1))[0]


def _meminfo():
    vals = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, rest = line.split(":", 1)
            vals[key] = int(rest.strip().split()[0])
    except Exception:
        return None
    return {
        "mem_available_kib": vals.get("MemAvailable", 0),
        "swap_total_kib": vals.get("SwapTotal", 0),
        "swap_free_kib": vals.get("SwapFree", 0),
    }


class Monitor:
    def __init__(self):
        self.stop = threading.Event()
        self.mem = []
        self.swap = []
        self.thread = None

    def _sample(self):
        m = _meminfo()
        if m:
            self.mem.append(int(m["mem_available_kib"]))
            self.swap.append(int(m["swap_total_kib"] - m["swap_free_kib"]))

    def _run(self):
        while not self.stop.wait(0.5):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        self._sample()

    def report(self):
        return {
            "min_mem_available_gib": (
                min(self.mem) / (1024 ** 2) if self.mem else None
            ),
            "max_swap_used_gib": (
                max(self.swap) / (1024 ** 2) if self.swap else None
            ),
        }


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--long-report", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    cp = args.checkpoint.resolve(strict=True)
    teacher = args.teacher.resolve(strict=True)
    solver = args.solver.resolve(strict=True)
    long_report = args.long_report.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    report_path = args.report.resolve()

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise RuntimeError("wrong post15315 protocol schema")
    if protocol.get("status") != "PREREGISTERED_BLOCKED_TEACHER_15315_AND_PERFORMANCE_GATE":
        raise RuntimeError("unexpected post15315 protocol lifecycle state")
    if int(protocol["teacher"]["required_iteration"]) != TARGET_ITERATION:
        raise RuntimeError("protocol teacher iteration drift")

    if sha256(cp) != distill.EXPECTED_SHA:
        raise RuntimeError("frozen 10105 checkpoint SHA mismatch")
    if sha256(teacher) != EXPECTED_TEACHER_SHA256:
        raise RuntimeError("frozen 15315 teacher SHA mismatch")

    lr = json.loads(long_report.read_text(encoding="utf-8"))
    if lr.get("schema") != LONG_SCHEMA:
        raise RuntimeError("wrong 10315->15315 long report schema")
    if lr.get("status") != "PASS" or int(lr.get("completed_iteration", -1)) != TARGET_ITERATION:
        raise RuntimeError("10315->15315 long report is not final PASS")
    if not lr.get("final_safety_guard_pass"):
        raise RuntimeError("15315 final safety guard failed")
    if str(lr.get("final_ensemble_sha256")) != EXPECTED_TEACHER_SHA256:
        raise RuntimeError("15315 report/teacher hash mismatch")

    members = list(protocol["rebuild_plan"]["member_seeds"])
    tasks4 = [(int(row["member"]), dict(row)) for row in members[:4]]

    with Monitor() as monitor:
        # Canonical 8-thread reduced reference.
        print("POST15315_PERF_REFERENCE_BEGIN", flush=True)
        reference = _isolated_serial(str(teacher), str(solver), [tasks4[0]], 8)
        print("POST15315_PERF_REFERENCE_PASS", flush=True)

        profiles = []
        for threads in THREAD_PROFILES:
            print(f"POST15315_PERF_PROFILE_BEGIN threads={threads}", flush=True)
            row = _isolated_serial(str(teacher), str(solver), [tasks4[0]], threads)
            exact = _rows_exact(reference["rows"], row["rows"])
            profiles.append(
                {
                    "threads": int(threads),
                    "wall_seconds": float(row["wall_seconds"]),
                    "exact_to_canonical_8t": bool(exact),
                    "row": row["rows"][0],
                }
            )
            print(
                "POST15315_PERF_PROFILE "
                + json.dumps(
                    {
                        "threads": int(threads),
                        "wall_seconds": float(row["wall_seconds"]),
                        "exact": bool(exact),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        eligible = [x for x in profiles if x["exact_to_canonical_8t"]]
        if not eligible:
            raise RuntimeError("no exact collection thread profile")
        selected = min(eligible, key=lambda x: x["wall_seconds"])
        selected_threads = int(selected["threads"])

        print(
            f"POST15315_PERF_SERIAL4_BEGIN threads={selected_threads}",
            flush=True,
        )
        serial4 = _isolated_serial(
            str(teacher), str(solver), tasks4, selected_threads
        )
        print("POST15315_PERF_SERIAL4_PASS", flush=True)

        print(
            f"POST15315_PERF_PARALLEL4_BEGIN workers=4 threads={selected_threads}",
            flush=True,
        )
        parallel4 = _parallel_members(
            str(teacher), str(solver), tasks4, selected_threads
        )
        parallel_exact = _rows_exact(serial4["rows"], parallel4["rows"])
        actual_parallel_speedup = (
            serial4["wall_seconds"] / parallel4["wall_seconds"]
            if parallel4["wall_seconds"] > 0
            else 0.0
        )
        print(
            "POST15315_PERF_PARALLEL4 "
            + json.dumps(
                {
                    "exact": bool(parallel_exact),
                    "speedup": float(actual_parallel_speedup),
                    "serial_seconds": float(serial4["wall_seconds"]),
                    "parallel_seconds": float(parallel4["wall_seconds"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        fit_task = (
            str(cp),
            str(teacher),
            str(solver),
            int(members[0]["base_collection_seed"]),
            int(members[0]["tail_fit_seed"]),
        )
        print("POST15315_PERF_FIT_PARITY_BEGIN", flush=True)
        fit_a = _isolated_fit(fit_task)
        fit_b = _isolated_fit(fit_task)
        fit_exact = bool(
            fit_a["collection_stats"] == fit_b["collection_stats"]
            and fit_a["train_digest"] == fit_b["train_digest"]
            and fit_a["holdout_digest"] == fit_b["holdout_digest"]
            and fit_a["loss_digest"] == fit_b["loss_digest"]
            and fit_a["model_tensor_digest"] == fit_b["model_tensor_digest"]
        )
        print(
            "POST15315_PERF_FIT_PARITY "
            + json.dumps(
                {
                    "exact": fit_exact,
                    "tensor_digest": fit_a["model_tensor_digest"],
                    "loss_digest": fit_a["loss_digest"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    resources = monitor.report()

    # Reduced workload is exactly 1/40 of one member's collection budget.
    # serial4/parallel4 represent four members; K=8 is two such waves.
    projection_factor = float(SCALE * 2)
    baseline_collection_seconds = float(serial4["wall_seconds"]) * projection_factor
    candidate_collection_seconds = float(parallel4["wall_seconds"]) * projection_factor

    # Fits stay canonical and sequential/non-fanned.  The mini fit measures a
    # 25-step tail-like semantic fit; scale to the frozen total 8200 fit steps.
    mini_fit_seconds = 0.5 * (float(fit_a["wall_seconds"]) + float(fit_b["wall_seconds"]))
    projected_fit_seconds = mini_fit_seconds * (8200.0 / float(MINI_FIT_STEPS))

    projected_baseline_total = baseline_collection_seconds + projected_fit_seconds
    projected_candidate_total = candidate_collection_seconds + projected_fit_seconds
    projected_speedup = (
        projected_baseline_total / projected_candidate_total
        if projected_candidate_total > 0
        else 0.0
    )
    projected_savings = projected_baseline_total - projected_candidate_total

    criteria = {
        "selected_collection_profile_exact": bool(selected["exact_to_canonical_8t"]),
        "parallel_four_member_collection_exact": bool(parallel_exact),
        "mini_fit_samples_losses_and_model_tensors_exact": bool(fit_exact),
        "parallel_profile_speedup_at_least_1_25x": bool(actual_parallel_speedup >= MIN_SPEEDUP),
        "projected_end_to_end_speedup_at_least_1_25x": bool(projected_speedup >= MIN_SPEEDUP),
        "projected_net_savings_at_least_30_minutes": bool(
            projected_savings >= MIN_PROJECTED_SAVINGS_SECONDS
        ),
        "min_mem_available_at_least_4gib": bool(
            resources["min_mem_available_gib"] is not None
            and resources["min_mem_available_gib"] >= MIN_MEM_AVAILABLE_GIB
        ),
        "swap_used_no_more_than_1gib": bool(
            resources["max_swap_used_gib"] is not None
            and resources["max_swap_used_gib"] <= MAX_SWAP_USED_GIB
        ),
    }
    passed = bool(all(criteria.values()))

    payload = {
        "schema": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "teacher_iteration": TARGET_ITERATION,
        "teacher_sha256": EXPECTED_TEACHER_SHA256,
        "protocol_sha256": sha256(protocol_path),
        "reduced_workload": {
            "scale_vs_full_member": SCALE,
            "ordinary_episodes": REDUCED_ORDINARY_EPISODES,
            "augmentation_episodes": REDUCED_AUGMENT_EPISODES,
            "specialist_episodes": REDUCED_SPECIALIST_EPISODES,
            "members_in_parallel_probe": [1, 2, 3, 4],
            "mini_fit_steps": MINI_FIT_STEPS,
        },
        "thread_profiles": profiles,
        "selected_collection_threads": selected_threads,
        "serial4": {
            "wall_seconds": serial4["wall_seconds"],
            "maxrss_kib": serial4["maxrss_kib"],
            "rows": serial4["rows"],
        },
        "parallel4": {
            **parallel4,
            "exact_to_serial": parallel_exact,
            "speedup": actual_parallel_speedup,
            "declared_threads": PARALLEL_WORKERS * selected_threads,
        },
        "fit_parity": {
            "exact": fit_exact,
            "reference": fit_a,
            "repeat": fit_b,
        },
        "projection": {
            "baseline_collection_seconds": baseline_collection_seconds,
            "candidate_collection_seconds": candidate_collection_seconds,
            "projected_fixed_fit_seconds": projected_fit_seconds,
            "projected_baseline_total_seconds": projected_baseline_total,
            "projected_candidate_total_seconds": projected_candidate_total,
            "projected_end_to_end_speedup": projected_speedup,
            "projected_net_savings_seconds": projected_savings,
        },
        "resources": resources,
        "criteria": criteria,
        "production_profile_if_pass": {
            "collection_workers": PARALLEL_WORKERS,
            "collection_threads_per_worker": selected_threads,
            "fit_threads": 8,
            "checkpoint_fanout_to_collection_workers": False,
            "member_seed_set": "FROZEN_MEMBERS_1_TO_8",
        },
        "interpretation": (
            "PASS authorizes implementing the post-15315 K=8 production runner "
            "with process-parallel independent collection streams and canonical "
            "8-thread fitting. It does not admit any rebuild member, does not "
            "change K=8/K=12 logic, and does not authorize policy promotion."
        ),
    }
    atomic_json(payload, report_path)

    print("POST15315_MULTIREBUILD_PERFORMANCE_GATE_" + ("PASS" if passed else "FAIL"))
    print(f"selected_collection_threads={selected_threads}")
    print(f"parallel_collection_speedup={actual_parallel_speedup}")
    print(f"projected_end_to_end_speedup={projected_speedup}")
    print(f"projected_net_savings_seconds={projected_savings}")
    print("criteria=" + json.dumps(criteria, sort_keys=True))
    print(f"report={report_path}")
    return 0 if passed else 5


if __name__ == "__main__":
    raise SystemExit(main())
