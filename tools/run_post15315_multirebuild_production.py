#!/usr/bin/env python3
from __future__ import annotations

"""Construct the frozen post-15315 K=8 rebuild set under the Ryzen PASS profile.

This runner is construction-only. It performs two waves of at most four
process-parallel collection jobs, each with the exact collection thread count
frozen by the target-Ryzen performance gate. Member fitting is then performed
one member at a time with the canonical eight-thread fit path. No stability,
quality, pathology, DC1/DC2, or strength benchmark is run here.

PROJECT_CONTRACT_IDS:
PERF-001,PERF-002,TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,
VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,VALID-038,
SAFE-001,SAFE-010,SAFE-011,CKPT-001,ART-001
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import build_post15315_rebuild_member as builder
import post15315_member_cache as member_cache
from spincore.solver import SolverLibrary

PROTOCOL_SCHEMA = "SPINCORE_POST15315_MULTIREBUILD_PREREG_V1"
PERF_SCHEMA = "SPINCORE_POST15315_MULTIREBUILD_PERFORMANCE_GATE_V1"
RUN_SCHEMA = "SPINCORE_POST15315_MULTIREBUILD_PRODUCTION_RUN_V1"
TARGET_ITERATION = 15315
EXPECTED_CHECKPOINT_SHA = "f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_TEACHER_SHA = "ac49b77c8ec3204e07f6d44d3718c68880e60675c5521f538438a3fa3ceae746"

_WORKER_PROTOCOL = None
_WORKER_PROTOCOL_PATH = None
_WORKER_TEACHER = None
_WORKER_TEACHER_SHA = None
_WORKER_SOLVER = None
_WORKER_OUT = None
_WORKER_THREADS = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


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


def validate_performance_evidence(path: Path, expected_sha: str, protocol_sha: str) -> dict:
    actual_sha = sha256(path)
    if actual_sha != str(expected_sha):
        raise RuntimeError(
            f"performance report SHA mismatch expected={expected_sha} actual={actual_sha}"
        )
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("schema") != PERF_SCHEMA or p.get("status") != "PASS":
        raise RuntimeError("performance evidence is not the frozen PASS schema/status")
    if int(p.get("teacher_iteration", -1)) != TARGET_ITERATION:
        raise RuntimeError("performance evidence teacher iteration mismatch")
    if str(p.get("teacher_sha256")) != EXPECTED_TEACHER_SHA:
        raise RuntimeError("performance evidence teacher SHA mismatch")
    if str(p.get("protocol_sha256")) != str(protocol_sha):
        raise RuntimeError("performance evidence protocol SHA mismatch")
    criteria = dict(p.get("criteria") or {})
    if not criteria or not all(bool(x) for x in criteria.values()):
        raise RuntimeError("performance evidence criteria are not all PASS")
    profile = dict(p.get("production_profile_if_pass") or {})
    expected = {
        "collection_workers": 4,
        "collection_threads_per_worker": 2,
        "fit_threads": 8,
        "checkpoint_fanout_to_collection_workers": False,
        "member_seed_set": "FROZEN_MEMBERS_1_TO_8",
    }
    if profile != expected:
        raise RuntimeError(f"performance production profile drift: {profile!r}")
    return p


def _collection_worker_init(
    protocol_path: str,
    teacher_path: str,
    teacher_sha: str,
    solver_path: str,
    out_dir: str,
    threads: int,
) -> None:
    global _WORKER_PROTOCOL, _WORKER_PROTOCOL_PATH, _WORKER_TEACHER
    global _WORKER_TEACHER_SHA, _WORKER_SOLVER, _WORKER_OUT, _WORKER_THREADS
    _set_threads(int(threads))
    _WORKER_PROTOCOL_PATH = Path(protocol_path).resolve(strict=True)
    _WORKER_PROTOCOL = json.loads(_WORKER_PROTOCOL_PATH.read_text(encoding="utf-8"))
    _WORKER_TEACHER, _, actual = builder.load_teacher(
        Path(teacher_path).resolve(strict=True), teacher_sha
    )
    if actual != str(teacher_sha):
        raise RuntimeError("worker teacher SHA mismatch")
    _WORKER_TEACHER_SHA = str(teacher_sha)
    _WORKER_SOLVER = SolverLibrary(Path(solver_path).resolve(strict=True))
    _WORKER_OUT = Path(out_dir).resolve()
    _WORKER_THREADS = int(threads)


def _collect_one(member: int) -> dict[str, Any]:
    path = member_cache.collect_member_cache(
        member=int(member),
        protocol=_WORKER_PROTOCOL,
        protocol_path=_WORKER_PROTOCOL_PATH,
        teacher_models=_WORKER_TEACHER,
        teacher_sha256=_WORKER_TEACHER_SHA,
        solver=_WORKER_SOLVER,
        out_dir=_WORKER_OUT,
        collection_threads=_WORKER_THREADS,
    )
    return {
        "member": int(member),
        "cache": str(path),
        "cache_sha256": sha256(path),
    }


def _member_seed(protocol: dict, member: int) -> dict:
    return builder.member_spec(protocol, int(member))


def _validate_existing_member(
    report_path: Path,
    *,
    member: int,
    teacher_sha: str,
    protocol_sha: str,
    seeds: dict,
) -> bool:
    if not report_path.exists():
        return False
    p = json.loads(report_path.read_text(encoding="utf-8"))
    if p.get("schema") != builder.REPORT_SCHEMA:
        raise RuntimeError(f"member {member}: existing report schema mismatch")
    if not bool(p.get("construction_pass")):
        raise RuntimeError(f"member {member}: existing report is not construction PASS")
    if int(p.get("member", -1)) != int(member):
        raise RuntimeError(f"member {member}: existing report member mismatch")
    if str(p.get("teacher_sha256")) != str(teacher_sha):
        raise RuntimeError(f"member {member}: existing report teacher SHA mismatch")
    if str(p.get("protocol_sha256")) != str(protocol_sha):
        raise RuntimeError(f"member {member}: existing report protocol SHA mismatch")
    if dict(p.get("member_seeds") or {}) != dict(seeds):
        raise RuntimeError(f"member {member}: existing report seed mismatch")
    if int(p.get("collection_threads", -1)) != 2 or int(p.get("fit_threads", -1)) != 8:
        raise RuntimeError(f"member {member}: existing report execution-profile mismatch")
    for key in ("tail", "fullpool", "specialist"):
        artifact = Path(str(p[f"{key}_artifact"]))
        if not artifact.is_file():
            raise RuntimeError(f"member {member}: missing durable {key} artifact")
        if sha256(artifact) != str(p[f"{key}_sha256"]):
            raise RuntimeError(f"member {member}: {key} artifact SHA mismatch")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--performance-report", type=Path, required=True)
    ap.add_argument("--performance-report-sha256", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    teacher = args.teacher.resolve(strict=True)
    solver = args.solver.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    perf_path = args.performance_report.resolve(strict=True)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("frozen 10105 checkpoint SHA mismatch")
    if sha256(teacher) != EXPECTED_TEACHER_SHA:
        raise RuntimeError("frozen 15315 teacher SHA mismatch")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise RuntimeError("wrong post15315 protocol schema")
    if int(protocol["teacher"]["required_iteration"]) != TARGET_ITERATION:
        raise RuntimeError("post15315 protocol target drift")
    if int(protocol["rebuild_plan"]["primary_rebuild_count"]) != 8:
        raise RuntimeError("post15315 primary K drift")
    protocol_sha = sha256(protocol_path)
    perf = validate_performance_evidence(
        perf_path, args.performance_report_sha256, protocol_sha
    )

    members = list(range(1, 9))
    report_path = out / "post15315_multirebuild_k8_construction.json"
    journal = {
        "schema": RUN_SCHEMA,
        "status": "IN_PROGRESS",
        "teacher_iteration": TARGET_ITERATION,
        "teacher_sha256": EXPECTED_TEACHER_SHA,
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "protocol_sha256": protocol_sha,
        "performance_report": str(perf_path),
        "performance_report_sha256": str(args.performance_report_sha256),
        "production_profile": dict(perf["production_profile_if_pass"]),
        "members_required": members,
        "members_collection_cached": [],
        "members_constructed": [],
        "strength_benchmark_performed": False,
        "stability_evaluation_performed": False,
    }

    constructed = []
    for member in members:
        rp = out / f"member_{member:02d}_report.json"
        if _validate_existing_member(
            rp,
            member=member,
            teacher_sha=EXPECTED_TEACHER_SHA,
            protocol_sha=protocol_sha,
            seeds=_member_seed(protocol, member),
        ):
            constructed.append(member)
    journal["members_constructed"] = list(constructed)
    atomic_json(journal, report_path)

    pending = [m for m in members if m not in constructed]
    cache_rows: dict[int, dict[str, Any]] = {}
    for member in pending:
        cp = member_cache.cache_path(out, member)
        if cp.exists():
            member_cache.load_member_cache(
                cp,
                expected_member=member,
                expected_teacher_sha256=EXPECTED_TEACHER_SHA,
                expected_protocol_sha256=protocol_sha,
                expected_member_seeds=_member_seed(protocol, member),
                expected_collection_threads=2,
            )
            cache_rows[member] = {
                "member": member,
                "cache": str(cp),
                "cache_sha256": sha256(cp),
            }

    missing_cache = [m for m in pending if m not in cache_rows]
    if missing_cache:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=4,
            mp_context=ctx,
            initializer=_collection_worker_init,
            initargs=(
                str(protocol_path),
                str(teacher),
                EXPECTED_TEACHER_SHA,
                str(solver),
                str(out),
                2,
            ),
        ) as pool:
            futures = {pool.submit(_collect_one, m): m for m in missing_cache}
            for fut in as_completed(futures):
                row = fut.result()
                cache_rows[int(row["member"])] = row
                journal["members_collection_cached"] = sorted(cache_rows)
                atomic_json(journal, report_path)

    journal["members_collection_cached"] = sorted(cache_rows)
    atomic_json(journal, report_path)

    # Canonical fits are deliberately sequential/non-fanned. The performance
    # gate projected this exact shape and did not validate concurrent 8-thread fits.
    for member in members:
        if member in constructed:
            continue
        cache = Path(cache_rows[member]["cache"]).resolve(strict=True)
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "build_post15315_rebuild_member.py"),
            "--member", str(member),
            "--protocol", str(protocol_path),
            "--checkpoint", str(checkpoint),
            "--semantic-advantage", str(teacher),
            "--teacher-sha256", EXPECTED_TEACHER_SHA,
            "--solver", str(solver),
            "--out-dir", str(out),
            "--collection-threads", "2",
            "--fit-threads", "8",
            "--collection-cache", str(cache),
        ]
        print(f"POST15315_K8_MEMBER_{member:02d}_FIT_BEGIN", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        rp = out / f"member_{member:02d}_report.json"
        if not _validate_existing_member(
            rp,
            member=member,
            teacher_sha=EXPECTED_TEACHER_SHA,
            protocol_sha=protocol_sha,
            seeds=_member_seed(protocol, member),
        ):
            raise RuntimeError(f"member {member}: build returned without durable PASS")
        constructed.append(member)
        journal["members_constructed"] = sorted(constructed)
        atomic_json(journal, report_path)
        print(f"POST15315_K8_MEMBER_{member:02d}_FIT_PASS", flush=True)

    if sorted(constructed) != members:
        raise RuntimeError(f"K=8 construction incomplete: {sorted(constructed)}")

    journal["status"] = "PASS"
    journal["members_constructed"] = members
    journal["next_gate"] = (
        "Build the frozen fresh stability bank at seed 153159001 and evaluate all "
        "eight members under evaluate_post15315_rebuild_stability.py. Do not open "
        "external strength evidence before the K=8/K=12 stabilization branch resolves."
    )
    atomic_json(journal, report_path)
    print("POST15315_MULTIREBUILD_K8_CONSTRUCTION_PASS")
    print(f"report={report_path}")
    print("STOP: K=8 construction only. Stability gate is the next stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
