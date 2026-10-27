#!/usr/bin/env python3
from __future__ import annotations

"""Construct exactly preregistered post-15315 extension members 9-12.

This stage is authorized only by a mechanically clean scientific K=8 stability
FAIL. It reuses the already-validated target-Ryzen execution shape: four
independent collection workers x two threads and sequential eight-thread fits.
It performs no K=12 stability evaluation and opens no external strength evidence.

PROJECT_CONTRACT_IDS:
PERF-001,PERF-002,TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,
VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,VALID-038,
SAFE-001,SAFE-010,SAFE-011,CKPT-001,ART-001
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import build_post15315_rebuild_member as builder
import post15315_member_cache as member_cache
import run_post15315_multirebuild_production as k8prod

PROTOCOL_SCHEMA = "SPINCORE_POST15315_MULTIREBUILD_PREREG_V1"
K8_GATE_SCHEMA = "SPINCORE_POST15315_K8_STABILITY_GATE_V1"
STABILITY_SCHEMA = "SPINCORE_POST15315_REBUILD_STABILITY_RESULT_V1"
RUN_SCHEMA = "SPINCORE_POST15315_MULTIREBUILD_K12_EXTENSION_RUN_V1"
TARGET_ITERATION = 15315
EXPECTED_CHECKPOINT_SHA = k8prod.EXPECTED_CHECKPOINT_SHA
EXPECTED_TEACHER_SHA = k8prod.EXPECTED_TEACHER_SHA
MEMBERS = (9, 10, 11, 12)


def sha256(path: Path) -> str:
    return k8prod.sha256(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    return k8prod.atomic_json(payload, path)


def member_seed(protocol: dict, member: int) -> dict:
    return builder.member_spec(protocol, int(member))


def validate_k8_gate(
    gate_path: Path,
    *,
    protocol_sha: str,
    adjudication_sha: str,
) -> dict:
    gate_path = gate_path.resolve(strict=True)
    g = json.loads(gate_path.read_text(encoding="utf-8"))
    if g.get("schema") != K8_GATE_SCHEMA or g.get("status") != "PASS":
        raise RuntimeError("K=8 stability gate schema/status mismatch")
    if not bool(g.get("mechanically_clean")):
        raise RuntimeError("K=8 stability gate is not mechanically clean")
    if str(g.get("protocol_sha256")) != protocol_sha:
        raise RuntimeError("K=8 stability gate protocol SHA mismatch")
    if str(g.get("coverage_adjudication_sha256")) != adjudication_sha:
        raise RuntimeError("K=8 stability gate coverage-adjudication SHA mismatch")
    if str(g.get("teacher_sha256")) != EXPECTED_TEACHER_SHA:
        raise RuntimeError("K=8 stability gate teacher SHA mismatch")
    if str(g.get("k8_stability_result")) != "FAIL":
        raise RuntimeError("K=8 stability result is not FAIL")
    if not bool(g.get("k12_extension_authorized")):
        raise RuntimeError("K=12 extension is not authorized")
    if bool(g.get("strength_benchmark_performed")):
        raise RuntimeError("K=8 gate unexpectedly opened strength evidence")

    report_path = Path(str(g.get("stability_report"))).resolve(strict=True)
    if sha256(report_path) != str(g.get("stability_report_sha256")):
        raise RuntimeError("K=8 stability report SHA mismatch")
    r = json.loads(report_path.read_text(encoding="utf-8"))
    if r.get("schema") != STABILITY_SCHEMA:
        raise RuntimeError("wrong K=8 stability report schema")
    if int(r.get("rebuild_count", -1)) != 8:
        raise RuntimeError("K=8 stability report rebuild-count drift")
    if bool(r.get("rebuild_stability_pass")):
        raise RuntimeError("K=8 stability report unexpectedly PASS")
    criteria = dict(r.get("precommitted_criteria") or {})
    if not bool(criteria.get("surface_coverage_minima")):
        raise RuntimeError("K=8 FAIL was not coverage-valid")
    failed = sorted(k for k, v in criteria.items() if not bool(v))
    if not failed:
        raise RuntimeError("K=8 report has no scientific failed criterion")
    return {
        "path": str(gate_path),
        "sha256": sha256(gate_path),
        "stability_report": str(report_path),
        "stability_report_sha256": sha256(report_path),
        "failed_criteria": failed,
    }


def validate_performance_profile(path: Path, expected_sha: str, protocol_sha: str) -> dict:
    path = path.resolve(strict=True)
    if sha256(path) != str(expected_sha):
        raise RuntimeError("performance evidence SHA mismatch")
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("schema") != k8prod.PERF_SCHEMA or p.get("status") != "PASS":
        raise RuntimeError("performance evidence schema/status mismatch")
    if int(p.get("teacher_iteration", -1)) != TARGET_ITERATION:
        raise RuntimeError("performance evidence teacher iteration mismatch")
    if str(p.get("teacher_sha256")) != EXPECTED_TEACHER_SHA:
        raise RuntimeError("performance evidence teacher SHA mismatch")
    if str(p.get("protocol_sha256")) != protocol_sha:
        raise RuntimeError("performance evidence protocol SHA mismatch")
    criteria = dict(p.get("criteria") or {})
    if not criteria or not all(bool(x) for x in criteria.values()):
        raise RuntimeError("performance evidence criteria are not all PASS")
    profile = dict(p.get("production_profile_if_pass") or {})
    expected_shape = {
        "collection_workers": 4,
        "collection_threads_per_worker": 2,
        "fit_threads": 8,
        "checkpoint_fanout_to_collection_workers": False,
    }
    for key, value in expected_shape.items():
        if profile.get(key) != value:
            raise RuntimeError(f"performance profile drift {key}={profile.get(key)!r}")
    if profile.get("member_seed_set") != "FROZEN_MEMBERS_1_TO_8":
        raise RuntimeError("unexpected original performance member-set label")
    return p


def validate_existing_member(
    out: Path,
    *,
    member: int,
    teacher_sha: str,
    protocol_sha: str,
    seeds: dict,
) -> bool:
    return k8prod._validate_existing_member(
        out / f"member_{member:02d}_report.json",
        member=member,
        teacher_sha=teacher_sha,
        protocol_sha=protocol_sha,
        seeds=seeds,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--coverage-adjudication", type=Path, required=True)
    ap.add_argument("--k8-stability-gate", type=Path, required=True)
    ap.add_argument("--performance-report", type=Path, required=True)
    ap.add_argument("--performance-report-sha256", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    teacher = args.teacher.resolve(strict=True)
    solver = args.solver.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    adjudication_path = args.coverage_adjudication.resolve(strict=True)
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
    rp = protocol["rebuild_plan"]
    if int(rp["primary_rebuild_count"]) != 8:
        raise RuntimeError("primary K drift")
    if int(rp["conditional_extension_rebuild_count"]) != 4:
        raise RuntimeError("conditional extension count drift")
    if int(rp["maximum_rebuild_count"]) != 12:
        raise RuntimeError("maximum K drift")
    protocol_sha = sha256(protocol_path)
    adjudication_sha = sha256(adjudication_path)

    gate = validate_k8_gate(
        args.k8_stability_gate,
        protocol_sha=protocol_sha,
        adjudication_sha=adjudication_sha,
    )
    perf = validate_performance_profile(
        perf_path, args.performance_report_sha256, protocol_sha
    )

    report_path = out / "post15315_multirebuild_k12_extension_construction.json"
    journal = {
        "schema": RUN_SCHEMA,
        "status": "IN_PROGRESS",
        "teacher_iteration": TARGET_ITERATION,
        "teacher_sha256": EXPECTED_TEACHER_SHA,
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "protocol_sha256": protocol_sha,
        "coverage_adjudication_sha256": adjudication_sha,
        "k8_stability_gate": gate,
        "performance_report": str(perf_path),
        "performance_report_sha256": str(args.performance_report_sha256),
        "execution_profile_reused_from_k8": {
            "collection_workers": 4,
            "collection_threads_per_worker": 2,
            "fit_threads": 8,
            "checkpoint_fanout_to_collection_workers": False,
            "rationale": (
                "Members 9-12 have the identical preregistered per-member workload. "
                "The target-Ryzen PASS profile is reused unchanged for one four-member "
                "collection wave and sequential fits."
            ),
        },
        "members_required": list(MEMBERS),
        "members_collection_cached": [],
        "members_constructed": [],
        "stability_evaluation_performed": False,
        "strength_benchmark_performed": False,
    }

    constructed = []
    for member in MEMBERS:
        if validate_existing_member(
            out,
            member=member,
            teacher_sha=EXPECTED_TEACHER_SHA,
            protocol_sha=protocol_sha,
            seeds=member_seed(protocol, member),
        ):
            constructed.append(member)
    journal["members_constructed"] = list(constructed)
    atomic_json(journal, report_path)

    pending = [m for m in MEMBERS if m not in constructed]
    cache_rows: dict[int, dict[str, Any]] = {}
    for member in pending:
        cp = member_cache.cache_path(out, member)
        if cp.exists():
            member_cache.load_member_cache(
                cp,
                expected_member=member,
                expected_teacher_sha256=EXPECTED_TEACHER_SHA,
                expected_protocol_sha256=protocol_sha,
                expected_member_seeds=member_seed(protocol, member),
                expected_collection_threads=2,
            )
            cache_rows[member] = {
                "member": member,
                "cache": str(cp),
                "cache_sha256": sha256(cp),
            }

    missing = [m for m in pending if m not in cache_rows]
    if missing:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=4,
            mp_context=ctx,
            initializer=k8prod._collection_worker_init,
            initargs=(
                str(protocol_path),
                str(teacher),
                EXPECTED_TEACHER_SHA,
                str(solver),
                str(out),
                2,
            ),
        ) as pool:
            futures = {pool.submit(k8prod._collect_one, m): m for m in missing}
            for fut in as_completed(futures):
                row = fut.result()
                cache_rows[int(row["member"])] = row
                journal["members_collection_cached"] = sorted(cache_rows)
                atomic_json(journal, report_path)

    journal["members_collection_cached"] = sorted(cache_rows)
    atomic_json(journal, report_path)

    for member in MEMBERS:
        if member in constructed:
            continue
        cache = Path(cache_rows[member]["cache"]).resolve(strict=True)
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "build_post15315_rebuild_member_cached.py"),
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
        print(f"POST15315_K12_MEMBER_{member:02d}_FIT_BEGIN", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        if not validate_existing_member(
            out,
            member=member,
            teacher_sha=EXPECTED_TEACHER_SHA,
            protocol_sha=protocol_sha,
            seeds=member_seed(protocol, member),
        ):
            raise RuntimeError(f"member {member}: build returned without durable PASS")
        constructed.append(member)
        journal["members_constructed"] = sorted(constructed)
        atomic_json(journal, report_path)
        print(f"POST15315_K12_MEMBER_{member:02d}_FIT_PASS", flush=True)

    if sorted(constructed) != list(MEMBERS):
        raise RuntimeError(f"K=12 extension construction incomplete: {sorted(constructed)}")

    journal["status"] = "PASS"
    journal["members_constructed"] = list(MEMBERS)
    journal["next_gate"] = (
        "Recompute the unchanged stability metrics over all 12 members on the exact "
        "seed-153159001 fixed-100k coverage-adjudicated state bank. No threshold "
        "change and no external strength evidence."
    )
    atomic_json(journal, report_path)
    print("POST15315_MULTIREBUILD_K12_EXTENSION_CONSTRUCTION_PASS", flush=True)
    print(f"report={report_path}", flush=True)
    print("STOP: members 9-12 construction only; K=12 stability is next.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
