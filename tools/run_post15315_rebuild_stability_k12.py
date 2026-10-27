#!/usr/bin/env python3
from __future__ import annotations

"""Run the unchanged post-15315 rebuild-stability gate over all 12 members.

This stage is permitted only after the K=8 scientific FAIL authorizes the
preregistered extension and members 9-12 complete construction. It regenerates
exactly the same fixed seed-153159001, 100000-episode coverage-adjudicated
teacher-rollout bank and evaluates all 12 members on every common state. The
thresholds and surfaces are unchanged. No external strength evidence is opened.

PROJECT_CONTRACT_IDS:
TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,
VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,VALID-038,
SAFE-001,SAFE-010,SAFE-011,CKPT-001,ART-001
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import build_post15315_rebuild_member as builder
import evaluate_post15315_rebuild_stability as stability
import post15315_collection as collect
import run_post15315_multirebuild_k12_extension as ext
import run_post15315_multirebuild_production as production
import run_post15315_rebuild_stability_k8 as k8stab
from audit_3h_average_policy_semantic_continuation_10105 import semantic_vector_from_obs
from spincore.solver import SolverLibrary

GATE_SCHEMA = "SPINCORE_POST15315_K12_STABILITY_GATE_V1"
BANK_META_SCHEMA = "SPINCORE_POST15315_K12_STABILITY_BANK_META_V1"
K8_COUNT = 8
K12_COUNT = 12
MEMBERS = tuple(range(1, 13))


def sha256(path: Path) -> str:
    return production.sha256(path)


def atomic_json(payload: dict, path: Path) -> None:
    return production.atomic_json(payload, path)


def validate_extension(
    extension_dir: Path,
    *,
    protocol: dict,
    protocol_sha: str,
    k8_gate_sha: str,
) -> tuple[dict, list[dict]]:
    extension_dir = extension_dir.resolve(strict=True)
    report_path = extension_dir / "post15315_multirebuild_k12_extension_construction.json"
    p = json.loads(report_path.read_text(encoding="utf-8"))
    if p.get("schema") != ext.RUN_SCHEMA or p.get("status") != "PASS":
        raise RuntimeError("K=12 extension construction report is not PASS")
    if int(p.get("teacher_iteration", -1)) != k8stab.TARGET_ITERATION:
        raise RuntimeError("K=12 extension teacher iteration mismatch")
    if str(p.get("teacher_sha256")) != k8stab.EXPECTED_TEACHER_SHA:
        raise RuntimeError("K=12 extension teacher SHA mismatch")
    if str(p.get("source_checkpoint_sha256")) != k8stab.EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("K=12 extension checkpoint SHA mismatch")
    if str(p.get("protocol_sha256")) != protocol_sha:
        raise RuntimeError("K=12 extension protocol SHA mismatch")
    gate_row = dict(p.get("k8_stability_gate") or {})
    if str(gate_row.get("sha256")) != str(k8_gate_sha):
        raise RuntimeError("K=12 extension K=8 authorization binding mismatch")
    if list(p.get("members_required") or []) != [9, 10, 11, 12]:
        raise RuntimeError("K=12 extension required-member set drift")
    if list(p.get("members_constructed") or []) != [9, 10, 11, 12]:
        raise RuntimeError("K=12 extension construction incomplete")
    if bool(p.get("stability_evaluation_performed")):
        raise RuntimeError("extension construction unexpectedly claims stability")
    if bool(p.get("strength_benchmark_performed")):
        raise RuntimeError("extension construction unexpectedly opened strength")

    rows = []
    for member in range(9, 13):
        seeds = builder.member_spec(protocol, member)
        rp = extension_dir / f"member_{member:02d}_report.json"
        if not production._validate_existing_member(
            rp,
            member=member,
            teacher_sha=k8stab.EXPECTED_TEACHER_SHA,
            protocol_sha=protocol_sha,
            seeds=seeds,
        ):
            raise RuntimeError(f"member {member}: durable extension member missing")
        q = json.loads(rp.read_text(encoding="utf-8"))
        rows.append(
            {
                "member": member,
                "report_path": str(rp.resolve()),
                "report_sha256": sha256(rp),
                "fullpool_path": str(Path(q["fullpool_artifact"]).resolve(strict=True)),
                "fullpool_sha256": str(q["fullpool_sha256"]),
                "specialist_path": str(Path(q["specialist_artifact"]).resolve(strict=True)),
                "specialist_sha256": str(q["specialist_sha256"]),
            }
        )
    return {"path": str(report_path.resolve()), "sha256": sha256(report_path)}, rows


def branch_decision(report: dict) -> dict:
    if bool(report.get("rebuild_stability_pass")):
        return {
            "k12_stability_result": "PASS",
            "stabilization_branch_resolved": True,
            "aggregate_quality_authorized": True,
            "next_gate": (
                "K=12 rebuild stability PASS. Stabilization branch is resolved; "
                "proceed to frozen aggregate quality, specialist confirmation, and "
                "safety/pathology gates. External strength remains separately gated."
            ),
        }
    return {
        "k12_stability_result": "FAIL",
        "stabilization_branch_resolved": True,
        "aggregate_quality_authorized": False,
        "next_gate": (
            "K=12 rebuild stability FAIL. Stop and report rebuild instability. "
            "Do not add members, change thresholds, select a subset, or inspect "
            "external strength."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--k8-construction-dir", type=Path, required=True)
    ap.add_argument("--extension-dir", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--coverage-adjudication", type=Path, required=True)
    ap.add_argument("--k8-stability-gate", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()

    if int(args.threads) <= 0 or int(args.batch_size) <= 0:
        raise SystemExit("threads and batch-size must be positive")
    k8stab.set_threads(int(args.threads))

    protocol_path = args.protocol.resolve(strict=True)
    protocol, protocol_sha = k8stab.validate_protocol(protocol_path)
    adjudication_path = args.coverage_adjudication.resolve(strict=True)
    _adj, adjudication_sha = k8stab.validate_coverage_adjudication(
        adjudication_path,
        protocol_blob_sha="5fd7667926c6112310fe5eca61afff396ad218de",
    )

    k8_gate = ext.validate_k8_gate(
        args.k8_stability_gate,
        protocol_sha=protocol_sha,
        adjudication_sha=adjudication_sha,
    )
    k8_gate_sha = str(k8_gate["sha256"])

    k8_construction, first_eight = k8stab.validate_construction(
        args.k8_construction_dir,
        protocol_sha=protocol_sha,
    )
    extension_report, last_four = validate_extension(
        args.extension_dir,
        protocol=protocol,
        protocol_sha=protocol_sha,
        k8_gate_sha=k8_gate_sha,
    )
    members = first_eight + last_four
    if [int(row["member"]) for row in members] != list(MEMBERS):
        raise RuntimeError("combined K=12 member identity/order drift")

    teacher_path = args.teacher.resolve(strict=True)
    teacher_models, _payload, teacher_sha = builder.load_teacher(
        teacher_path, k8stab.EXPECTED_TEACHER_SHA
    )
    if teacher_sha != k8stab.EXPECTED_TEACHER_SHA:
        raise RuntimeError("teacher SHA validation drift")
    solver = SolverLibrary(args.solver.resolve(strict=True))

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("POST15315_K12_STABILITY_BANK_BEGIN", flush=True)
    train, hold, collection = collect.collect_fresh_split(
        solver,
        teacher_models,
        k8stab.BANK_EPISODES,
        master_seed=k8stab.BANK_SEED,
        sample_iteration=k8stab.TARGET_ITERATION,
        progress_prefix="POST15315_K12_STABILITY_BANK",
    )
    samples = list(train) + list(hold)
    if not samples:
        raise RuntimeError("empty K=12 stability bank")

    legal, s72, trips, high_card_no_draw = k8stab.surface_masks(samples)
    coverage = k8stab.validate_surface_coverage(
        protocol, s72, trips, high_card_no_draw
    )
    semantic_rows = np.asarray(
        [semantic_vector_from_obs(sample.observation) for sample in samples],
        dtype=np.float32,
    )

    probs = np.zeros((K12_COUNT, len(samples), 10), dtype=np.float32)
    for idx, row in enumerate(members):
        member = int(row["member"])
        print(f"POST15315_K12_STABILITY_MEMBER_{member:02d}_BEGIN", flush=True)
        probs[idx] = k8stab.member_probabilities(
            samples,
            semantic_rows,
            trips,
            fullpool_path=Path(row["fullpool_path"]),
            specialist_path=Path(row["specialist_path"]),
            member=member,
            protocol_sha=protocol_sha,
            batch_size=int(args.batch_size),
        )
        print(f"POST15315_K12_STABILITY_MEMBER_{member:02d}_PASS", flush=True)

    bank_path = out / "post15315_k12_stability_bank.npz"
    k8stab.atomic_npz(
        bank_path,
        probs=probs,
        legal=legal,
        surface_72o=s72,
        surface_trips_fold=trips,
        surface_high_card_no_draw=high_card_no_draw,
    )
    bank_sha = sha256(bank_path)

    bank_meta = {
        "schema": BANK_META_SCHEMA,
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "coverage_adjudication_sha256": adjudication_sha,
        "k8_stability_gate_sha256": k8_gate_sha,
        "k8_construction_report": k8_construction,
        "k12_extension_report": extension_report,
        "member_artifacts": members,
        "teacher_iteration": k8stab.TARGET_ITERATION,
        "teacher_sha256": k8stab.EXPECTED_TEACHER_SHA,
        "seed": k8stab.BANK_SEED,
        "forced_three_handed_episodes": k8stab.BANK_EPISODES,
        "state_count": len(samples),
        "collection": collect.canonical_collection_stats(collection),
        "surface_coverage": coverage,
        "state_generator": (
            "Exact fixed seed-153159001 100k coverage-adjudicated teacher-15315 "
            "rollout, regenerated from episode 0. All 12 rebuild policies are "
            "evaluated on every resulting common state and never influence generation."
        ),
        "bank_path": str(bank_path),
        "bank_sha256": bank_sha,
        "training_performed": False,
        "member_selection_performed": False,
        "strength_benchmark_performed": False,
    }
    meta_path = out / "post15315_k12_stability_bank_meta.json"
    atomic_json(bank_meta, meta_path)

    report = stability.evaluate(
        probs, legal, s72, trips, high_card_no_draw, protocol
    )
    if int(report.get("rebuild_count", -1)) != 12:
        raise RuntimeError("K=12 evaluator rebuild-count mismatch")
    report_path = out / "post15315_k12_stability_report.json"
    atomic_json(report, report_path)

    decision = branch_decision(report)
    gate = {
        "schema": GATE_SCHEMA,
        "status": "PASS",
        "mechanically_clean": True,
        "protocol_sha256": protocol_sha,
        "coverage_adjudication_sha256": adjudication_sha,
        "k8_stability_gate_sha256": k8_gate_sha,
        "extension_construction_report_sha256": extension_report["sha256"],
        "bank_sha256": bank_sha,
        "bank_meta_sha256": sha256(meta_path),
        "stability_report": str(report_path),
        "stability_report_sha256": sha256(report_path),
        **decision,
        "strength_benchmark_performed": False,
    }
    gate_path = out / "post15315_k12_stability_gate.json"
    atomic_json(gate, gate_path)

    print("POST15315_K12_STABILITY_RESULT_" + decision["k12_stability_result"], flush=True)
    print("coverage=" + json.dumps(coverage, sort_keys=True), flush=True)
    print("criteria=" + json.dumps(report["precommitted_criteria"], sort_keys=True), flush=True)
    print("summary=" + json.dumps(report["stability_summary"], sort_keys=True), flush=True)
    print(f"report={report_path}", flush=True)
    print(f"gate={gate_path}", flush=True)
    if bool(decision["aggregate_quality_authorized"]):
        print("POST15315_K12_STABILITY_PASS", flush=True)
        print("STOP: aggregate quality/specialist/safety gates are next.", flush=True)
    else:
        print("POST15315_K12_STABILITY_FINAL_FAIL", flush=True)
        print("STOP: rebuild instability. No K>12, threshold change, subset selection, or strength.", flush=True)
    print("POST15315_K12_STABILITY_RUNNER_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
