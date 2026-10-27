#!/usr/bin/env python3
from __future__ import annotations

"""Run the frozen post-15315 K=8 common-state rebuild-stability gate.

This stage is evaluation-only. It validates the completed K=8 construction,
creates the coverage-adjudicated 100,000-episode forced-THREE_HANDED common-state bank
with seed 153159001 under the frozen teacher rollout, evaluates all eight
rebuild members on exactly those same states after each member applies the
frozen fullpool/specialist route, and delegates the stability decision to
evaluate_post15315_rebuild_stability.py.

A mechanically clean K=8 stability FAIL authorizes construction of exactly
members 9-12 under the already-frozen seeds. It does not run K=12 itself and
never opens external strength evidence.

PROJECT_CONTRACT_IDS:
TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,
VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,VALID-038,
SAFE-001,SAFE-010,SAFE-011,CKPT-001,ART-001
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import audit_3h_semantic_sidecar_attribution_10105 as sidecar
import build_post15315_rebuild_member as builder
import evaluate_post15315_rebuild_stability as stability
import post15315_collection as collect
import run_post15315_multirebuild_production as production
from audit_3h_average_policy_semantic_continuation_10105 import (
    V1SemanticPolicyNet,
    semantic_vector_from_obs,
)
from spincore.decision_sanity import preflop_class
from spincore_nn.lean_batch import vectorized_batch
from spincore.solver import SolverLibrary

PROTOCOL_SCHEMA = "SPINCORE_POST15315_MULTIREBUILD_PREREG_V1"
RUN_SCHEMA = production.RUN_SCHEMA
EXPECTED_TEACHER_SHA = production.EXPECTED_TEACHER_SHA
EXPECTED_CHECKPOINT_SHA = production.EXPECTED_CHECKPOINT_SHA
TARGET_ITERATION = 15315
K = 8
FOLD = 0
ALL_IN = 9
BANK_SEED = 153159001
ORIGINAL_BANK_EPISODES = 80000
BANK_EPISODES = 100000
ADJUDICATION_SCHEMA = "SPINCORE_POST15315_STABILITY_COVERAGE_ADJUDICATION_V1"
BANK_SCHEMA = "SPINCORE_POST15315_K8_STABILITY_BANK_META_V1"
GATE_SCHEMA = "SPINCORE_POST15315_K8_STABILITY_GATE_V1"


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


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def set_threads(n: int) -> None:
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


def validate_protocol(path: Path) -> tuple[dict, str]:
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("schema") != PROTOCOL_SCHEMA:
        raise RuntimeError("wrong post15315 preregistration schema")
    if int(p["teacher"]["required_iteration"]) != TARGET_ITERATION:
        raise RuntimeError("teacher iteration drift")
    if int(p["rebuild_plan"]["primary_rebuild_count"]) != K:
        raise RuntimeError("primary rebuild count drift")
    bank = p["shared_fresh_evaluation"]["stability_state_bank"]
    if int(bank["seed"]) != BANK_SEED:
        raise RuntimeError("stability bank seed drift")
    if int(bank["forced_three_handed_episodes"]) != ORIGINAL_BANK_EPISODES:
        raise RuntimeError("original stability bank episode budget drift")
    return p, sha256(path)


def validate_coverage_adjudication(path: Path, *, protocol_blob_sha: str) -> tuple[dict, str]:
    a = json.loads(path.read_text(encoding="utf-8"))
    if a.get("schema") != ADJUDICATION_SCHEMA:
        raise RuntimeError("wrong stability coverage adjudication schema")
    if a.get("status") != "FROZEN_BEFORE_ANY_REBUILD_STABILITY_PREDICTIONS":
        raise RuntimeError("coverage adjudication status drift")
    original = dict(a.get("original_preregistration") or {})
    if str(original.get("blob_sha")) != str(protocol_blob_sha):
        raise RuntimeError("coverage adjudication original-protocol blob drift")
    if int(original.get("state_bank_seed", -1)) != BANK_SEED:
        raise RuntimeError("coverage adjudication seed drift")
    if int(original.get("forced_three_handed_episodes", -1)) != ORIGINAL_BANK_EPISODES:
        raise RuntimeError("coverage adjudication original episode budget drift")
    rule = str((a.get("adjudication") or {}).get("replacement_bank_rule") or "")
    if "exactly 100000" not in rule or "seed-153159001" not in rule:
        raise RuntimeError("coverage adjudication replacement-bank rule drift")
    observed = dict(a.get("observed_execution_fact") or {})
    if bool(observed.get("member_probability_evaluation_started")):
        raise RuntimeError("coverage adjudication claims member evaluation had already started")
    if bool(observed.get("stability_metrics_observed")):
        raise RuntimeError("coverage adjudication claims stability metrics were already observed")
    if bool(observed.get("k12_authorized_by_this_failure")):
        raise RuntimeError("coverage adjudication incorrectly authorizes K=12")
    return a, sha256(path)


def validate_construction(
    construction_dir: Path,
    *,
    protocol_sha: str,
) -> tuple[dict, list[dict[str, Any]]]:
    construction_dir = construction_dir.resolve(strict=True)
    run_path = construction_dir / "post15315_multirebuild_k8_construction.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("schema") != RUN_SCHEMA or run.get("status") != "PASS":
        raise RuntimeError("K=8 construction report is not durable PASS")
    if int(run.get("teacher_iteration", -1)) != TARGET_ITERATION:
        raise RuntimeError("K=8 construction teacher iteration mismatch")
    if str(run.get("teacher_sha256")) != EXPECTED_TEACHER_SHA:
        raise RuntimeError("K=8 construction teacher SHA mismatch")
    if str(run.get("source_checkpoint_sha256")) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("K=8 construction source checkpoint mismatch")
    if str(run.get("protocol_sha256")) != protocol_sha:
        raise RuntimeError("K=8 construction protocol SHA mismatch")
    if list(run.get("members_required") or []) != list(range(1, K + 1)):
        raise RuntimeError("K=8 required-member set drift")
    if list(run.get("members_constructed") or []) != list(range(1, K + 1)):
        raise RuntimeError("K=8 construction incomplete")
    if bool(run.get("strength_benchmark_performed")):
        raise RuntimeError("construction report unexpectedly opened strength evidence")
    if bool(run.get("stability_evaluation_performed")):
        raise RuntimeError("construction report unexpectedly claims prior stability")

    rows: list[dict[str, Any]] = []
    for member in range(1, K + 1):
        report_path = construction_dir / f"member_{member:02d}_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema") != builder.REPORT_SCHEMA:
            raise RuntimeError(f"member {member}: wrong construction report schema")
        if not bool(report.get("construction_pass")):
            raise RuntimeError(f"member {member}: construction PASS missing")
        if int(report.get("member", -1)) != member:
            raise RuntimeError(f"member {member}: report identity mismatch")
        if str(report.get("teacher_sha256")) != EXPECTED_TEACHER_SHA:
            raise RuntimeError(f"member {member}: teacher SHA mismatch")
        if str(report.get("protocol_sha256")) != protocol_sha:
            raise RuntimeError(f"member {member}: protocol SHA mismatch")
        if int(report.get("collection_threads", -1)) != 2:
            raise RuntimeError(f"member {member}: collection-thread drift")
        if int(report.get("fit_threads", -1)) != 8:
            raise RuntimeError(f"member {member}: fit-thread drift")
        if bool(report.get("selection_performed")):
            raise RuntimeError(f"member {member}: forbidden member selection recorded")
        if bool(report.get("strength_benchmark_performed")):
            raise RuntimeError(f"member {member}: forbidden strength evidence recorded")

        expected = {
            "fullpool": construction_dir / f"member_{member:02d}_fullpool.pt",
            "specialist": construction_dir / f"member_{member:02d}_specialist.pt",
        }
        for key, expected_path in expected.items():
            reported_path = Path(str(report[f"{key}_artifact"])).resolve(strict=True)
            if reported_path != expected_path.resolve(strict=True):
                raise RuntimeError(f"member {member}: {key} artifact path drift")
            actual = sha256(reported_path)
            if actual != str(report[f"{key}_sha256"]):
                raise RuntimeError(f"member {member}: {key} artifact SHA mismatch")

        rows.append(
            {
                "member": member,
                "report_path": str(report_path.resolve()),
                "report_sha256": sha256(report_path),
                "fullpool_path": str(expected["fullpool"].resolve()),
                "fullpool_sha256": str(report["fullpool_sha256"]),
                "specialist_path": str(expected["specialist"].resolve()),
                "specialist_sha256": str(report["specialist_sha256"]),
            }
        )
    return {
        "path": str(run_path.resolve()),
        "sha256": sha256(run_path),
    }, rows


def load_member_model(
    path: Path,
    *,
    schema: str,
    member: int,
    protocol_sha: str,
) -> tuple[V1SemanticPolicyNet, dict]:
    p = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if p.get("schema") != schema:
        raise RuntimeError(f"member {member}: wrong {schema} artifact schema")
    if int(p.get("member", -1)) != int(member):
        raise RuntimeError(f"member {member}: artifact member mismatch")
    if str(p.get("teacher_sha256")) != EXPECTED_TEACHER_SHA:
        raise RuntimeError(f"member {member}: artifact teacher SHA mismatch")
    if str(p.get("source_checkpoint_sha256")) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError(f"member {member}: artifact checkpoint SHA mismatch")
    if str(p.get("protocol_sha256")) != protocol_sha:
        raise RuntimeError(f"member {member}: artifact protocol SHA mismatch")
    if int(p.get("semantic_completed_iteration", -1)) != TARGET_ITERATION:
        raise RuntimeError(f"member {member}: artifact iteration mismatch")
    if str(p.get("production_status")) != "BUILT_UNVALIDATED_NOT_PROMOTED":
        raise RuntimeError(f"member {member}: unexpected production status")
    model = V1SemanticPolicyNet()
    model.load_state_dict(p["model_state"])
    model.eval()
    return model, p


def surface_masks(samples) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(samples)
    legal = np.asarray(
        [[bool(x) for x in sample.legal] for sample in samples],
        dtype=np.bool_,
    )
    if legal.shape != (n, 10):
        raise RuntimeError(f"universal legal-mask shape drift: {legal.shape}")

    s72 = np.zeros(n, dtype=np.bool_)
    trips = np.zeros(n, dtype=np.bool_)
    high_card_no_draw = np.zeros(n, dtype=np.bool_)

    for i, sample in enumerate(samples):
        hole, board, _numeric, cat = sidecar.decode_obs(sample.observation)
        ps = sidecar.private_semantics(hole, board)
        street = int(cat[1])
        if street == 0 and bool(legal[i, ALL_IN]) and len(hole) == 2:
            s72[i] = preflop_class(tuple(hole)) == "72o"
        if street > 0 and bool(legal[i, FOLD]) and int(ps["made"]) >= 3:
            trips[i] = True
        if (
            street > 0
            and bool(legal[i, ALL_IN])
            and bool(ps["high_card_no_draw"])
        ):
            high_card_no_draw[i] = True
    return legal, s72, trips, high_card_no_draw


def validate_surface_coverage(
    protocol: dict,
    s72: np.ndarray,
    trips: np.ndarray,
    high_card_no_draw: np.ndarray,
) -> dict[str, int]:
    actual = {
        "preflop_72o_allin_opportunities": int(s72.sum()),
        "postflop_trips_plus_fold_legal": int(trips.sum()),
        "postflop_high_card_no_immediate_draw_allin_legal": int(
            high_card_no_draw.sum()
        ),
    }
    required = {
        key: int(value)
        for key, value in protocol["shared_fresh_evaluation"]["stability_state_bank"][
            "required_surfaces"
        ].items()
    }
    for key, minimum in required.items():
        if actual.get(key, 0) < minimum:
            raise RuntimeError(
                f"stability bank surface coverage below frozen minimum: "
                f"{key}={actual.get(key, 0)}/{minimum}"
            )
    return actual


def member_probabilities(
    samples,
    semantic_rows: np.ndarray,
    route_specialist: np.ndarray,
    *,
    fullpool_path: Path,
    specialist_path: Path,
    member: int,
    protocol_sha: str,
    batch_size: int,
) -> np.ndarray:
    fullpool, full_payload = load_member_model(
        fullpool_path,
        schema=builder.FULLPOOL_SCHEMA,
        member=member,
        protocol_sha=protocol_sha,
    )
    specialist, specialist_payload = load_member_model(
        specialist_path,
        schema=builder.SPECIALIST_SCHEMA,
        member=member,
        protocol_sha=protocol_sha,
    )
    if str(specialist_payload.get("base_fullpool_sha256")) != sha256(fullpool_path):
        raise RuntimeError(f"member {member}: specialist/fullpool binding mismatch")

    out = np.zeros((len(samples), 10), dtype=np.float32)
    for start in range(0, len(samples), int(batch_size)):
        stop = min(len(samples), start + int(batch_size))
        chunk = samples[start:stop]
        b, _target, _weights = vectorized_batch(chunk, "cpu")
        sb = dict(b)
        sb["semantic"] = torch.tensor(
            semantic_rows[start:stop],
            dtype=torch.float32,
        )
        with torch.no_grad():
            fp = fullpool.probabilities(sb).cpu().numpy().astype(np.float32)
            sp = specialist.probabilities(sb).cpu().numpy().astype(np.float32)
        use_sp = route_specialist[start:stop, None]
        chosen = np.where(use_sp, sp, fp)
        out[start:stop] = chosen
    del fullpool, specialist, full_payload, specialist_payload
    return out


def branch_decision(report: dict) -> dict[str, Any]:
    passed = bool(report.get("rebuild_stability_pass"))
    if passed:
        return {
            "k8_stability_result": "PASS",
            "k12_extension_authorized": False,
            "next_gate": (
                "K=8 stabilization branch resolved PASS. Proceed only to the frozen "
                "aggregate quality/specialist-confirmation/safety sequence; external "
                "strength remains separately gated."
            ),
        }
    return {
        "k8_stability_result": "FAIL",
        "k12_extension_authorized": True,
        "next_gate": (
            "Mechanically clean K=8 stability FAIL activates exactly preregistered "
            "members 9-12. Build all four with frozen seeds, then rerun unchanged "
            "stability metrics over all 12. Do not inspect external strength first."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--construction-dir", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--coverage-adjudication", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()

    if int(args.threads) <= 0 or int(args.batch_size) <= 0:
        raise SystemExit("threads and batch-size must be positive")
    set_threads(int(args.threads))

    protocol_path = args.protocol.resolve(strict=True)
    protocol, protocol_sha = validate_protocol(protocol_path)
    adjudication_path = args.coverage_adjudication.resolve(strict=True)
    _adjudication, adjudication_sha = validate_coverage_adjudication(
        adjudication_path, protocol_blob_sha="5fd7667926c6112310fe5eca61afff396ad218de"
    )
    construction_run, members = validate_construction(
        args.construction_dir,
        protocol_sha=protocol_sha,
    )

    teacher_path = args.teacher.resolve(strict=True)
    teacher_models, _teacher_payload, teacher_sha = builder.load_teacher(
        teacher_path, EXPECTED_TEACHER_SHA
    )
    if teacher_sha != EXPECTED_TEACHER_SHA:
        raise RuntimeError("teacher SHA validation drift")

    solver = SolverLibrary(args.solver.resolve(strict=True))
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("POST15315_K8_STABILITY_BANK_BEGIN", flush=True)
    train, hold, collection = collect.collect_fresh_split(
        solver,
        teacher_models,
        BANK_EPISODES,
        master_seed=BANK_SEED,
        sample_iteration=TARGET_ITERATION,
        progress_prefix="POST15315_K8_STABILITY_BANK",
    )
    samples = list(train) + list(hold)
    if not samples:
        raise RuntimeError("empty stability state bank")

    legal, s72, trips, high_card_no_draw = surface_masks(samples)
    coverage = validate_surface_coverage(
        protocol, s72, trips, high_card_no_draw
    )
    semantic_rows = np.asarray(
        [semantic_vector_from_obs(sample.observation) for sample in samples],
        dtype=np.float32,
    )

    probs = np.zeros((K, len(samples), 10), dtype=np.float32)
    for idx, row in enumerate(members):
        member = int(row["member"])
        print(f"POST15315_K8_STABILITY_MEMBER_{member:02d}_BEGIN", flush=True)
        probs[idx] = member_probabilities(
            samples,
            semantic_rows,
            trips,
            fullpool_path=Path(row["fullpool_path"]),
            specialist_path=Path(row["specialist_path"]),
            member=member,
            protocol_sha=protocol_sha,
            batch_size=int(args.batch_size),
        )
        print(f"POST15315_K8_STABILITY_MEMBER_{member:02d}_PASS", flush=True)

    bank_path = out / "post15315_k8_stability_bank.npz"
    atomic_npz(
        bank_path,
        probs=probs,
        legal=legal,
        surface_72o=s72,
        surface_trips_fold=trips,
        surface_high_card_no_draw=high_card_no_draw,
    )
    bank_sha = sha256(bank_path)

    bank_meta = {
        "schema": BANK_SCHEMA,
        "status": "PASS",
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "coverage_adjudication": str(adjudication_path),
        "coverage_adjudication_sha256": adjudication_sha,
        "original_preregistered_episode_budget": ORIGINAL_BANK_EPISODES,
        "teacher_iteration": TARGET_ITERATION,
        "teacher": str(teacher_path),
        "teacher_sha256": EXPECTED_TEACHER_SHA,
        "construction_report": construction_run,
        "member_artifacts": members,
        "seed": BANK_SEED,
        "forced_three_handed_episodes": BANK_EPISODES,
        "state_count": len(samples),
        "collection": collect.canonical_collection_stats(collection),
        "state_generator": (
            "Frozen 15315 semantic-Advantage teacher rollout through "
            "post15315_collection.collect_fresh_split; evaluation-only common bank; "
            "rebuild policies do not influence state generation."
        ),
        "surface_coverage": coverage,
        "bank_path": str(bank_path),
        "bank_sha256": bank_sha,
        "training_performed": False,
        "member_selection_performed": False,
        "strength_benchmark_performed": False,
    }
    meta_path = out / "post15315_k8_stability_bank_meta.json"
    atomic_json(bank_meta, meta_path)

    report = stability.evaluate(
        probs,
        legal,
        s72,
        trips,
        high_card_no_draw,
        protocol,
    )
    report_path = out / "post15315_k8_stability_report.json"
    atomic_json(report, report_path)

    decision = branch_decision(report)
    gate = {
        "schema": GATE_SCHEMA,
        "status": "PASS",
        "mechanically_clean": True,
        "protocol_sha256": protocol_sha,
        "coverage_adjudication_sha256": adjudication_sha,
        "teacher_sha256": EXPECTED_TEACHER_SHA,
        "construction_report_sha256": construction_run["sha256"],
        "bank_sha256": bank_sha,
        "bank_meta_sha256": sha256(meta_path),
        "stability_report_sha256": sha256(report_path),
        "stability_report": str(report_path),
        **decision,
        "strength_benchmark_performed": False,
    }
    gate_path = out / "post15315_k8_stability_gate.json"
    atomic_json(gate, gate_path)

    print("POST15315_K8_STABILITY_RESULT_" + decision["k8_stability_result"], flush=True)
    print("coverage=" + json.dumps(coverage, sort_keys=True), flush=True)
    print("criteria=" + json.dumps(report["precommitted_criteria"], sort_keys=True), flush=True)
    print("summary=" + json.dumps(report["stability_summary"], sort_keys=True), flush=True)
    print(f"report={report_path}", flush=True)
    print(f"gate={gate_path}", flush=True)
    if bool(decision["k12_extension_authorized"]):
        print("POST15315_K12_EXTENSION_AUTHORIZED_BY_FROZEN_RULE", flush=True)
        print("STOP: do not launch strength; prepare exactly members 9-12.", flush=True)
    else:
        print("POST15315_K8_STABILITY_PASS", flush=True)
        print("STOP: K=8 stability only; aggregate quality is the next gate.", flush=True)
    print("POST15315_K8_STABILITY_RUNNER_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
