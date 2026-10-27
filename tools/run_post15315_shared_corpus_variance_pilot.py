#!/usr/bin/env python3
from __future__ import annotations

"""Research-only shared-corpus variance attribution after closed K=12 FAIL.

The experiment pools the already-created member collection caches into one
common teacher-target corpus, then trains eight independent fit replicas on the
exact same ordered corpus. The original closed-branch 12 policies and the
shared-corpus 8 replicas are evaluated on one fresh research-only common bank.

This cannot promote or reopen the failed K=12 candidate and opens no external
strength evidence.

PROJECT_CONTRACT_IDS:
TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,
VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,VALID-038,
SAFE-001,SAFE-010,SAFE-011,CKPT-001,ART-001
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import build_post15315_rebuild_member as builder
import evaluate_post15315_rebuild_stability as stability
import post15315_collection as collect
import post15315_member_cache as member_cache
import run_post15315_multirebuild_production as production
import run_post15315_rebuild_stability_k8 as k8stab
from audit_3h_average_policy_semantic_continuation_10105 import semantic_vector_from_obs
from spincore.solver import SolverLibrary

PILOT_SCHEMA = "SPINCORE_POST15315_SHARED_CORPUS_VARIANCE_PILOT_RESULT_V1"
PREREG_SCHEMA = "SPINCORE_POST15315_SHARED_CORPUS_VARIANCE_PILOT_V1"
EXPECTED_PROTOCOL_SHA = "5fd7667926c6112310fe5eca61afff396ad218de"
EXPECTED_TEACHER_SHA = "ac49b77c8ec3204e07f6d44d3718c68880e60675c5521f538438a3fa3ceae746"
EXPECTED_CHECKPOINT_SHA = "f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
FINAL_ITERATION = 15315
K12_GATE_SCHEMA = "SPINCORE_POST15315_K12_STABILITY_GATE_V1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sample_key(sample):
    return bytes(sample.observation), tuple(bool(x) for x in sample.legal)


def same_target(a, b) -> bool:
    return np.allclose(
        np.asarray(a.target, dtype=np.float64),
        np.asarray(b.target, dtype=np.float64),
        atol=1e-12,
        rtol=0.0,
    )


def dedup_extend(dst: list, seen: dict, samples, *, exclude: set | None = None) -> int:
    added = 0
    for sample in samples:
        key = sample_key(sample)
        if exclude is not None and key in exclude:
            continue
        prior = seen.get(key)
        if prior is not None:
            if not same_target(prior, sample):
                raise RuntimeError("same state/legal key has teacher-target drift")
            continue
        seen[key] = sample
        dst.append(sample)
        added += 1
    return added


def validate_closed_k12(k12_dir: Path) -> dict:
    gate_path = k12_dir / "post15315_k12_stability_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("schema") != K12_GATE_SCHEMA or gate.get("status") != "PASS":
        raise RuntimeError("K=12 gate is not mechanically PASS")
    if not bool(gate.get("mechanically_clean")):
        raise RuntimeError("K=12 gate is not mechanically clean")
    if str(gate.get("k12_stability_result")) != "FAIL":
        raise RuntimeError("pilot is allowed only after immutable K=12 FAIL")
    if bool(gate.get("aggregate_quality_authorized")):
        raise RuntimeError("closed K=12 unexpectedly authorizes aggregate quality")
    if bool(gate.get("strength_benchmark_performed")):
        raise RuntimeError("closed K=12 unexpectedly opened external strength")
    diag_path = k12_dir / "post15315_k12_instability_diagnostic.json"
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    if diag.get("schema") != "SPINCORE_POST15315_K12_INSTABILITY_DIAGNOSTIC_V1":
        raise RuntimeError("missing canonical K=12 instability diagnostic")
    if diag.get("status") != "PASS" or not bool(diag.get("frozen_k12_failure_is_immutable")):
        raise RuntimeError("K=12 diagnostic does not preserve immutable failure")
    if bool(diag.get("external_strength_benchmark_performed")):
        raise RuntimeError("diagnostic unexpectedly opened strength")
    return {
        "gate_path": str(gate_path.resolve()),
        "gate_sha256": sha256(gate_path),
        "diagnostic_path": str(diag_path.resolve()),
        "diagnostic_sha256": sha256(diag_path),
    }


def validate_prereg(path: Path) -> dict:
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("schema") != PREREG_SCHEMA:
        raise RuntimeError("wrong shared-corpus pilot preregistration schema")
    if p.get("status") != "PREREGISTERED_RESEARCH_ONLY_NOT_PROMOTIONAL":
        raise RuntimeError("shared-corpus pilot preregistration status drift")
    design = dict(p.get("fit_design") or {})
    bank = dict(p.get("fresh_research_bank") or {})
    if int(design.get("replicas", -1)) != 8:
        raise RuntimeError("pilot replica count drift")
    if int(design.get("fullpool_fit_steps", -1)) != 500:
        raise RuntimeError("pilot fullpool-step drift")
    if int(design.get("specialist_fit_steps", -1)) != 25:
        raise RuntimeError("pilot specialist-step drift")
    if list(design.get("fullpool_fit_seeds") or []) != list(range(153160001, 153160009)):
        raise RuntimeError("pilot fullpool fit-seed drift")
    if list(design.get("specialist_fit_seeds") or []) != list(range(153161001, 153161009)):
        raise RuntimeError("pilot specialist fit-seed drift")
    if int(bank.get("seed", -1)) != 153159501:
        raise RuntimeError("pilot research-bank seed drift")
    if int(bank.get("forced_three_handed_episodes", -1)) != 40000:
        raise RuntimeError("pilot research-bank episode drift")
    return p


def original_member_rows(protocol: dict, protocol_sha: str, k8_dir: Path, ext_dir: Path):
    _run, first8 = k8stab.validate_construction(k8_dir, protocol_sha=protocol_sha)
    rows = list(first8)
    for member in range(9, 13):
        seeds = builder.member_spec(protocol, member)
        rp = ext_dir / f"member_{member:02d}_report.json"
        if not production._validate_existing_member(
            rp,
            member=member,
            teacher_sha=EXPECTED_TEACHER_SHA,
            protocol_sha=protocol_sha,
            seeds=seeds,
        ):
            raise RuntimeError(f"member {member}: durable extension member missing")
        q = json.loads(rp.read_text(encoding="utf-8"))
        rows.append(
            {
                "member": member,
                "fullpool_path": str(Path(q["fullpool_artifact"]).resolve(strict=True)),
                "specialist_path": str(Path(q["specialist_artifact"]).resolve(strict=True)),
            }
        )
    if [int(x["member"]) for x in rows] != list(range(1, 13)):
        raise RuntimeError("original member order drift")
    return rows


def load_shared_corpus(
    protocol: dict,
    protocol_sha: str,
    k8_dir: Path,
    ext_dir: Path,
) -> tuple[list, list, list, dict]:
    ordinary: list = []
    ordinary_seen: dict = {}
    augmentation_raw: list = []
    augmentation_seen: dict = {}
    specialist_raw: list = []
    specialist_seen: dict = {}
    cache_rows = []

    for member in range(1, 13):
        base = k8_dir if member <= 8 else ext_dir
        cp = member_cache.cache_path(base, member)
        seeds = builder.member_spec(protocol, member)
        payload = member_cache.load_member_cache(
            cp,
            expected_member=member,
            expected_teacher_sha256=EXPECTED_TEACHER_SHA,
            expected_protocol_sha256=protocol_sha,
            expected_member_seeds=seeds,
            expected_collection_threads=2,
        )
        ordinary_added = dedup_extend(
            ordinary, ordinary_seen, list(payload["ordinary"]["train"])
        )
        augmentation_added = dedup_extend(
            augmentation_raw,
            augmentation_seen,
            list(payload["augmentation"]["train"]) + list(payload["augmentation"]["holdout"]),
        )
        specialist_added = dedup_extend(
            specialist_raw,
            specialist_seen,
            list(payload["specialist"]["train"]) + list(payload["specialist"]["holdout"]),
        )
        cache_rows.append(
            {
                "member": member,
                "path": str(cp.resolve()),
                "sha256": sha256(cp),
                "ordinary_unique_added": ordinary_added,
                "augmentation_unique_added": augmentation_added,
                "specialist_unique_added": specialist_added,
            }
        )
        del payload
        gc.collect()

    ordinary_strong_keys = {sample_key(s) for s in ordinary if builder.strong(s)}
    novel = [s for s in augmentation_raw if sample_key(s) not in ordinary_strong_keys]
    full_train = list(ordinary) + novel
    specialist_unique = builder.spec.unique_strong(specialist_raw)

    if len(ordinary) < 20000:
        raise RuntimeError("pooled ordinary support unexpectedly small")
    if len(novel) < 700:
        raise RuntimeError("pooled novel strong support unexpectedly small")
    if len(specialist_unique) < 1200:
        raise RuntimeError("pooled specialist support unexpectedly small")

    meta = {
        "cache_rows": cache_rows,
        "ordinary_unique_train_samples": len(ordinary),
        "ordinary_strong_keys": len(ordinary_strong_keys),
        "augmentation_unique_raw": len(augmentation_raw),
        "novel_strong_states": len(novel),
        "fullpool_train_samples": len(full_train),
        "specialist_unique_strong_states": len(specialist_unique),
        "ordinary_digest": collect.sample_digest(ordinary),
        "novel_digest": collect.sample_digest(novel),
        "specialist_digest": collect.sample_digest(specialist_unique),
    }
    return ordinary, full_train, specialist_unique, meta


def save_pilot_member(
    out: Path,
    *,
    replica: int,
    fullpool,
    specialist,
    protocol_sha: str,
    full_seed: int,
    specialist_seed: int,
    corpus_meta: dict,
) -> dict:
    full_path = out / f"shared_member_{replica:02d}_fullpool.pt"
    sp_path = out / f"shared_member_{replica:02d}_specialist.pt"
    common = {
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "teacher_sha256": EXPECTED_TEACHER_SHA,
        "teacher_schema": builder.ADV_SCHEMA,
        "semantic_completed_iteration": FINAL_ITERATION,
        "protocol_sha256": protocol_sha,
        "member": replica,
        "member_seeds": {
            "shared_fullpool_fit_seed": int(full_seed),
            "shared_specialist_fit_seed": int(specialist_seed),
        },
        "collection_threads": 0,
        "fit_threads": 8,
        "production_status": "BUILT_UNVALIDATED_NOT_PROMOTED",
        "research_only_shared_corpus": True,
    }
    builder.atomic_torch_save(
        {
            "schema": builder.FULLPOOL_SCHEMA,
            **common,
            "selected_steps": 500,
            "ordinary_train_samples": int(corpus_meta["ordinary_unique_train_samples"]),
            "extra_unique_strong_states": int(corpus_meta["novel_strong_states"]),
            "model_state": {k: v.detach().cpu() for k, v in fullpool.state_dict().items()},
        },
        full_path,
    )
    full_sha = sha256(full_path)
    builder.atomic_torch_save(
        {
            "schema": builder.SPECIALIST_SCHEMA,
            **common,
            "selected_steps": 25,
            "selected_training_mode": "STRATIFIED",
            "specialist_train_episodes": 0,
            "specialist_unique_strong_states": int(corpus_meta["specialist_unique_strong_states"]),
            "specialist_unique_strong_strata": {},
            "base_fullpool_sha256": full_sha,
            "route_contract": "postflop made_category>=TRIPS AND FOLD legal -> specialist; otherwise fullpool",
            "model_state": {k: v.detach().cpu() for k, v in specialist.state_dict().items()},
        },
        sp_path,
    )
    return {
        "member": replica,
        "fullpool_path": str(full_path.resolve()),
        "fullpool_sha256": full_sha,
        "specialist_path": str(sp_path.resolve()),
        "specialist_sha256": sha256(sp_path),
        "fullpool_fit_seed": int(full_seed),
        "specialist_fit_seed": int(specialist_seed),
    }


def train_shared_replicas(
    checkpoint: Path,
    protocol_sha: str,
    prereg: dict,
    full_train: list,
    specialist_unique: list,
    corpus_meta: dict,
    out: Path,
) -> list[dict]:
    payload = torch.load(checkpoint.resolve(strict=True), map_location="cpu", weights_only=False)
    d3 = (payload.get("domains") or {}).get(builder.DOMAIN) or {}
    cfg = dict(payload.get("config") or {})
    if not d3 or not cfg:
        raise RuntimeError("frozen 10105 THREE_HANDED state/config missing")

    design = prereg["fit_design"]
    full_seeds = list(design["fullpool_fit_seeds"])
    sp_seeds = list(design["specialist_fit_seeds"])
    torch.set_num_threads(int(design["fit_threads"]))
    rows = []

    for idx in range(8):
        replica = idx + 1
        print(f"POST15315_SHARED_CORPUS_REPLICA_{replica:02d}_FIT_BEGIN", flush=True)
        _, _, fullpool, fullopt = builder.confirm.initial_policy_pair(d3, cfg)
        builder.train_semantic(
            fullpool,
            fullopt,
            full_train,
            cfg,
            steps=int(design["fullpool_fit_steps"]),
            seed=int(full_seeds[idx]),
            label=f"POST15315_SHARED_REPLICA_{replica:02d}_FULLPOOL_FIT",
        )
        fullpool.eval()

        specialist = builder.spec.clone_model(fullpool)
        specialist_opt = torch.optim.Adam(
            specialist.parameters(), lr=float(cfg["learning_rate"])
        )
        import random
        rng = random.Random(int(sp_seeds[idx]))
        builder.spec.train_range(
            specialist,
            specialist_opt,
            specialist_unique,
            cfg,
            0,
            int(design["specialist_fit_steps"]),
            rng,
            "STRATIFIED",
        )
        specialist.eval()
        row = save_pilot_member(
            out,
            replica=replica,
            fullpool=fullpool,
            specialist=specialist,
            protocol_sha=protocol_sha,
            full_seed=int(full_seeds[idx]),
            specialist_seed=int(sp_seeds[idx]),
            corpus_meta=corpus_meta,
        )
        rows.append(row)
        print(f"POST15315_SHARED_CORPUS_REPLICA_{replica:02d}_FIT_PASS", flush=True)
        del fullpool, fullopt, specialist, specialist_opt
        gc.collect()
    return rows


def action_dispersion(probs: np.ndarray) -> list[dict]:
    ensemble = probs.mean(axis=0)
    dev = np.abs(probs - ensemble[None, :, :]).mean(axis=0)
    totals = dev.sum(axis=0)
    denom = float(totals.sum())
    rows = []
    for action, name in enumerate(stability.ACTION_NAMES):
        x = dev[:, action]
        rows.append(
            {
                "action": name,
                "mean_abs_member_probability_deviation": float(x.mean()),
                "p95_abs_member_probability_deviation": float(np.quantile(x, 0.95)),
                "share_of_total_absolute_dispersion": (
                    float(totals[action] / denom) if denom > 0.0 else 0.0
                ),
            }
        )
    return sorted(rows, key=lambda r: r["share_of_total_absolute_dispersion"], reverse=True)


def safe_ratio(a: float, b: float) -> float:
    if b <= 0.0:
        raise RuntimeError("cannot form attribution ratio with nonpositive denominator")
    return float(a / b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--pilot-prereg", type=Path, required=True)
    ap.add_argument("--k8-dir", type=Path, required=True)
    ap.add_argument("--extension-dir", type=Path, required=True)
    ap.add_argument("--k12-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    teacher_path = args.teacher.resolve(strict=True)
    solver_path = args.solver.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    pilot_prereg_path = args.pilot_prereg.resolve(strict=True)
    k8_dir = args.k8_dir.resolve(strict=True)
    ext_dir = args.extension_dir.resolve(strict=True)
    k12_dir = args.k12_dir.resolve(strict=True)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("source checkpoint SHA mismatch")
    if sha256(teacher_path) != EXPECTED_TEACHER_SHA:
        raise RuntimeError("teacher SHA mismatch")
    if sha256(protocol_path) != EXPECTED_PROTOCOL_SHA:
        raise RuntimeError("original post15315 protocol SHA mismatch")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    prereg = validate_prereg(pilot_prereg_path)
    closed = validate_closed_k12(k12_dir)
    original_rows = original_member_rows(protocol, EXPECTED_PROTOCOL_SHA, k8_dir, ext_dir)

    print("POST15315_SHARED_CORPUS_POOL_BEGIN", flush=True)
    _ordinary, full_train, specialist_unique, corpus_meta = load_shared_corpus(
        protocol, EXPECTED_PROTOCOL_SHA, k8_dir, ext_dir
    )
    print(
        "POST15315_SHARED_CORPUS_POOL_PASS "
        + json.dumps(
            {
                "ordinary": corpus_meta["ordinary_unique_train_samples"],
                "fullpool": corpus_meta["fullpool_train_samples"],
                "specialist": corpus_meta["specialist_unique_strong_states"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    shared_rows = train_shared_replicas(
        checkpoint,
        EXPECTED_PROTOCOL_SHA,
        prereg,
        full_train,
        specialist_unique,
        corpus_meta,
        out,
    )

    teacher_models, _teacher_payload, teacher_sha = builder.load_teacher(
        teacher_path, EXPECTED_TEACHER_SHA
    )
    if teacher_sha != EXPECTED_TEACHER_SHA:
        raise RuntimeError("teacher validation drift")
    solver = SolverLibrary(solver_path)
    bank = prereg["fresh_research_bank"]
    print("POST15315_SHARED_CORPUS_RESEARCH_BANK_BEGIN", flush=True)
    train, hold, collection_raw = collect.collect_fresh_split(
        solver,
        teacher_models,
        int(bank["forced_three_handed_episodes"]),
        master_seed=int(bank["seed"]),
        sample_iteration=FINAL_ITERATION,
        progress_prefix="POST15315_SHARED_CORPUS_RESEARCH_BANK",
    )
    samples = list(train) + list(hold)
    legal, s72, trips, high = k8stab.surface_masks(samples)
    semantic_rows = np.asarray(
        [semantic_vector_from_obs(s.observation) for s in samples], dtype=np.float32
    )
    surfaces = {
        "preflop_72o": int(s72.sum()),
        "trips_plus_fold": int(trips.sum()),
        "high_card_no_draw": int(high.sum()),
    }

    original_probs = np.zeros((12, len(samples), 10), dtype=np.float32)
    for idx, row in enumerate(original_rows):
        member = int(row["member"])
        print(f"POST15315_ORIGINAL12_RESEARCH_EVAL_{member:02d}_BEGIN", flush=True)
        original_probs[idx] = k8stab.member_probabilities(
            samples,
            semantic_rows,
            trips,
            fullpool_path=Path(row["fullpool_path"]),
            specialist_path=Path(row["specialist_path"]),
            member=member,
            protocol_sha=EXPECTED_PROTOCOL_SHA,
            batch_size=int(args.batch_size),
        )
        print(f"POST15315_ORIGINAL12_RESEARCH_EVAL_{member:02d}_PASS", flush=True)

    shared_probs = np.zeros((8, len(samples), 10), dtype=np.float32)
    for idx, row in enumerate(shared_rows):
        member = int(row["member"])
        print(f"POST15315_SHARED8_RESEARCH_EVAL_{member:02d}_BEGIN", flush=True)
        shared_probs[idx] = k8stab.member_probabilities(
            samples,
            semantic_rows,
            trips,
            fullpool_path=Path(row["fullpool_path"]),
            specialist_path=Path(row["specialist_path"]),
            member=member,
            protocol_sha=EXPECTED_PROTOCOL_SHA,
            batch_size=int(args.batch_size),
        )
        print(f"POST15315_SHARED8_RESEARCH_EVAL_{member:02d}_PASS", flush=True)

    original_report = stability.evaluate(original_probs, legal, s72, trips, high, protocol)
    shared_report = stability.evaluate(shared_probs, legal, s72, trips, high, protocol)
    osum = dict(original_report["stability_summary"])
    ssum = dict(shared_report["stability_summary"])

    ratio_mean = safe_ratio(
        float(ssum["broad_worst_split_mean_tv"]),
        float(osum["broad_worst_split_mean_tv"]),
    )
    ratio_p95 = safe_ratio(
        float(ssum["broad_worst_split_statewise_p95_tv"]),
        float(osum["broad_worst_split_statewise_p95_tv"]),
    )
    ratio_loo = safe_ratio(
        float(ssum["leave_one_out_worst_mean_tv"]),
        float(osum["leave_one_out_worst_mean_tv"]),
    )
    if ratio_mean <= 0.70 and ratio_p95 <= 0.70:
        attribution = "CORPUS_SUPPORT_DOMINANT_HYPOTHESIS_SUPPORTED"
        next_branch = (
            "Design a separate preregistered shared-corpus candidate using fresh training "
            "seeds and a never-observed promotion bank."
        )
    elif ratio_mean >= 0.90 and ratio_p95 >= 0.90:
        attribution = "FIT_ARCHITECTURE_DOMINANT"
        next_branch = (
            "Do not scale collection; investigate optimization/model/loss stability before "
            "another candidate."
        )
    else:
        attribution = "MIXED_VARIANCE"
        next_branch = (
            "Run a second factorial research pilot separating corpus and fit components "
            "before candidate construction."
        )

    pred_path = out / "post15315_shared_corpus_variance_predictions.npz"
    np.savez_compressed(
        pred_path,
        original_probs=original_probs,
        shared_probs=shared_probs,
        legal=legal,
        surface_72o=s72,
        surface_trips_fold=trips,
        surface_high_card_no_draw=high,
    )

    result = {
        "schema": PILOT_SCHEMA,
        "status": "PASS",
        "purpose": "research_variance_attribution_only",
        "closed_k12_failure_is_immutable": True,
        "promotion_decision_performed": False,
        "external_strength_benchmark_performed": False,
        "member_selection_performed": False,
        "member_weighting_performed": False,
        "k_greater_than_12_performed": False,
        "original_protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "pilot_preregistration": str(pilot_prereg_path),
        "pilot_preregistration_sha256": sha256(pilot_prereg_path),
        "closed_k12_evidence": closed,
        "shared_corpus": corpus_meta,
        "shared_replicas": shared_rows,
        "research_bank": {
            "seed": int(bank["seed"]),
            "episodes": int(bank["forced_three_handed_episodes"]),
            "state_count": len(samples),
            "surface_counts": surfaces,
            "collection": collect.canonical_collection_stats(collection_raw),
        },
        "original12_stability_summary": osum,
        "shared8_stability_summary": ssum,
        "original12_action_dispersion": action_dispersion(original_probs),
        "shared8_action_dispersion": action_dispersion(shared_probs),
        "attribution_ratios_shared8_over_original12": {
            "broad_worst_split_mean_tv": ratio_mean,
            "broad_worst_split_statewise_p95_tv": ratio_p95,
            "leave_one_out_worst_mean_tv": ratio_loo,
        },
        "attribution_result": attribution,
        "next_branch": next_branch,
        "prediction_archive": str(pred_path.resolve()),
        "prediction_archive_sha256": sha256(pred_path),
        "interpretation": (
            "This development-only experiment compares combined corpus+fit variance of the "
            "closed original rebuilds with fit-only variance under a common pooled corpus. "
            "It cannot promote any policy. Any future candidate requires separate preregistration "
            "and a never-observed validation bank."
        ),
    }
    report_path = out / "post15315_shared_corpus_variance_pilot_report.json"
    atomic_json(result, report_path)

    print("POST15315_SHARED_CORPUS_VARIANCE_PILOT_PASS", flush=True)
    print("attribution_result=" + attribution, flush=True)
    print(
        "ratios="
        + json.dumps(result["attribution_ratios_shared8_over_original12"], sort_keys=True),
        flush=True,
    )
    print("original12_summary=" + json.dumps(osum, sort_keys=True), flush=True)
    print("shared8_summary=" + json.dumps(ssum, sort_keys=True), flush=True)
    print("surface_counts=" + json.dumps(surfaces, sort_keys=True), flush=True)
    print(f"report={report_path}", flush=True)
    print("STOP: research attribution only; no strength or promotion.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
