#!/usr/bin/env python3
from __future__ import annotations

"""Research-only canonical-corpus stabilization pilot after factorial corpus dominance.

Three matched eight-replica groups are compared on one fresh 100k common bank:
(1) existing V2 member-01 single-anchor control, (2) deterministic multi-anchor
canonical corpus at exactly the control corpus size, and (3) the full expanded
unique multi-anchor canonical corpus. Canonical treatment replicas reuse the
same eight fit-seed pairs as the V2 control.

No result from this script can promote a policy, reopen the failed K=12
candidate, or open external strength evidence.

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
import random
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
import run_post15315_rebuild_stability_k8 as k8stab
from audit_3h_average_policy_semantic_continuation_10105 import semantic_vector_from_obs
from spincore.solver import SolverLibrary

RESULT_SCHEMA = "SPINCORE_POST15315_CANONICAL_CORPUS_PILOT_RESULT_V1"
PREREG_SCHEMA = "SPINCORE_POST15315_CANONICAL_CORPUS_PILOT_V1"
FACTORIAL_SCHEMA = "SPINCORE_POST15315_FACTORIAL_VARIANCE_PILOT_RESULT_V1"
CONTROL_SCHEMA = "SPINCORE_POST15315_SHARED_CORPUS_VARIANCE_PILOT_RESULT_V2"
EXPECTED_PROTOCOL_BLOB_SHA = "5fd7667926c6112310fe5eca61afff396ad218de"
EXPECTED_TEACHER_SHA = "ac49b77c8ec3204e07f6d44d3718c68880e60675c5521f538438a3fa3ceae746"
EXPECTED_CHECKPOINT_SHA = "f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
FINAL_ITERATION = 15315


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


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


def rank_key(sample, salt: str) -> bytes:
    h = hashlib.sha256()
    h.update(str(salt).encode("utf-8"))
    h.update(b"\0")
    h.update(bytes(sample.observation))
    h.update(bytes(1 if bool(x) else 0 for x in sample.legal))
    return h.digest()


def canonical_unique(samples, *, salt: str) -> list:
    seen = {}
    for sample in samples:
        key = sample_key(sample)
        prior = seen.get(key)
        if prior is not None:
            if not same_target(prior, sample):
                raise RuntimeError("duplicate canonical state/legal key has teacher-target drift")
            continue
        seen[key] = sample
    return sorted(seen.values(), key=lambda s: rank_key(s, salt))


def validate_prereg(path: Path) -> dict:
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("schema") != PREREG_SCHEMA:
        raise RuntimeError("wrong canonical-corpus preregistration schema")
    if p.get("status") != "PREREGISTERED_RESEARCH_ONLY_NOT_PROMOTIONAL":
        raise RuntimeError("canonical-corpus preregistration status drift")
    sources = dict(p.get("canonical_sources") or {})
    if list(sources.get("anchor_members") or []) != [1, 4, 7, 10]:
        raise RuntimeError("canonical source-anchor drift")
    salts = dict(sources.get("stream_salts") or {})
    expected_salts = {
        "ordinary": "POST15315_CANONICAL_ORDINARY_V1",
        "augmentation": "POST15315_CANONICAL_AUGMENTATION_V1",
        "specialist": "POST15315_CANONICAL_SPECIALIST_V1",
    }
    if salts != expected_salts:
        raise RuntimeError("canonical stream-salt drift")

    fixed = dict((p.get("treatments") or {}).get("fixed_size_canonical") or {})
    expected_counts = {
        "ordinary_train_samples": 26798,
        "novel_strong_samples": 2645,
        "fullpool_train_samples": 29443,
        "specialist_unique_strong_samples": 4125,
    }
    for key, value in expected_counts.items():
        if int(fixed.get(key, -1)) != value:
            raise RuntimeError(f"fixed-size canonical count drift: {key}")

    fit = dict(p.get("fit_design") or {})
    if int(fit.get("replicas_per_treatment", -1)) != 8:
        raise RuntimeError("canonical replica-count drift")
    if list(fit.get("fullpool_fit_seeds") or []) != list(range(153160001, 153160009)):
        raise RuntimeError("canonical fullpool fit-seed drift")
    if list(fit.get("specialist_fit_seeds") or []) != list(range(153161001, 153161009)):
        raise RuntimeError("canonical specialist fit-seed drift")
    if int(fit.get("fullpool_fit_steps", -1)) != 500:
        raise RuntimeError("canonical fullpool-step drift")
    if int(fit.get("specialist_fit_steps", -1)) != 25:
        raise RuntimeError("canonical specialist-step drift")
    if int(fit.get("fit_threads", -1)) != 8:
        raise RuntimeError("canonical fit-thread drift")

    bank = dict(p.get("fresh_confirmation_bank") or {})
    if int(bank.get("seed", -1)) != 153159503:
        raise RuntimeError("canonical confirmation-bank seed drift")
    if int(bank.get("forced_three_handed_episodes", -1)) != 100000:
        raise RuntimeError("canonical confirmation-bank budget drift")
    return p


def validate_factorial(path: Path, prereg: dict) -> dict:
    p = json.loads(path.read_text(encoding="utf-8"))
    trigger = dict(prereg.get("trigger_evidence") or {})
    if p.get("schema") != FACTORIAL_SCHEMA or p.get("status") != "PASS":
        raise RuntimeError("factorial trigger evidence is not canonical PASS")
    if p.get("attribution_result") != "CORPUS_CONDITIONED_LARGER":
        raise RuntimeError("canonical-corpus pilot requires corpus-conditioned-larger result")
    ratios = dict(p.get("corpus_over_fit_ratios") or {})
    if float(ratios.get("mean_tv", 0.0)) < float(trigger["required_min_corpus_over_fit_mean_ratio"]):
        raise RuntimeError("factorial mean ratio below canonical-pilot trigger")
    if float(ratios.get("p95_tv", 0.0)) < float(trigger["required_min_corpus_over_fit_p95_ratio"]):
        raise RuntimeError("factorial p95 ratio below canonical-pilot trigger")
    if not bool(p.get("closed_k12_failure_is_immutable")):
        raise RuntimeError("factorial evidence does not preserve closed K=12 failure")
    if bool(p.get("promotion_decision_performed")) or bool(p.get("external_strength_benchmark_performed")):
        raise RuntimeError("factorial evidence unexpectedly opened promotion/strength")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "attribution_result": p["attribution_result"],
        "ratios": ratios,
        "decomposition": dict(p.get("two_way_probability_variance") or {}),
    }


def validate_control(path: Path) -> tuple[dict, list[dict]]:
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("schema") != CONTROL_SCHEMA or p.get("status") != "PASS":
        raise RuntimeError("shared-corpus V2 control report is not canonical PASS")
    if p.get("attribution_result") != "MIXED_VARIANCE":
        raise RuntimeError("unexpected V2 control attribution identity")
    if not bool(p.get("closed_k12_failure_is_immutable")):
        raise RuntimeError("V2 control does not preserve closed K=12 failure")
    rows = list(p.get("shared_replicas") or [])
    if len(rows) != 8 or [int(r.get("member", -1)) for r in rows] != list(range(1, 9)):
        raise RuntimeError("V2 control replica identity drift")
    for row in rows:
        fp = Path(str(row["fullpool_path"])).resolve(strict=True)
        sp = Path(str(row["specialist_path"])).resolve(strict=True)
        if sha256(fp) != str(row["fullpool_sha256"]):
            raise RuntimeError("V2 control fullpool artifact SHA mismatch")
        if sha256(sp) != str(row["specialist_sha256"]):
            raise RuntimeError("V2 control specialist artifact SHA mismatch")
    return (
        {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "prior_summary": dict(p.get("shared8_stability_summary") or {}),
            "prior_bank": dict(p.get("research_bank") or {}),
        },
        rows,
    )


def cache_base(anchor: int, k8_dir: Path, extension_dir: Path) -> Path:
    return k8_dir if int(anchor) <= 8 else extension_dir


def load_canonical_corpora(
    protocol: dict,
    protocol_sha: str,
    prereg: dict,
    k8_dir: Path,
    extension_dir: Path,
) -> tuple[dict, dict, dict]:
    anchors = list(prereg["canonical_sources"]["anchor_members"])
    salts = dict(prereg["canonical_sources"]["stream_salts"])
    ordinary_all = []
    augmentation_all = []
    specialist_all = []
    cache_rows = []

    for anchor in anchors:
        cp = member_cache.cache_path(cache_base(int(anchor), k8_dir, extension_dir), int(anchor))
        payload = member_cache.load_member_cache(
            cp,
            expected_member=int(anchor),
            expected_teacher_sha256=EXPECTED_TEACHER_SHA,
            expected_protocol_sha256=protocol_sha,
            expected_member_seeds=builder.member_spec(protocol, int(anchor)),
            expected_collection_threads=2,
        )
        ordinary = list(payload["ordinary"]["train"])
        augmentation = list(payload["augmentation"]["train"]) + list(payload["augmentation"]["holdout"])
        specialist = list(payload["specialist"]["train"]) + list(payload["specialist"]["holdout"])
        ordinary_all.extend(ordinary)
        augmentation_all.extend(augmentation)
        specialist_all.extend(specialist)
        cache_rows.append(
            {
                "anchor_member": int(anchor),
                "cache_path": str(cp.resolve()),
                "cache_sha256": sha256(cp),
                "ordinary_train_samples_raw": len(ordinary),
                "augmentation_samples_raw": len(augmentation),
                "specialist_samples_raw": len(specialist),
            }
        )

    ordinary_union = canonical_unique(ordinary_all, salt=salts["ordinary"])
    augmentation_union = canonical_unique(augmentation_all, salt=salts["augmentation"])
    specialist_union = canonical_unique(specialist_all, salt=salts["specialist"])

    fixed_cfg = prereg["treatments"]["fixed_size_canonical"]
    n_ord = int(fixed_cfg["ordinary_train_samples"])
    n_novel = int(fixed_cfg["novel_strong_samples"])
    n_spec = int(fixed_cfg["specialist_unique_strong_samples"])
    if len(ordinary_union) < n_ord or len(specialist_union) < n_spec:
        raise RuntimeError("canonical unique union below fixed-size requirements")

    fixed_ordinary = list(ordinary_union[:n_ord])
    fixed_strong_keys = {sample_key(s) for s in fixed_ordinary if builder.strong(s)}
    fixed_novel_pool = [
        s for s in augmentation_union if sample_key(s) not in fixed_strong_keys
    ]
    if len(fixed_novel_pool) < n_novel:
        raise RuntimeError("canonical augmentation union below fixed novel-strong requirement")
    fixed_novel = list(fixed_novel_pool[:n_novel])
    fixed_specialist = list(specialist_union[:n_spec])
    fixed_full = fixed_ordinary + fixed_novel

    expanded_ordinary = list(ordinary_union)
    expanded_strong_keys = {
        sample_key(s) for s in expanded_ordinary if builder.strong(s)
    }
    expanded_novel = [
        s for s in augmentation_union if sample_key(s) not in expanded_strong_keys
    ]
    expanded_specialist = list(specialist_union)
    expanded_full = expanded_ordinary + expanded_novel

    if len(fixed_full) != int(fixed_cfg["fullpool_train_samples"]):
        raise RuntimeError("fixed canonical fullpool size drift")
    if len(expanded_ordinary) < len(fixed_ordinary):
        raise RuntimeError("expanded ordinary unexpectedly smaller than fixed")
    if len(expanded_novel) < len(fixed_novel):
        raise RuntimeError("expanded novel unexpectedly smaller than fixed")
    if len(expanded_specialist) < len(fixed_specialist):
        raise RuntimeError("expanded specialist unexpectedly smaller than fixed")

    fixed = {
        "ordinary": fixed_ordinary,
        "novel": fixed_novel,
        "fullpool": fixed_full,
        "specialist": fixed_specialist,
    }
    expanded = {
        "ordinary": expanded_ordinary,
        "novel": expanded_novel,
        "fullpool": expanded_full,
        "specialist": expanded_specialist,
    }
    meta = {
        "anchors": anchors,
        "source_caches": cache_rows,
        "unique_union": {
            "ordinary": len(ordinary_union),
            "augmentation": len(augmentation_union),
            "specialist": len(specialist_union),
        },
        "fixed_size": {
            "ordinary": len(fixed_ordinary),
            "novel": len(fixed_novel),
            "fullpool": len(fixed_full),
            "specialist": len(fixed_specialist),
            "ordinary_digest": collect.sample_digest(fixed_ordinary),
            "novel_digest": collect.sample_digest(fixed_novel),
            "specialist_digest": collect.sample_digest(fixed_specialist),
        },
        "expanded": {
            "ordinary": len(expanded_ordinary),
            "novel": len(expanded_novel),
            "fullpool": len(expanded_full),
            "specialist": len(expanded_specialist),
            "ordinary_digest": collect.sample_digest(expanded_ordinary),
            "novel_digest": collect.sample_digest(expanded_novel),
            "specialist_digest": collect.sample_digest(expanded_specialist),
        },
    }
    return fixed, expanded, meta


def save_member(
    out: Path,
    *,
    treatment: str,
    replica: int,
    artifact_member: int,
    fullpool,
    specialist,
    protocol_sha: str,
    full_seed: int,
    specialist_seed: int,
    corpus_meta: dict,
) -> dict:
    stem = f"{treatment}_replica_{replica:02d}"
    full_path = out / f"{stem}_fullpool.pt"
    sp_path = out / f"{stem}_specialist.pt"
    common = {
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "teacher_sha256": EXPECTED_TEACHER_SHA,
        "teacher_schema": builder.ADV_SCHEMA,
        "semantic_completed_iteration": FINAL_ITERATION,
        "protocol_sha256": protocol_sha,
        "member": int(artifact_member),
        "member_seeds": {
            "canonical_fullpool_fit_seed": int(full_seed),
            "canonical_specialist_fit_seed": int(specialist_seed),
        },
        "collection_threads": 0,
        "fit_threads": 8,
        "production_status": "BUILT_UNVALIDATED_NOT_PROMOTED",
        "research_only_canonical_corpus": True,
        "canonical_treatment": treatment,
        "canonical_replica": int(replica),
    }
    builder.atomic_torch_save(
        {
            "schema": builder.FULLPOOL_SCHEMA,
            **common,
            "selected_steps": 500,
            "ordinary_train_samples": int(corpus_meta["ordinary"]),
            "extra_unique_strong_states": int(corpus_meta["novel"]),
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
            "specialist_unique_strong_states": int(corpus_meta["specialist"]),
            "specialist_unique_strong_strata": {},
            "base_fullpool_sha256": full_sha,
            "route_contract": "postflop made_category>=TRIPS AND FOLD legal -> specialist; otherwise fullpool",
            "model_state": {k: v.detach().cpu() for k, v in specialist.state_dict().items()},
        },
        sp_path,
    )
    return {
        "treatment": treatment,
        "replica": int(replica),
        "member": int(artifact_member),
        "fullpool_fit_seed": int(full_seed),
        "specialist_fit_seed": int(specialist_seed),
        "fullpool_path": str(full_path.resolve()),
        "fullpool_sha256": full_sha,
        "specialist_path": str(sp_path.resolve()),
        "specialist_sha256": sha256(sp_path),
    }


def train_treatment(
    checkpoint: Path,
    protocol_sha: str,
    prereg: dict,
    treatment: str,
    corpus: dict,
    corpus_meta: dict,
    out: Path,
    member_base: int,
) -> list[dict]:
    payload = torch.load(checkpoint.resolve(strict=True), map_location="cpu", weights_only=False)
    d3 = (payload.get("domains") or {}).get(builder.DOMAIN) or {}
    cfg = dict(payload.get("config") or {})
    if not d3 or not cfg:
        raise RuntimeError("frozen 10105 THREE_HANDED state/config missing")

    fit = prereg["fit_design"]
    full_seeds = list(fit["fullpool_fit_seeds"])
    sp_seeds = list(fit["specialist_fit_seeds"])
    torch.set_num_threads(int(fit["fit_threads"]))
    rows = []

    for idx in range(8):
        replica = idx + 1
        artifact_member = int(member_base) + replica
        label = treatment.upper()
        print(f"POST15315_CANONICAL_{label}_REPLICA_{replica:02d}_FIT_BEGIN", flush=True)
        _, _, fullpool, fullopt = builder.confirm.initial_policy_pair(d3, cfg)
        builder.train_semantic(
            fullpool,
            fullopt,
            corpus["fullpool"],
            cfg,
            steps=int(fit["fullpool_fit_steps"]),
            seed=int(full_seeds[idx]),
            label=f"POST15315_CANONICAL_{label}_R{replica:02d}_FULLPOOL",
        )
        fullpool.eval()

        specialist = builder.spec.clone_model(fullpool)
        specialist_opt = torch.optim.Adam(
            specialist.parameters(), lr=float(cfg["learning_rate"])
        )
        rng = random.Random(int(sp_seeds[idx]))
        builder.spec.train_range(
            specialist,
            specialist_opt,
            corpus["specialist"],
            cfg,
            0,
            int(fit["specialist_fit_steps"]),
            rng,
            "STRATIFIED",
        )
        specialist.eval()

        row = save_member(
            out,
            treatment=treatment,
            replica=replica,
            artifact_member=artifact_member,
            fullpool=fullpool,
            specialist=specialist,
            protocol_sha=protocol_sha,
            full_seed=int(full_seeds[idx]),
            specialist_seed=int(sp_seeds[idx]),
            corpus_meta=corpus_meta,
        )
        rows.append(row)
        print(f"POST15315_CANONICAL_{label}_REPLICA_{replica:02d}_FIT_PASS", flush=True)
        del fullpool, fullopt, specialist, specialist_opt
        gc.collect()
    return rows


def evaluate_rows(
    rows: list[dict],
    *,
    group: str,
    samples,
    semantic_rows: np.ndarray,
    trips: np.ndarray,
    protocol_sha: str,
    batch_size: int,
) -> np.ndarray:
    probs = np.zeros((8, len(samples), 10), dtype=np.float32)
    for idx, row in enumerate(rows):
        member = int(row["member"])
        print(f"POST15315_CANONICAL_{group}_EVAL_{idx + 1:02d}_BEGIN", flush=True)
        probs[idx] = k8stab.member_probabilities(
            samples,
            semantic_rows,
            trips,
            fullpool_path=Path(row["fullpool_path"]),
            specialist_path=Path(row["specialist_path"]),
            member=member,
            protocol_sha=protocol_sha,
            batch_size=int(batch_size),
        )
        print(f"POST15315_CANONICAL_{group}_EVAL_{idx + 1:02d}_PASS", flush=True)
    return probs


def safe_ratio(a: float, b: float) -> float:
    if b <= 0.0:
        raise RuntimeError("cannot form canonical-corpus ratio with nonpositive denominator")
    return float(a / b)


def summary_ratios(num: dict, den: dict) -> dict:
    return {
        "broad_worst_split_mean_tv": safe_ratio(
            float(num["broad_worst_split_mean_tv"]),
            float(den["broad_worst_split_mean_tv"]),
        ),
        "broad_worst_split_statewise_p95_tv": safe_ratio(
            float(num["broad_worst_split_statewise_p95_tv"]),
            float(den["broad_worst_split_statewise_p95_tv"]),
        ),
        "leave_one_out_worst_mean_tv": safe_ratio(
            float(num["leave_one_out_worst_mean_tv"]),
            float(den["leave_one_out_worst_mean_tv"]),
        ),
        "trips_plus_fold_worst_split_mean_probability_delta": safe_ratio(
            float(num["trips_plus_fold_worst_split_mean_probability_delta"]),
            float(den["trips_plus_fold_worst_split_mean_probability_delta"]),
        ),
        "trips_plus_fold_worst_split_statewise_p95_probability_delta": safe_ratio(
            float(num["trips_plus_fold_worst_split_statewise_p95_probability_delta"]),
            float(den["trips_plus_fold_worst_split_statewise_p95_probability_delta"]),
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--canonical-prereg", type=Path, required=True)
    ap.add_argument("--factorial-report", type=Path, required=True)
    ap.add_argument("--control-report", type=Path, required=True)
    ap.add_argument("--k8-dir", type=Path, required=True)
    ap.add_argument("--extension-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    teacher_path = args.teacher.resolve(strict=True)
    solver_path = args.solver.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    prereg_path = args.canonical_prereg.resolve(strict=True)
    factorial_path = args.factorial_report.resolve(strict=True)
    control_path = args.control_report.resolve(strict=True)
    k8_dir = args.k8_dir.resolve(strict=True)
    extension_dir = args.extension_dir.resolve(strict=True)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("source checkpoint SHA mismatch")
    if sha256(teacher_path) != EXPECTED_TEACHER_SHA:
        raise RuntimeError("teacher SHA mismatch")
    if git_blob_sha1(protocol_path) != EXPECTED_PROTOCOL_BLOB_SHA:
        raise RuntimeError("original post15315 protocol Git blob SHA mismatch")
    protocol_sha = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    prereg = validate_prereg(prereg_path)
    factorial = validate_factorial(factorial_path, prereg)
    control_meta, control_rows = validate_control(control_path)

    print("POST15315_CANONICAL_CORPUS_BUILD_BEGIN", flush=True)
    fixed, expanded, corpus_meta = load_canonical_corpora(
        protocol,
        protocol_sha,
        prereg,
        k8_dir,
        extension_dir,
    )
    print(
        "POST15315_CANONICAL_CORPUS_BUILD_PASS "
        + json.dumps(
            {
                "fixed_fullpool": corpus_meta["fixed_size"]["fullpool"],
                "fixed_specialist": corpus_meta["fixed_size"]["specialist"],
                "expanded_fullpool": corpus_meta["expanded"]["fullpool"],
                "expanded_specialist": corpus_meta["expanded"]["specialist"],
                "ordinary_union": corpus_meta["unique_union"]["ordinary"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    fixed_rows = train_treatment(
        checkpoint,
        protocol_sha,
        prereg,
        "fixed",
        fixed,
        corpus_meta["fixed_size"],
        out,
        member_base=1100,
    )
    expanded_rows = train_treatment(
        checkpoint,
        protocol_sha,
        prereg,
        "expanded",
        expanded,
        corpus_meta["expanded"],
        out,
        member_base=1200,
    )

    teacher_models, _teacher_payload, teacher_sha = builder.load_teacher(
        teacher_path, EXPECTED_TEACHER_SHA
    )
    if teacher_sha != EXPECTED_TEACHER_SHA:
        raise RuntimeError("teacher validation drift")
    solver = SolverLibrary(solver_path)
    bank = prereg["fresh_confirmation_bank"]
    print("POST15315_CANONICAL_CONFIRMATION_BANK_BEGIN", flush=True)
    train, hold, collection_raw = collect.collect_fresh_split(
        solver,
        teacher_models,
        int(bank["forced_three_handed_episodes"]),
        master_seed=int(bank["seed"]),
        sample_iteration=FINAL_ITERATION,
        progress_prefix="POST15315_CANONICAL_CONFIRMATION_BANK",
    )
    samples = list(train) + list(hold)
    legal, s72, trips, high = k8stab.surface_masks(samples)
    coverage = k8stab.validate_surface_coverage(protocol, s72, trips, high)
    semantic_rows = np.asarray(
        [semantic_vector_from_obs(s.observation) for s in samples], dtype=np.float32
    )
    print(
        "POST15315_CANONICAL_CONFIRMATION_BANK_PASS "
        + json.dumps(coverage, sort_keys=True),
        flush=True,
    )

    control_probs = evaluate_rows(
        control_rows,
        group="CONTROL",
        samples=samples,
        semantic_rows=semantic_rows,
        trips=trips,
        protocol_sha=protocol_sha,
        batch_size=int(args.batch_size),
    )
    fixed_probs = evaluate_rows(
        fixed_rows,
        group="FIXED",
        samples=samples,
        semantic_rows=semantic_rows,
        trips=trips,
        protocol_sha=protocol_sha,
        batch_size=int(args.batch_size),
    )
    expanded_probs = evaluate_rows(
        expanded_rows,
        group="EXPANDED",
        samples=samples,
        semantic_rows=semantic_rows,
        trips=trips,
        protocol_sha=protocol_sha,
        batch_size=int(args.batch_size),
    )

    control_report = stability.evaluate(control_probs, legal, s72, trips, high, protocol)
    fixed_report = stability.evaluate(fixed_probs, legal, s72, trips, high, protocol)
    expanded_report = stability.evaluate(expanded_probs, legal, s72, trips, high, protocol)

    csum = dict(control_report["stability_summary"])
    fsum = dict(fixed_report["stability_summary"])
    esum = dict(expanded_report["stability_summary"])
    control_pass = bool(control_report["rebuild_stability_pass"])
    fixed_pass = bool(fixed_report["rebuild_stability_pass"])
    expanded_pass = bool(expanded_report["rebuild_stability_pass"])

    if control_pass:
        classification = "BANK_SENSITIVITY_STOP"
        next_branch = (
            "Do not choose a canonical treatment. Preregister a second independent 100k "
            "confirmation bank because the reused V2 control unexpectedly passed."
        )
    elif fixed_pass:
        classification = "STANDARDIZATION_SUFFICIENT"
        next_branch = (
            "Design a new candidate family around the frozen deterministic fixed-size "
            "canonical corpus, then preregister fresh stability/quality/safety before strength."
        )
    elif expanded_pass:
        classification = "EXPANSION_REQUIRED"
        next_branch = (
            "Design a new candidate family around the frozen expanded canonical corpus, "
            "then preregister fresh stability/quality/safety before strength."
        )
    else:
        classification = "CANONICAL_CORPUS_INSUFFICIENT"
        next_branch = (
            "Do not build a candidate yet. Combine canonical corpus construction with an "
            "explicit fit/optimization stability intervention and test research-only."
        )

    predictions = out / "post15315_canonical_corpus_predictions.npz"
    np.savez_compressed(
        predictions,
        control_probs=control_probs,
        fixed_probs=fixed_probs,
        expanded_probs=expanded_probs,
        legal=legal,
        surface_72o=s72,
        surface_trips_fold=trips,
        surface_high_card_no_draw=high,
    )

    result = {
        "schema": RESULT_SCHEMA,
        "status": "PASS",
        "purpose": "canonical_corpus_standardization_and_volume_research_only",
        "closed_k12_failure_is_immutable": True,
        "promotion_decision_performed": False,
        "external_strength_benchmark_performed": False,
        "member_selection_performed": False,
        "member_weighting_performed": False,
        "k_greater_than_12_performed": False,
        "new_teacher_collection_performed": False,
        "original_protocol_sha256": protocol_sha,
        "original_protocol_git_blob_sha1": EXPECTED_PROTOCOL_BLOB_SHA,
        "canonical_preregistration": str(prereg_path),
        "canonical_preregistration_sha256": sha256(prereg_path),
        "factorial_trigger_evidence": factorial,
        "control_evidence": control_meta,
        "canonical_corpus": corpus_meta,
        "fixed_replicas": fixed_rows,
        "expanded_replicas": expanded_rows,
        "confirmation_bank": {
            "seed": int(bank["seed"]),
            "episodes": int(bank["forced_three_handed_episodes"]),
            "state_count": len(samples),
            "surface_coverage": coverage,
            "collection": collect.canonical_collection_stats(collection_raw),
        },
        "control8": {
            "pass": control_pass,
            "summary": csum,
            "criteria": dict(control_report["precommitted_criteria"]),
        },
        "fixed_size_canonical8": {
            "pass": fixed_pass,
            "summary": fsum,
            "criteria": dict(fixed_report["precommitted_criteria"]),
        },
        "expanded_canonical8": {
            "pass": expanded_pass,
            "summary": esum,
            "criteria": dict(expanded_report["precommitted_criteria"]),
        },
        "ratios": {
            "fixed_over_control": summary_ratios(fsum, csum),
            "expanded_over_control": summary_ratios(esum, csum),
            "expanded_over_fixed": summary_ratios(esum, fsum),
        },
        "classification": classification,
        "next_branch": next_branch,
        "prediction_archive": str(predictions.resolve()),
        "prediction_archive_sha256": sha256(predictions),
        "interpretation": (
            "Research-only matched-seed confirmation. Control, fixed-size canonical, "
            "and expanded canonical groups are evaluated on identical fresh states with "
            "the unchanged original K=8 stability evaluator and thresholds. A treatment "
            "PASS is evidence for future candidate design only; it is not promotion."
        ),
    }
    report_path = out / "post15315_canonical_corpus_pilot_report.json"
    atomic_json(result, report_path)

    print("POST15315_CANONICAL_CORPUS_PILOT_PASS", flush=True)
    print("classification=" + classification, flush=True)
    print(
        "passes="
        + json.dumps(
            {
                "control8": control_pass,
                "fixed_size_canonical8": fixed_pass,
                "expanded_canonical8": expanded_pass,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("control8_summary=" + json.dumps(csum, sort_keys=True), flush=True)
    print("fixed8_summary=" + json.dumps(fsum, sort_keys=True), flush=True)
    print("expanded8_summary=" + json.dumps(esum, sort_keys=True), flush=True)
    print("ratios=" + json.dumps(result["ratios"], sort_keys=True), flush=True)
    print("surface_coverage=" + json.dumps(coverage, sort_keys=True), flush=True)
    print(f"report={report_path}", flush=True)
    print("STOP: research confirmation only; no strength or promotion.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
