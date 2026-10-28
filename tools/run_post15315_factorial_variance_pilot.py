#!/usr/bin/env python3
from __future__ import annotations

"""Crossed corpus x fit-seed variance attribution after shared-corpus MIXED result.

Research-only. Four deterministic original-size corpus anchors are crossed with
four matched fit-seed pairs (16 cells). The experiment compares within-corpus
fit-only variation against across-corpus variation at matched fit seeds on one
fresh common bank, and reports a balanced two-way probability-space variance
decomposition. It cannot promote or reopen the failed K=12 candidate.

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
from itertools import combinations
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import build_post15315_rebuild_member as builder
import post15315_collection as collect
import post15315_member_cache as member_cache
import run_post15315_rebuild_stability_k8 as k8stab
from audit_3h_average_policy_semantic_continuation_10105 import semantic_vector_from_obs
from spincore.solver import SolverLibrary

RESULT_SCHEMA = "SPINCORE_POST15315_FACTORIAL_VARIANCE_PILOT_RESULT_V1"
PREREG_SCHEMA = "SPINCORE_POST15315_FACTORIAL_VARIANCE_PILOT_V1"
PRIOR_SCHEMA = "SPINCORE_POST15315_SHARED_CORPUS_VARIANCE_PILOT_RESULT_V2"
EXPECTED_PROTOCOL_BLOB_SHA = "5fd7667926c6112310fe5eca61afff396ad218de"
EXPECTED_TEACHER_SHA = "ac49b77c8ec3204e07f6d44d3718c68880e60675c5521f538438a3fa3ceae746"
EXPECTED_CHECKPOINT_SHA = "f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
FINAL_ITERATION = 15315
ACTION_NAMES = (
    "FOLD", "CHECK_CALL", "MIN_RAISE", "POT_33", "POT_40",
    "POT_50", "POT_66", "POT_75", "POT_100", "ALL_IN",
)


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


def validate_prereg(path: Path) -> dict:
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("schema") != PREREG_SCHEMA:
        raise RuntimeError("wrong factorial preregistration schema")
    if p.get("status") != "PREREGISTERED_RESEARCH_ONLY_NOT_PROMOTIONAL":
        raise RuntimeError("factorial preregistration status drift")
    anchors = list((p.get("corpus_factor") or {}).get("anchor_members") or [])
    if anchors != [1, 4, 7, 10]:
        raise RuntimeError("factorial corpus-anchor drift")
    design = dict(p.get("fit_factor") or {})
    if list(design.get("matched_fit_seed_ids") or []) != [1, 2, 3, 4]:
        raise RuntimeError("factorial fit-seed-id drift")
    if list(design.get("fullpool_fit_seeds") or []) != list(range(153162001, 153162005)):
        raise RuntimeError("factorial fullpool fit-seed drift")
    if list(design.get("specialist_fit_seeds") or []) != list(range(153163001, 153163005)):
        raise RuntimeError("factorial specialist fit-seed drift")
    if int(design.get("fullpool_fit_steps", -1)) != 500:
        raise RuntimeError("factorial fullpool-step drift")
    if int(design.get("specialist_fit_steps", -1)) != 25:
        raise RuntimeError("factorial specialist-step drift")
    bank = dict(p.get("fresh_research_bank") or {})
    if int(bank.get("seed", -1)) != 153159502:
        raise RuntimeError("factorial bank-seed drift")
    if int(bank.get("forced_three_handed_episodes", -1)) != 40000:
        raise RuntimeError("factorial bank-budget drift")
    return p


def validate_prior_mixed(path: Path) -> dict:
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("schema") != PRIOR_SCHEMA or p.get("status") != "PASS":
        raise RuntimeError("prior shared-corpus pilot is not canonical PASS")
    if p.get("attribution_result") != "MIXED_VARIANCE":
        raise RuntimeError("factorial pilot requires prior MIXED_VARIANCE result")
    if not bool(p.get("closed_k12_failure_is_immutable")):
        raise RuntimeError("prior pilot does not preserve closed K=12 failure")
    if bool(p.get("promotion_decision_performed")) or bool(p.get("external_strength_benchmark_performed")):
        raise RuntimeError("prior research pilot unexpectedly opened promotion/strength")
    bank = dict(p.get("research_bank") or {})
    if int(bank.get("seed", -1)) != 153159501 or int(bank.get("episodes", -1)) != 40000:
        raise RuntimeError("prior shared-corpus evidence identity drift")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "attribution_result": p["attribution_result"],
        "ratios": dict(p.get("attribution_ratios_shared8_over_original12") or {}),
    }


def cache_dir_for_anchor(anchor: int, k8_dir: Path, extension_dir: Path) -> Path:
    return k8_dir if int(anchor) <= 8 else extension_dir


def load_anchor_corpus(
    protocol: dict,
    protocol_sha: str,
    anchor: int,
    k8_dir: Path,
    extension_dir: Path,
) -> tuple[list, list, dict]:
    base = cache_dir_for_anchor(anchor, k8_dir, extension_dir)
    cp = member_cache.cache_path(base, int(anchor))
    seeds = builder.member_spec(protocol, int(anchor))
    payload = member_cache.load_member_cache(
        cp,
        expected_member=int(anchor),
        expected_teacher_sha256=EXPECTED_TEACHER_SHA,
        expected_protocol_sha256=protocol_sha,
        expected_member_seeds=seeds,
        expected_collection_threads=2,
    )

    ordinary = list(payload["ordinary"]["train"])
    base_strong_keys = {
        sample_key(sample) for sample in ordinary if builder.strong(sample)
    }
    seen = set()
    novel = []
    for sample in list(payload["augmentation"]["train"]) + list(payload["augmentation"]["holdout"]):
        key = sample_key(sample)
        if key in base_strong_keys or key in seen:
            continue
        seen.add(key)
        novel.append(sample)

    full_train = list(ordinary) + novel
    specialist_unique = builder.spec.unique_strong(
        list(payload["specialist"]["train"]) + list(payload["specialist"]["holdout"])
    )
    if len(ordinary) < 20000 or len(novel) < 700 or len(specialist_unique) < 1200:
        raise RuntimeError(f"anchor {anchor}: frozen corpus support below construction minima")

    meta = {
        "anchor_member": int(anchor),
        "cache_path": str(cp.resolve()),
        "cache_sha256": sha256(cp),
        "ordinary_train_samples": len(ordinary),
        "novel_strong_states": len(novel),
        "fullpool_train_samples": len(full_train),
        "specialist_unique_strong_states": len(specialist_unique),
        "ordinary_digest": collect.sample_digest(ordinary),
        "novel_digest": collect.sample_digest(novel),
        "specialist_digest": collect.sample_digest(specialist_unique),
    }
    return full_train, specialist_unique, meta


def save_cell(
    out: Path,
    *,
    cell_member: int,
    anchor: int,
    fit_id: int,
    fullpool,
    specialist,
    protocol_sha: str,
    full_seed: int,
    specialist_seed: int,
    corpus_meta: dict,
) -> dict:
    stem = f"anchor_{anchor:02d}_fit_{fit_id:02d}"
    full_path = out / f"{stem}_fullpool.pt"
    sp_path = out / f"{stem}_specialist.pt"
    common = {
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "teacher_sha256": EXPECTED_TEACHER_SHA,
        "teacher_schema": builder.ADV_SCHEMA,
        "semantic_completed_iteration": FINAL_ITERATION,
        "protocol_sha256": protocol_sha,
        "member": int(cell_member),
        "member_seeds": {
            "factorial_fullpool_fit_seed": int(full_seed),
            "factorial_specialist_fit_seed": int(specialist_seed),
        },
        "collection_threads": 0,
        "fit_threads": 8,
        "production_status": "BUILT_UNVALIDATED_NOT_PROMOTED",
        "research_only_factorial_cell": True,
        "factorial_anchor_member": int(anchor),
        "factorial_fit_seed_id": int(fit_id),
    }
    builder.atomic_torch_save(
        {
            "schema": builder.FULLPOOL_SCHEMA,
            **common,
            "selected_steps": 500,
            "ordinary_train_samples": int(corpus_meta["ordinary_train_samples"]),
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
        "cell_member": int(cell_member),
        "anchor_member": int(anchor),
        "fit_seed_id": int(fit_id),
        "fullpool_fit_seed": int(full_seed),
        "specialist_fit_seed": int(specialist_seed),
        "fullpool_path": str(full_path.resolve()),
        "fullpool_sha256": full_sha,
        "specialist_path": str(sp_path.resolve()),
        "specialist_sha256": sha256(sp_path),
    }


def train_cells(
    checkpoint: Path,
    protocol_sha: str,
    prereg: dict,
    protocol: dict,
    k8_dir: Path,
    extension_dir: Path,
    out: Path,
) -> tuple[list[dict], list[dict]]:
    payload = torch.load(checkpoint.resolve(strict=True), map_location="cpu", weights_only=False)
    d3 = (payload.get("domains") or {}).get(builder.DOMAIN) or {}
    cfg = dict(payload.get("config") or {})
    if not d3 or not cfg:
        raise RuntimeError("frozen 10105 THREE_HANDED state/config missing")

    anchors = list(prereg["corpus_factor"]["anchor_members"])
    fit_ids = list(prereg["fit_factor"]["matched_fit_seed_ids"])
    full_seeds = list(prereg["fit_factor"]["fullpool_fit_seeds"])
    sp_seeds = list(prereg["fit_factor"]["specialist_fit_seeds"])
    torch.set_num_threads(int(prereg["fit_factor"]["fit_threads"]))

    rows = []
    corpus_rows = []
    for anchor in anchors:
        full_train, specialist_unique, corpus_meta = load_anchor_corpus(
            protocol, protocol_sha, int(anchor), k8_dir, extension_dir
        )
        corpus_rows.append(corpus_meta)
        print(
            f"POST15315_FACTORIAL_ANCHOR_{anchor:02d}_CORPUS_PASS "
            + json.dumps(
                {
                    "fullpool": corpus_meta["fullpool_train_samples"],
                    "ordinary": corpus_meta["ordinary_train_samples"],
                    "specialist": corpus_meta["specialist_unique_strong_states"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for j, fit_id in enumerate(fit_ids):
            cell_member = int(anchor) * 100 + int(fit_id)
            print(f"POST15315_FACTORIAL_A{anchor:02d}_F{fit_id:02d}_FIT_BEGIN", flush=True)
            _, _, fullpool, fullopt = builder.confirm.initial_policy_pair(d3, cfg)
            builder.train_semantic(
                fullpool,
                fullopt,
                full_train,
                cfg,
                steps=int(prereg["fit_factor"]["fullpool_fit_steps"]),
                seed=int(full_seeds[j]),
                label=f"POST15315_FACTORIAL_A{anchor:02d}_F{fit_id:02d}_FULLPOOL",
            )
            fullpool.eval()
            specialist = builder.spec.clone_model(fullpool)
            specialist_opt = torch.optim.Adam(
                specialist.parameters(), lr=float(cfg["learning_rate"])
            )
            rng = random.Random(int(sp_seeds[j]))
            builder.spec.train_range(
                specialist,
                specialist_opt,
                specialist_unique,
                cfg,
                0,
                int(prereg["fit_factor"]["specialist_fit_steps"]),
                rng,
                "STRATIFIED",
            )
            specialist.eval()
            rows.append(
                save_cell(
                    out,
                    cell_member=cell_member,
                    anchor=int(anchor),
                    fit_id=int(fit_id),
                    fullpool=fullpool,
                    specialist=specialist,
                    protocol_sha=protocol_sha,
                    full_seed=int(full_seeds[j]),
                    specialist_seed=int(sp_seeds[j]),
                    corpus_meta=corpus_meta,
                )
            )
            print(f"POST15315_FACTORIAL_A{anchor:02d}_F{fit_id:02d}_FIT_PASS", flush=True)
            del fullpool, fullopt, specialist, specialist_opt
            gc.collect()
        del full_train, specialist_unique
        gc.collect()
    return rows, corpus_rows


def pairwise_tv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(a - b).sum(axis=-1)


def aggregate_pairwise(values: list[np.ndarray]) -> dict:
    if not values:
        raise RuntimeError("empty pairwise metric collection")
    x = np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1) for v in values])
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "p95": float(np.quantile(x, 0.95)),
        "median": float(np.median(x)),
        "max": float(x.max()),
    }


def factor_tv_metrics(probs: np.ndarray) -> tuple[dict, dict]:
    c_count, s_count = probs.shape[:2]
    fit_vals = []
    fit_by_anchor = []
    for c in range(c_count):
        vals = [pairwise_tv(probs[c, a], probs[c, b]) for a, b in combinations(range(s_count), 2)]
        fit_vals.extend(vals)
        fit_by_anchor.append(aggregate_pairwise(vals))

    corpus_vals = []
    corpus_by_fit = []
    for s in range(s_count):
        vals = [pairwise_tv(probs[a, s], probs[b, s]) for a, b in combinations(range(c_count), 2)]
        corpus_vals.extend(vals)
        corpus_by_fit.append(aggregate_pairwise(vals))

    return (
        {"aggregate": aggregate_pairwise(fit_vals), "by_anchor": fit_by_anchor},
        {"aggregate": aggregate_pairwise(corpus_vals), "by_fit_seed": corpus_by_fit},
    )


def action_surface_pairwise(
    probs: np.ndarray,
    mask: np.ndarray,
    action: int,
) -> tuple[dict, dict]:
    if int(mask.sum()) <= 0:
        return ({"aggregate": None}, {"aggregate": None})
    x = probs[:, :, mask, int(action)]
    c_count, s_count = x.shape[:2]
    fit_vals = []
    for c in range(c_count):
        for a, b in combinations(range(s_count), 2):
            fit_vals.append(np.abs(x[c, a] - x[c, b]))
    corpus_vals = []
    for s in range(s_count):
        for a, b in combinations(range(c_count), 2):
            corpus_vals.append(np.abs(x[a, s] - x[b, s]))
    return (
        {"aggregate": aggregate_pairwise(fit_vals)},
        {"aggregate": aggregate_pairwise(corpus_vals)},
    )


def two_way_decomposition(probs: np.ndarray) -> dict:
    x = probs.astype(np.float64, copy=False)
    c_count, s_count = x.shape[:2]
    grand = x.mean(axis=(0, 1), keepdims=True)
    corpus_mean = x.mean(axis=1, keepdims=True)
    fit_mean = x.mean(axis=0, keepdims=True)
    residual = x - corpus_mean - fit_mean + grand

    ss_total = float(np.square(x - grand).sum())
    ss_corpus = float(s_count * np.square(corpus_mean - grand).sum())
    ss_fit = float(c_count * np.square(fit_mean - grand).sum())
    ss_interaction = float(np.square(residual).sum())
    explained = ss_corpus + ss_fit + ss_interaction
    if ss_total <= 0.0:
        raise RuntimeError("nonpositive factorial total variance")
    closure_error = abs(explained - ss_total) / ss_total
    if closure_error > 1e-10:
        raise RuntimeError(f"factorial SS decomposition closure drift: {closure_error}")
    return {
        "ss_total": ss_total,
        "ss_corpus": ss_corpus,
        "ss_fit_seed": ss_fit,
        "ss_interaction": ss_interaction,
        "share_corpus": ss_corpus / ss_total,
        "share_fit_seed": ss_fit / ss_total,
        "share_interaction": ss_interaction / ss_total,
        "relative_closure_error": closure_error,
    }


def per_action_factor_metrics(probs: np.ndarray) -> list[dict]:
    rows = []
    for action, name in enumerate(ACTION_NAMES):
        fp, cp = factor_tv_metrics(probs[..., action:action+1])
        rows.append(
            {
                "action": name,
                "fit_only_mean_abs_delta": fp["aggregate"]["mean"] * 2.0,
                "fit_only_p95_abs_delta": fp["aggregate"]["p95"] * 2.0,
                "corpus_conditioned_mean_abs_delta": cp["aggregate"]["mean"] * 2.0,
                "corpus_conditioned_p95_abs_delta": cp["aggregate"]["p95"] * 2.0,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--factorial-prereg", type=Path, required=True)
    ap.add_argument("--prior-pilot-report", type=Path, required=True)
    ap.add_argument("--k8-dir", type=Path, required=True)
    ap.add_argument("--extension-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    args = ap.parse_args()

    checkpoint = args.checkpoint.resolve(strict=True)
    teacher_path = args.teacher.resolve(strict=True)
    solver_path = args.solver.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    prereg_path = args.factorial_prereg.resolve(strict=True)
    prior_path = args.prior_pilot_report.resolve(strict=True)
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
    prior = validate_prior_mixed(prior_path)

    rows, corpus_rows = train_cells(
        checkpoint,
        protocol_sha,
        prereg,
        protocol,
        k8_dir,
        extension_dir,
        out,
    )

    anchors = list(prereg["corpus_factor"]["anchor_members"])
    fit_ids = list(prereg["fit_factor"]["matched_fit_seed_ids"])
    if len(rows) != len(anchors) * len(fit_ids):
        raise RuntimeError("factorial cell count mismatch")

    teacher_models, _teacher_payload, teacher_sha = builder.load_teacher(
        teacher_path, EXPECTED_TEACHER_SHA
    )
    if teacher_sha != EXPECTED_TEACHER_SHA:
        raise RuntimeError("teacher validation drift")
    solver = SolverLibrary(solver_path)
    bank = prereg["fresh_research_bank"]
    print("POST15315_FACTORIAL_RESEARCH_BANK_BEGIN", flush=True)
    train, hold, collection_raw = collect.collect_fresh_split(
        solver,
        teacher_models,
        int(bank["forced_three_handed_episodes"]),
        master_seed=int(bank["seed"]),
        sample_iteration=FINAL_ITERATION,
        progress_prefix="POST15315_FACTORIAL_RESEARCH_BANK",
    )
    samples = list(train) + list(hold)
    if not samples:
        raise RuntimeError("empty factorial research bank")
    legal, s72, trips, high = k8stab.surface_masks(samples)
    semantic_rows = np.asarray(
        [semantic_vector_from_obs(s.observation) for s in samples], dtype=np.float32
    )
    surfaces = {
        "preflop_72o": int(s72.sum()),
        "trips_plus_fold": int(trips.sum()),
        "high_card_no_draw": int(high.sum()),
    }

    probs = np.zeros((len(anchors), len(fit_ids), len(samples), 10), dtype=np.float32)
    row_map = {(int(r["anchor_member"]), int(r["fit_seed_id"])): r for r in rows}
    for ci, anchor in enumerate(anchors):
        for si, fit_id in enumerate(fit_ids):
            row = row_map[(int(anchor), int(fit_id))]
            member = int(row["cell_member"])
            print(f"POST15315_FACTORIAL_A{anchor:02d}_F{fit_id:02d}_EVAL_BEGIN", flush=True)
            probs[ci, si] = k8stab.member_probabilities(
                samples,
                semantic_rows,
                trips,
                fullpool_path=Path(row["fullpool_path"]),
                specialist_path=Path(row["specialist_path"]),
                member=member,
                protocol_sha=protocol_sha,
                batch_size=int(args.batch_size),
            )
            print(f"POST15315_FACTORIAL_A{anchor:02d}_F{fit_id:02d}_EVAL_PASS", flush=True)

    fit_tv, corpus_tv = factor_tv_metrics(probs)
    fit_mean = float(fit_tv["aggregate"]["mean"])
    fit_p95 = float(fit_tv["aggregate"]["p95"])
    corpus_mean = float(corpus_tv["aggregate"]["mean"])
    corpus_p95 = float(corpus_tv["aggregate"]["p95"])
    if fit_mean <= 0.0 or fit_p95 <= 0.0:
        raise RuntimeError("nonpositive fit-only denominator")
    ratio_mean = corpus_mean / fit_mean
    ratio_p95 = corpus_p95 / fit_p95

    if ratio_mean >= 1.25 and ratio_p95 >= 1.25:
        attribution = "CORPUS_CONDITIONED_LARGER"
        next_branch = (
            "Investigate standardized/expanded teacher-target corpus construction before "
            "any new candidate; confirm on a separately preregistered bank."
        )
    elif ratio_mean <= 0.80 and ratio_p95 <= 0.80:
        attribution = "FIT_SEED_LARGER"
        next_branch = (
            "Investigate optimization/model/loss stability before any new candidate; "
            "do not spend more teacher collection first."
        )
    else:
        attribution = "MIXED_INTERACTION"
        next_branch = (
            "Design a targeted intervention addressing both fit stability and corpus "
            "coverage, then validate it as a new candidate under a new preregistration."
        )

    decomposition = two_way_decomposition(probs)
    surface_metrics = {}
    for name, mask, action in (
        ("preflop_72o_allin", s72, 9),
        ("trips_plus_fold", trips, 0),
        ("high_card_no_draw_allin", high, 9),
    ):
        fit_s, corpus_s = action_surface_pairwise(probs, mask, action)
        surface_metrics[name] = {
            "state_count": int(mask.sum()),
            "fit_only": fit_s,
            "corpus_conditioned": corpus_s,
        }

    pred_path = out / "post15315_factorial_variance_predictions.npz"
    np.savez_compressed(
        pred_path,
        probs=probs,
        legal=legal,
        surface_72o=s72,
        surface_trips_fold=trips,
        surface_high_card_no_draw=high,
        anchors=np.asarray(anchors, dtype=np.int32),
        fit_seed_ids=np.asarray(fit_ids, dtype=np.int32),
    )

    result = {
        "schema": RESULT_SCHEMA,
        "status": "PASS",
        "purpose": "crossed_corpus_fit_variance_attribution_only",
        "closed_k12_failure_is_immutable": True,
        "promotion_decision_performed": False,
        "external_strength_benchmark_performed": False,
        "member_selection_performed": False,
        "member_weighting_performed": False,
        "k_greater_than_12_performed": False,
        "original_protocol_sha256": protocol_sha,
        "original_protocol_git_blob_sha1": EXPECTED_PROTOCOL_BLOB_SHA,
        "factorial_preregistration": str(prereg_path),
        "factorial_preregistration_sha256": sha256(prereg_path),
        "prior_mixed_evidence": prior,
        "corpus_anchors": corpus_rows,
        "factorial_cells": rows,
        "research_bank": {
            "seed": int(bank["seed"]),
            "episodes": int(bank["forced_three_handed_episodes"]),
            "state_count": len(samples),
            "surface_counts": surfaces,
            "collection": collect.canonical_collection_stats(collection_raw),
        },
        "fit_only_tv": fit_tv,
        "corpus_conditioned_tv": corpus_tv,
        "corpus_over_fit_ratios": {
            "mean_tv": ratio_mean,
            "p95_tv": ratio_p95,
        },
        "two_way_probability_variance": decomposition,
        "surface_metrics": surface_metrics,
        "per_action_factor_metrics": per_action_factor_metrics(probs),
        "attribution_result": attribution,
        "next_branch": next_branch,
        "prediction_archive": str(pred_path.resolve()),
        "prediction_archive_sha256": sha256(pred_path),
        "interpretation": (
            "Balanced crossed research design: within-anchor comparisons vary fit seed "
            "while holding corpus fixed; across-anchor comparisons hold matched fit-seed "
            "id while varying original-size corpus. The two-way decomposition reports "
            "corpus, fit-seed and interaction shares in probability space. No cell is a "
            "deployable candidate and no strength evidence is opened."
        ),
    }
    report_path = out / "post15315_factorial_variance_pilot_report.json"
    atomic_json(result, report_path)

    print("POST15315_FACTORIAL_VARIANCE_PILOT_PASS", flush=True)
    print("attribution_result=" + attribution, flush=True)
    print("ratios=" + json.dumps(result["corpus_over_fit_ratios"], sort_keys=True), flush=True)
    print("fit_only=" + json.dumps(fit_tv["aggregate"], sort_keys=True), flush=True)
    print("corpus_conditioned=" + json.dumps(corpus_tv["aggregate"], sort_keys=True), flush=True)
    print("decomposition=" + json.dumps(decomposition, sort_keys=True), flush=True)
    print("surface_counts=" + json.dumps(surfaces, sort_keys=True), flush=True)
    print(f"report={report_path}", flush=True)
    print("STOP: factorial research attribution only; no strength or promotion.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
