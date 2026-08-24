from __future__ import annotations

"""Phase2B9: fit-only robust Advantage regression screen on frozen Phase2B6 memories.

No solver traversal or reservoir mutation is permitted. For each completed Phase2B6
training seed, this tool fits paired four-member Advantage ensembles from the exact
same final Advantage memory: canonical MSE versus one precommitted Huber/Smooth-L1
candidate with beta=0.02. It then compares cross-seed behavior stability on the
frozen H2/3H heldouts using the canonical uncertainty-damped regret-matching algebra.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Sequence

import numpy as np
import torch

import r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot as b6
import r7_5_arch_reset_v1plus_phase2b7_residual_localization as b7
from spincore.r7_5_action_cfr import NUM_ACTIONS, legal_mask
from spincore.r7_5_action_uncertainty import uncertainty_damped_policy_from_advantages
from spincore.r7_5_representation_v3 import H2_FINAL
from spincore.r7_5_representation_v3_checkpoint import load_representation_v3_checkpoint
from spincore.r7_5_representation_v3_fit import (
    audit_v3_advantage_model,
    ensemble_v3_advantage_nrmse,
)
from spincore.r7_5_representation_v3_phase2_eval import equal_group_stratified_bootstrap_mean_ci
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    ADVANTAGE_NRMSE_MAX,
    ADVANTAGE_STEPS,
    AUDIT_SIZE,
    BATCH_SIZE,
    ENSEMBLE_SIZE,
    EPSILON_CAP,
    EPSILON_SCALE,
    EVALUATION_SEEDS,
    LEARNING_RATE,
    MODEL_FINGERPRINTS,
    TORCH_THREADS,
    TRAINING_SEEDS,
    primary_reset_seed,
    side_member_seeds,
    validate_phase2_v3_contract,
)
from spincore_nn.models_v3_final import collate_v3_observations, make_h2_final_v3
from spincore_nn.training import train_step

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B9_ROBUST_ADVANTAGE_REGRESSION_V1"
SEED_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B9_SEED_V1"
PAIR_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B9_PAIRED_FIT_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
POLICY_COUNT = 1024
HUBER_BETA = 0.02
PHASE2B6_RESULT_SHA256 = "33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a"
PHASE2B8_RESULT_SHA256 = "1fd9144a488cea6de0a7500320d552abf994908b5200146d4baa4bd6f81c4d98"
PHASE2B6_EXECUTION_SHA = "4fa96434321c32efc734a55ae75982018ff2d091"
PHASE2B6_CHECKPOINT_SCHEMA = b6.CHECKPOINT_EXTRA_SCHEMA
FINAL_ITERATION = 3
FINAL_STAGE_INDEX = 12
FINAL_GLOBAL_ROOT = 768
MEMBER0_BATCH_XOR = 0x2B900B17
BOOTSTRAP_REPLICATES = 2000
ABS_IMPROVEMENT_MIN = 0.03
REL_IMPROVEMENT_MIN = 0.10
P95_MAX_DEGRADE = 0.02
REGION_MAX_DEGRADE = 0.01
SIDE_REPRO_MAX_ABS = 1e-7
MASK64 = (1 << 64) - 1


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    b6._atomic_json(payload, path)


def _atomic_torch_save(payload, path: Path) -> None:
    b6._atomic_torch_save(payload, path)


def _member_seeds(training_seed: int, member: int) -> tuple[int, int]:
    seed = int(training_seed)
    m = int(member)
    if m == 0:
        init_seed = primary_reset_seed(seed, FINAL_ITERATION)
        batch_seed = (seed ^ MEMBER0_BATCH_XOR ^ (FINAL_ITERATION * 0x85EBCA77)) & MASK64
        return int(init_seed), int(batch_seed)
    return tuple(map(int, side_member_seeds(seed, FINAL_ITERATION, m)))


def _batch(samples: Sequence, *, device: str = "cpu"):
    batch = collate_v3_observations(
        [sample.observation for sample in samples],
        [sample.legal for sample in samples],
        with_semantics=False,
        device=device,
    )
    target = torch.tensor([sample.target for sample in samples], dtype=torch.float32, device=device)
    weights = torch.tensor([sample.weight for sample in samples], dtype=torch.float32, device=device)
    return batch, target, weights


def _huber_train_step(model, optimizer, batch, target, weights, *, beta: float) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(batch)
    mask = batch["legal"].float()
    w = weights / weights.mean().clamp_min(1e-12)
    diff = out - target
    abs_diff = diff.abs()
    b = float(beta)
    element = torch.where(abs_diff < b, 0.5 * diff.square() / b, abs_diff - 0.5 * b)
    per = (element * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    loss = (per * w).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return float(loss.detach().cpu())


def _state_dict_max_abs(left: dict, right: dict) -> float:
    if set(left) != set(right):
        return math.inf
    maximum = 0.0
    for key in left:
        a = left[key].detach().cpu()
        b = right[key].detach().cpu()
        if a.shape != b.shape:
            return math.inf
        if a.numel():
            maximum = max(maximum, float((a - b).abs().max()))
    return maximum


def _load_phase2b6_checkpoint(repo_root: Path, b6_root: Path, training_seed: int):
    validate_phase2_v3_contract(
        repo_root,
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(training_seed),
    )
    path = b6_root / f"seed_{int(training_seed)}" / "resume_checkpoint.pt"
    if not path.is_file():
        raise RuntimeError(f"Phase2B9 missing Phase2B6 resume checkpoint: {path}")
    bundle, progress, _spec, extra = load_representation_v3_checkpoint(
        path,
        repo_root=repo_root,
        expected_domain=DOMAIN,
        expected_representation=REPRESENTATION,
        expected_seed=int(training_seed),
        expected_action_candidate=ACTION_CANDIDATE,
        expected_execution_sha=PHASE2B6_EXECUTION_SHA,
        expected_architecture_fingerprint_sha256=MODEL_FINGERPRINTS[REPRESENTATION],
        device="cpu",
    )
    if progress.phase != "phase2b6_resume":
        raise RuntimeError("Phase2B9 source checkpoint phase mismatch")
    if int(progress.iteration) != FINAL_ITERATION or int(progress.global_root) != FINAL_GLOBAL_ROOT:
        raise RuntimeError("Phase2B9 source checkpoint progress mismatch")
    if extra.get("schema") != PHASE2B6_CHECKPOINT_SCHEMA:
        raise RuntimeError("Phase2B9 source checkpoint extra schema mismatch")
    if int(extra.get("stage_index", -1)) != FINAL_STAGE_INDEX:
        raise RuntimeError("Phase2B9 source checkpoint stage mismatch")
    intervention = dict(extra.get("intervention") or {})
    if float(intervention.get("floor", -1.0)) != 0.25:
        raise RuntimeError("Phase2B9 source checkpoint Phase2B6 floor mismatch")
    states = list(extra.get("behavior_model_states") or [])
    if len(states) != ENSEMBLE_SIZE:
        raise RuntimeError("Phase2B9 source checkpoint requires four final behavior states")
    return path, bundle, states


def _valid_pair_artifact(path: Path, *, execution_sha: str, source_sha: str, training_seed: int, member: int):
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    expected = {
        "schema": PAIR_SCHEMA,
        "status": "PAIR_FIT_COMPLETE",
        "diagnostic_execution_sha": str(execution_sha),
        "source_checkpoint_sha256": str(source_sha),
        "training_seed": int(training_seed),
        "member": int(member),
        "huber_beta": HUBER_BETA,
        "steps": ADVANTAGE_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            return None
    init_seed, batch_seed = _member_seeds(int(training_seed), int(member))
    if int(payload.get("init_seed", -1)) != init_seed or int(payload.get("batch_seed", -1)) != batch_seed:
        return None
    if not isinstance(payload.get("mse_model_state"), dict) or not isinstance(payload.get("huber_model_state"), dict):
        return None
    return payload


def _fit_pair(memory_items: Sequence, *, training_seed: int, member: int):
    items = list(memory_items)
    if not items:
        raise RuntimeError("Phase2B9 cannot fit empty Advantage memory")
    init_seed, batch_seed = _member_seeds(int(training_seed), int(member))
    _cfg_mse, mse = make_h2_final_v3(device="cpu", seed=init_seed)
    _cfg_huber, huber = make_h2_final_v3(device="cpu", seed=init_seed)
    mse_opt = torch.optim.Adam(mse.parameters(), lr=LEARNING_RATE)
    huber_opt = torch.optim.Adam(huber.parameters(), lr=LEARNING_RATE)
    rng = random.Random(batch_seed)
    count = min(BATCH_SIZE, len(items))
    mse_losses = []
    huber_losses = []
    started = time.perf_counter()
    for _ in range(ADVANTAGE_STEPS):
        samples = rng.sample(items, count)
        batch, target, weights = _batch(samples)
        mse_losses.append(train_step(mse, mse_opt, batch, target, weights, "advantage"))
        huber_losses.append(_huber_train_step(huber, huber_opt, batch, target, weights, beta=HUBER_BETA))
    return mse, huber, {
        "init_seed": int(init_seed),
        "batch_seed": int(batch_seed),
        "steps": ADVANTAGE_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "huber_beta": HUBER_BETA,
        "mse_mean_loss": float(sum(mse_losses) / len(mse_losses)),
        "mse_final_loss": float(mse_losses[-1]),
        "huber_mean_loss": float(sum(huber_losses) / len(huber_losses)),
        "huber_final_loss": float(huber_losses[-1]),
        "seconds": float(time.perf_counter() - started),
        "paired_identical_batch_sequence": True,
    }


def _load_model_from_state(state: dict):
    _cfg, model = make_h2_final_v3(device="cpu", seed=0)
    model.load_state_dict(state)
    model.eval()
    return model


def _run_seed(args, training_seed: int) -> int:
    repo_root = Path(args.repo_root).resolve()
    b6_root = Path(args.phase2b6_root).resolve()
    out_root = Path(args.output_root).resolve()
    seed_root = out_root / f"seed_{int(training_seed)}"
    seed_root.mkdir(parents=True, exist_ok=True)
    result_path = seed_root / "seed_result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("schema") == SEED_SCHEMA and existing.get("status") == "SEED_COMPLETE" and existing.get("diagnostic_execution_sha") == str(args.execution_sha):
            print(f"[Phase2B9 seed resume] seed={training_seed} already complete", flush=True)
            return 0

    checkpoint, bundle, stored_states = _load_phase2b6_checkpoint(repo_root, b6_root, int(training_seed))
    checkpoint_sha = _sha256(checkpoint)
    memory = list(bundle.adv_mem.items)
    if len(memory) != int(bundle.adv_mem.capacity):
        raise RuntimeError("Phase2B9 expects saturated frozen Phase2B6 Advantage reservoir")

    models = {"MSE_PAIRED_CONTROL": [], "HUBER_BETA_002": []}
    fit_rows = []
    side_repro = []
    for member in range(ENSEMBLE_SIZE):
        artifact = seed_root / f"paired_member_{member}.pt"
        payload = _valid_pair_artifact(
            artifact,
            execution_sha=str(args.execution_sha),
            source_sha=checkpoint_sha,
            training_seed=int(training_seed),
            member=member,
        )
        if payload is None:
            print(f"[Phase2B9 fit] seed={training_seed} member={member} MSE+HUBER", flush=True)
            mse, huber, report = _fit_pair(memory, training_seed=int(training_seed), member=member)
            payload = {
                "schema": PAIR_SCHEMA,
                "status": "PAIR_FIT_COMPLETE",
                "diagnostic_execution_sha": str(args.execution_sha),
                "source_checkpoint_sha256": checkpoint_sha,
                "training_seed": int(training_seed),
                "member": int(member),
                "init_seed": int(report["init_seed"]),
                "batch_seed": int(report["batch_seed"]),
                "huber_beta": HUBER_BETA,
                "steps": ADVANTAGE_STEPS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "fit_report": report,
                "mse_model_state": mse.state_dict(),
                "huber_model_state": huber.state_dict(),
            }
            _atomic_torch_save(payload, artifact)
        else:
            print(f"[Phase2B9 fit resume] seed={training_seed} member={member}", flush=True)
        mse = _load_model_from_state(payload["mse_model_state"])
        huber = _load_model_from_state(payload["huber_model_state"])
        models["MSE_PAIRED_CONTROL"].append(mse)
        models["HUBER_BETA_002"].append(huber)
        row = {
            "member": member,
            "artifact": str(artifact),
            "artifact_sha256": _sha256(artifact),
            **dict(payload.get("fit_report") or {}),
        }
        fit_rows.append(row)
        if member > 0:
            max_abs = _state_dict_max_abs(payload["mse_model_state"], stored_states[member])
            passed = bool(max_abs <= SIDE_REPRO_MAX_ABS)
            side_repro.append({"member": member, "max_abs_parameter_diff": max_abs, "tolerance": SIDE_REPRO_MAX_ABS, "pass": passed})
            if not passed:
                raise RuntimeError(f"Phase2B9 canonical MSE side-member reproduction failed seed={training_seed} member={member}: {max_abs}")

    audit_seed = int(training_seed) ^ 0x2B9A0D17
    audits = {}
    for name, ensemble in models.items():
        members = []
        for index, model in enumerate(ensemble):
            value = audit_v3_advantage_model(
                model,
                memory,
                representation=REPRESENTATION,
                sample_size=AUDIT_SIZE,
                seed=audit_seed,
            )
            members.append({"member": index, "weighted_nrmse": float(value)})
        ensemble_value = ensemble_v3_advantage_nrmse(
            ensemble,
            memory,
            representation=REPRESENTATION,
            sample_size=AUDIT_SIZE,
            seed=audit_seed,
        )
        audits[name] = {
            "audit_seed": audit_seed,
            "audit_size": AUDIT_SIZE,
            "members": members,
            "ensemble_weighted_nrmse": float(ensemble_value),
            "gate_max": ADVANTAGE_NRMSE_MAX,
            "gate_pass": bool(float(ensemble_value) <= ADVANTAGE_NRMSE_MAX),
        }

    result = {
        "schema": SEED_SCHEMA,
        "status": "SEED_COMPLETE",
        "diagnostic_execution_sha": str(args.execution_sha),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_sha,
        "training_seed": int(training_seed),
        "advantage_memory": {
            "capacity": int(bundle.adv_mem.capacity),
            "retained": len(memory),
            "seen": int(bundle.adv_mem.seen),
        },
        "paired_fits": fit_rows,
        "canonical_side_member_reproduction": {
            "tolerance": SIDE_REPRO_MAX_ABS,
            "rows": side_repro,
            "pass": bool(len(side_repro) == 3 and all(row["pass"] for row in side_repro)),
        },
        "audits": audits,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    _atomic_json(result, result_path)
    print(json.dumps({
        "status": result["status"],
        "training_seed": int(training_seed),
        "mse_ensemble_nrmse": audits["MSE_PAIRED_CONTROL"]["ensemble_weighted_nrmse"],
        "huber_ensemble_nrmse": audits["HUBER_BETA_002"]["ensemble_weighted_nrmse"],
        "side_reproduction_pass": result["canonical_side_member_reproduction"]["pass"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


def _load_pair_models(output_root: Path, training_seed: int, loss_name: str):
    key = "mse_model_state" if loss_name == "MSE_PAIRED_CONTROL" else "huber_model_state"
    models = []
    for member in range(ENSEMBLE_SIZE):
        payload = torch.load(output_root / f"seed_{int(training_seed)}" / f"paired_member_{member}.pt", map_location="cpu", weights_only=False)
        models.append(_load_model_from_state(payload[key]))
    return models


def _behavior_probabilities(models, descriptors) -> tuple[list[list[float]], dict]:
    if not descriptors:
        return [], {"count": 0}
    legal_sets = [tuple(int(x) for x in item.legal_slots) for item in descriptors]
    masks = [legal_mask(legal) for legal in legal_sets]
    batch = collate_v3_observations(
        [item.observation_v3 for item in descriptors],
        masks,
        with_semantics=False,
        device="cpu",
    )
    raw = []
    for model in models:
        model.eval()
        with torch.no_grad():
            raw.append(model(batch).detach().cpu().numpy())
    policies = []
    eps = []
    disagreement = []
    cap_hits = 0
    for index, legal in enumerate(legal_sets):
        rows = [raw[member][index].tolist() for member in range(len(models))]
        policy, stats = uncertainty_damped_policy_from_advantages(
            rows,
            legal,
            action_count=NUM_ACTIONS,
            epsilon_scale=EPSILON_SCALE,
            epsilon_cap=EPSILON_CAP,
        )
        policies.append([float(x) for x in policy])
        eps.append(float(stats["epsilon"]))
        disagreement.append(float(stats["disagreement"]))
        cap_hits += int(bool(stats["cap_hit"]))
    return policies, {
        "count": len(policies),
        "mean_epsilon": float(sum(eps) / len(eps)),
        "max_epsilon": float(max(eps)),
        "mean_disagreement": float(sum(disagreement) / len(disagreement)),
        "cap_hit_fraction": float(cap_hits / len(policies)),
    }


def _tv_rows(left, right) -> list[float]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return [float(x) for x in (0.5 * np.abs(a - b).sum(axis=1)).tolist()]


def _metric(tv: Sequence[float], left, right) -> dict:
    arr = np.asarray(list(tv), dtype=np.float64)
    if not arr.size:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None, "dominant_action_mismatch_rate": None}
    l = np.asarray(left, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    mismatch = np.argmax(l, axis=1) != np.argmax(r, axis=1)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50, method="linear")),
        "p95": float(np.quantile(arr, 0.95, method="linear")),
        "max": float(arr.max()),
        "dominant_action_mismatch_rate": float(mismatch.mean()),
    }


def _region_indices(descriptors) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(descriptors):
        meta = b7._decode_observation(item.observation_v3)
        region = str(meta["region"])
        groups.setdefault(region, []).append(index)
    return groups


def _slice_rows(rows, indices):
    return [rows[i] for i in indices]


def _evaluate_loss_pair(models_by_seed: dict, descriptors_by_eval: dict) -> tuple[dict, dict, dict]:
    seed_a, seed_b = map(int, TRAINING_SEEDS)
    evaluations = {}
    tv_vectors = {}
    behavior_telemetry = {}
    for evaluation_seed in map(int, EVALUATION_SEEDS):
        desc = descriptors_by_eval[evaluation_seed]
        evaluations[str(evaluation_seed)] = {}
        tv_vectors[str(evaluation_seed)] = {}
        behavior_telemetry[str(evaluation_seed)] = {}
        for loss_name in ("MSE_PAIRED_CONTROL", "HUBER_BETA_002"):
            left, left_stats = _behavior_probabilities(models_by_seed[(seed_a, loss_name)], desc)
            right, right_stats = _behavior_probabilities(models_by_seed[(seed_b, loss_name)], desc)
            tv = _tv_rows(left, right)
            regions = {}
            for region, indices in _region_indices(desc).items():
                regions[region] = _metric(tv=[tv[i] for i in indices], left=_slice_rows(left, indices), right=_slice_rows(right, indices))
            evaluations[str(evaluation_seed)][loss_name] = {
                "overall": _metric(tv, left, right),
                "regions": regions,
            }
            tv_vectors[str(evaluation_seed)][loss_name] = tv
            behavior_telemetry[str(evaluation_seed)][loss_name] = {
                str(seed_a): left_stats,
                str(seed_b): right_stats,
            }
    return evaluations, tv_vectors, behavior_telemetry


def _pooled_region(evaluations: dict, loss_name: str, region_names: Sequence[str]) -> dict:
    weighted_sum = 0.0
    count = 0
    for evaluation_seed in map(str, EVALUATION_SEEDS):
        regions = evaluations[evaluation_seed][loss_name]["regions"]
        for name in region_names:
            row = regions.get(name)
            if not row or not row.get("count"):
                continue
            weighted_sum += float(row["mean"]) * int(row["count"])
            count += int(row["count"])
    return {"count": count, "mean": float(weighted_sum / count) if count else None}


def _validate_prerequisites(b6_result_path: Path, b8_result_path: Path) -> tuple[dict, dict]:
    if _sha256(b6_result_path) != PHASE2B6_RESULT_SHA256:
        raise RuntimeError("Phase2B9 exact Phase2B6 result SHA drift")
    if _sha256(b8_result_path) != PHASE2B8_RESULT_SHA256:
        raise RuntimeError("Phase2B9 exact Phase2B8 result SHA drift")
    r6 = json.loads(b6_result_path.read_text(encoding="utf-8"))
    r8 = json.loads(b8_result_path.read_text(encoding="utf-8"))
    if r6.get("status") != "PREFLOP_DAMPING_CAUSAL_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE":
        raise RuntimeError("Phase2B9 Phase2B6 status mismatch")
    if r8.get("status") != "LAGGED_ANCHOR_EFFECT_NOT_SUPPORTED":
        raise RuntimeError("Phase2B9 Phase2B8 status mismatch")
    if not bool((r8.get("decision") or {}).get("equivalence_before_divergence_pass")):
        raise RuntimeError("Phase2B9 requires clean Phase2B8 causal equivalence evidence")
    return r6, r8


def _evaluate_parent(args) -> dict:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    b6_root = Path(args.phase2b6_root).resolve()
    heldout_root = Path(args.heldout_root).resolve()
    b6_result_path = Path(args.phase2b6_result).resolve()
    b8_result_path = Path(args.phase2b8_result).resolve()
    r6, r8 = _validate_prerequisites(b6_result_path, b8_result_path)

    seed_results = {}
    for seed in map(int, TRAINING_SEEDS):
        path = output_root / f"seed_{seed}" / "seed_result.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema") != SEED_SCHEMA or row.get("status") != "SEED_COMPLETE" or row.get("diagnostic_execution_sha") != str(args.execution_sha):
            raise RuntimeError(f"Phase2B9 invalid completed seed result {seed}")
        if not bool((row.get("canonical_side_member_reproduction") or {}).get("pass")):
            raise RuntimeError(f"Phase2B9 side-member reproduction not PASS for seed {seed}")
        seed_results[seed] = row

    descriptors = {}
    heldout_identity = []
    for evaluation_seed in map(int, EVALUATION_SEEDS):
        heldout = b6._find_heldout(heldout_root, evaluation_seed)
        rows = load_heldout_v3_artifact(
            heldout,
            expected_domain=DOMAIN,
            expected_evaluation_seed=evaluation_seed,
            expected_count=2048,
        )[:POLICY_COUNT]
        if len(rows) != POLICY_COUNT:
            raise RuntimeError("Phase2B9 heldout policy count drift")
        descriptors[evaluation_seed] = rows
        heldout_identity.append({"evaluation_seed": evaluation_seed, "path": str(heldout), "sha256": _sha256(heldout)})

    models = {}
    for seed in map(int, TRAINING_SEEDS):
        for loss_name in ("MSE_PAIRED_CONTROL", "HUBER_BETA_002"):
            models[(seed, loss_name)] = _load_pair_models(output_root, seed, loss_name)
    evaluations, tv_vectors, behavior_telemetry = _evaluate_loss_pair(models, descriptors)

    pooled = {}
    for loss_name in ("MSE_PAIRED_CONTROL", "HUBER_BETA_002"):
        means = [float(evaluations[str(es)][loss_name]["overall"]["mean"]) for es in EVALUATION_SEEDS]
        mismatches = [float(evaluations[str(es)][loss_name]["overall"]["dominant_action_mismatch_rate"]) for es in EVALUATION_SEEDS]
        pooled[loss_name] = {
            "equal_heldout_mean_tv": float(sum(means) / len(means)),
            "equal_heldout_dominant_action_mismatch_rate": float(sum(mismatches) / len(mismatches)),
            "root": _pooled_region(evaluations, loss_name, ("PREFLOP_ROOT",)),
            "preflop_continuation": _pooled_region(evaluations, loss_name, ("PREFLOP_CONTINUATION_1", "PREFLOP_CONTINUATION_2PLUS")),
        }
    improvement = float(pooled["MSE_PAIRED_CONTROL"]["equal_heldout_mean_tv"] - pooled["HUBER_BETA_002"]["equal_heldout_mean_tv"])
    relative = float(improvement / pooled["MSE_PAIRED_CONTROL"]["equal_heldout_mean_tv"]) if pooled["MSE_PAIRED_CONTROL"]["equal_heldout_mean_tv"] > 0 else -math.inf

    groups = {
        str(es): [float(a - b) for a, b in zip(tv_vectors[str(es)]["MSE_PAIRED_CONTROL"], tv_vectors[str(es)]["HUBER_BETA_002"])]
        for es in EVALUATION_SEEDS
    }
    bootstrap = equal_group_stratified_bootstrap_mean_ci(
        groups,
        seed_parts=("R7.5_ARCH_RESET", "PHASE2B9", "MSE_MINUS_HUBER"),
        replicates=BOOTSTRAP_REPLICATES,
        confidence_level=0.95,
    )

    huber_nrmse_pass = bool(all(float(seed_results[seed]["audits"]["HUBER_BETA_002"]["ensemble_weighted_nrmse"]) <= ADVANTAGE_NRMSE_MAX for seed in TRAINING_SEEDS))
    both_eval_improve = bool(all(float(evaluations[str(es)]["HUBER_BETA_002"]["overall"]["mean"]) < float(evaluations[str(es)]["MSE_PAIRED_CONTROL"]["overall"]["mean"]) for es in EVALUATION_SEEDS))
    material = bool(improvement >= ABS_IMPROVEMENT_MIN or relative >= REL_IMPROVEMENT_MIN)
    ci_positive = bool(float(bootstrap["ci_low"]) > 0.0)
    p95_ok = bool(all(float(evaluations[str(es)]["HUBER_BETA_002"]["overall"]["p95"]) - float(evaluations[str(es)]["MSE_PAIRED_CONTROL"]["overall"]["p95"]) <= P95_MAX_DEGRADE for es in EVALUATION_SEEDS))
    mismatch_ok = bool(pooled["HUBER_BETA_002"]["equal_heldout_dominant_action_mismatch_rate"] <= pooled["MSE_PAIRED_CONTROL"]["equal_heldout_dominant_action_mismatch_rate"])
    root_mse = float(pooled["MSE_PAIRED_CONTROL"]["root"]["mean"])
    root_huber = float(pooled["HUBER_BETA_002"]["root"]["mean"])
    cont_mse = float(pooled["MSE_PAIRED_CONTROL"]["preflop_continuation"]["mean"])
    cont_huber = float(pooled["HUBER_BETA_002"]["preflop_continuation"]["mean"])
    root_ok = bool(root_huber - root_mse <= REGION_MAX_DEGRADE)
    continuation_ok = bool(cont_huber - cont_mse <= REGION_MAX_DEGRADE)
    passed = bool(huber_nrmse_pass and both_eval_improve and material and ci_positive and p95_ok and mismatch_ok and root_ok and continuation_ok)
    status = "HUBER_ROBUSTNESS_SCREEN_PASS_ELIGIBLE_FOR_SMALL_CAUSAL_PILOT" if passed else "HUBER_ROBUSTNESS_SCREEN_FAIL_DO_NOT_TRAIN"
    next_route = "PRECOMMIT_PHASE2B10_HUBER_CAUSAL_TRAINING_PILOT" if passed else "DESIGN_STRATIFIED_CHANCE_SUPPORT_OR_SOLVER_LEVEL_VARIANCE_REDUCTION"

    return {
        "schema": SCHEMA,
        "status": status,
        "diagnostic_execution_sha": str(args.execution_sha),
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "training_seeds": list(map(int, TRAINING_SEEDS)),
        "evaluation_seeds": list(map(int, EVALUATION_SEEDS)),
        "policy_count_per_evaluation_seed": POLICY_COUNT,
        "candidate": "HUBER_BETA_002",
        "huber_beta": HUBER_BETA,
        "fit_contract": {
            "steps": ADVANTAGE_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "ensemble_size": ENSEMBLE_SIZE,
            "paired_identical_batch_sequence_per_member": True,
            "gradient_clip_norm": 10.0,
            "solver_traversal": False,
            "reservoir_mutation": False,
            "average_policy_fit": False,
        },
        "frozen_inputs": {
            "phase2b6_result_sha256": PHASE2B6_RESULT_SHA256,
            "phase2b8_result_sha256": PHASE2B8_RESULT_SHA256,
            "heldout": heldout_identity,
            "source_checkpoints": [
                {"training_seed": seed, "sha256": seed_results[seed]["source_checkpoint_sha256"], "advantage_memory": seed_results[seed]["advantage_memory"]}
                for seed in map(int, TRAINING_SEEDS)
            ],
        },
        "seed_results": {str(seed): seed_results[seed] for seed in map(int, TRAINING_SEEDS)},
        "heldout_evaluations": evaluations,
        "behavior_telemetry": behavior_telemetry,
        "pooled": pooled,
        "paired_improvement": {
            "absolute_mse_minus_huber": improvement,
            "relative_mse_minus_huber": relative,
            "bootstrap_95_ci": bootstrap,
        },
        "decision": {
            "huber_nrmse_gate_pass": huber_nrmse_pass,
            "both_evaluation_seed_means_improve": both_eval_improve,
            "materiality_pass": material,
            "bootstrap_ci_strictly_positive": ci_positive,
            "p95_non_degradation_pass": p95_ok,
            "dominant_action_mismatch_nonincrease_pass": mismatch_ok,
            "root_non_degradation_pass": root_ok,
            "preflop_continuation_non_degradation_pass": continuation_ok,
            "screen_pass": passed,
            "classification": status,
            "next_route": next_route,
            "small_causal_training_pilot_authorized": passed,
            "architecture_winner_selected": False,
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
        "source_phase2b6_status": r6.get("status"),
        "source_phase2b8_status": r8.get("status"),
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def _run_parent(args) -> int:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    entrypoint = str(Path(__file__).resolve())
    commands = []
    for seed in map(int, TRAINING_SEEDS):
        commands.append((seed, [
            sys.executable,
            entrypoint,
            "--repo-root", str(Path(args.repo_root).resolve()),
            "--phase2b6-root", str(Path(args.phase2b6_root).resolve()),
            "--phase2b6-result", str(Path(args.phase2b6_result).resolve()),
            "--phase2b8-result", str(Path(args.phase2b8_result).resolve()),
            "--heldout-root", str(Path(args.heldout_root).resolve()),
            "--output-root", str(output_root),
            "--execution-sha", str(args.execution_sha),
            "--single-seed", str(seed),
        ]))
    with ThreadPoolExecutor(max_workers=min(int(args.seed_workers), len(commands))) as pool:
        futures = {pool.submit(subprocess.run, cmd, check=False): seed for seed, cmd in commands}
        for future in as_completed(futures):
            seed = futures[future]
            completed = future.result()
            if int(completed.returncode) != 0:
                raise RuntimeError(f"Phase2B9 seed worker {seed} failed with exit code {completed.returncode}")
    result = _evaluate_parent(args)
    out = output_root / "R7_5_ARCH_RESET_V1PLUS_PHASE2B9_ROBUST_ADVANTAGE_REGRESSION.json"
    _atomic_json(result, out)
    print(json.dumps({
        "status": result["status"],
        "mse_pooled_mean_tv": result["pooled"]["MSE_PAIRED_CONTROL"]["equal_heldout_mean_tv"],
        "huber_pooled_mean_tv": result["pooled"]["HUBER_BETA_002"]["equal_heldout_mean_tv"],
        "absolute_improvement": result["paired_improvement"]["absolute_mse_minus_huber"],
        "relative_improvement": result["paired_improvement"]["relative_mse_minus_huber"],
        "bootstrap_ci": [result["paired_improvement"]["bootstrap_95_ci"]["ci_low"], result["paired_improvement"]["bootstrap_95_ci"]["ci_high"]],
        "screen_pass": result["decision"]["screen_pass"],
        "next_route": result["decision"]["next_route"],
        "result": str(out),
        "result_sha256": _sha256(out),
    }, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5 architecture-reset Phase2B9 robust Advantage regression screen")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--phase2b6-root", type=Path, required=True)
    parser.add_argument("--phase2b6-result", type=Path, required=True)
    parser.add_argument("--phase2b8-result", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--seed-workers", type=int, default=2)
    parser.add_argument("--single-seed", type=int, choices=TRAINING_SEEDS)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    for seed in map(int, TRAINING_SEEDS):
        validate_phase2_v3_contract(repo_root, representation=REPRESENTATION, domain=DOMAIN, training_seed=seed)
    if HUBER_BETA != 0.02 or ADVANTAGE_STEPS != 4096 or BATCH_SIZE != 256 or LEARNING_RATE != 0.001:
        raise RuntimeError("Phase2B9 frozen fit contract drift")
    _validate_prerequisites(Path(args.phase2b6_result).resolve(), Path(args.phase2b8_result).resolve())
    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("Phase2B9 torch thread contract drift")

    if args.single_seed is not None:
        return _run_seed(args, int(args.single_seed))
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
