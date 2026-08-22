from __future__ import annotations

"""Read-only Advantage-trajectory forensic for the completed V1+ Phase 2A run.

This tool does not traverse the solver, mutate a checkpoint, replay a reservoir,
fit a model, or perform an optimizer step.  It inspects the two completed H2/3H
Phase 2A resume checkpoints and the already-frozen heldout states to separate
Advantage-memory pressure, target noise, trajectory-support divergence, and
representation fragmentation before any causal Phase 2B training is designed.

Governance:
validation/R7_5_ARCH_RESET_V1PLUS_ADVANTAGE_FORENSIC_PRECOMMIT_20260822.md
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from spincore.r7_5_action_cfr import legal_mask, regret_matching_policy
from spincore.r7_5_representation_v3 import H2_FINAL
from spincore.r7_5_representation_v3_checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    EVALUATION_SEEDS,
    ITERATIONS,
    MODEL_FINGERPRINTS,
    TORCH_THREADS,
    TRAINING_SEEDS,
)
from spincore_nn.codec_v3 import DecodedInputV3, decode_spnniv3
from spincore_nn.models_v3_final import collate_v3_observations, make_h2_final_v3

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_ADVANTAGE_FORENSIC_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
PHASE2A_EXTRA_SCHEMA = "SPINCORE_R7_5_3D_V1PLUS_PHASE2A_RESUME_V1"
EXPECTED_STAGE_INDEX = 12
EXPECTED_ROOTS = 768
EXPECTED_ADV_CAPACITY = 100_000
POLICY_COUNT = 1024
STREET_NAMES = {0: "PREFLOP", 1: "FLOP", 2: "TURN", 3: "RIVER"}
PROJECTION_NAMES = (
    "exact_observation",
    "cards_only",
    "current_without_history",
    "geometry_without_cards",
    "history_exact",
    "history_structured",
    "history_v1_like",
    "current_plus_v1_like_history",
    "geometry_plus_v1_like_history",
)
COARSE_TARGET_PROJECTIONS = (
    "current_without_history",
    "current_plus_v1_like_history",
    "geometry_plus_v1_like_history",
)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q), method="linear"))


def _summary(values: Iterable[float]) -> dict:
    rows = [float(value) for value in values if _finite(value)]
    if not rows:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": _quantile(rows, 0.50),
        "p95": _quantile(rows, 0.95),
        "max": float(arr.max()),
    }


def _canonical_digest(material: object) -> bytes:
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _hex(value: bytes) -> str:
    return value.hex()


def _history_v1_like(decoded: DecodedInputV3) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(event.categorical[1]), int(event.categorical[2]))
        for event in decoded.history[-32:]
    )


def _history_structured(decoded: DecodedInputV3) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(tuple(int(value) for value in event.categorical) for event in decoded.history)


def _history_exact(decoded: DecodedInputV3) -> tuple:
    return tuple(
        (
            tuple(int(value) for value in event.categorical),
            tuple(float(value) for value in event.numeric),
        )
        for event in decoded.history
    )


def _current_without_history(decoded: DecodedInputV3) -> tuple:
    return (
        tuple(int(value) for value in decoded.categorical),
        tuple(int(value) for value in decoded.rank_tokens),
        tuple(int(value) for value in decoded.same_suit),
        tuple(float(value) for value in decoded.numeric),
        tuple(int(value) for value in decoded.primitive_legal),
    )


def _geometry_without_cards(decoded: DecodedInputV3) -> tuple:
    return (
        tuple(int(value) for value in decoded.categorical),
        tuple(float(value) for value in decoded.numeric),
        tuple(int(value) for value in decoded.primitive_legal),
    )


def _projection_digests(observation: bytes, decoded: DecodedInputV3) -> dict[str, bytes]:
    v1 = _history_v1_like(decoded)
    current = _current_without_history(decoded)
    geometry = _geometry_without_cards(decoded)
    return {
        "exact_observation": hashlib.sha256(observation).digest(),
        "cards_only": _canonical_digest((decoded.rank_tokens, decoded.same_suit)),
        "current_without_history": _canonical_digest(current),
        "geometry_without_cards": _canonical_digest(geometry),
        "history_exact": _canonical_digest(_history_exact(decoded)),
        "history_structured": _canonical_digest(_history_structured(decoded)),
        "history_v1_like": _canonical_digest(v1),
        "current_plus_v1_like_history": _canonical_digest((current, v1)),
        "geometry_plus_v1_like_history": _canonical_digest((geometry, v1)),
    }


def _history_bin(length: int) -> str:
    value = int(length)
    if value == 0:
        return "0"
    if value <= 2:
        return "1-2"
    if value <= 4:
        return "3-4"
    if value <= 8:
        return "5-8"
    if value <= 16:
        return "9-16"
    return "17+"


def _legal_signature(mask: Sequence[int]) -> str:
    bits = 0
    for index, value in enumerate(mask):
        if int(value):
            bits |= 1 << index
    return f"0x{bits:03x}"


def _counter_tv(left: Counter, right: Counter) -> dict:
    left_total = sum(int(value) for value in left.values())
    right_total = sum(int(value) for value in right.values())
    keys = set(left) | set(right)
    if left_total <= 0 or right_total <= 0:
        return {"tv": None, "left_total": left_total, "right_total": right_total}
    tv = 0.5 * sum(
        abs(float(left.get(key, 0)) / left_total - float(right.get(key, 0)) / right_total)
        for key in keys
    )
    return {
        "tv": float(tv),
        "left_total": int(left_total),
        "right_total": int(right_total),
        "support_union": len(keys),
    }


def _jaccard(left: set[bytes], right: set[bytes]) -> dict:
    intersection = left & right
    union = left | right
    return {
        "left_unique": len(left),
        "right_unique": len(right),
        "intersection": len(intersection),
        "union": len(union),
        "jaccard": float(len(intersection) / len(union)) if union else 1.0,
    }


def _new_exact_aggregate(mask: tuple[int, ...], decoded: DecodedInputV3) -> dict:
    return {
        "count": 0,
        "weight_sum": 0.0,
        "weighted_sum": [0.0] * 10,
        "sum": [0.0] * 10,
        "sumsq": [0.0] * 10,
        "positive": [0] * 10,
        "negative": [0] * 10,
        "legal": tuple(int(value) for value in mask),
        "street": int(decoded.categorical[1]),
        "history_len": int(decoded.history_len),
    }


def _add_exact_aggregate(aggregate: dict, target: Sequence[float], weight: float) -> None:
    aggregate["count"] += 1
    aggregate["weight_sum"] += float(weight)
    for index in range(10):
        value = float(target[index])
        aggregate["weighted_sum"][index] += float(weight) * value
        aggregate["sum"][index] += value
        aggregate["sumsq"][index] += value * value
        if value > 0.0:
            aggregate["positive"][index] += 1
        elif value < 0.0:
            aggregate["negative"][index] += 1


def _new_coarse_aggregate(mask: tuple[int, ...]) -> dict:
    return {
        "count": 0,
        "weight_sum": 0.0,
        "weighted_sum": [0.0] * 10,
        "legal": tuple(int(value) for value in mask),
    }


def _add_coarse_aggregate(aggregate: dict, target: Sequence[float], weight: float) -> None:
    aggregate["count"] += 1
    aggregate["weight_sum"] += float(weight)
    for index in range(10):
        aggregate["weighted_sum"][index] += float(weight) * float(target[index])


def _weighted_target(aggregate: dict) -> list[float]:
    total = float(aggregate["weight_sum"])
    if total <= 0.0:
        count = max(1, int(aggregate["count"]))
        return [float(value) / count for value in aggregate.get("sum", [0.0] * 10)]
    return [float(value) / total for value in aggregate["weighted_sum"]]


def _legal_indices(mask: Sequence[int]) -> tuple[int, ...]:
    out = tuple(index for index, value in enumerate(mask) if int(value))
    if not out:
        raise RuntimeError("Advantage sample contains empty legal mask")
    return out


def _target_policy(target: Sequence[float], mask: Sequence[int]) -> tuple[float, ...]:
    return regret_matching_policy(tuple(float(value) for value in target), _legal_indices(mask))


def _policy_tv(left: Sequence[float], right: Sequence[float]) -> float:
    return float(0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right)))


def _within_seed_exact_noise(exact: dict) -> dict:
    rms_std = []
    sign_instability = []
    duplicate_groups = 0
    duplicate_items = 0
    for aggregate in exact.values():
        count = int(aggregate["count"])
        if count < 2:
            continue
        duplicate_groups += 1
        duplicate_items += count
        legal = _legal_indices(aggregate["legal"])
        variances = []
        unstable = 0
        for slot in legal:
            mean = float(aggregate["sum"][slot]) / count
            variance = max(0.0, float(aggregate["sumsq"][slot]) / count - mean * mean)
            variances.append(variance)
            if int(aggregate["positive"][slot]) > 0 and int(aggregate["negative"][slot]) > 0:
                unstable += 1
        rms_std.append(math.sqrt(float(sum(variances) / len(variances))) if variances else 0.0)
        sign_instability.append(float(unstable / len(legal)) if legal else 0.0)
    retained = sum(int(row["count"]) for row in exact.values())
    return {
        "unique_exact_groups": len(exact),
        "duplicate_exact_groups": int(duplicate_groups),
        "retained_items_in_duplicate_groups": int(duplicate_items),
        "fraction_retained_items_in_duplicate_groups": (
            float(duplicate_items / retained) if retained else 0.0
        ),
        "per_duplicate_group_legal_slot_rms_target_std": _summary(rms_std),
        "per_duplicate_group_fraction_legal_slots_with_both_target_signs": _summary(sign_instability),
    }


def _process_advantage_memory(items: Sequence[object], *, seen: int, capacity: int) -> dict:
    projections = {name: set() for name in PROJECTION_NAMES}
    exact: dict[tuple[bytes, tuple[int, ...]], dict] = {}
    coarse = {name: {} for name in COARSE_TARGET_PROJECTIONS}
    by_iteration = Counter()
    by_street = Counter()
    by_history_bin = Counter()
    by_legal = Counter()

    for index, item in enumerate(items, start=1):
        observation = bytes(item.observation)
        decoded = decode_spnniv3(observation)
        digests = _projection_digests(observation, decoded)
        for name, digest in digests.items():
            projections[name].add(digest)

        mask = tuple(int(value) for value in item.legal)
        if len(mask) != 10 or not any(mask):
            raise RuntimeError("malformed retained Advantage legal mask")
        target = tuple(float(value) for value in item.target)
        if len(target) != 10 or not all(_finite(value) for value in target):
            raise RuntimeError("malformed retained Advantage target")
        weight = float(item.weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise RuntimeError("malformed retained Advantage sample weight")

        exact_key = (digests["exact_observation"], mask)
        aggregate = exact.get(exact_key)
        if aggregate is None:
            aggregate = _new_exact_aggregate(mask, decoded)
            exact[exact_key] = aggregate
        _add_exact_aggregate(aggregate, target, weight)

        for name in COARSE_TARGET_PROJECTIONS:
            key = (digests[name], mask)
            row = coarse[name].get(key)
            if row is None:
                row = _new_coarse_aggregate(mask)
                coarse[name][key] = row
            _add_coarse_aggregate(row, target, weight)

        by_iteration[int(item.iteration)] += 1
        by_street[STREET_NAMES.get(int(decoded.categorical[1]), str(decoded.categorical[1]))] += 1
        by_history_bin[_history_bin(decoded.history_len)] += 1
        by_legal[_legal_signature(mask)] += 1

        if index % 25_000 == 0:
            print(f"[Adv forensic] decoded {index}/{len(items)} retained Advantage samples", flush=True)

    retained = len(items)
    return {
        "capacity": int(capacity),
        "seen": int(seen),
        "retained": int(retained),
        "retention_fraction": float(retained / seen) if seen else 0.0,
        "saturation_seen_over_capacity": float(seen / capacity) if capacity else None,
        "projections": projections,
        "exact": exact,
        "coarse": coarse,
        "distributions": {
            "iteration": by_iteration,
            "street": by_street,
            "history_length_bin": by_history_bin,
            "legal_mask": by_legal,
        },
        "within_seed_exact_target_noise": _within_seed_exact_noise(exact),
    }


def _projection_overlap(left: dict, right: dict) -> dict:
    out = {}
    for name in PROJECTION_NAMES:
        out[name] = _jaccard(left["projections"][name], right["projections"][name])
    exact_j = float(out["exact_observation"]["jaccard"])
    for name in PROJECTION_NAMES:
        out[name]["absolute_jaccard_gain_vs_exact"] = float(out[name]["jaccard"] - exact_j)
    return out


def _shared_group_comparison(left: dict, right: dict, *, include_top: bool) -> dict:
    shared = set(left) & set(right)
    tvs = []
    maes = []
    sign_disagreement = []
    weighted_tv_num = 0.0
    weighted_tv_den = 0.0
    top = []
    left_items_shared = 0
    right_items_shared = 0

    for key in shared:
        a = left[key]
        b = right[key]
        if tuple(a["legal"]) != tuple(b["legal"]):
            raise RuntimeError("shared Advantage group has inconsistent legal mask")
        mask = a["legal"]
        legal = _legal_indices(mask)
        ta = _weighted_target(a)
        tb = _weighted_target(b)
        pa = _target_policy(ta, mask)
        pb = _target_policy(tb, mask)
        tv = _policy_tv(pa, pb)
        mae = float(sum(abs(ta[slot] - tb[slot]) for slot in legal) / len(legal))
        sign = float(sum((ta[slot] > 0.0) != (tb[slot] > 0.0) for slot in legal) / len(legal))
        weight = float(min(int(a["count"]), int(b["count"])))
        weighted_tv_num += weight * tv
        weighted_tv_den += weight
        tvs.append(tv)
        maes.append(mae)
        sign_disagreement.append(sign)
        left_items_shared += int(a["count"])
        right_items_shared += int(b["count"])
        if include_top:
            digest = key[0]
            top.append({
                "group_sha256": _hex(digest),
                "left_count": int(a["count"]),
                "right_count": int(b["count"]),
                "street": STREET_NAMES.get(int(a.get("street", -1)), str(a.get("street"))),
                "history_len": int(a.get("history_len", -1)),
                "target_policy_tv": float(tv),
                "legal_target_mae": float(mae),
                "positive_sign_disagreement_fraction": float(sign),
                "left_weighted_target": [float(value) for value in ta],
                "right_weighted_target": [float(value) for value in tb],
                "left_regret_matching_policy": [float(value) for value in pa],
                "right_regret_matching_policy": [float(value) for value in pb],
            })

    left_total = sum(int(row["count"]) for row in left.values())
    right_total = sum(int(row["count"]) for row in right.values())
    result = {
        "left_groups": len(left),
        "right_groups": len(right),
        "shared_groups": len(shared),
        "group_jaccard": float(len(shared) / len(set(left) | set(right))) if (left or right) else 1.0,
        "left_retained_item_coverage_by_shared_groups": float(left_items_shared / left_total) if left_total else 0.0,
        "right_retained_item_coverage_by_shared_groups": float(right_items_shared / right_total) if right_total else 0.0,
        "target_derived_regret_matching_policy_tv": _summary(tvs),
        "count_min_weighted_mean_target_policy_tv": (
            float(weighted_tv_num / weighted_tv_den) if weighted_tv_den > 0.0 else None
        ),
        "legal_target_mae": _summary(maes),
        "positive_sign_disagreement_fraction": _summary(sign_disagreement),
    }
    if include_top:
        result["top_50_shared_groups_by_target_policy_tv"] = sorted(
            top,
            key=lambda row: float(row["target_policy_tv"]),
            reverse=True,
        )[:50]
    return result


def _stage_report_summary(payload: dict) -> list[dict]:
    extra = dict(payload.get("extra") or {})
    state = dict(extra.get("stage_state") or {})
    reports = list(state.get("iteration_reports") or [])
    out = []
    for report in reports:
        roots = int(report.get("roots_added", 0))
        adv = int(report.get("advantage_seen_added", 0))
        out.append({
            "iteration": int(report.get("iteration", -1)),
            "roots_added": roots,
            "advantage_seen_added": adv,
            "advantage_samples_per_root": float(adv / roots) if roots else None,
            "strategy_seen_added": int(report.get("strategy_seen_added", 0)),
            "ensemble_weighted_nrmse": float(report.get("ensemble_weighted_nrmse", math.nan)),
            "ensemble_advantage_gate_pass": bool(report.get("ensemble_advantage_gate_pass")),
            "tree_collection_seconds": float(report.get("tree_collection_seconds", 0.0)),
            "branch_geometry": dict(report.get("branch_geometry") or {}),
        })
    return out


def _load_seed_checkpoint(input_root: Path, seed: int, source_execution_sha: str) -> tuple[dict, object, dict]:
    seed_root = input_root / f"seed_{int(seed)}"
    checkpoint = seed_root / "resume_checkpoint.pt"
    seed_result = seed_root / "seed_result.json"
    if not checkpoint.is_file() or not seed_result.is_file():
        raise RuntimeError(f"missing completed Phase2A artifacts for seed {seed}")

    result = json.loads(seed_result.read_text(encoding="utf-8"))
    if result.get("status") != "SEED_COMPLETE":
        raise RuntimeError(f"Phase2A seed result is not complete for {seed}")
    if result.get("execution_sha") != str(source_execution_sha):
        raise RuntimeError(f"Phase2A seed result execution SHA mismatch for {seed}")
    if result.get("representation") != REPRESENTATION or result.get("domain") != DOMAIN:
        raise RuntimeError(f"Phase2A seed result representation/domain mismatch for {seed}")
    if int(result.get("roots", -1)) != EXPECTED_ROOTS or int(result.get("iterations", -1)) != ITERATIONS:
        raise RuntimeError(f"Phase2A seed result completed-shape mismatch for {seed}")
    if result.get("all_advantage_gates_pass") is not True:
        raise RuntimeError(f"Phase2A local Advantage gates did not pass for {seed}")

    torch_rng = torch.get_rng_state().clone()
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    finally:
        torch.set_rng_state(torch_rng)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"wrong Phase2A checkpoint schema for {seed}")
    expected = {
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "seed": int(seed),
        "action_candidate": ACTION_CANDIDATE,
        "execution_sha": str(source_execution_sha),
        "architecture_fingerprint_sha256": MODEL_FINGERPRINTS[REPRESENTATION],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Phase2A checkpoint identity mismatch {seed}/{key}")
    if bool(payload.get("production_training_authorized")) or bool(payload.get("ready_for_tables")):
        raise RuntimeError("Phase2A checkpoint illegally authorizes production/table use")

    progress = dict(payload.get("progress") or {})
    if progress.get("phase") != "phase2a_resume":
        raise RuntimeError(f"Phase2A resume phase mismatch for {seed}")
    if int(progress.get("iteration", -1)) != ITERATIONS or int(progress.get("global_root", -1)) != EXPECTED_ROOTS:
        raise RuntimeError(f"Phase2A resume progress mismatch for {seed}")
    extra = dict(payload.get("extra") or {})
    if extra.get("schema") != PHASE2A_EXTRA_SCHEMA or int(extra.get("stage_index", -1)) != EXPECTED_STAGE_INDEX:
        raise RuntimeError(f"Phase2A resume extra/stage mismatch for {seed}")
    adv_state = dict(payload.get("adv_mem") or {})
    if int(adv_state.get("capacity", -1)) != EXPECTED_ADV_CAPACITY:
        raise RuntimeError(f"Phase2A Advantage reservoir capacity mismatch for {seed}")

    _cfg, model = make_h2_final_v3(device="cpu", seed=0)
    model.load_state_dict(payload["advantage"])
    model.eval()

    memory = _process_advantage_memory(
        list(adv_state.get("items") or []),
        seen=int(adv_state.get("seen", 0)),
        capacity=int(adv_state.get("capacity", 0)),
    )
    memory["iteration_reports"] = _stage_report_summary(payload)
    memory["seed_result_advantage_gates"] = list(result.get("advantage_gates") or [])
    memory["checkpoint"] = str(checkpoint)
    del payload
    return memory, model, result


def _find_heldout(root: Path, evaluation_seed: int) -> Path:
    matches = []
    for path in root.rglob("states.json.gz"):
        try:
            states = load_heldout_v3_artifact(
                path,
                expected_domain=DOMAIN,
                expected_evaluation_seed=int(evaluation_seed),
                expected_count=2048,
            )
        except Exception:
            continue
        if states:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"heldout identity mismatch for {evaluation_seed}: {matches}")
    return matches[0]


def _advantage_logits(model, descriptors) -> list[list[float]]:
    out = []
    for start in range(0, len(descriptors), 256):
        rows = descriptors[start : start + 256]
        batch = collate_v3_observations(
            [item.observation_v3 for item in rows],
            [legal_mask(item.legal_slots) for item in rows],
            with_semantics=False,
            device="cpu",
        )
        with torch.no_grad():
            logits = model(batch).detach().cpu().tolist()
        out.extend([[float(value) for value in row] for row in logits])
    return out


def _heldout_advantage_comparison(models: dict[int, object], heldout_root: Path) -> list[dict]:
    seed_a, seed_b = map(int, TRAINING_SEEDS)
    output = []
    for evaluation_seed in EVALUATION_SEEDS:
        path = _find_heldout(heldout_root, int(evaluation_seed))
        descriptors = list(load_heldout_v3_artifact(
            path,
            expected_domain=DOMAIN,
            expected_evaluation_seed=int(evaluation_seed),
            expected_count=2048,
        )[:POLICY_COUNT])
        left = _advantage_logits(models[seed_a], descriptors)
        right = _advantage_logits(models[seed_b], descriptors)
        local = []
        by_street = defaultdict(list)
        for descriptor, a, b in zip(descriptors, left, right):
            legal = tuple(int(value) for value in descriptor.legal_slots)
            pa = regret_matching_policy(a, legal)
            pb = regret_matching_policy(b, legal)
            tv = _policy_tv(pa, pb)
            mae = float(sum(abs(a[slot] - b[slot]) for slot in legal) / len(legal))
            sign = float(sum((a[slot] > 0.0) != (b[slot] > 0.0) for slot in legal) / len(legal))
            dominant_a = max(legal, key=lambda slot: float(pa[slot]))
            dominant_b = max(legal, key=lambda slot: float(pb[slot]))
            decoded = decode_spnniv3(descriptor.observation_v3)
            street = STREET_NAMES.get(int(decoded.categorical[1]), str(decoded.categorical[1]))
            row = {
                "state_index": int(descriptor.state_index),
                "street": street,
                "behavior_policy_tv": float(tv),
                "legal_raw_advantage_mae": float(mae),
                "positive_sign_disagreement_fraction": float(sign),
                "dominant_legal_action_mismatch": int(dominant_a != dominant_b),
            }
            local.append(row)
            by_street[street].append(row)

        def summarize(rows: Sequence[dict]) -> dict:
            return {
                "count": len(rows),
                "behavior_policy_tv": _summary(row["behavior_policy_tv"] for row in rows),
                "legal_raw_advantage_mae": _summary(row["legal_raw_advantage_mae"] for row in rows),
                "positive_sign_disagreement_fraction": _summary(
                    row["positive_sign_disagreement_fraction"] for row in rows
                ),
                "dominant_legal_action_mismatch_rate": (
                    float(sum(row["dominant_legal_action_mismatch"] for row in rows) / len(rows))
                    if rows else None
                ),
            }

        output.append({
            "evaluation_seed": int(evaluation_seed),
            "heldout": str(path),
            "overall": summarize(local),
            "by_street": {name: summarize(rows) for name, rows in sorted(by_street.items())},
            "top_50_states_by_behavior_policy_tv": sorted(
                local,
                key=lambda row: float(row["behavior_policy_tv"]),
                reverse=True,
            )[:50],
        })
    return output


def _public_memory_summary(memory: dict) -> dict:
    return {
        "capacity": int(memory["capacity"]),
        "seen": int(memory["seen"]),
        "retained": int(memory["retained"]),
        "retention_fraction": float(memory["retention_fraction"]),
        "saturation_seen_over_capacity": float(memory["saturation_seen_over_capacity"]),
        "unique_projection_counts": {
            name: len(memory["projections"][name]) for name in PROJECTION_NAMES
        },
        "retained_by_iteration": {
            str(key): int(value)
            for key, value in sorted(memory["distributions"]["iteration"].items())
        },
        "retained_by_street": {
            str(key): int(value)
            for key, value in sorted(memory["distributions"]["street"].items())
        },
        "retained_by_history_length_bin": {
            str(key): int(value)
            for key, value in sorted(memory["distributions"]["history_length_bin"].items())
        },
        "retained_by_legal_mask": {
            str(key): int(value)
            for key, value in sorted(memory["distributions"]["legal_mask"].items())
        },
        "within_seed_exact_target_noise": memory["within_seed_exact_target_noise"],
        "iteration_reports": memory["iteration_reports"],
        "seed_result_advantage_gates": memory["seed_result_advantage_gates"],
        "checkpoint": memory["checkpoint"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only V1+ Phase2A Advantage-trajectory forensic")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--source-execution-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not str(args.source_execution_sha).strip():
        raise SystemExit("--source-execution-sha is required")
    input_root = args.input_root.resolve()
    heldout_root = args.heldout_root.resolve()
    output = args.out.resolve()

    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("Advantage forensic torch-thread contract drift")

    memories = {}
    models = {}
    seed_results = {}
    for seed in map(int, TRAINING_SEEDS):
        print(f"[Adv forensic] loading completed Phase2A seed {seed}", flush=True)
        memory, model, seed_result = _load_seed_checkpoint(
            input_root,
            int(seed),
            str(args.source_execution_sha),
        )
        memories[int(seed)] = memory
        models[int(seed)] = model
        seed_results[int(seed)] = seed_result

    seed_a, seed_b = map(int, TRAINING_SEEDS)
    left = memories[seed_a]
    right = memories[seed_b]
    projection_overlap = _projection_overlap(left, right)
    distribution_divergence = {
        "street": _counter_tv(left["distributions"]["street"], right["distributions"]["street"]),
        "history_length_bin": _counter_tv(
            left["distributions"]["history_length_bin"], right["distributions"]["history_length_bin"]
        ),
        "legal_mask": _counter_tv(left["distributions"]["legal_mask"], right["distributions"]["legal_mask"]),
        "retained_iteration": _counter_tv(
            left["distributions"]["iteration"], right["distributions"]["iteration"]
        ),
    }

    print("[Adv forensic] comparing shared exact Advantage infosets", flush=True)
    shared_exact = _shared_group_comparison(left["exact"], right["exact"], include_top=True)
    coarse_targets = {}
    for name in COARSE_TARGET_PROJECTIONS:
        print(f"[Adv forensic] coarse target diagnostic {name}", flush=True)
        coarse_targets[name] = _shared_group_comparison(
            left["coarse"][name],
            right["coarse"][name],
            include_top=False,
        )
        coarse_targets[name]["diagnostic_only"] = True
        coarse_targets[name]["warning"] = (
            "This projection aliases strategically distinct exact observations and cannot justify production compression by itself."
        )

    heldout = _heldout_advantage_comparison(models, heldout_root)

    generated_by_iteration = []
    left_reports = {int(row["iteration"]): row for row in left["iteration_reports"]}
    right_reports = {int(row["iteration"]): row for row in right["iteration_reports"]}
    for iteration in range(1, ITERATIONS + 1):
        a = left_reports.get(iteration, {})
        b = right_reports.get(iteration, {})
        av = int(a.get("advantage_seen_added", 0))
        bv = int(b.get("advantage_seen_added", 0))
        mean = 0.5 * (av + bv)
        generated_by_iteration.append({
            "iteration": iteration,
            "left_advantage_seen_added": av,
            "right_advantage_seen_added": bv,
            "absolute_difference": abs(av - bv),
            "relative_difference_to_pair_mean": float(abs(av - bv) / mean) if mean > 0 else 0.0,
            "left_advantage_samples_per_root": a.get("advantage_samples_per_root"),
            "right_advantage_samples_per_root": b.get("advantage_samples_per_root"),
        })

    result = {
        "schema": SCHEMA,
        "status": "ADVANTAGE_FORENSIC_COMPLETE_NO_CAUSAL_REMEDY_SELECTED",
        "governance_scope": "Post-R7.5.3 architecture-reset read-only diagnosis; R7.5.3 remains closed and is not reopened.",
        "source_execution_sha": str(args.source_execution_sha),
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "training_seeds": [seed_a, seed_b],
        "evaluation_seeds": [int(value) for value in EVALUATION_SEEDS],
        "roots_per_seed": EXPECTED_ROOTS,
        "advantage_memory_by_seed": {
            str(seed): _public_memory_summary(memories[seed]) for seed in (seed_a, seed_b)
        },
        "advantage_samples_generated_cross_seed_by_iteration": generated_by_iteration,
        "retained_support_projection_overlap": projection_overlap,
        "retained_distribution_cross_seed_tv": distribution_divergence,
        "shared_exact_infoset_target_diagnostics": shared_exact,
        "coarse_projection_target_diagnostics": coarse_targets,
        "final_advantage_network_common_heldout_behavior_diagnostics": heldout,
        "interpretation_guardrails": [
            "This output is diagnostic and does not itself prove a causal remedy.",
            "A larger Advantage reservoir was not passively shadowed in Phase2A; capacity causality therefore cannot be claimed from this artifact.",
            "Coarse projection target comparisons intentionally alias exact poker states and are diagnostic only.",
            "Heldout Advantage behavior TV measures upstream behavior-policy divergence; it is not the final AveragePolicy admission metric.",
            "No threshold, seed, model weight, reservoir, traversal state or action abstraction was modified.",
            "No H2/H3/V1+ winner is selected and no production training is authorized.",
        ],
        "production_training_authorized": False,
        "ready_for_tables": False,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(json.dumps({
        "status": result["status"],
        "out": str(output),
        "shared_exact_groups": int(shared_exact["shared_groups"]),
        "exact_support_jaccard": float(projection_overlap["exact_observation"]["jaccard"]),
        "current_support_jaccard": float(projection_overlap["current_without_history"]["jaccard"]),
        "geometry_support_jaccard": float(projection_overlap["geometry_without_cards"]["jaccard"]),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
