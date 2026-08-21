from __future__ import annotations

"""Read-only forensic post-mortem for the final R7.5.3C x16 checkpoints.

This tool performs no training and changes no model.  It measures where the two
frozen training seeds disagree, how that disagreement relates to V3 history
richness/current geometry, and whether the final reservoirs show evidence of
state-space fragmentation or replacement pressure.

Governance is frozen in:
validation/R7_5_3D_V1PLUS_POSTMORTEM_PRECOMMIT_20260821.md
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

from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from spincore.r7_5_representation_v3_final_policy import (
    extract_final_v3_policy_light,
    load_finalized_v3_policy_light,
)
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact
from spincore.r7_5_representation_v3_stage_contract import (
    DOMAINS,
    EVALUATION_SEEDS,
    MODEL_FINGERPRINTS,
    TORCH_THREADS,
    TRAINING_SEEDS,
)
from spincore_nn.codec_v3 import DecodedInputV3, decode_spnniv3

SCHEMA = "SPINCORE_R7_5_3D_V1PLUS_FORENSIC_POSTMORTEM_V1"
REPRESENTATIONS = (H2_FINAL, H3_FINAL)
POLICY_COUNT = 1024
STREET_NAMES = {0: "PREFLOP", 1: "FLOP", 2: "TURN", 3: "RIVER"}


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_float(value: object, default: float = 0.0) -> float:
    return float(value) if _finite(value) else float(default)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires non-empty values")
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


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = 0.5 * (cursor + end - 1) + 1.0
        ranks[order[cursor:end]] = rank
        cursor = end
    return ranks


def _spearman(rows: Sequence[dict], feature: str) -> dict:
    pairs = []
    for row in rows:
        value = row.get(feature)
        tv = row.get("tv")
        if _finite(value) and _finite(tv):
            pairs.append((float(value), float(tv)))
    if len(pairs) < 3:
        return {"feature": feature, "count": len(pairs), "rho": None}
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return {"feature": feature, "count": int(x.size), "rho": None}
    rho = float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])
    return {"feature": feature, "count": int(x.size), "rho": rho}


def _digest(material: object) -> str:
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _v1_history_projection(decoded: DecodedInputV3) -> tuple[tuple[int, int], ...]:
    # Mirrors only the information content of V1 public_history: last 32
    # (street, action_type) events.  It deliberately does NOT claim byte parity
    # with SPNNIV1, whose card representation is different.
    return tuple(
        (int(event.categorical[1]), int(event.categorical[2]))
        for event in decoded.history[-32:]
    )


def _structured_history_projection(decoded: DecodedInputV3) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(tuple(int(value) for value in event.categorical) for event in decoded.history)


def _exact_history_projection(decoded: DecodedInputV3) -> tuple:
    return tuple(
        (
            tuple(int(value) for value in event.categorical),
            tuple(float(value) for value in event.numeric),
        )
        for event in decoded.history
    )


def _history_bins(value: int) -> str:
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


def _variant_bins(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3-4"
    if value <= 8:
        return "5-8"
    return "9+"


def _spr_bin(value: float | None) -> str:
    if value is None or not _finite(value):
        return "NA"
    x = float(value)
    if x <= 1.0:
        return "<=1"
    if x <= 2.0:
        return "1-2"
    if x <= 5.0:
        return "2-5"
    if x <= 10.0:
        return "5-10"
    return ">10"


def _decode_features(decoded: DecodedInputV3, *, action_path_len: int) -> dict:
    domain, street, dealer_rel, sb_rel, bb_rel, live_count, visible_board, *statuses = decoded.categorical
    pot_bb = float(decoded.numeric[0])
    to_call_bb = float(decoded.numeric[1])
    current_bet_bb = float(decoded.numeric[2])
    stacks = [float(decoded.numeric[3 + index]) for index in range(3)]
    live_stacks = [stack for index, stack in enumerate(stacks[: int(live_count)]) if stack > 0.0]
    min_positive_stack = min(live_stacks) if live_stacks else 0.0
    spr = (min_positive_stack / pot_bb) if pot_bb > 1e-12 else None

    actor_counts = Counter()
    action_counts = Counter()
    forced_count = 0
    paid_ratios: list[float] = []
    commitment_ratios: list[float] = []
    for event in decoded.history:
        actor, _event_street, action_type, forced = (int(value) for value in event.categorical)
        actor_counts[actor] += 1
        action_counts[action_type] += 1
        forced_count += int(forced)
        paid, commitment, pot_before, _pot_after = (float(value) for value in event.numeric)
        if pot_before > 1e-9:
            paid_ratios.append(paid / pot_before)
            commitment_ratios.append(commitment / pot_before)

    last_actor = int(decoded.history[-1].categorical[0]) if decoded.history else None
    last_action_type = int(decoded.history[-1].categorical[2]) if decoded.history else None
    paid_mean = float(np.mean(paid_ratios)) if paid_ratios else 0.0
    paid_std = float(np.std(paid_ratios)) if paid_ratios else 0.0
    paid_max = max(paid_ratios) if paid_ratios else 0.0
    commitment_max = max(commitment_ratios) if commitment_ratios else 0.0

    v1_projection = _v1_history_projection(decoded)
    structured_projection = _structured_history_projection(decoded)
    exact_projection = _exact_history_projection(decoded)

    out = {
        "domain_id": int(domain),
        "street": int(street),
        "street_name": STREET_NAMES.get(int(street), str(street)),
        "dealer_rel": int(dealer_rel),
        "small_blind_rel": int(sb_rel),
        "big_blind_rel": int(bb_rel),
        "live_count": int(live_count),
        "visible_board": int(visible_board),
        "pot_bb": pot_bb,
        "to_call_bb": to_call_bb,
        "current_bet_bb": current_bet_bb,
        "min_positive_stack_bb": min_positive_stack,
        "spr": spr,
        "legal_primitive_count": int(sum(int(value) for value in decoded.primitive_legal)),
        "history_len": int(decoded.history_len),
        "action_path_len": int(action_path_len),
        "forced_count": int(forced_count),
        "nonforced_count": int(decoded.history_len - forced_count),
        "unique_history_actors": int(len(actor_counts)),
        "last_actor": last_actor,
        "last_action_type": last_action_type,
        "history_paid_over_pot_mean": paid_mean,
        "history_paid_over_pot_std": paid_std,
        "history_paid_over_pot_max": paid_max,
        "history_commitment_over_pot_max": commitment_max,
        "v1_history_projection_sha256": _digest(v1_projection),
        "structured_history_projection_sha256": _digest(structured_projection),
        "exact_history_sha256": _digest(exact_projection),
    }
    for action_type in range(6):
        out[f"history_action_type_{action_type}_count"] = int(action_counts[action_type])
    return out


def _find_heldout(root: Path, domain: str, evaluation_seed: int) -> Path:
    matches = []
    for path in root.rglob("states.json.gz"):
        try:
            payload = load_heldout_v3_artifact(
                path,
                expected_domain=domain,
                expected_evaluation_seed=int(evaluation_seed),
                expected_count=2048,
            )
        except Exception:
            continue
        if payload:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"heldout identity mismatch for {domain}/{evaluation_seed}: {matches}")
    return matches[0]


def _find_final_checkpoints(input_root: Path) -> list[Path]:
    finals = []
    for checkpoint in sorted(input_root.rglob("checkpoint.pt")):
        report = checkpoint.parent / "report.json"
        if not report.exists():
            continue
        payload = json.loads(report.read_text(encoding="utf-8"))
        if bool(payload.get("finalized")):
            finals.append(checkpoint)
    if len(finals) != 8:
        raise RuntimeError(f"expected exactly 8 final x16 checkpoints, found {len(finals)}")
    return finals


def _extract_inventory(
    *,
    repo_root: Path,
    input_root: Path,
    output_root: Path,
    training_execution_sha: str,
) -> tuple[dict, dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_by_key = {}
    policy_by_key = {}
    for index, checkpoint in enumerate(_find_final_checkpoints(input_root)):
        temp = output_root / f"extracting_{index}.pt"
        metadata = extract_final_v3_policy_light(
            checkpoint,
            temp,
            expected_training_execution_sha=training_execution_sha,
        )
        key = (
            str(metadata["representation"]),
            str(metadata["domain"]),
            int(metadata["training_seed"]),
        )
        if key in checkpoint_by_key:
            raise RuntimeError(f"duplicate final x16 checkpoint {key}")
        destination = output_root / f"{key[0]}__{key[1]}__{key[2]}.pt"
        temp.replace(destination)
        checkpoint_by_key[key] = checkpoint
        policy_by_key[key] = load_finalized_v3_policy_light(
            destination,
            repo_root=repo_root,
            expected_training_execution_sha=training_execution_sha,
            expected_representation=key[0],
            expected_domain=key[1],
            expected_training_seed=key[2],
        )
    expected = {
        (rep, domain, int(seed))
        for rep in REPRESENTATIONS
        for domain in DOMAINS
        for seed in TRAINING_SEEDS
    }
    if set(checkpoint_by_key) != expected:
        raise RuntimeError(
            f"final inventory mismatch missing={sorted(expected-set(checkpoint_by_key))} "
            f"extra={sorted(set(checkpoint_by_key)-expected)}"
        )
    return checkpoint_by_key, policy_by_key


def _projection_group_enrichment(rows: list[dict]) -> None:
    by_v1: dict[str, list[dict]] = defaultdict(list)
    by_structured: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_v1[str(row["v1_history_projection_sha256"])].append(row)
        by_structured[str(row["structured_history_projection_sha256"])].append(row)
    for group in by_v1.values():
        variants = len({str(row["exact_history_sha256"]) for row in group})
        for row in group:
            row["v1_projection_group_size"] = len(group)
            row["v1_projection_exact_variants"] = int(variants)
    for group in by_structured.values():
        variants = len({str(row["exact_history_sha256"]) for row in group})
        for row in group:
            row["structured_projection_group_size"] = len(group)
            row["structured_projection_exact_variants"] = int(variants)


def _grouped_tv(rows: Sequence[dict], field: str, transform=None) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        raw = row.get(field)
        key = transform(raw) if transform is not None else raw
        groups[str(key)].append(float(row["tv"]))
    return {key: _summary(values) for key, values in sorted(groups.items())}


def _state_readout(
    *,
    policies: dict,
    heldout_root: Path,
) -> tuple[list[dict], list[dict], dict]:
    seed_a, seed_b = map(int, TRAINING_SEEDS)
    state_rows: list[dict] = []
    row_summaries: list[dict] = []
    action_slot_summary: dict[str, dict] = {}

    for representation in REPRESENTATIONS:
        for domain in DOMAINS:
            for evaluation_seed in EVALUATION_SEEDS:
                heldout_path = _find_heldout(heldout_root, domain, int(evaluation_seed))
                descriptors = load_heldout_v3_artifact(
                    heldout_path,
                    expected_domain=domain,
                    expected_evaluation_seed=int(evaluation_seed),
                    expected_count=2048,
                )[:POLICY_COUNT]
                observations = [item.observation_v3 for item in descriptors]
                legal_sets = [item.legal_slots for item in descriptors]
                left = policies[(representation, domain, seed_a)].batch_probabilities(observations, legal_sets)
                right = policies[(representation, domain, seed_b)].batch_probabilities(observations, legal_sets)

                local_rows: list[dict] = []
                slot_abs = np.zeros(10, dtype=np.float64)
                slot_legal = np.zeros(10, dtype=np.int64)
                for descriptor, left_policy, right_policy in zip(descriptors, left, right):
                    a = np.asarray(left_policy, dtype=np.float64)
                    b = np.asarray(right_policy, dtype=np.float64)
                    abs_delta = np.abs(a - b)
                    tv = float(0.5 * abs_delta.sum())
                    slot_abs += abs_delta
                    for slot in descriptor.legal_slots:
                        slot_legal[int(slot)] += 1
                    decoded = decode_spnniv3(descriptor.observation_v3)
                    features = _decode_features(decoded, action_path_len=len(descriptor.action_path))
                    row = {
                        "representation": representation,
                        "domain": domain,
                        "evaluation_seed": int(evaluation_seed),
                        "state_index": int(descriptor.state_index),
                        "hand_index": int(descriptor.hand_index),
                        "scenario_index": int(descriptor.scenario_index),
                        "actor": int(descriptor.actor),
                        "legal_slots": [int(value) for value in descriptor.legal_slots],
                        "action_path": [int(value) for value in descriptor.action_path],
                        "tv": tv,
                        "left_policy": [float(value) for value in left_policy],
                        "right_policy": [float(value) for value in right_policy],
                        "abs_delta_by_slot": [float(value) for value in abs_delta],
                        "dominant_delta_slot": int(np.argmax(abs_delta)),
                        **features,
                    }
                    local_rows.append(row)

                _projection_group_enrichment(local_rows)
                state_rows.extend(local_rows)
                key = f"{representation}|{domain}|{int(evaluation_seed)}"
                total_abs = float(slot_abs.sum())
                action_slot_summary[key] = {
                    "sum_absolute_probability_delta_by_slot": [float(value) for value in slot_abs],
                    "share_of_total_l1_by_slot": [
                        float(value / total_abs) if total_abs > 0 else 0.0 for value in slot_abs
                    ],
                    "legal_state_count_by_slot": [int(value) for value in slot_legal],
                }
                correlations = [
                    _spearman(local_rows, feature)
                    for feature in (
                        "history_len",
                        "action_path_len",
                        "unique_history_actors",
                        "forced_count",
                        "history_paid_over_pot_mean",
                        "history_paid_over_pot_std",
                        "history_paid_over_pot_max",
                        "pot_bb",
                        "to_call_bb",
                        "current_bet_bb",
                        "min_positive_stack_bb",
                        "spr",
                        "v1_projection_exact_variants",
                        "structured_projection_exact_variants",
                    )
                ]
                row_summaries.append({
                    "representation": representation,
                    "domain": domain,
                    "evaluation_seed": int(evaluation_seed),
                    "tv": _summary(row["tv"] for row in local_rows),
                    "by_street": _grouped_tv(local_rows, "street_name"),
                    "by_history_len": _grouped_tv(local_rows, "history_len", lambda value: _history_bins(int(value))),
                    "by_action_path_len": _grouped_tv(local_rows, "action_path_len", lambda value: _history_bins(int(value))),
                    "by_legal_slot_count": _grouped_tv(local_rows, "legal_slots", lambda value: len(value)),
                    "by_spr": _grouped_tv(local_rows, "spr", _spr_bin),
                    "by_last_action_type": _grouped_tv(local_rows, "last_action_type"),
                    "by_v1_projection_exact_variants": _grouped_tv(
                        local_rows,
                        "v1_projection_exact_variants",
                        lambda value: _variant_bins(int(value)),
                    ),
                    "by_structured_projection_exact_variants": _grouped_tv(
                        local_rows,
                        "structured_projection_exact_variants",
                        lambda value: _variant_bins(int(value)),
                    ),
                    "spearman_tv_correlations": correlations,
                    "top_25_state_indices_by_tv": [
                        int(row["state_index"])
                        for row in sorted(local_rows, key=lambda item: float(item["tv"]), reverse=True)[:25]
                    ],
                })
    return state_rows, row_summaries, action_slot_summary


def _memory_projection_sets(items: Sequence[object]) -> tuple[set[str], set[str], set[str], Counter]:
    exact: set[str] = set()
    v1: set[str] = set()
    structured: set[str] = set()
    iterations: Counter = Counter()
    for item in items:
        observation = bytes(item.observation)
        exact.add(hashlib.sha256(observation).hexdigest())
        decoded = decode_spnniv3(observation)
        v1.add(_digest(_v1_history_projection(decoded)))
        structured.add(_digest(_structured_history_projection(decoded)))
        iterations[int(item.iteration)] += 1
    return exact, v1, structured, iterations


def _memory_single(state: dict) -> tuple[dict, dict[str, set[str]]]:
    items = list(state.get("items") or [])
    capacity = int(state.get("capacity", 0))
    seen = int(state.get("seen", 0))
    exact, v1, structured, iterations = _memory_projection_sets(items)
    retained = len(items)
    return {
        "capacity": capacity,
        "seen": seen,
        "retained": retained,
        "saturation_factor_seen_over_capacity": float(seen / capacity) if capacity > 0 else None,
        "retention_fraction_retained_over_seen": float(retained / seen) if seen > 0 else None,
        "unique_exact_observations": len(exact),
        "exact_duplicate_fraction": float(1.0 - len(exact) / retained) if retained > 0 else 0.0,
        "unique_v1_history_projections": len(v1),
        "unique_structured_history_projections": len(structured),
        "retained_by_iteration": {str(key): int(value) for key, value in sorted(iterations.items())},
    }, {"exact": exact, "v1": v1, "structured": structured}


def _jaccard(left: set[str], right: set[str]) -> dict:
    union = left | right
    inter = left & right
    return {
        "left_unique": len(left),
        "right_unique": len(right),
        "intersection": len(inter),
        "union": len(union),
        "jaccard": float(len(inter) / len(union)) if union else 1.0,
    }


def _reservoir_readout(checkpoints: dict, training_execution_sha: str) -> tuple[list[dict], list[dict]]:
    per_cell: list[dict] = []
    sets_by_key: dict[tuple[str, str, int, str], dict[str, set[str]]] = {}
    for key, checkpoint in sorted(checkpoints.items()):
        representation, domain, seed = key
        torch_rng = torch.get_rng_state().clone()
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        finally:
            torch.set_rng_state(torch_rng)
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise RuntimeError(f"wrong checkpoint schema {checkpoint}")
        if payload.get("execution_sha") != str(training_execution_sha):
            raise RuntimeError(f"checkpoint execution SHA mismatch {checkpoint}")
        if payload.get("architecture_fingerprint_sha256") != MODEL_FINGERPRINTS[representation]:
            raise RuntimeError(f"checkpoint model fingerprint mismatch {checkpoint}")
        for memory_name, payload_key in (("ADVANTAGE", "adv_mem"), ("STRATEGY", "pol_mem")):
            summary, projection_sets = _memory_single(dict(payload[payload_key]))
            per_cell.append({
                "representation": representation,
                "domain": domain,
                "training_seed": int(seed),
                "memory": memory_name,
                **summary,
            })
            sets_by_key[(representation, domain, int(seed), memory_name)] = projection_sets

    seed_a, seed_b = map(int, TRAINING_SEEDS)
    overlaps: list[dict] = []
    for representation in REPRESENTATIONS:
        for domain in DOMAINS:
            for memory_name in ("ADVANTAGE", "STRATEGY"):
                left = sets_by_key[(representation, domain, seed_a, memory_name)]
                right = sets_by_key[(representation, domain, seed_b, memory_name)]
                overlaps.append({
                    "representation": representation,
                    "domain": domain,
                    "memory": memory_name,
                    "training_seed_pair": [seed_a, seed_b],
                    "exact_observation_overlap": _jaccard(left["exact"], right["exact"]),
                    "v1_history_projection_overlap": _jaccard(left["v1"], right["v1"]),
                    "structured_history_projection_overlap": _jaccard(left["structured"], right["structured"]),
                })
    return per_cell, overlaps


def _h3_minus_h2_readout(state_rows: Sequence[dict]) -> list[dict]:
    by_key = {
        (
            str(row["representation"]),
            str(row["domain"]),
            int(row["evaluation_seed"]),
            int(row["state_index"]),
        ): row
        for row in state_rows
    }
    output: list[dict] = []
    for domain in DOMAINS:
        for evaluation_seed in EVALUATION_SEEDS:
            deltas = []
            by_street: dict[str, list[float]] = defaultdict(list)
            for state_index in range(POLICY_COUNT):
                h2 = by_key[(H2_FINAL, domain, int(evaluation_seed), state_index)]
                h3 = by_key[(H3_FINAL, domain, int(evaluation_seed), state_index)]
                delta = float(h3["tv"]) - float(h2["tv"])
                deltas.append(delta)
                by_street[str(h2["street_name"])].append(delta)
            output.append({
                "domain": domain,
                "evaluation_seed": int(evaluation_seed),
                "meaning": "negative favors H3 stability; positive means H3 is less stable than H2 on the same heldout state",
                "h3_minus_h2_tv": _summary(deltas),
                "by_street": {name: _summary(values) for name, values in sorted(by_street.items())},
                "fraction_states_h3_more_stable": float(sum(value < 0.0 for value in deltas) / len(deltas)),
                "fraction_states_h3_less_stable": float(sum(value > 0.0 for value in deltas) / len(deltas)),
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only V1+/V3 x16 forensic post-mortem")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--input-root", type=Path, required=True, help="final x16 checkpoint tree")
    parser.add_argument("--heldout-root", type=Path, required=True, help="frozen heldout artifact tree")
    parser.add_argument("--training-execution-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    input_root = args.input_root.resolve()
    heldout_root = args.heldout_root.resolve()
    output = args.out.resolve()
    if not args.training_execution_sha.strip():
        raise SystemExit("--training-execution-sha is required")

    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("frozen torch thread contract was not applied")

    light_root = output.parent / "v1plus_forensic_light_policies"
    checkpoints, policies = _extract_inventory(
        repo_root=repo_root,
        input_root=input_root,
        output_root=light_root,
        training_execution_sha=str(args.training_execution_sha),
    )

    state_rows, stability_rows, action_slot_summary = _state_readout(
        policies=policies,
        heldout_root=heldout_root,
    )
    reservoir_cells, reservoir_overlaps = _reservoir_readout(
        checkpoints,
        str(args.training_execution_sha),
    )
    h3_minus_h2 = _h3_minus_h2_readout(state_rows)

    result = {
        "schema": SCHEMA,
        "status": "FORENSIC_READOUT_COMPLETE_NO_ARCHITECTURE_SELECTED",
        "purpose": "Explain x16 cross-seed instability before designing V1+. This tool performs no training and cannot authorize production.",
        "training_execution_sha": str(args.training_execution_sha),
        "policy_count_per_representation_domain_evaluation_seed": POLICY_COUNT,
        "training_seeds": [int(value) for value in TRAINING_SEEDS],
        "evaluation_seeds": [int(value) for value in EVALUATION_SEEDS],
        "representations": list(REPRESENTATIONS),
        "domains": list(DOMAINS),
        "state_level_stability_summaries": stability_rows,
        "action_slot_disagreement": action_slot_summary,
        "h3_minus_h2_paired_stability": h3_minus_h2,
        "reservoir_cells": reservoir_cells,
        "reservoir_cross_seed_overlap": reservoir_overlaps,
        "state_rows": state_rows,
        "interpretation_guardrails": [
            "Correlation is diagnostic and does not by itself prove causation.",
            "V1-like history projection is an information-content projection only; it is not byte-equivalent SPNNIV1 reconstruction.",
            "No H2/H3/V1+ winner is selected by this output.",
            "No stability threshold, seed, domain, action candidate, or trained weight is changed.",
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
        "state_rows": len(state_rows),
        "reservoir_cells": len(reservoir_cells),
        "reservoir_overlap_rows": len(reservoir_overlaps),
        "out": str(output),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
