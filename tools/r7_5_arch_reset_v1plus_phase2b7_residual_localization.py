from __future__ import annotations

"""Read-only Phase2B7 localization of residual Phase2B6 cross-seed instability."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

import r7_5_3d_v1plus_phase2a_strategy_capacity as phase2a
import r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot as b6
from spincore.r7_5_representation_v3 import H2_FINAL
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact
from spincore.r7_5_representation_v3_stage_contract import (
    CROSS_SEED_MEAN_TV_MAX,
    CROSS_SEED_P95_TV_MAX,
    EVALUATION_SEEDS,
    TRAINING_SEEDS,
    validate_phase2_v3_contract,
)

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B7_RESIDUAL_LOCALIZATION_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
PHASE2A_RESULT_SHA256 = "65f691e6b9cf7fbbddf88852c5ac6e0dcd2211af45f53cc4bb3e8271dbaa6149"
PHASE2B6_RESULT_SHA256 = "33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a"
PHASE2B6_EXECUTION_SHA = "4fa96434321c32efc734a55ae75982018ff2d091"
POLICY_COUNT = 1024
TAIL_TV = 0.35
DOMINANCE_MIN = 0.35
SCENARIO_TOP3_MIN = 0.50
REPRO_TOL = 1e-12

STREET_NAMES = {0: "PREFLOP", 1: "FLOP", 2: "TURN", 3: "RIVER"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _linear_quantile(values: Iterable[float], q: float) -> float | None:
    arr = np.asarray(list(values), dtype=np.float64)
    if not arr.size:
        return None
    return float(np.quantile(arr, q, method="linear"))


def _dist(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if not arr.size:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": _linear_quantile(arr, 0.50),
        "p95": _linear_quantile(arr, 0.95),
        "max": float(arr.max()),
    }


def _decode_observation(observation: bytes) -> dict:
    if len(observation) < 120 or not observation.startswith(b"SPNNIV3\x00"):
        raise RuntimeError("Phase2B7 requires authoritative SPNNIV3 bytes")
    history_count = int.from_bytes(observation[116:120], "little", signed=False)
    expected = 120 + 20 * history_count
    if len(observation) != expected:
        raise RuntimeError(f"Phase2B7 SPNNIV3 length drift: {len(observation)} != {expected}")
    street = int(observation[9])
    if street not in STREET_NAMES:
        raise RuntimeError(f"Phase2B7 invalid street {street}")
    nonforced_preflop = 0
    for index in range(history_count):
        offset = 120 + 20 * index
        event_street = int(observation[offset + 1])
        forced = int(observation[offset + 3])
        if event_street == 0 and forced == 0:
            nonforced_preflop += 1
    if street == 0:
        if nonforced_preflop == 0:
            region = "PREFLOP_ROOT"
        elif nonforced_preflop == 1:
            region = "PREFLOP_CONTINUATION_1"
        else:
            region = "PREFLOP_CONTINUATION_2PLUS"
    else:
        region = STREET_NAMES[street]
    return {
        "street": street,
        "street_name": STREET_NAMES[street],
        "history_count": history_count,
        "nonforced_preflop_count": nonforced_preflop,
        "region": region,
    }


def _path_bin(length: int) -> str:
    n = int(length)
    if n <= 3:
        return str(n)
    if n <= 5:
        return "4-5"
    return "6+"


def _history_bin(count: int) -> str:
    n = int(count)
    if n <= 3:
        return str(n)
    if n <= 5:
        return "4-5"
    if n <= 9:
        return "6-9"
    return "10+"


def _group_summary(rows: list[dict], *, total_tv_mass: float, total_tail: int) -> dict:
    baseline = [float(row["baseline_tv"]) for row in rows]
    pilot = [float(row["pilot_tv"]) for row in rows]
    bd = _dist(baseline)
    pd = _dist(pilot)
    base_mean = float(bd["mean"]) if bd["mean"] is not None else math.nan
    pilot_mean = float(pd["mean"]) if pd["mean"] is not None else math.nan
    improvement = base_mean - pilot_mean
    tail_count = sum(value > TAIL_TV for value in pilot)
    pilot_mass = float(sum(pilot))
    return {
        "count": len(rows),
        "baseline_tv": bd,
        "pilot_tv": pd,
        "absolute_mean_improvement": float(improvement),
        "relative_mean_improvement": float(improvement / base_mean) if base_mean > 0.0 else None,
        "pilot_tv_mass": pilot_mass,
        "pilot_tv_mass_share": float(pilot_mass / total_tv_mass) if total_tv_mass > 0.0 else 0.0,
        "pilot_tail_gt_035_count": int(tail_count),
        "pilot_tail_gt_035_fraction_within_group": float(tail_count / len(rows)) if rows else 0.0,
        "pilot_tail_gt_035_share_of_all_tail": float(tail_count / total_tail) if total_tail else 0.0,
    }


def _summarize_axis(rows: list[dict], key: str) -> dict:
    total_tv_mass = float(sum(float(row["pilot_tv"]) for row in rows))
    total_tail = sum(float(row["pilot_tv"]) > TAIL_TV for row in rows)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        name: _group_summary(group, total_tv_mass=total_tv_mass, total_tail=total_tail)
        for name, group in sorted(groups.items(), key=lambda kv: kv[0])
    }


def _broad_region_shares(region_groups: dict) -> dict:
    def val(name: str, field: str) -> float:
        return float((region_groups.get(name) or {}).get(field, 0.0))

    return {
        "ROOT": {
            "pilot_tv_mass_share": val("PREFLOP_ROOT", "pilot_tv_mass_share"),
            "pilot_tail_share": val("PREFLOP_ROOT", "pilot_tail_gt_035_share_of_all_tail"),
        },
        "PREFLOP_CONTINUATION": {
            "pilot_tv_mass_share": val("PREFLOP_CONTINUATION_1", "pilot_tv_mass_share") + val("PREFLOP_CONTINUATION_2PLUS", "pilot_tv_mass_share"),
            "pilot_tail_share": val("PREFLOP_CONTINUATION_1", "pilot_tail_gt_035_share_of_all_tail") + val("PREFLOP_CONTINUATION_2PLUS", "pilot_tail_gt_035_share_of_all_tail"),
        },
        "POSTFLOP": {
            "pilot_tv_mass_share": sum(val(name, "pilot_tv_mass_share") for name in ("FLOP", "TURN", "RIVER")),
            "pilot_tail_share": sum(val(name, "pilot_tail_gt_035_share_of_all_tail") for name in ("FLOP", "TURN", "RIVER")),
        },
    }


def _route_decision(region_groups: dict, scenario_groups: dict) -> dict:
    broad = _broad_region_shares(region_groups)
    eligible = []
    for name, row in broad.items():
        score = min(float(row["pilot_tv_mass_share"]), float(row["pilot_tail_share"]))
        row["dominance_score"] = score
        row["dominant_eligible"] = bool(
            float(row["pilot_tv_mass_share"]) >= DOMINANCE_MIN
            and float(row["pilot_tail_share"]) >= DOMINANCE_MIN
        )
        if row["dominant_eligible"]:
            eligible.append((score, name))
    if eligible:
        eligible.sort(reverse=True)
        winner = eligible[0][1]
        classification = {
            "ROOT": "ROOT_DOMINANT",
            "PREFLOP_CONTINUATION": "PREFLOP_CONTINUATION_DOMINANT",
            "POSTFLOP": "POSTFLOP_DOMINANT",
        }[winner]
    else:
        ranked = sorted(
            scenario_groups.items(),
            key=lambda kv: float(kv[1].get("pilot_tv_mass_share", 0.0)),
            reverse=True,
        )[:3]
        top3_mass = sum(float(row.get("pilot_tv_mass_share", 0.0)) for _, row in ranked)
        top3_tail = sum(float(row.get("pilot_tail_gt_035_share_of_all_tail", 0.0)) for _, row in ranked)
        if top3_mass >= SCENARIO_TOP3_MIN and top3_tail >= SCENARIO_TOP3_MIN:
            classification = "SCENARIO_CONCENTRATED"
        else:
            classification = "BROAD_MIXED_RESIDUAL"
    next_route = {
        "ROOT_DOMINANT": "PRECOMMIT_ROOT_PREFLOP_ANCHOR_OR_LAGGED_BEHAVIOR_SCREEN",
        "PREFLOP_CONTINUATION_DOMINANT": "PRECOMMIT_EARLY_PREFLOP_LAGGED_TARGET_OR_ANCHOR_SCREEN",
        "POSTFLOP_DOMINANT": "LOCALIZE_POSTFLOP_BY_STREET_AND_SUPPORT_BEFORE_TRAINING",
        "SCENARIO_CONCENTRATED": "PRECOMMIT_SCENARIO_STRATIFIED_CHANCE_SUPPORT_SCREEN",
        "BROAD_MIXED_RESIDUAL": "REASSESS_REPRESENTATION_AND_VARIANCE_WITHOUT_MORE_GLOBAL_DAMPING",
    }[classification]
    ranked = sorted(
        scenario_groups.items(),
        key=lambda kv: float(kv[1].get("pilot_tv_mass_share", 0.0)),
        reverse=True,
    )[:3]
    return {
        "classification": classification,
        "next_route": next_route,
        "broad_region_shares": broad,
        "top3_scenarios": [name for name, _ in ranked],
        "top3_scenario_tv_mass_share": float(sum(float(row.get("pilot_tv_mass_share", 0.0)) for _, row in ranked)),
        "top3_scenario_tail_share": float(sum(float(row.get("pilot_tail_gt_035_share_of_all_tail", 0.0)) for _, row in ranked)),
        "training_authorized": False,
        "higher_floor_training_authorized": False,
        "architecture_winner_selected": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def _expected_b6_metric(b6_result: dict, mode: str, evaluation_seed: int) -> dict:
    for row in b6_result["heldout_comparisons"]:
        if row["learner_mode"] == mode and int(row["evaluation_seed"]) == int(evaluation_seed):
            return row["pilot_phase2b6"]
    raise RuntimeError(f"Phase2B7 missing Phase2B6 expected metric {mode}/{evaluation_seed}")


def _load_and_validate_inputs(args) -> tuple[dict, dict, dict, list[dict]]:
    repo_root = Path(args.repo_root).resolve()
    phase2a_root = Path(args.phase2a_root).resolve()
    b6_root = Path(args.phase2b6_root).resolve()
    heldout_root = Path(args.heldout_root).resolve()
    phase2a_result_path = Path(args.phase2a_result).resolve()
    b6_result_path = Path(args.phase2b6_result).resolve()

    if _sha256(phase2a_result_path) != PHASE2A_RESULT_SHA256:
        raise RuntimeError("Phase2B7 exact Phase2A result SHA drift")
    if _sha256(b6_result_path) != PHASE2B6_RESULT_SHA256:
        raise RuntimeError("Phase2B7 exact Phase2B6 result SHA drift")
    b6_result = json.loads(b6_result_path.read_text(encoding="utf-8"))
    if b6_result.get("schema") != b6.SCHEMA or b6_result.get("status") != "PREFLOP_DAMPING_CAUSAL_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE":
        raise RuntimeError("Phase2B7 Phase2B6 schema/status mismatch")
    if b6_result.get("execution_sha") != PHASE2B6_EXECUTION_SHA:
        raise RuntimeError("Phase2B7 Phase2B6 execution SHA mismatch")
    if not bool((b6_result.get("decision") or {}).get("causal_effect_supported")):
        raise RuntimeError("Phase2B7 requires completed supported Phase2B6 causal effect")
    if (b6_result.get("decision") or {}).get("next_route") != "LOCALIZE_RESIDUAL_WITHOUT_ESCALATING_DAMPING_FLOOR":
        raise RuntimeError("Phase2B7 route mismatch")

    for seed in map(int, TRAINING_SEEDS):
        validate_phase2_v3_contract(repo_root, representation=REPRESENTATION, domain=DOMAIN, training_seed=seed)
    baseline_identity = b6._validate_phase2a_baseline(phase2a_root, phase2a_result_path)

    pilot_identity = []
    for seed in map(int, TRAINING_SEEDS):
        seed_result_path = b6_root / f"seed_{seed}" / "seed_result.json"
        seed_result = json.loads(seed_result_path.read_text(encoding="utf-8"))
        if seed_result.get("schema") != b6.SEED_SCHEMA or seed_result.get("status") != "SEED_COMPLETE":
            raise RuntimeError(f"Phase2B7 invalid Phase2B6 seed result {seed}")
        if seed_result.get("execution_sha") != PHASE2B6_EXECUTION_SHA:
            raise RuntimeError(f"Phase2B7 Phase2B6 seed execution drift {seed}")
        for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
            meta_path = b6_root / f"seed_{seed}" / "policies" / f"{mode}.json"
            artifact = b6_root / f"seed_{seed}" / "policies" / f"{mode}.pt"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            actual = _sha256(artifact)
            if meta.get("schema") != b6.POLICY_SCHEMA or meta.get("status") != "POLICY_FIT_COMPLETE":
                raise RuntimeError(f"Phase2B7 invalid pilot policy metadata {seed}/{mode}")
            if meta.get("artifact_sha256") != actual:
                raise RuntimeError(f"Phase2B7 pilot policy artifact SHA drift {seed}/{mode}")
            if float(meta.get("floor_training", -1.0)) != 0.25 or float(meta.get("floor_inference", -1.0)) != 0.0:
                raise RuntimeError(f"Phase2B7 pilot policy floor drift {seed}/{mode}")
            pilot_identity.append({"training_seed": seed, "learner_mode": mode, "sha256": actual})

    expected_heldout = {int(row["evaluation_seed"]): str(row["sha256"]) for row in b6_result["frozen_inputs"]["heldout"]}
    heldout_identity = []
    for evaluation_seed in map(int, EVALUATION_SEEDS):
        path = b6._find_heldout(heldout_root, evaluation_seed)
        actual = _sha256(path)
        if actual != expected_heldout[evaluation_seed]:
            raise RuntimeError(f"Phase2B7 heldout SHA drift {evaluation_seed}")
        heldout_identity.append({"evaluation_seed": evaluation_seed, "sha256": actual, "path": str(path)})
    return b6_result, baseline_identity, {"policy_artifacts": pilot_identity}, heldout_identity


def _analyze_mode(args, b6_result: dict, mode: str) -> tuple[dict, list[dict]]:
    phase2a_root = Path(args.phase2a_root).resolve()
    b6_root = Path(args.phase2b6_root).resolve()
    heldout_root = Path(args.heldout_root).resolve()
    seed_a, seed_b = map(int, TRAINING_SEEDS)
    baseline_models = {}
    pilot_models = {}
    for seed in (seed_a, seed_b):
        baseline_models[seed], _ = b6._load_baseline_policy(
            phase2a_root / f"seed_{seed}" / "policies" / f"{mode}__S100K_CONTROL.pt"
        )
        pilot_models[seed], _ = b6._load_pilot_policy(
            b6_root / f"seed_{seed}" / "policies" / f"{mode}.pt",
            training_seed=seed,
            mode=mode,
        )

    all_rows = []
    reproduction = []
    for evaluation_seed in map(int, EVALUATION_SEEDS):
        path = b6._find_heldout(heldout_root, evaluation_seed)
        descriptors = load_heldout_v3_artifact(
            path,
            expected_domain=DOMAIN,
            expected_evaluation_seed=evaluation_seed,
            expected_count=2048,
        )[:POLICY_COUNT]
        base_a = b6._probabilities_fixed(baseline_models[seed_a], descriptors)
        base_b = b6._probabilities_fixed(baseline_models[seed_b], descriptors)
        pilot_a = b6._probabilities_fixed(pilot_models[seed_a], descriptors)
        pilot_b = b6._probabilities_fixed(pilot_models[seed_b], descriptors)
        baseline_tv = b6._tv_vector(base_a, base_b)
        pilot_tv = b6._tv_vector(pilot_a, pilot_b)
        metric = b6.cross_seed_policy_stability(pilot_a, pilot_b)
        expected = _expected_b6_metric(b6_result, mode, evaluation_seed)
        mean_err = abs(float(metric["mean"]) - float(expected["mean"]))
        p95_err = abs(float(metric["p95"]) - float(expected["p95"]))
        if mean_err > REPRO_TOL or p95_err > REPRO_TOL:
            raise RuntimeError(
                f"Phase2B7 Phase2B6 metric reproduction drift {mode}/{evaluation_seed}: mean_err={mean_err} p95_err={p95_err}"
            )
        reproduction.append({
            "evaluation_seed": evaluation_seed,
            "mean": float(metric["mean"]),
            "p95": float(metric["p95"]),
            "mean_reproduction_abs_error": mean_err,
            "p95_reproduction_abs_error": p95_err,
        })
        for descriptor, btv, ptv in zip(descriptors, baseline_tv, pilot_tv):
            decoded = _decode_observation(descriptor.observation_v3)
            all_rows.append({
                "evaluation_seed": evaluation_seed,
                "state_index": int(descriptor.state_index),
                "scenario_index": int(descriptor.scenario_index),
                "actor": int(descriptor.actor),
                "action_path_length": len(descriptor.action_path),
                "path_bin": _path_bin(len(descriptor.action_path)),
                "legal_count": len(descriptor.legal_slots),
                "legal_slots": list(map(int, descriptor.legal_slots)),
                "history_count": int(decoded["history_count"]),
                "history_bin": _history_bin(decoded["history_count"]),
                "nonforced_preflop_count": int(decoded["nonforced_preflop_count"]),
                "street": int(decoded["street"]),
                "street_name": str(decoded["street_name"]),
                "region": str(decoded["region"]),
                "baseline_tv": float(btv),
                "pilot_tv": float(ptv),
                "baseline_minus_pilot": float(btv - ptv),
            })

    total_mass = float(sum(row["pilot_tv"] for row in all_rows))
    total_tail = sum(row["pilot_tv"] > TAIL_TV for row in all_rows)
    overall = _group_summary(all_rows, total_tv_mass=total_mass, total_tail=total_tail)
    groups = {
        "region": _summarize_axis(all_rows, "region"),
        "street": _summarize_axis(all_rows, "street_name"),
        "actor": _summarize_axis(all_rows, "actor"),
        "scenario_index": _summarize_axis(all_rows, "scenario_index"),
        "action_path_length_bin": _summarize_axis(all_rows, "path_bin"),
        "legal_action_count": _summarize_axis(all_rows, "legal_count"),
        "history_count_bin": _summarize_axis(all_rows, "history_bin"),
    }
    return {
        "reproduced_phase2b6_heldouts": reproduction,
        "overall": overall,
        "groups": groups,
    }, all_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Phase2B7 residual localization")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--phase2a-root", type=Path, required=True)
    parser.add_argument("--phase2a-result", type=Path, required=True)
    parser.add_argument("--phase2b6-root", type=Path, required=True)
    parser.add_argument("--phase2b6-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(2)
    b6_result, baseline_identity, pilot_identity, heldout_identity = _load_and_validate_inputs(args)
    mode_results = {}
    common_rows = None
    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        summary, rows = _analyze_mode(args, b6_result, mode)
        mode_results[mode] = summary
        if mode == "COMMON_LEARNER":
            common_rows = rows
    if common_rows is None:
        raise RuntimeError("Phase2B7 missing COMMON rows")

    common_groups = mode_results["COMMON_LEARNER"]["groups"]
    decision = _route_decision(common_groups["region"], common_groups["scenario_index"])
    top_states = sorted(common_rows, key=lambda row: row["pilot_tv"], reverse=True)[:50]

    result = {
        "schema": SCHEMA,
        "status": decision["classification"],
        "purpose": "Read-only localization of residual Phase2B6 AveragePolicy cross-seed instability; no training authorization.",
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "training_seeds": list(map(int, TRAINING_SEEDS)),
        "evaluation_seeds": list(map(int, EVALUATION_SEEDS)),
        "policy_count_per_evaluation_seed": POLICY_COUNT,
        "tail_tv_reference": TAIL_TV,
        "hard_stability_reference": {
            "mean_tv_max": CROSS_SEED_MEAN_TV_MAX,
            "p95_tv_max": CROSS_SEED_P95_TV_MAX,
        },
        "frozen_inputs": {
            "phase2a_result_sha256": PHASE2A_RESULT_SHA256,
            "phase2b6_result_sha256": PHASE2B6_RESULT_SHA256,
            "phase2b6_execution_sha": PHASE2B6_EXECUTION_SHA,
            "phase2a_baseline": baseline_identity,
            "phase2b6_policies": pilot_identity,
            "heldout": heldout_identity,
        },
        "learner_modes": mode_results,
        "top50_common_residual_states": top_states,
        "decision": decision,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    _atomic_json(result, Path(args.out).resolve())
    print(json.dumps({
        "status": result["status"],
        "common_pilot_mean_tv": result["learner_modes"]["COMMON_LEARNER"]["overall"]["pilot_tv"]["mean"],
        "common_pilot_p95_tv": result["learner_modes"]["COMMON_LEARNER"]["overall"]["pilot_tv"]["p95"],
        "broad_region_shares": decision["broad_region_shares"],
        "top3_scenarios": decision["top3_scenarios"],
        "next_route": decision["next_route"],
        "out": str(Path(args.out).resolve()),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
