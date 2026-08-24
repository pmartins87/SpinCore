from __future__ import annotations

"""Phase2B14: read-only localization after the sub-material Phase2B13 root-IID64 pilot."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

import r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot as b6
import r7_5_arch_reset_v1plus_phase2b7_residual_localization as b7
import r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training as b13
from spincore.r7_5_representation_v3 import H2_FINAL
from spincore.r7_5_representation_v3_phase2_eval import cross_seed_policy_stability
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact
from spincore.r7_5_representation_v3_stage_contract import (
    EVALUATION_SEEDS,
    TRAINING_SEEDS,
    validate_phase2_v3_contract,
)

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B14_B13_RESIDUAL_LOCALIZATION_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
B13_RESULT_SHA256 = "6de7996282236d34adf5e8e53416fd8a443a1fbf5abc89fc807492d0cb3dbf80"
B13_EXECUTION_SHA = "2cd7d1ece46a20d2b8937fe5135a415f6bbe54c2"
POLICY_COUNT = 1024
REPRO_TOL = 1e-12
TAIL_TV = 0.35
DOMINANCE_MIN = 0.35
SCENARIO_TOP3_MIN = 0.50
CONTROL_ARM = b13.CONTROL_ARM
CANDIDATE_ARM = b13.CANDIDATE_ARM


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


def _expected_row(result: dict, mode: str, evaluation_seed: int) -> dict:
    for row in result.get("heldout_comparisons") or []:
        if row.get("learner_mode") == mode and int(row.get("evaluation_seed", -1)) == int(evaluation_seed):
            return row
    raise RuntimeError(f"Phase2B14 missing Phase2B13 comparison {mode}/{evaluation_seed}")


def _validate_b13_result(path: Path) -> dict:
    actual = _sha256(path)
    if actual != B13_RESULT_SHA256:
        raise RuntimeError(f"Phase2B14 exact Phase2B13 result SHA drift: {actual}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema") != b13.SCHEMA:
        raise RuntimeError("Phase2B14 Phase2B13 schema mismatch")
    if result.get("status") != "ROOT_IID64_TRAINING_EFFECT_NOT_SUPPORTED":
        raise RuntimeError("Phase2B14 requires exact failed Phase2B13 causal screen")
    if result.get("execution_sha") != B13_EXECUTION_SHA:
        raise RuntimeError("Phase2B14 Phase2B13 execution SHA mismatch")
    decision = dict(result.get("decision") or {})
    if bool(decision.get("causal_effect_supported")):
        raise RuntimeError("Phase2B14 expected Phase2B13 causal_effect_supported=false")
    if bool(decision.get("common_materiality_pass")):
        raise RuntimeError("Phase2B14 expected Phase2B13 materiality gate failure")
    if decision.get("next_route") != "REASSESS_CONTINUATION_CONDITIONAL_CHANCE_OR_REPRESENTATION_SUPPORT_NO_SCALEUP":
        raise RuntimeError("Phase2B14 Phase2B13 route mismatch")
    return result


def _load_policy_pair(root: Path, arm: str, mode: str) -> tuple[object, object, list[dict]]:
    models = []
    identities = []
    for seed in map(int, TRAINING_SEEDS):
        seed_root = root / arm / f"seed_{seed}"
        seed_result_path = seed_root / "seed_result.json"
        meta_path = seed_root / "policies" / f"{mode}.json"
        artifact = seed_root / "policies" / f"{mode}.pt"
        if not seed_result_path.is_file() or not meta_path.is_file() or not artifact.is_file():
            raise RuntimeError(f"Phase2B14 missing Phase2B13 local artifact {arm}/{seed}/{mode}")
        seed_result = json.loads(seed_result_path.read_text(encoding="utf-8"))
        if (
            seed_result.get("schema") != b13.SEED_SCHEMA
            or seed_result.get("status") != "SEED_COMPLETE"
            or seed_result.get("execution_sha") != B13_EXECUTION_SHA
            or seed_result.get("arm") != arm
            or int(seed_result.get("training_seed", -1)) != seed
            or int(seed_result.get("roots", -1)) != b13.TOTAL_ROOTS
        ):
            raise RuntimeError(f"Phase2B14 invalid B13 seed result {arm}/{seed}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        artifact_sha = _sha256(artifact)
        if (
            meta.get("schema") != b13.POLICY_SCHEMA
            or meta.get("status") != "POLICY_FIT_COMPLETE"
            or meta.get("execution_sha") != B13_EXECUTION_SHA
            or meta.get("arm") != arm
            or int(meta.get("training_seed", -1)) != seed
            or meta.get("learner_mode") != mode
            or int(meta.get("k", -1)) != b13.K
            or meta.get("artifact_sha256") != artifact_sha
        ):
            raise RuntimeError(f"Phase2B14 policy metadata/hash drift {arm}/{seed}/{mode}")
        model, payload = b13._load_policy(
            artifact,
            arm=arm,
            training_seed=seed,
            mode=mode,
            execution_sha=B13_EXECUTION_SHA,
        )
        if float(payload.get("floor_training", -1.0)) != 0.25 or float(payload.get("floor_inference", -1.0)) != 0.0:
            raise RuntimeError("Phase2B14 B13 policy floor identity drift")
        models.append(model)
        identities.append({
            "arm": arm,
            "training_seed": seed,
            "learner_mode": mode,
            "artifact_sha256": artifact_sha,
        })
    return models[0], models[1], identities


def _region_summary(rows: list[dict]) -> dict:
    total_mass = float(sum(float(row["pilot_tv"]) for row in rows))
    total_tail = sum(float(row["pilot_tv"]) > TAIL_TV for row in rows)
    return b7._group_summary(rows, total_tv_mass=total_mass, total_tail=total_tail)


def _route(common_rows: list[dict], region_groups: dict, scenario_groups: dict) -> dict:
    broad = b7._broad_region_shares(region_groups)
    eligible = []
    for name, row in broad.items():
        row["dominance_score"] = min(float(row["pilot_tv_mass_share"]), float(row["pilot_tail_share"]))
        row["dominant_eligible"] = bool(
            float(row["pilot_tv_mass_share"]) >= DOMINANCE_MIN
            and float(row["pilot_tail_share"]) >= DOMINANCE_MIN
        )
        if row["dominant_eligible"]:
            eligible.append((float(row["dominance_score"]), name))

    root_by_eval = {}
    root_improvements = []
    for evaluation_seed in map(int, EVALUATION_SEEDS):
        subset = [
            row for row in common_rows
            if int(row["evaluation_seed"]) == evaluation_seed and row["region"] == "PREFLOP_ROOT"
        ]
        if not subset:
            raise RuntimeError(f"Phase2B14 no root states for heldout {evaluation_seed}")
        control_mean = sum(float(row["baseline_tv"]) for row in subset) / len(subset)
        candidate_mean = sum(float(row["pilot_tv"]) for row in subset) / len(subset)
        improvement = control_mean - candidate_mean
        root_by_eval[str(evaluation_seed)] = {
            "count": len(subset),
            "control_mean_tv": float(control_mean),
            "candidate_mean_tv": float(candidate_mean),
            "improvement": float(improvement),
        }
        root_improvements.append(float(improvement))
    pooled_root_improvement = sum(root_improvements) / len(root_improvements)
    root_effect_consistent = bool(all(value > 0.0 for value in root_improvements) and pooled_root_improvement > 0.0)

    if eligible:
        eligible.sort(reverse=True)
        winner = eligible[0][1]
        if winner == "PREFLOP_CONTINUATION":
            if root_effect_consistent:
                classification = "PREFLOP_CONTINUATION_RESIDUAL_DOMINANT_AFTER_ROOT_IID64"
                next_route = "PRECOMMIT_POSTERIOR_WEIGHTED_PREFLOP_CONTINUATION_CHANCE_SCREEN"
            else:
                classification = "PREFLOP_CONTINUATION_DOMINANT_ROOT_EFFECT_NOT_LOCALIZED"
                next_route = "REASSESS_REPRESENTATION_SUPPORT_BEFORE_MORE_CHANCE_INTEGRATION"
        elif winner == "ROOT":
            classification = "ROOT_RESIDUAL_DOMINANT_AFTER_IID64"
            next_route = "REASSESS_ROOT_ESTIMATOR_OR_REPRESENTATION_SUPPORT_NO_SCALEUP"
        else:
            classification = "POSTFLOP_RESIDUAL_DOMINANT_AFTER_ROOT_IID64"
            next_route = "LOCALIZE_POSTFLOP_SUPPORT_BEFORE_ANY_NEW_TRAINING"
    else:
        ranked = sorted(
            scenario_groups.items(),
            key=lambda kv: float(kv[1].get("pilot_tv_mass_share", 0.0)),
            reverse=True,
        )[:3]
        top3_mass = sum(float(row.get("pilot_tv_mass_share", 0.0)) for _, row in ranked)
        top3_tail = sum(float(row.get("pilot_tail_gt_035_share_of_all_tail", 0.0)) for _, row in ranked)
        if top3_mass >= SCENARIO_TOP3_MIN and top3_tail >= SCENARIO_TOP3_MIN:
            classification = "SCENARIO_CONCENTRATED_RESIDUAL_AFTER_ROOT_IID64"
            next_route = "PRECOMMIT_SCENARIO_STRATIFIED_SUPPORT_SCREEN"
        else:
            classification = "BROAD_MIXED_RESIDUAL_AFTER_ROOT_IID64"
            next_route = "REASSESS_REPRESENTATION_SUPPORT_AND_VARIANCE_NO_SCALEUP"

    ranked = sorted(
        scenario_groups.items(),
        key=lambda kv: float(kv[1].get("pilot_tv_mass_share", 0.0)),
        reverse=True,
    )[:3]
    return {
        "classification": classification,
        "next_route": next_route,
        "broad_region_shares": broad,
        "root_effect_consistent": root_effect_consistent,
        "root_by_evaluation_seed": root_by_eval,
        "pooled_root_mean_improvement": float(pooled_root_improvement),
        "top3_scenarios": [name for name, _ in ranked],
        "top3_scenario_tv_mass_share": float(sum(float(row.get("pilot_tv_mass_share", 0.0)) for _, row in ranked)),
        "top3_scenario_tail_share": float(sum(float(row.get("pilot_tail_gt_035_share_of_all_tail", 0.0)) for _, row in ranked)),
        "training_authorized": False,
        "full_x4_confirmation_authorized": False,
        "architecture_winner_selected": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def run(args) -> dict:
    repo_root = Path(args.repo_root).resolve()
    b13_root = Path(args.phase2b13_root).resolve()
    b13_result_path = Path(args.phase2b13_result).resolve()
    heldout_root = Path(args.heldout_root).resolve()
    b13_result = _validate_b13_result(b13_result_path)

    for seed in map(int, TRAINING_SEEDS):
        validate_phase2_v3_contract(
            repo_root,
            representation=REPRESENTATION,
            domain=DOMAIN,
            training_seed=seed,
        )
    torch.set_num_threads(2)

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
            raise RuntimeError("Phase2B14 heldout count drift")
        descriptors[evaluation_seed] = rows
        heldout_identity.append({
            "evaluation_seed": evaluation_seed,
            "path": str(heldout),
            "sha256": _sha256(heldout),
        })

    modes = {}
    reproduction = []
    policy_identity = []
    common_rows = None
    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        control_a, control_b, ids = _load_policy_pair(b13_root, CONTROL_ARM, mode)
        policy_identity.extend(ids)
        candidate_a, candidate_b, ids = _load_policy_pair(b13_root, CANDIDATE_ARM, mode)
        policy_identity.extend(ids)
        all_rows = []
        for evaluation_seed in map(int, EVALUATION_SEEDS):
            desc = descriptors[evaluation_seed]
            control_left = b6._probabilities_fixed(control_a, desc)
            control_right = b6._probabilities_fixed(control_b, desc)
            candidate_left = b6._probabilities_fixed(candidate_a, desc)
            candidate_right = b6._probabilities_fixed(candidate_b, desc)
            control_metric = cross_seed_policy_stability(control_left, control_right)
            candidate_metric = cross_seed_policy_stability(candidate_left, candidate_right)
            expected = _expected_row(b13_result, mode, evaluation_seed)
            checks = {
                "control_mean": abs(float(control_metric["mean"]) - float(expected["control"]["mean"])),
                "control_p95": abs(float(control_metric["p95"]) - float(expected["control"]["p95"])),
                "candidate_mean": abs(float(candidate_metric["mean"]) - float(expected["candidate"]["mean"])),
                "candidate_p95": abs(float(candidate_metric["p95"]) - float(expected["candidate"]["p95"])),
            }
            if max(checks.values()) > REPRO_TOL:
                raise RuntimeError(
                    f"Phase2B14 Phase2B13 metric reproduction failed {mode}/{evaluation_seed}: {checks}"
                )
            reproduction.append({
                "learner_mode": mode,
                "evaluation_seed": evaluation_seed,
                "absolute_errors": checks,
                "pass": True,
            })
            control_tv = b6._tv_vector(control_left, control_right)
            candidate_tv = b6._tv_vector(candidate_left, candidate_right)
            for descriptor, ctv, ktv in zip(desc, control_tv, candidate_tv):
                decoded = b7._decode_observation(descriptor.observation_v3)
                all_rows.append({
                    "evaluation_seed": evaluation_seed,
                    "state_index": int(descriptor.state_index),
                    "scenario_index": int(descriptor.scenario_index),
                    "actor": int(descriptor.actor),
                    "action_path_length": len(descriptor.action_path),
                    "path_bin": b7._path_bin(len(descriptor.action_path)),
                    "legal_count": len(descriptor.legal_slots),
                    "history_bin": b7._history_bin(int(decoded["history_count"])),
                    "baseline_tv": float(ctv),
                    "pilot_tv": float(ktv),
                    "control_tv": float(ctv),
                    "candidate_tv": float(ktv),
                    "improvement_control_minus_candidate": float(ctv - ktv),
                    **decoded,
                })
        total_mass = float(sum(float(row["pilot_tv"]) for row in all_rows))
        total_tail = sum(float(row["pilot_tv"]) > TAIL_TV for row in all_rows)
        groups = {
            "region": b7._summarize_axis(all_rows, "region"),
            "street": b7._summarize_axis(all_rows, "street_name"),
            "actor": b7._summarize_axis(all_rows, "actor"),
            "scenario_index": b7._summarize_axis(all_rows, "scenario_index"),
            "action_path_length_bin": b7._summarize_axis(all_rows, "path_bin"),
            "legal_action_count": b7._summarize_axis(all_rows, "legal_count"),
            "history_count_bin": b7._summarize_axis(all_rows, "history_bin"),
        }
        modes[mode] = {
            "overall": b7._group_summary(all_rows, total_tv_mass=total_mass, total_tail=total_tail),
            "groups": groups,
        }
        if mode == "COMMON_LEARNER":
            common_rows = all_rows

    if common_rows is None:
        raise RuntimeError("Phase2B14 missing COMMON rows")
    common_groups = modes["COMMON_LEARNER"]["groups"]
    decision = _route(common_rows, common_groups["region"], common_groups["scenario_index"])
    top_states = sorted(common_rows, key=lambda row: float(row["candidate_tv"]), reverse=True)[:50]

    return {
        "schema": SCHEMA,
        "status": decision["classification"],
        "source_phase2b13_result_sha256": B13_RESULT_SHA256,
        "source_phase2b13_execution_sha": B13_EXECUTION_SHA,
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "policy_count_per_heldout": POLICY_COUNT,
        "tail_threshold": TAIL_TV,
        "reproduction_tolerance": REPRO_TOL,
        "reproduction": reproduction,
        "frozen_inputs": {
            "phase2b13_result_sha256": B13_RESULT_SHA256,
            "heldout": heldout_identity,
            "policy_artifacts": policy_identity,
        },
        "learner_modes": modes,
        "decision": decision,
        "top_common_candidate_tail_states": top_states,
        "training_authorized": False,
        "full_x4_confirmation_authorized": False,
        "architecture_winner_selected": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5 Phase2B14 read-only B13 residual localization")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--phase2b13-root", type=Path, required=True)
    parser.add_argument("--phase2b13-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    _atomic_json(result, Path(args.out).resolve())
    print(json.dumps({
        "status": result["status"],
        "common_candidate_mean_tv": result["learner_modes"]["COMMON_LEARNER"]["overall"]["pilot_tv"]["mean"],
        "common_candidate_p95_tv": result["learner_modes"]["COMMON_LEARNER"]["overall"]["pilot_tv"]["p95"],
        "broad_region_shares": result["decision"]["broad_region_shares"],
        "root_effect_consistent": result["decision"]["root_effect_consistent"],
        "pooled_root_mean_improvement": result["decision"]["pooled_root_mean_improvement"],
        "next_route": result["decision"]["next_route"],
        "out": str(Path(args.out).resolve()),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
