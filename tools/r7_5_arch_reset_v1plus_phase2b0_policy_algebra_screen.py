from __future__ import annotations

"""Read-only Phase 2B0 screen for a variance-to-policy feedback algebra change.

No solver traversal, optimizer step, reservoir replay or model mutation occurs.
The candidate averages the four frozen raw Advantage vectors before regret
matching and then reuses the exact control uncertainty epsilon for damping.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from spincore.r7_5_action_cfr import legal_mask, regret_matching_policy
from spincore.r7_5_action_uncertainty import uncertainty_damped_policy_from_advantages
from spincore.r7_5_representation_v3 import H2_FINAL
from spincore.r7_5_representation_v3_checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    EVALUATION_SEEDS,
    MODEL_FINGERPRINTS,
    TORCH_THREADS,
    TRAINING_SEEDS,
)
from spincore_nn.codec_v3 import decode_spnniv3
from spincore_nn.models_v3_final import collate_v3_observations, make_h2_final_v3

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B0_POLICY_ALGEBRA_SCREEN_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
SOURCE_EXECUTION_SHA = "4bfa55d69029cd69536fa6dbfcadd162719cb887"
PHASE2A_EXTRA_SCHEMA = "SPINCORE_R7_5_3D_V1PLUS_PHASE2A_RESUME_V1"
EXPECTED_ROOTS = 768
EXPECTED_STAGE_INDEX = 12
ENSEMBLE_SIZE = 4
POLICY_COUNT = 1024
EPSILON_SCALE = 1.75
EPSILON_CAP = 0.5
STREET_NAMES = {0: "PREFLOP", 1: "FLOP", 2: "TURN", 3: "RIVER"}


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q), method="linear"))


def _summary(values: Sequence[float]) -> dict:
    rows = [float(value) for value in values]
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


def _policy_tv(left: Sequence[float], right: Sequence[float]) -> float:
    return float(0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right)))


def raw_mean_then_regret_match_same_epsilon(
    member_advantages: Sequence[Sequence[float]],
    legal: tuple[int, ...],
) -> tuple[tuple[float, ...], tuple[float, ...], dict]:
    """Return control, candidate and control stats for one state.

    The control path calls the canonical frozen helper.  Candidate exploitation
    uses regret matching on the arithmetic raw-Advantage ensemble mean, then
    mixes toward the same uniform policy using the *identical* control epsilon.
    """
    control, stats = uncertainty_damped_policy_from_advantages(
        member_advantages,
        legal,
        action_count=10,
        epsilon_scale=EPSILON_SCALE,
        epsilon_cap=EPSILON_CAP,
    )
    if len(member_advantages) != ENSEMBLE_SIZE:
        raise ValueError("Phase2B0 requires exactly four ensemble members")
    raw_mean = [
        sum(float(row[slot]) for row in member_advantages) / len(member_advantages)
        for slot in range(10)
    ]
    exploit = regret_matching_policy(raw_mean, legal)
    epsilon = float(stats["epsilon"])
    uniform = {slot: 1.0 / len(legal) for slot in legal}
    candidate = [0.0] * 10
    for slot in legal:
        candidate[slot] = (1.0 - epsilon) * float(exploit[slot]) + epsilon * float(uniform[slot])
    total = sum(candidate[slot] for slot in legal)
    if total <= 0.0:
        raise RuntimeError("candidate policy lost legal probability mass")
    for slot in legal:
        candidate[slot] /= total
    return tuple(control), tuple(candidate), {
        **dict(stats),
        "raw_mean": [float(value) for value in raw_mean],
        "raw_mean_positive_legal_count": int(sum(raw_mean[slot] > 0.0 for slot in legal)),
    }


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


def _load_ensemble(input_root: Path, seed: int, source_sha: str) -> list[object]:
    checkpoint = input_root / f"seed_{int(seed)}" / "resume_checkpoint.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"missing Phase2A checkpoint for seed {seed}: {checkpoint}")
    rng = torch.get_rng_state().clone()
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    finally:
        torch.set_rng_state(rng)
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "seed": int(seed),
        "action_candidate": ACTION_CANDIDATE,
        "execution_sha": str(source_sha),
        "architecture_fingerprint_sha256": MODEL_FINGERPRINTS[REPRESENTATION],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Phase2B0 checkpoint identity mismatch {seed}/{key}")
    progress = dict(payload.get("progress") or {})
    extra = dict(payload.get("extra") or {})
    if progress.get("phase") != "phase2a_resume" or int(progress.get("global_root", -1)) != EXPECTED_ROOTS:
        raise RuntimeError(f"Phase2B0 incomplete Phase2A checkpoint for seed {seed}")
    if extra.get("schema") != PHASE2A_EXTRA_SCHEMA or int(extra.get("stage_index", -1)) != EXPECTED_STAGE_INDEX:
        raise RuntimeError(f"Phase2B0 checkpoint extra mismatch for seed {seed}")
    if bool(payload.get("production_training_authorized")) or bool(payload.get("ready_for_tables")):
        raise RuntimeError("source checkpoint illegally authorizes production/table use")
    states = list(extra.get("behavior_model_states") or [])
    if len(states) != ENSEMBLE_SIZE:
        raise RuntimeError(f"Phase2B0 expected four behavior model states for seed {seed}, got {len(states)}")
    models = []
    for index, state in enumerate(states):
        _cfg, model = make_h2_final_v3(device="cpu", seed=0x2B0000 + index)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def _member_outputs(models: Sequence[object], descriptors: Sequence[object]) -> list[list[list[float]]]:
    # state-major -> member-major -> slot
    output: list[list[list[float]]] = []
    for start in range(0, len(descriptors), 256):
        rows = list(descriptors[start : start + 256])
        batch = collate_v3_observations(
            [item.observation_v3 for item in rows],
            [legal_mask(item.legal_slots) for item in rows],
            with_semantics=False,
            device="cpu",
        )
        member_batches = []
        for model in models:
            with torch.no_grad():
                member_batches.append(model(batch).detach().cpu().tolist())
        for row_index in range(len(rows)):
            output.append([
                [float(value) for value in member_batches[member][row_index]]
                for member in range(len(models))
            ])
    return output


def _mode_summary(rows: Sequence[dict], mode: str) -> dict:
    tv_key = f"{mode}_cross_seed_tv"
    mismatch_key = f"{mode}_dominant_action_mismatch"
    return {
        "behavior_policy_tv": _summary([float(row[tv_key]) for row in rows]),
        "dominant_legal_action_mismatch_rate": float(sum(int(row[mismatch_key]) for row in rows) / len(rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only V1+ Phase2B0 policy-algebra screen")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--source-execution-sha", default=SOURCE_EXECUTION_SHA)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if str(args.source_execution_sha) != SOURCE_EXECUTION_SHA:
        raise SystemExit("Phase2B0 source execution SHA drift")
    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("Phase2B0 torch-thread contract drift")

    input_root = args.input_root.resolve()
    heldout_root = args.heldout_root.resolve()
    output = args.out.resolve()
    seed_a, seed_b = map(int, TRAINING_SEEDS)
    ensembles = {
        seed: _load_ensemble(input_root, seed, str(args.source_execution_sha))
        for seed in (seed_a, seed_b)
    }

    evaluation_rows = []
    all_state_rows = []
    epsilon_identity_checks = 0
    for evaluation_seed in EVALUATION_SEEDS:
        path = _find_heldout(heldout_root, int(evaluation_seed))
        descriptors = list(load_heldout_v3_artifact(
            path,
            expected_domain=DOMAIN,
            expected_evaluation_seed=int(evaluation_seed),
            expected_count=2048,
        )[:POLICY_COUNT])
        outputs = {
            seed: _member_outputs(ensembles[seed], descriptors)
            for seed in (seed_a, seed_b)
        }
        local = []
        by_street = defaultdict(list)
        for index, descriptor in enumerate(descriptors):
            legal = tuple(int(value) for value in descriptor.legal_slots)
            per_seed = {}
            for seed in (seed_a, seed_b):
                control, candidate, stats = raw_mean_then_regret_match_same_epsilon(outputs[seed][index], legal)
                # Candidate explicitly reuses this exact scalar; count every state/seed identity.
                epsilon_identity_checks += 1
                per_seed[seed] = {
                    "control": control,
                    "candidate": candidate,
                    "epsilon": float(stats["epsilon"]),
                    "disagreement": float(stats["disagreement"]),
                    "within_tv": _policy_tv(control, candidate),
                    "raw_mean_positive_legal_count": int(stats["raw_mean_positive_legal_count"]),
                }
            control_tv = _policy_tv(per_seed[seed_a]["control"], per_seed[seed_b]["control"])
            candidate_tv = _policy_tv(per_seed[seed_a]["candidate"], per_seed[seed_b]["candidate"])
            control_dom_a = max(legal, key=lambda slot: float(per_seed[seed_a]["control"][slot]))
            control_dom_b = max(legal, key=lambda slot: float(per_seed[seed_b]["control"][slot]))
            candidate_dom_a = max(legal, key=lambda slot: float(per_seed[seed_a]["candidate"][slot]))
            candidate_dom_b = max(legal, key=lambda slot: float(per_seed[seed_b]["candidate"][slot]))
            decoded = decode_spnniv3(descriptor.observation_v3)
            street = STREET_NAMES.get(int(decoded.categorical[1]), str(decoded.categorical[1]))
            row = {
                "evaluation_seed": int(evaluation_seed),
                "state_index": int(descriptor.state_index),
                "street": street,
                "control_cross_seed_tv": float(control_tv),
                "candidate_cross_seed_tv": float(candidate_tv),
                "candidate_minus_control_cross_seed_tv": float(candidate_tv - control_tv),
                "control_dominant_action_mismatch": int(control_dom_a != control_dom_b),
                "candidate_dominant_action_mismatch": int(candidate_dom_a != candidate_dom_b),
                "seed_1342191342_epsilon": per_seed[seed_a]["epsilon"],
                "seed_1801739323_epsilon": per_seed[seed_b]["epsilon"],
                "seed_1342191342_control_candidate_tv": per_seed[seed_a]["within_tv"],
                "seed_1801739323_control_candidate_tv": per_seed[seed_b]["within_tv"],
            }
            local.append(row)
            by_street[street].append(row)
        eval_summary = {
            "evaluation_seed": int(evaluation_seed),
            "heldout": str(path),
            "CONTROL_RM_THEN_MEAN": _mode_summary(local, "control"),
            "CANDIDATE_RAW_MEAN_THEN_RM": _mode_summary(local, "candidate"),
            "by_street": {
                street: {
                    "CONTROL_RM_THEN_MEAN": _mode_summary(rows, "control"),
                    "CANDIDATE_RAW_MEAN_THEN_RM": _mode_summary(rows, "candidate"),
                }
                for street, rows in sorted(by_street.items())
            },
        }
        evaluation_rows.append(eval_summary)
        all_state_rows.extend(local)

    pooled_control = _mode_summary(all_state_rows, "control")
    pooled_candidate = _mode_summary(all_state_rows, "candidate")
    control_mean = float(pooled_control["behavior_policy_tv"]["mean"])
    candidate_mean = float(pooled_candidate["behavior_policy_tv"]["mean"])
    absolute_improvement = control_mean - candidate_mean
    relative_improvement = absolute_improvement / control_mean if control_mean > 0.0 else 0.0
    both_eval_improve = all(
        float(row["CANDIDATE_RAW_MEAN_THEN_RM"]["behavior_policy_tv"]["mean"])
        < float(row["CONTROL_RM_THEN_MEAN"]["behavior_policy_tv"]["mean"])
        for row in evaluation_rows
    )
    p95_ok = all(
        float(row["CANDIDATE_RAW_MEAN_THEN_RM"]["behavior_policy_tv"]["p95"])
        <= float(row["CONTROL_RM_THEN_MEAN"]["behavior_policy_tv"]["p95"]) + 0.02
        for row in evaluation_rows
    )
    dominant_ok = (
        float(pooled_candidate["dominant_legal_action_mismatch_rate"])
        <= float(pooled_control["dominant_legal_action_mismatch_rate"])
    )
    material = bool(absolute_improvement >= 0.05 or relative_improvement >= 0.10)
    screen_pass = bool(both_eval_improve and material and p95_ok and dominant_ok)

    result = {
        "schema": SCHEMA,
        "status": (
            "SCREEN_PASS_ELIGIBLE_FOR_CAUSAL_PHASE2B_TRAINING"
            if screen_pass else "SCREEN_FAIL_DO_NOT_TRAIN_CANDIDATE"
        ),
        "governance_scope": "Post-R7.5.3 architecture-reset read-only screen; R7.5.3 remains closed.",
        "source_execution_sha": str(args.source_execution_sha),
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "training_seeds": [seed_a, seed_b],
        "evaluation_seeds": [int(value) for value in EVALUATION_SEEDS],
        "policy_count_per_evaluation_seed": POLICY_COUNT,
        "candidate": "RAW_MEAN_THEN_REGRET_MATCH_WITH_CONTROL_EPSILON",
        "epsilon_scale": EPSILON_SCALE,
        "epsilon_cap": EPSILON_CAP,
        "epsilon_identity_checks": int(epsilon_identity_checks),
        "epsilon_identity_pass": bool(epsilon_identity_checks == 2 * POLICY_COUNT * len(EVALUATION_SEEDS)),
        "evaluation_rows": evaluation_rows,
        "pooled": {
            "CONTROL_RM_THEN_MEAN": pooled_control,
            "CANDIDATE_RAW_MEAN_THEN_RM": pooled_candidate,
            "absolute_mean_tv_improvement": float(absolute_improvement),
            "relative_mean_tv_improvement": float(relative_improvement),
            "both_evaluation_seeds_improve": bool(both_eval_improve),
            "p95_nonworsening_rule_pass": bool(p95_ok),
            "dominant_action_mismatch_nonworsening_pass": bool(dominant_ok),
            "materiality_rule_pass": bool(material),
        },
        "screen_rule_pass": bool(screen_pass),
        "state_rows": all_state_rows,
        "interpretation_guardrails": [
            "This is a read-only algebra screen using already-trained ensembles; it is not a causal training result.",
            "Passing only authorizes precommitting one controlled Phase2B training ablation.",
            "Failing forbids training this candidate and redirects work to direct chance/return variance reduction or stratified chance support.",
            "No model, reservoir, seed, heldout state, epsilon coefficient or threshold was changed.",
        ],
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    if not result["epsilon_identity_pass"]:
        raise RuntimeError("Phase2B0 epsilon identity accounting failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(json.dumps({
        "status": result["status"],
        "control_pooled_mean_tv": control_mean,
        "candidate_pooled_mean_tv": candidate_mean,
        "absolute_improvement": absolute_improvement,
        "relative_improvement": relative_improvement,
        "out": str(output),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
