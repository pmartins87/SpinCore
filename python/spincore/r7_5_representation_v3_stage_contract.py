from __future__ import annotations

import hashlib
import json
from pathlib import Path

from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL

TRAINING_FREEZE_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_TRAINING_FREEZE_V1"
MODEL_FREEZE_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_MODEL_FREEZE_V1"
PRECOMMIT_SCHEMA = "SPINCORE_R7_5_3C_HYBRID_REPRESENTATION_PRECOMMIT_V1"
RESOURCE_EVIDENCE_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_RESOURCE_PREFLIGHT_EVIDENCE_V1"

REPRESENTATIONS = (H2_FINAL, H3_FINAL)
DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")
TRAINING_SEEDS = (1342191342, 1801739323)
EVALUATION_SEEDS = (2029384436, 1150634112)
ACTION_CANDIDATE = "PF0_CONTROL_33_75_AI"
ITERATIONS = 3
ROOTS_PER_ITERATION = 64
EXACT_OPPONENT_LEVELS = 2
RESERVOIR_CAPACITY = 100000
ADVANTAGE_STEPS = 4096
POLICY_STEPS = 16384
BATCH_SIZE = 256
LEARNING_RATE = 0.001
ENSEMBLE_SIZE = 4
AUDIT_SIZE = 2048
CROSS_SEED_OBSERVATIONS = 1024
EPSILON_SCALE = 1.75
EPSILON_CAP = 0.5
TORCH_THREADS = 2
PAYOUT = (0.5, 0.3, 0.2)
ADVANTAGE_NRMSE_MAX = 0.75
POLICY_TV_MAX = 0.12
CROSS_SEED_MEAN_TV_MAX = 0.15
CROSS_SEED_P95_TV_MAX = 0.35

MODEL_FINGERPRINTS = {
    H2_FINAL: "1362caf9f893cee88bfb2f3f26e8054c4932e112b27e0a5dc112a08a727a9f97",
    H3_FINAL: "3568865e93366a793c3ecddb5ceeaad9de73540ca6985a20653c925b2a7576cf",
}
MODEL_PARAMETER_COUNTS = {
    H2_FINAL: 190958,
    H3_FINAL: 228110,
}

MEMBER_INIT_XOR = 0x0E115EED
MEMBER_BATCH_XOR = 0xBA7C8A11


def primary_reset_seed(training_seed: int, iteration: int) -> int:
    return (int(training_seed) ^ (int(iteration) * 0x9E3779B1)) & 0x7FFFFFFF


def side_member_seeds(training_seed: int, iteration: int, member: int) -> tuple[int, int]:
    if int(member) not in (1, 2, 3):
        raise ValueError("side member must be 1, 2 or 3")
    init_seed = (
        int(training_seed)
        ^ MEMBER_INIT_XOR
        ^ (int(iteration) * 0x9E3779B1)
        ^ (int(member) * 0x045D9F3B)
    ) & 0x7FFFFFFF
    batch_seed = (
        int(training_seed)
        ^ MEMBER_BATCH_XOR
        ^ (int(iteration) * 0x85EBCA77)
        ^ (int(member) * 0xC2B2AE3D)
    ) & ((1 << 64) - 1)
    return int(init_seed), int(batch_seed)


def deck_seed(training_seed: int, global_root: int, iteration: int) -> int:
    return (
        int(training_seed) * 1_000_003 + int(global_root) * 97 + int(iteration)
    ) & ((1 << 64) - 1)


def _read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _live_model_evidence(root: Path, representation: str, frozen_model: dict, frozen_sources: dict) -> dict:
    from spincore_nn.models_v3_final import make_h2_final_v3, make_h3_final_v3

    live_sources = {}
    for relative_path, expected in frozen_sources.items():
        actual = _sha256(root / relative_path)
        if actual != expected:
            raise ValueError(
                f"Phase 2 frozen source drift: {relative_path} {actual} != {expected}"
            )
        live_sources[relative_path] = actual

    factory = make_h2_final_v3 if representation == H2_FINAL else make_h3_final_v3
    config, network = factory(device="cpu", seed=0)
    parameter_count = int(sum(parameter.numel() for parameter in network.parameters()))
    if parameter_count != MODEL_PARAMETER_COUNTS[representation]:
        raise ValueError("Phase 2 live model parameter-count drift")
    if parameter_count != int(frozen_model["parameter_count"]):
        raise ValueError("Phase 2 live model disagrees with frozen parameter count")
    config_dict = config.to_dict()
    if config_dict != dict(frozen_model["config"]):
        raise ValueError("Phase 2 live model config drift")

    material = {
        "representation": representation,
        "config": config_dict,
        "parameter_count": parameter_count,
        "source_sha256": live_sources,
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(blob).hexdigest()
    if fingerprint != MODEL_FINGERPRINTS[representation]:
        raise ValueError("Phase 2 live architecture fingerprint drift")
    if fingerprint != frozen_model["architecture_fingerprint_sha256"]:
        raise ValueError("Phase 2 live fingerprint disagrees with model freeze")
    return {
        "config": config_dict,
        "parameter_count": parameter_count,
        "source_sha256": live_sources,
        "architecture_fingerprint_sha256": fingerprint,
    }


def validate_phase2_v3_contract(
    repo_root: str | Path,
    *,
    representation: str,
    domain: str,
    training_seed: int,
) -> dict:
    root = Path(repo_root)
    validation = root / "validation"
    training = _read(validation / "R7_5_3C_PHASE2_TRAINING_FREEZE_20260815.json")
    model = _read(validation / "R7_5_3C_PHASE2_MODEL_FREEZE_20260815.json")
    precommit = _read(validation / "R7_5_3C_HYBRID_REPRESENTATION_PRECOMMIT.json")
    resource = _read(validation / "R7_5_3C_PHASE2_RESOURCE_PREFLIGHT_EVIDENCE_20260815.json")

    if training.get("schema") != TRAINING_FREEZE_SCHEMA:
        raise ValueError("Phase 2 V3 training freeze mismatch")
    if model.get("schema") != MODEL_FREEZE_SCHEMA:
        raise ValueError("Phase 2 V3 model freeze mismatch")
    if precommit.get("schema") != PRECOMMIT_SCHEMA:
        raise ValueError("Phase 2 V3 precommit mismatch")
    if resource.get("schema") != RESOURCE_EVIDENCE_SCHEMA or not resource.get("resource_preflight_pass"):
        raise ValueError("Phase 2 V3 resource admission is not PASS")
    if bool(training.get("ready_for_tables")) or bool(training.get("production_training_authorized")):
        raise ValueError("Phase 2 training freeze illegally authorizes production/table use")

    if representation not in REPRESENTATIONS or representation not in training["representations"]:
        raise ValueError("non-frozen Phase 2 representation")
    if domain not in DOMAINS or domain not in training["domains"]:
        raise ValueError("non-frozen Phase 2 domain")
    if int(training_seed) not in TRAINING_SEEDS:
        raise ValueError("non-frozen Phase 2 training seed")
    if tuple(training["training_seeds"]) != TRAINING_SEEDS:
        raise ValueError("Phase 2 training seed set drift")
    if tuple(training["evaluation_seeds"]) != EVALUATION_SEEDS:
        raise ValueError("Phase 2 evaluation seed set drift")
    if training["action_candidate"] != ACTION_CANDIDATE:
        raise ValueError("Phase 2 action candidate drift")

    cfr = training["cfr_execution"]
    expected_cfr = {
        "iterations": ITERATIONS,
        "roots_per_iteration": ROOTS_PER_ITERATION,
        "roots_per_seed": ITERATIONS * ROOTS_PER_ITERATION,
        "exact_opponent_levels": EXACT_OPPONENT_LEVELS,
    }
    for key, expected in expected_cfr.items():
        if cfr.get(key) != expected:
            raise ValueError(f"Phase 2 CFR contract drift for {key}")
    if tuple(cfr.get("payout") or ()) != PAYOUT:
        raise ValueError("Phase 2 payout drift")

    behavior = training["behavior"]
    if behavior.get("ensemble_size") != ENSEMBLE_SIZE:
        raise ValueError("Phase 2 ensemble size drift")
    if behavior.get("epsilon_scale") != EPSILON_SCALE or behavior.get("epsilon_cap") != EPSILON_CAP:
        raise ValueError("Phase 2 uncertainty coefficient drift")

    advantage = training["advantage_fit"]
    exact_advantage = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "optimizer_steps_per_member_per_iteration": ADVANTAGE_STEPS,
        "reservoir_capacity": RESERVOIR_CAPACITY,
        "ensemble_members": ENSEMBLE_SIZE,
    }
    for key, expected in exact_advantage.items():
        if advantage.get(key) != expected:
            raise ValueError(f"Phase 2 advantage-fit drift for {key}")

    policy = training["average_policy_fit"]
    exact_policy = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "optimizer_steps": POLICY_STEPS,
    }
    for key, expected in exact_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"Phase 2 policy-fit drift for {key}")

    diagnostics = training["diagnostics"]
    expected_diagnostics = {
        "advantage_audit_sample_size": AUDIT_SIZE,
        "advantage_weighted_nrmse_max": ADVANTAGE_NRMSE_MAX,
        "policy_audit_sample_size": AUDIT_SIZE,
        "policy_weighted_mean_tv_max": POLICY_TV_MAX,
        "cross_seed_observations_per_training_seed": CROSS_SEED_OBSERVATIONS,
        "cross_seed_mean_tv_max": CROSS_SEED_MEAN_TV_MAX,
        "cross_seed_p95_tv_max": CROSS_SEED_P95_TV_MAX,
    }
    for key, expected in expected_diagnostics.items():
        if diagnostics.get(key) != expected:
            raise ValueError(f"Phase 2 diagnostic drift for {key}")

    frozen_model = model["production_candidates"][representation]
    if int(frozen_model["parameter_count"]) != MODEL_PARAMETER_COUNTS[representation]:
        raise ValueError("Phase 2 model parameter-count drift")
    if frozen_model["architecture_fingerprint_sha256"] != MODEL_FINGERPRINTS[representation]:
        raise ValueError("Phase 2 model fingerprint drift")
    if not model["admission"].get("phase2_strategic_training_allowed"):
        raise ValueError("Phase 2 model not strategically admitted")
    live_model = _live_model_evidence(
        root,
        representation,
        frozen_model,
        dict(model["shared_source_sha256"]),
    )

    from spincore.r7_5_action_contract import postflop_candidate_specs
    specs = postflop_candidate_specs(root)
    if ACTION_CANDIDATE not in specs:
        raise ValueError("Phase 2 PF0 action spec missing")

    return {
        "training": training,
        "model": model,
        "precommit": precommit,
        "resource": resource,
        "live_model": live_model,
        "action_spec": specs[ACTION_CANDIDATE],
    }
