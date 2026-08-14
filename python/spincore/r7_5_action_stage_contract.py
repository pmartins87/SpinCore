from __future__ import annotations

import json
from pathlib import Path

PHASE = "R7_5_4A_POSTFLOP"
ROOT_LEVEL = 160
ROOTS_PER_ITERATION = 32
ITERATIONS = 5
EXACT_OPPONENT_LEVELS = 2
RESERVOIR_CAPACITY = 100000
ADVANTAGE_STEPS = 4096
POLICY_STEPS = 16384
BATCH_SIZE = 256
LEARNING_RATE = 0.001
ENSEMBLE_SIZE = 4
AUDIT_SIZE = 2048
CROSS_SEED_PER_SEED = 1024
EPSILON_SCALE = 1.75
EPSILON_CAP = 0.5
TORCH_THREADS = 2
SELECTED_REPRESENTATION = "C0_V1_FROZEN_CONTROL"
PAYOUT = (0.5, 0.3, 0.2)
POSTFLOP_TRAINING_SEEDS = (1737995611, 645939859, 1311335590)
PAIRED_EVALUATION_SEEDS = (1817694185, 1617273629)
MEMBER_INIT_XOR = 0x0E115EED
MEMBER_BATCH_XOR = 0xBA7C8A11

RUNNER_FREEZE_SCHEMA = "SPINCORE_R7_5_4_RUNNER_IMPLEMENTATION_FREEZE_V1"
TRAINING_FREEZE_SCHEMA = "SPINCORE_R7_5_4_TRAINING_IMPLEMENTATION_FREEZE_V1"
PREFLIGHT_SCHEMA = "SPINCORE_R7_5_4_STRATEGIC_PREFLIGHT_V5"
REP_RESULT_SCHEMA = "SPINCORE_R7_5_3_REPRESENTATION_ABLATION_RESULT_V1"
COST_FREEZE_SCHEMA = "SPINCORE_R7_5_4_COST_TELEMETRY_SEMANTIC_FREEZE_V1"


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


def validate_action_stage_contract(
    repo_root: str | Path,
    *,
    candidate_id: str,
    training_seed: int,
) -> dict:
    root = Path(repo_root)
    validation = root / "validation"
    runner = _read(validation / "R7_5_4_RUNNER_IMPLEMENTATION_FREEZE_20260814.json")
    training = _read(validation / "R7_5_4_TRAINING_IMPLEMENTATION_FREEZE.json")
    preflight = _read(validation / "R7_5_4A_160_STRATEGIC_PREFLIGHT.json")
    rep = _read(validation / "R7_5_3_REPRESENTATION_ABLATION_RESULT.json")
    v3 = _read(validation / "R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V3.json")
    cost = _read(validation / "R7_5_4_COST_TELEMETRY_SEMANTIC_FREEZE_20260814.json")

    if runner.get("schema") != RUNNER_FREEZE_SCHEMA:
        raise ValueError("runner implementation freeze mismatch")
    if training.get("schema") != TRAINING_FREEZE_SCHEMA:
        raise ValueError("training implementation freeze mismatch")
    if cost.get("schema") != COST_FREEZE_SCHEMA:
        raise ValueError("cost telemetry freeze mismatch")
    if preflight.get("schema") != PREFLIGHT_SCHEMA or not bool(preflight.get("ready_to_start")):
        raise ValueError("durable R7.5.4A 160 preflight is not PASS")
    if rep.get("schema") != REP_RESULT_SCHEMA or not bool(rep.get("r7_5_3_representation_ablation_pass")):
        raise ValueError("R7.5.3 representation gate is not PASS")
    if rep.get("selected_candidate") != SELECTED_REPRESENTATION:
        raise ValueError("R7.5.4A requires the durable C0 representation winner")
    if preflight.get("selected_representation") != SELECTED_REPRESENTATION:
        raise ValueError("preflight representation differs from durable C0 winner")

    seed_contract = v3.get("seed_derivation") or {}
    if tuple(seed_contract.get("postflop_training_seeds") or ()) != POSTFLOP_TRAINING_SEEDS:
        raise ValueError("V3 postflop seed contract drift")
    if tuple(seed_contract.get("paired_evaluation_seeds") or ()) != PAIRED_EVALUATION_SEEDS:
        raise ValueError("V3 paired-evaluation seed contract drift")
    if int(training_seed) not in POSTFLOP_TRAINING_SEEDS:
        raise ValueError("non-frozen R7.5.4A training seed")

    frozen = runner.get("training") or {}
    expected_training = {
        "reservoir_capacity": RESERVOIR_CAPACITY,
        "final_fit_audit_size": AUDIT_SIZE,
        "cross_seed_observations_per_seed": CROSS_SEED_PER_SEED,
        "iterations": ITERATIONS,
        "exact_opponent_levels": EXACT_OPPONENT_LEVELS,
        "advantage_optimizer_steps_per_member_per_iteration": ADVANTAGE_STEPS,
        "average_policy_optimizer_steps": POLICY_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "ensemble_size": ENSEMBLE_SIZE,
        "epsilon_scale": EPSILON_SCALE,
        "epsilon_cap": EPSILON_CAP,
    }
    for key, expected in expected_training.items():
        if frozen.get(key) != expected:
            raise ValueError(f"runner freeze drift for {key}")
    if int(runner["root_levels"][str(ROOT_LEVEL)]["roots_per_iteration"]) != ROOTS_PER_ITERATION:
        raise ValueError("root-level contract drift")
    if int(runner["runtime_for_github_ablation"]["torch_threads"]) != TORCH_THREADS:
        raise ValueError("GitHub thread contract drift")

    from spincore.r7_5_action_contract import postflop_candidate_specs

    specs = postflop_candidate_specs(root)
    if str(candidate_id) not in specs:
        raise ValueError("unknown R7.5.4A candidate")

    return {
        "runner": runner,
        "training": training,
        "preflight": preflight,
        "representation": rep,
        "v3": v3,
        "cost": cost,
        "candidate_spec": specs[str(candidate_id)],
    }
