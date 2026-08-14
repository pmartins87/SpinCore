from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from spincore.r7_5_action_stage import ActionStageConfig
from spincore.r7_5_action_stage_contract import (
    ADVANTAGE_STEPS,
    AUDIT_SIZE,
    BATCH_SIZE,
    CROSS_SEED_PER_SEED,
    ENSEMBLE_SIZE,
    EPSILON_CAP,
    EPSILON_SCALE,
    EXACT_OPPONENT_LEVELS,
    ITERATIONS,
    LEARNING_RATE,
    POLICY_STEPS,
    POSTFLOP_TRAINING_SEEDS,
    RESERVOIR_CAPACITY,
    SELECTED_REPRESENTATION,
    TORCH_THREADS,
    validate_action_stage_contract,
)

ROOT_LEVEL_320 = 320
ROOTS_PER_ITERATION_320 = 64
EXPECTED_PARENT_TRAINING_SHA = "457996944f76e9f1fa0475691df978f450259641"
EXPECTED_PARENT_EVALUATOR_SHA = "4752a951e53c6f195fb12676a417a0b690c8e4cf"
RESULT_160_SCHEMA = "SPINCORE_R7_5_4A_160_RESULT_V1"
PRECOMMIT_320_SCHEMA = "SPINCORE_R7_5_4A_320_RUNNER_PRECOMMIT_V1"
CARD_SYMMETRY_RESULT_SCHEMA = "SPINCORE_R7_5_3B_CARD_SYMMETRY_RESULT_V1"
CARD_SYMMETRY_REQUIRED_WINNER = "S0_V1_FROZEN_CONTROL"
CONTROL = "PF0_CONTROL_33_75_AI"
REFEREE = "PF_DENSE_REFERENCE"
ELIGIBLE_POSTFLOP = {
    "PF0_CONTROL_33_75_AI",
    "PF1_33_50_75_AI",
    "PF2_33_50_75_100_AI",
    "PF3_COMPACT_33_66_100_AI",
    "PF4_CRUSHER_COMPACT_40_66_100_AI",
}


@dataclass(frozen=True)
class ExecutionCandidate320:
    candidate_id: str
    strategically_eligible: bool
    role: str


@dataclass(frozen=True)
class ExecutionPlan320:
    survivors: tuple[str, ...]
    execution_candidates: tuple[ExecutionCandidate320, ...]

    @property
    def execution_ids(self) -> tuple[str, ...]:
        return tuple(row.candidate_id for row in self.execution_candidates)


def frozen_config_320() -> ActionStageConfig:
    return ActionStageConfig(
        roots_per_iteration=ROOTS_PER_ITERATION_320,
        total_iterations=ITERATIONS,
        exact_opponent_levels=EXACT_OPPONENT_LEVELS,
        reservoir_capacity=RESERVOIR_CAPACITY,
        advantage_steps=ADVANTAGE_STEPS,
        policy_steps=POLICY_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        ensemble_size=ENSEMBLE_SIZE,
        audit_size=AUDIT_SIZE,
        epsilon_scale=EPSILON_SCALE,
        epsilon_cap=EPSILON_CAP,
    )


def validate_resume_mode_320(*, target_iteration: int, resume_supplied: bool, finalize: bool) -> None:
    target = int(target_iteration)
    if target not in (1, 2, 3, 4, 5):
        raise ValueError("R7.5.4A-320 target iteration must be 1..5")
    if target == 1 and resume_supplied:
        raise ValueError("R7.5.4A-320 iteration 1 must start fresh; no prior-level checkpoint is accepted")
    if target > 1 and not resume_supplied:
        raise ValueError("R7.5.4A-320 iterations 2..5 require a 320 checkpoint from the immediately prior iteration")
    if target == ITERATIONS and not finalize:
        raise ValueError("R7.5.4A-320 iteration 5 must include final AveragePolicy fit")
    if finalize and target != ITERATIONS:
        raise ValueError("R7.5.4A-320 finalize is legal only on iteration 5")


def execution_plan_from_160_result(result: Mapping) -> ExecutionPlan320:
    if result.get("schema") != RESULT_160_SCHEMA:
        raise ValueError("wrong durable R7.5.4A-160 result schema")
    if int(result.get("root_level", -1)) != 160:
        raise ValueError("parent result is not the 160-root result")
    if str(result.get("training_execution_sha")) != EXPECTED_PARENT_TRAINING_SHA:
        raise ValueError("parent 160 training SHA mismatch")
    if str(result.get("evaluator_sha")) != EXPECTED_PARENT_EVALUATOR_SHA:
        raise ValueError("parent 160 evaluator SHA mismatch")
    if bool(result.get("r7_5_4a_postflop_selected")) or result.get("r7_5_4a_postflop_selected_candidate") is not None:
        raise ValueError("160 result illegally contains a final postflop selection")
    if bool(result.get("production_training_authorized")) or bool(result.get("ready_for_tables")):
        raise ValueError("160 result illegally authorizes production/table use")

    selection = dict(result.get("selection") or {})
    if selection.get("status") != "PASS_LEVEL":
        raise ValueError("320 requires durable 160 PASS_LEVEL")
    if int(selection.get("root_level", -1)) != 160:
        raise ValueError("160 selection root-level mismatch")
    if selection.get("selected_candidate") is not None:
        raise ValueError("160 selection must not have a final candidate")
    if int(selection.get("next_level", -1)) != ROOT_LEVEL_320:
        raise ValueError("160 result does not authorize escalation to 320")
    if bool(selection.get("production_training_authorized")) or bool(selection.get("ready_for_tables")):
        raise ValueError("160 selection illegally authorizes production/table use")

    survivors = tuple(sorted(str(value) for value in (selection.get("survivors") or ())))
    if not survivors:
        raise ValueError("160 PASS_LEVEL must contain at least one strategic survivor")
    if len(set(survivors)) != len(survivors) or not set(survivors).issubset(ELIGIBLE_POSTFLOP):
        raise ValueError("160 survivor set contains duplicate or non-eligible candidate")

    expected_execution = tuple(sorted(set(survivors) | {CONTROL, REFEREE}))
    actual_execution = tuple(sorted(str(value) for value in (selection.get("mandatory_next_level_execution") or ())))
    if actual_execution != expected_execution:
        raise ValueError("160 mandatory 320 execution set drift")
    expected_control_only = tuple(sorted(value for value in expected_execution if value not in survivors and value != REFEREE))
    actual_control_only = tuple(sorted(str(value) for value in (selection.get("control_only_noneligible") or ())))
    if actual_control_only != expected_control_only:
        raise ValueError("160 control-only classification drift")

    execution: list[ExecutionCandidate320] = []
    for candidate_id in expected_execution:
        strategically_eligible = candidate_id in survivors and candidate_id != REFEREE
        if candidate_id == REFEREE:
            role = "REFEREE_CONTROL_ONLY"
        elif strategically_eligible and candidate_id == CONTROL:
            role = "STRATEGIC_SURVIVOR_AND_CONTROL"
        elif strategically_eligible:
            role = "STRATEGIC_SURVIVOR"
        else:
            role = "MANDATORY_CONTROL_ONLY"
        execution.append(ExecutionCandidate320(candidate_id, strategically_eligible, role))
    return ExecutionPlan320(survivors=survivors, execution_candidates=tuple(execution))


def validate_card_symmetry_parent_320(repo_root: str | Path) -> dict:
    """Fail closed unless R7.5.3B retained the exact V1 representation.

    Existing R7.5.4A evidence was generated under C0/V1. If the lossless card
    symmetry candidate wins, that evidence is representation-conditioned and
    cannot be escalated to 320 as though nothing changed.
    """
    path = Path(repo_root) / "validation" / "R7_5_3B_CARD_SYMMETRY_RESULT.json"
    if not path.exists():
        raise FileNotFoundError(
            "R7.5.4A-320 is gated by R7.5.3B; durable card-symmetry result is not yet persisted"
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema") != CARD_SYMMETRY_RESULT_SCHEMA:
        raise ValueError("wrong durable R7.5.3B card-symmetry result schema")
    if result.get("status") != "PASS":
        raise ValueError("R7.5.4A-320 requires R7.5.3B PASS")
    if bool(result.get("production_training_authorized")) or bool(result.get("ready_for_tables")):
        raise ValueError("R7.5.3B result illegally authorizes production/table use")
    if result.get("winner_id") != CARD_SYMMETRY_REQUIRED_WINNER:
        raise ValueError(
            "R7.5.3B changed the representation; existing R7.5.4A sizing evidence cannot escalate to 320"
        )
    decision = dict(result.get("representation_decision") or {})
    if decision.get("existing_R7_5_4A_evidence_representation_consistent") is not True:
        raise ValueError("R7.5.3B representation-consistency declaration mismatch")
    return result


def load_execution_plan_320(repo_root: str | Path) -> ExecutionPlan320:
    validate_card_symmetry_parent_320(repo_root)
    path = Path(repo_root) / "validation" / "R7_5_4A_160_RESULT.json"
    if not path.exists():
        raise FileNotFoundError("durable R7.5.4A-160 result is not yet persisted")
    return execution_plan_from_160_result(json.loads(path.read_text(encoding="utf-8")))


def validate_action_320_stage_contract(
    repo_root: str | Path,
    *,
    candidate_id: str,
    training_seed: int,
) -> dict:
    root = Path(repo_root)
    # Reuse all already-frozen 160/common science: seeds, representation, model,
    # training steps, epsilon, reservoir and runtime. This does NOT authorize
    # reuse of a 160 checkpoint; only the scientific constants are inherited.
    common = validate_action_stage_contract(
        root,
        candidate_id=str(candidate_id),
        training_seed=int(training_seed),
    )
    precommit = json.loads(
        (root / "validation" / "R7_5_4A_320_RUNNER_PRECOMMIT_20260814.json").read_text(encoding="utf-8")
    )
    if precommit.get("schema") != PRECOMMIT_320_SCHEMA:
        raise ValueError("R7.5.4A-320 runner precommit mismatch")
    if bool(precommit.get("launch_authorized")):
        raise ValueError("static 320 precommit must remain non-authorizing; durable 160 result is the runtime authorization gate")
    if int(precommit.get("root_level", -1)) != ROOT_LEVEL_320:
        raise ValueError("320 precommit root-level drift")
    if int(precommit.get("roots_per_iteration", -1)) != ROOTS_PER_ITERATION_320:
        raise ValueError("320 precommit roots-per-iteration drift")
    if not bool(precommit.get("fresh_training_required")) or bool(precommit.get("resume_from_160_checkpoint_allowed")):
        raise ValueError("320 fresh-run contract drift")

    runner = common["runner"]
    root_row = dict((runner.get("root_levels") or {}).get(str(ROOT_LEVEL_320)) or {})
    if int(root_row.get("roots_per_iteration", -1)) != ROOTS_PER_ITERATION_320:
        raise ValueError("runner freeze 320 roots-per-iteration drift")
    if int(root_row.get("roots_per_seed", -1)) != ROOT_LEVEL_320:
        raise ValueError("runner freeze 320 roots-per-seed drift")
    if int(training_seed) not in POSTFLOP_TRAINING_SEEDS:
        raise ValueError("non-frozen R7.5.4A training seed")
    if SELECTED_REPRESENTATION != "C0_V1_FROZEN_CONTROL" or TORCH_THREADS != 2:
        raise ValueError("shared representation/runtime contract drift")

    plan = load_execution_plan_320(root)
    by_id = {row.candidate_id: row for row in plan.execution_candidates}
    if str(candidate_id) not in by_id:
        raise ValueError("candidate was pruned at 160 and is not a mandatory 320 control")
    return {
        **common,
        "execution_plan": plan,
        "execution_candidate": by_id[str(candidate_id)],
        "root_level": ROOT_LEVEL_320,
        "roots_per_iteration": ROOTS_PER_ITERATION_320,
    }
