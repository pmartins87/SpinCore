from __future__ import annotations

import copy

import pytest

from spincore.r7_5_action_320_contract import (
    CONTROL,
    EXPECTED_PARENT_EVALUATOR_SHA,
    EXPECTED_PARENT_TRAINING_SHA,
    REFEREE,
    ROOTS_PER_ITERATION_320,
    execution_plan_from_160_result,
    frozen_config_320,
    validate_resume_mode_320,
)
from spincore.r7_5_action_stage_contract import (
    ADVANTAGE_STEPS,
    AUDIT_SIZE,
    BATCH_SIZE,
    ENSEMBLE_SIZE,
    EPSILON_CAP,
    EPSILON_SCALE,
    EXACT_OPPONENT_LEVELS,
    ITERATIONS,
    LEARNING_RATE,
    POLICY_STEPS,
    RESERVOIR_CAPACITY,
    ROOTS_PER_ITERATION as ROOTS_PER_ITERATION_160,
)


def _result(survivors=("PF2_33_50_75_100_AI", "PF4_CRUSHER_COMPACT_40_66_100_AI")):
    survivors = tuple(sorted(survivors))
    mandatory = tuple(sorted(set(survivors) | {CONTROL, REFEREE}))
    control_only = tuple(sorted(value for value in mandatory if value not in survivors and value != REFEREE))
    return {
        "schema": "SPINCORE_R7_5_4A_160_RESULT_V1",
        "root_level": 160,
        "training_execution_sha": EXPECTED_PARENT_TRAINING_SHA,
        "evaluator_sha": EXPECTED_PARENT_EVALUATOR_SHA,
        "r7_5_4a_postflop_selected": False,
        "r7_5_4a_postflop_selected_candidate": None,
        "production_training_authorized": False,
        "ready_for_tables": False,
        "selection": {
            "status": "PASS_LEVEL",
            "root_level": 160,
            "survivors": list(survivors),
            "selected_candidate": None,
            "next_level": 320,
            "mandatory_next_level_execution": list(mandatory),
            "control_only_noneligible": list(control_only),
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
    }


def test_320_plan_preserves_survivors_and_adds_controls_without_resurrection() -> None:
    plan = execution_plan_from_160_result(_result())
    assert plan.survivors == (
        "PF2_33_50_75_100_AI",
        "PF4_CRUSHER_COMPACT_40_66_100_AI",
    )
    assert plan.execution_ids == (
        CONTROL,
        "PF2_33_50_75_100_AI",
        "PF4_CRUSHER_COMPACT_40_66_100_AI",
        REFEREE,
    )
    rows = {row.candidate_id: row for row in plan.execution_candidates}
    assert rows[CONTROL].strategically_eligible is False
    assert rows[CONTROL].role == "MANDATORY_CONTROL_ONLY"
    assert rows[REFEREE].strategically_eligible is False
    assert rows["PF2_33_50_75_100_AI"].strategically_eligible is True


def test_pf0_remains_eligible_only_when_it_actually_survived_160() -> None:
    plan = execution_plan_from_160_result(_result((CONTROL, "PF3_COMPACT_33_66_100_AI")))
    rows = {row.candidate_id: row for row in plan.execution_candidates}
    assert rows[CONTROL].strategically_eligible is True
    assert rows[CONTROL].role == "STRATEGIC_SURVIVOR_AND_CONTROL"


def test_320_plan_fails_closed_on_parent_provenance_or_escalation_drift() -> None:
    bad = _result()
    bad["training_execution_sha"] = "wrong"
    with pytest.raises(ValueError, match="training SHA"):
        execution_plan_from_160_result(bad)
    bad = _result()
    bad["evaluator_sha"] = "wrong"
    with pytest.raises(ValueError, match="evaluator SHA"):
        execution_plan_from_160_result(bad)
    bad = _result()
    bad["selection"]["next_level"] = 640
    with pytest.raises(ValueError, match="does not authorize escalation to 320"):
        execution_plan_from_160_result(bad)
    bad = _result()
    bad["selection"]["selected_candidate"] = "PF2_33_50_75_100_AI"
    with pytest.raises(ValueError, match="must not have a final candidate"):
        execution_plan_from_160_result(bad)


def test_320_execution_set_and_control_only_fields_are_recomputed_not_trusted() -> None:
    bad = _result()
    bad["selection"]["mandatory_next_level_execution"] = [CONTROL, REFEREE]
    with pytest.raises(ValueError, match="execution set drift"):
        execution_plan_from_160_result(bad)
    bad = _result()
    bad["selection"]["control_only_noneligible"] = []
    with pytest.raises(ValueError, match="control-only classification drift"):
        execution_plan_from_160_result(bad)


def test_320_config_differs_from_160_only_by_frozen_root_count() -> None:
    config = frozen_config_320()
    assert ROOTS_PER_ITERATION_160 == 32
    assert ROOTS_PER_ITERATION_320 == 64
    assert config.roots_per_iteration == 64
    assert config.total_iterations == ITERATIONS
    assert config.exact_opponent_levels == EXACT_OPPONENT_LEVELS
    assert config.reservoir_capacity == RESERVOIR_CAPACITY
    assert config.advantage_steps == ADVANTAGE_STEPS
    assert config.policy_steps == POLICY_STEPS
    assert config.batch_size == BATCH_SIZE
    assert config.learning_rate == LEARNING_RATE
    assert config.ensemble_size == ENSEMBLE_SIZE
    assert config.audit_size == AUDIT_SIZE
    assert config.epsilon_scale == EPSILON_SCALE
    assert config.epsilon_cap == EPSILON_CAP


def test_320_resume_contract_forbids_any_prior_level_checkpoint_at_iteration1() -> None:
    with pytest.raises(ValueError, match="iteration 1 must start fresh"):
        validate_resume_mode_320(target_iteration=1, resume_supplied=True, finalize=False)
    validate_resume_mode_320(target_iteration=1, resume_supplied=False, finalize=False)
    with pytest.raises(ValueError, match="require a 320 checkpoint"):
        validate_resume_mode_320(target_iteration=2, resume_supplied=False, finalize=False)
    validate_resume_mode_320(target_iteration=2, resume_supplied=True, finalize=False)
    with pytest.raises(ValueError, match="must include final"):
        validate_resume_mode_320(target_iteration=5, resume_supplied=True, finalize=False)
    validate_resume_mode_320(target_iteration=5, resume_supplied=True, finalize=True)
