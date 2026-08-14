from __future__ import annotations

import json
from pathlib import Path

from spincore.production_trainability import (
    HARD_CAP_DAYS,
    RESERVE_MULTIPLIER,
    MIN_TIMING_SAMPLES_PER_STREAM,
    FULL_ITERATION_SCOPE,
    MATURE_COST_BASIS,
    NON_ITERATION_SCOPE,
    SCHEMA,
)

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"


def test_trainability_contract_matches_implementation_and_selected_representation() -> None:
    contract = json.loads(
        (VALIDATION / "R8_TRAINABILITY_TIME_BUDGET_CONTRACT_20260814.json").read_text(
            encoding="utf-8"
        )
    )
    result = json.loads(
        (VALIDATION / "R7_5_3_REPRESENTATION_ABLATION_RESULT.json").read_text(
            encoding="utf-8"
        )
    )

    assert contract["schema"] == "SPINCORE_R8_TRAINABILITY_TIME_BUDGET_CONTRACT_V2"
    assert contract["hard_cap"]["wall_clock_days"] == HARD_CAP_DAYS == 90.0
    assert contract["planning_reserve"]["multiplier"] == RESERVE_MULTIPLIER == 1.20
    assert contract["planning_reserve"]["implied_nominal_budget_days"] == 75.0
    physical = contract["physical_measurement_contract"]
    assert physical["minimum_complete_iteration_timing_samples_per_stream"] == MIN_TIMING_SAMPLES_PER_STREAM == 3
    assert physical["timing_unit"] == FULL_ITERATION_SCOPE
    assert physical["timing_cost_basis"] == MATURE_COST_BASIS
    assert physical["non_iteration_scope"] == NON_ITERATION_SCOPE

    projection = contract["projection_contract"]
    assert projection["schema"] == SCHEMA == "SPINCORE_R8_PRODUCTION_TRAINABILITY_V2"
    assert projection["required_domains"] == ["TRUE_HEADS_UP", "THREE_HANDED"]
    assert projection["all_non_iteration_training_and_freeze_work_required"] is True
    assert projection["scheduling_upper_bound"] == "LPT_STREAM_PINNED_CONSERVATIVE_UPPER_BOUND_PLUS_NON_ITERATION_WORK"

    selected = result["selected_candidate"]
    assert result["r7_5_3_representation_ablation_pass"] is True
    assert selected == "C0_V1_FROZEN_CONTROL"
    assert contract["representation_state_at_freeze"]["selected_candidate"] == selected
    assert contract["representation_state_at_freeze"]["serialized_observation_bytes"] == 126
    assert contract["representation_state_at_freeze"]["model_parameter_count"] == 152438

    forbidden = "\n".join(contract["forbidden_deadline_shortcuts"])
    assert "early low-cost" in forbidden
    assert "AveragePolicy" in forbidden
    assert contract["current_trainability_status"] == "NOT_MEASURED_PHYSICALLY / NOT_PASS"
    assert contract["production_training_authorized"] is False
    assert contract["ready_for_tables"] is False
