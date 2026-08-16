from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "r7_5_3c_post_x4_frontier_guard.py"
spec = importlib.util.spec_from_file_location("post_x4_guard", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def base_result(status: str) -> dict:
    passing = status == "STABILITY_PASS"
    return {
        "schema": "SPINCORE_R7_5_3C_CHANCE_COVERAGE_X4_STABILITY_RESULT_V1",
        "status": status,
        "chance_coverage": {
            "multiplier": 4,
            "roots_per_iteration": 256,
            "iterations": 3,
            "roots_per_seed": 768,
            "independent_training_seeds": [1342191342, 1801739323],
            "production_deck_seed_semantics_preserved": True,
        },
        "frozen_hard_gates": {
            "cross_seed_mean_tv_max": 0.15,
            "cross_seed_p95_tv_max": 0.35,
            "all_local_training_gates_required": True,
        },
        "summary": {
            "training_cells": 8,
            "training_gate_pass_count": 8,
            "cross_seed_rows": 8,
            "cross_seed_gate_pass_count": 8 if passing else 7,
            "all_local_training_gates_pass": True,
            "all_cross_seed_gates_pass": passing,
            "stability_readmission_pass": passing,
        },
        "representation_winner": None,
        "selection_rule_changed": False,
        "changes_frozen_thresholds": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def test_pass_unlocks_only_complete_strategic_evaluation():
    verdict = mod.evaluate_frontier(base_result("STABILITY_PASS"))
    assert verdict["status"] == "VALID"
    assert verdict["may_run_complete_frozen_phase2_strategic_evaluation"] is True
    assert verdict["may_select_h2_or_h3_now"] is False
    assert verdict["may_run_r7_5_4_now"] is False
    assert verdict["may_authorize_r8"] is False
    assert verdict["ready_for_tables"] is False


def test_blocked_keeps_strategic_evaluation_locked():
    verdict = mod.evaluate_frontier(base_result("STABILITY_BLOCKED"))
    assert verdict["status"] == "VALID"
    assert verdict["may_run_complete_frozen_phase2_strategic_evaluation"] is False
    assert verdict["next_frontier"] == "FINAL_WINNER_INDEPENDENT_REMEDIATION_OR_CLOSE_R7_5_3_BLOCKED"


def test_gate_relaxation_is_rejected():
    result = base_result("STABILITY_PASS")
    result["frozen_hard_gates"]["cross_seed_mean_tv_max"] = 0.16
    verdict = mod.evaluate_frontier(result)
    assert verdict["status"] == "INVALID"
    assert "MEAN_TV_GATE_DRIFT" in verdict["failures"]


def test_illegal_representation_winner_is_rejected():
    result = base_result("STABILITY_PASS")
    result["representation_winner"] = "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL"
    verdict = mod.evaluate_frontier(result)
    assert verdict["status"] == "INVALID"
    assert "STABILITY_RESULT_ILLEGALLY_SELECTS_REPRESENTATION" in verdict["failures"]


def test_seed_or_chance_semantics_drift_is_rejected():
    result = deepcopy(base_result("STABILITY_PASS"))
    result["chance_coverage"]["independent_training_seeds"] = [1342191342, 1342191342]
    result["chance_coverage"]["production_deck_seed_semantics_preserved"] = False
    verdict = mod.evaluate_frontier(result)
    assert verdict["status"] == "INVALID"
    assert "TRAINING_SEED_IDENTITY_DRIFT" in verdict["failures"]
    assert "DECK_SEED_SEMANTICS_NOT_PRESERVED" in verdict["failures"]
