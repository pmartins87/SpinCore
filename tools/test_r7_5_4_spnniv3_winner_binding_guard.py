from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("r7_5_4_spnniv3_winner_binding_guard.py")
spec = importlib.util.spec_from_file_location("winner_binding_guard", MODULE_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def transfer_precommit() -> dict:
    return {
        "schema": mod.TRANSFER_SCHEMA,
        "status": "FROZEN_BEFORE_PHASE2_STRATEGIC_AGGREGATE_OUTPUT",
        "phase2_strategic_aggregate_result_seen_before_freeze": False,
        "phase2_winner_known_before_freeze": False,
        "winner_independent_transfer_rule": {
            "structural_transfer_gate_required": True,
            "strategic_action_revalidation_required": True,
            "historical_action_result_may_not_be_relabelled_as_spnniv3_result": True,
        },
    }


def selected(winner: str) -> dict:
    return {
        "schema": mod.PHASE2_RESULT_SCHEMA,
        "status": "SELECTED",
        "winner": winner,
        "decision": {"status": "SELECTED", "winner": winner},
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def blocked() -> dict:
    return {
        "schema": mod.PHASE2_RESULT_SCHEMA,
        "status": "BLOCKED",
        "winner": None,
        "decision": {"status": "BLOCKED", "winner": None},
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def expect_pass(winner: str, representation: str) -> None:
    result = mod.evaluate(selected(winner), transfer_precommit())
    assert result["winner_binding_pass"] is True
    assert result["winner"] == winner
    assert result["provisional_representation"] == representation
    assert result["r7_5_4_strategic_execution_authorized"] is False
    assert result["structural_transfer_gate_still_required"] is True
    assert result["r7_5_4_strategic_revalidation_still_required"] is True
    assert result["production_training_authorized"] is False
    assert result["ready_for_tables"] is False


def expect_blocked(strategic: dict, transfer: dict, failure: str) -> None:
    result = mod.evaluate(strategic, transfer)
    assert result["winner_binding_pass"] is False
    assert result["winner"] is None
    assert result["provisional_representation"] is None
    assert failure in result["failures"]
    assert result["r7_5_4_strategic_execution_authorized"] is False


def main() -> int:
    expect_pass("H2", "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL")
    expect_pass("H3", "H3_HYBRID_EXACT_SEMANTIC_FINAL")
    expect_blocked(blocked(), transfer_precommit(), "REPRESENTATION_NOT_SELECTED")

    bad = selected("H2")
    bad["decision"] = {"status": "SELECTED", "winner": "H3"}
    expect_blocked(bad, transfer_precommit(), "DECISION_TOP_LEVEL_IDENTITY_MISMATCH")

    bad = selected("H2")
    bad["production_training_authorized"] = True
    expect_blocked(bad, transfer_precommit(), "STRATEGIC_RESULT_IMPROPERLY_AUTHORIZES_PRODUCTION")

    bad_transfer = copy.deepcopy(transfer_precommit())
    bad_transfer["phase2_winner_known_before_freeze"] = True
    expect_blocked(selected("H2"), bad_transfer, "TRANSFER_PRECOMMIT_WINNER_WAS_KNOWN")

    bad_transfer = copy.deepcopy(transfer_precommit())
    bad_transfer["winner_independent_transfer_rule"]["structural_transfer_gate_required"] = False
    expect_blocked(selected("H3"), bad_transfer, "STRUCTURAL_TRANSFER_GATE_NOT_REQUIRED")

    print("R7.5.4 SPNNIV3 winner-binding guard tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
