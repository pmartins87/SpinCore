from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("r8_2_calibration_readiness_guard.py")
spec = importlib.util.spec_from_file_location("r8_2_guard", MODULE_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def precommit() -> dict:
    return {
        "schema": mod.PRECOMMIT_SCHEMA,
        "status": "FROZEN_BEFORE_R8_2_PHYSICAL_CALIBRATION",
        "execution_seen_before_freeze": False,
        "physical_calibration_result_seen_before_freeze": False,
        "prerequisite_contract": {
            "r8_selected_state": {
                "required_schema": mod.SELECTED_STATE_SCHEMA,
                "required_status": "PASS",
            },
            "r7_5_5_architecture_closure": {
                "required_schema": mod.ARCHITECTURE_SCHEMA,
                "require_universal_action_width": 10,
            },
        },
    }


def selected_state() -> dict:
    return {
        "schema": mod.SELECTED_STATE_SCHEMA,
        "status": "PASS",
        "r8_0_status": "PASS",
        "selected_state_hash": "selected-state-sha256",
        "primary_evidence_sha256": "primary-evidence-sha256",
    }


def architecture(rep: str = "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL") -> dict:
    return {
        "schema": mod.ARCHITECTURE_SCHEMA,
        "status": "PASS",
        "architecture_finalized": True,
        "representation": rep,
        "action_sizing_finalized": True,
        "universal_action_width": 10,
        "production_profile_sha256": "production-profile-sha256",
    }


def expect_blocked(pc: dict, ss: dict, arch: dict, failure: str) -> None:
    result = mod.evaluate(pc, ss, arch)
    assert result["status"] == "BLOCKED"
    assert result["r8_2_physical_calibration_authorized"] is False
    assert failure in result["failures"]
    assert result["production_training_authorized"] is False
    assert result["ready_for_tables"] is False


def main() -> int:
    for rep in sorted(mod.ALLOWED_REPRESENTATIONS):
        result = mod.evaluate(precommit(), selected_state(), architecture(rep))
        assert result["status"] == "PASS"
        assert result["r8_2_physical_calibration_authorized"] is True
        assert result["runtime_machine_guard_still_required"] is True
        assert result["calibration_may_change_strategy_architecture"] is False
        assert result["production_training_authorized"] is False
        assert result["ready_for_tables"] is False

    bad = copy.deepcopy(selected_state())
    bad["status"] = "BLOCKED"
    expect_blocked(precommit(), bad, architecture(), "R8_0_NOT_PASS")

    bad_arch = copy.deepcopy(architecture())
    bad_arch["architecture_finalized"] = False
    expect_blocked(precommit(), selected_state(), bad_arch, "ARCHITECTURE_NOT_FINALIZED")

    bad_arch = copy.deepcopy(architecture())
    bad_arch["universal_action_width"] = 6
    expect_blocked(precommit(), selected_state(), bad_arch, "ACTION_WIDTH_NOT_10")

    bad_arch = copy.deepcopy(architecture())
    bad_arch["representation"] = "C0_V1_FROZEN_CONTROL"
    expect_blocked(precommit(), selected_state(), bad_arch, "REPRESENTATION_NOT_FINAL_H2_H3")

    bad_pc = copy.deepcopy(precommit())
    bad_pc["execution_seen_before_freeze"] = True
    expect_blocked(bad_pc, selected_state(), architecture(), "EXECUTION_SEEN_BEFORE_FREEZE")

    print("R8.2 calibration readiness guard tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
