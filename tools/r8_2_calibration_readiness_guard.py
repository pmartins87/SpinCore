from __future__ import annotations

import argparse
import json
from pathlib import Path

PRECOMMIT_SCHEMA = "SPINCORE_R8_2_RYZEN_CALIBRATION_HANDOFF_PRECOMMIT_V1"
SELECTED_STATE_SCHEMA = "SPINCORE_R8_SELECTED_STATE_PACKET_V1"
ARCHITECTURE_SCHEMA = "SPINCORE_R7_5_5_ARCHITECTURE_CLOSURE_V1"
OUTPUT_SCHEMA = "SPINCORE_R8_2_CALIBRATION_READINESS_V1"
ALLOWED_REPRESENTATIONS = {
    "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL",
    "H3_HYBRID_EXACT_SEMANTIC_FINAL",
}


def _read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(precommit: dict, selected_state: dict, architecture: dict) -> dict:
    failures: list[str] = []

    if precommit.get("schema") != PRECOMMIT_SCHEMA:
        failures.append("PRECOMMIT_SCHEMA")
    if precommit.get("status") != "FROZEN_BEFORE_R8_2_PHYSICAL_CALIBRATION":
        failures.append("PRECOMMIT_STATUS")
    if bool(precommit.get("execution_seen_before_freeze")):
        failures.append("EXECUTION_SEEN_BEFORE_FREEZE")
    if bool(precommit.get("physical_calibration_result_seen_before_freeze")):
        failures.append("CALIBRATION_RESULT_SEEN_BEFORE_FREEZE")

    selected_contract = (precommit.get("prerequisite_contract") or {}).get("r8_selected_state") or {}
    if selected_state.get("schema") != SELECTED_STATE_SCHEMA:
        failures.append("SELECTED_STATE_SCHEMA")
    if selected_state.get("status") != "PASS" or selected_state.get("r8_0_status") not in (None, "PASS"):
        failures.append("R8_0_NOT_PASS")
    if not selected_state.get("selected_state_hash"):
        failures.append("SELECTED_STATE_HASH_MISSING")
    if not selected_state.get("primary_evidence_sha256"):
        failures.append("PRIMARY_EVIDENCE_HASH_MISSING")
    if selected_contract.get("required_schema") != SELECTED_STATE_SCHEMA:
        failures.append("PRECOMMIT_SELECTED_SCHEMA_MISMATCH")
    if selected_contract.get("required_status") != "PASS":
        failures.append("PRECOMMIT_SELECTED_STATUS_MISMATCH")

    arch_contract = (precommit.get("prerequisite_contract") or {}).get("r7_5_5_architecture_closure") or {}
    if architecture.get("schema") != ARCHITECTURE_SCHEMA:
        failures.append("ARCHITECTURE_SCHEMA")
    if architecture.get("status") != "PASS":
        failures.append("R7_5_5_NOT_PASS")
    if not bool(architecture.get("architecture_finalized")):
        failures.append("ARCHITECTURE_NOT_FINALIZED")
    representation = architecture.get("representation")
    if representation not in ALLOWED_REPRESENTATIONS:
        failures.append("REPRESENTATION_NOT_FINAL_H2_H3")
    if not bool(architecture.get("action_sizing_finalized")):
        failures.append("ACTION_SIZING_NOT_FINALIZED")
    if architecture.get("universal_action_width") != 10:
        failures.append("ACTION_WIDTH_NOT_10")
    if not architecture.get("production_profile_sha256"):
        failures.append("PRODUCTION_PROFILE_HASH_MISSING")
    if arch_contract.get("required_schema") != ARCHITECTURE_SCHEMA:
        failures.append("PRECOMMIT_ARCHITECTURE_SCHEMA_MISMATCH")
    if arch_contract.get("require_universal_action_width") != 10:
        failures.append("PRECOMMIT_ACTION_WIDTH_MISMATCH")

    passed = not failures
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "PASS" if passed else "BLOCKED",
        "failures": failures,
        "r8_0_selected_state_hash": selected_state.get("selected_state_hash") if passed else None,
        "r7_5_5_representation": representation if passed else None,
        "r7_5_5_production_profile_sha256": architecture.get("production_profile_sha256") if passed else None,
        "universal_action_width": 10 if passed else None,
        "r8_2_physical_calibration_authorized": passed,
        "runtime_machine_guard_still_required": True,
        "calibration_may_change_strategy_architecture": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed R8.2 calibration prerequisite guard.")
    parser.add_argument("--precommit", type=Path, required=True)
    parser.add_argument("--selected-state", type=Path, required=True)
    parser.add_argument("--architecture-closure", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate(_read(args.precommit), _read(args.selected_state), _read(args.architecture_closure))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["r8_2_physical_calibration_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
