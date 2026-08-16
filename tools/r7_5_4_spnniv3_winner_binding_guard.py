from __future__ import annotations

import argparse
import json
from pathlib import Path

PHASE2_RESULT_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_EVALUATION_RESULT_V1"
TRANSFER_SCHEMA = "SPINCORE_R7_5_4_SPNNIV3_ACTION_SIZING_TRANSFER_PRECOMMIT_V1"
OUTPUT_SCHEMA = "SPINCORE_R7_5_4_SPNNIV3_WINNER_BINDING_GUARD_V1"

WINNER_TO_REPRESENTATION = {
    "H2": "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL",
    "H3": "H3_HYBRID_EXACT_SEMANTIC_FINAL",
}


def _read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(strategic_result: dict, transfer: dict) -> dict:
    failures: list[str] = []

    if strategic_result.get("schema") != PHASE2_RESULT_SCHEMA:
        failures.append("STRATEGIC_RESULT_SCHEMA")
    if transfer.get("schema") != TRANSFER_SCHEMA:
        failures.append("TRANSFER_PRECOMMIT_SCHEMA")
    if transfer.get("status") != "FROZEN_BEFORE_PHASE2_STRATEGIC_AGGREGATE_OUTPUT":
        failures.append("TRANSFER_PRECOMMIT_STATUS")
    if bool(transfer.get("phase2_strategic_aggregate_result_seen_before_freeze")):
        failures.append("TRANSFER_PRECOMMIT_NOT_PREOUTPUT")
    if bool(transfer.get("phase2_winner_known_before_freeze")):
        failures.append("TRANSFER_PRECOMMIT_WINNER_WAS_KNOWN")

    status = strategic_result.get("status")
    winner = strategic_result.get("winner")
    decision = strategic_result.get("decision") or {}
    if status != "SELECTED":
        failures.append("REPRESENTATION_NOT_SELECTED")
    if winner not in WINNER_TO_REPRESENTATION:
        failures.append("WINNER_NOT_H2_OR_H3")
    if decision.get("status") != status or decision.get("winner") != winner:
        failures.append("DECISION_TOP_LEVEL_IDENTITY_MISMATCH")
    if bool(strategic_result.get("production_training_authorized")):
        failures.append("STRATEGIC_RESULT_IMPROPERLY_AUTHORIZES_PRODUCTION")
    if bool(strategic_result.get("ready_for_tables")):
        failures.append("STRATEGIC_RESULT_IMPROPERLY_AUTHORIZES_TABLES")

    provisional_representation = WINNER_TO_REPRESENTATION.get(str(winner))
    transfer_rule = transfer.get("winner_independent_transfer_rule") or {}
    if not bool(transfer_rule.get("structural_transfer_gate_required")):
        failures.append("STRUCTURAL_TRANSFER_GATE_NOT_REQUIRED")
    if not bool(transfer_rule.get("strategic_action_revalidation_required")):
        failures.append("STRATEGIC_ACTION_REVALIDATION_NOT_REQUIRED")
    if not bool(transfer_rule.get("historical_action_result_may_not_be_relabelled_as_spnniv3_result")):
        failures.append("HISTORICAL_RESULT_RELABEL_GUARD_MISSING")

    winner_binding_pass = not failures
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "WINNER_BOUND_FOR_TRANSFER_PREPARATION" if winner_binding_pass else "BLOCKED",
        "failures": failures,
        "winner": winner if winner_binding_pass else None,
        "provisional_representation": provisional_representation if winner_binding_pass else None,
        "winner_binding_pass": winner_binding_pass,
        "structural_transfer_gate_still_required": True,
        "r7_5_4_strategic_revalidation_still_required": True,
        "r7_5_4_strategic_execution_authorized": False,
        "selection_rule_changed": False,
        "thresholds_relaxed": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed bridge from a frozen SPNNIV3 H2/H3 strategic winner to R7.5.4 transfer preparation."
    )
    parser.add_argument("--strategic-result", type=Path, required=True)
    parser.add_argument("--transfer-precommit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate(_read(args.strategic_result), _read(args.transfer_precommit))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["winner_binding_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
