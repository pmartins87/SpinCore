from __future__ import annotations

"""Fail-closed transition guard for the R7.5.3C x4 stability result.

This tool does not select H2/H3 and never authorizes production.  It only
materializes the already-frozen dependency transition after the x4 stability
artifact exists:

STABILITY_PASS    -> complete frozen Phase-2 strategic evaluation may run.
STABILITY_BLOCKED -> strategic evaluation stays locked; finite-closure policy
                     requires the final permitted winner-independent remediation
                     or closure of R7.5.3 as FAIL/BLOCKED.
"""

import argparse
import json
from pathlib import Path

EXPECTED_SCHEMA = "SPINCORE_R7_5_3C_CHANCE_COVERAGE_X4_STABILITY_RESULT_V1"
EXPECTED_MEAN_TV_MAX = 0.15
EXPECTED_P95_TV_MAX = 0.35
EXPECTED_TRAINING_CELLS = 8
EXPECTED_CROSS_SEED_ROWS = 8


def evaluate_frontier(result: dict) -> dict:
    failures: list[str] = []

    if result.get("schema") != EXPECTED_SCHEMA:
        failures.append("WRONG_SCHEMA")

    status = result.get("status")
    if status not in {"STABILITY_PASS", "STABILITY_BLOCKED"}:
        failures.append("WRONG_STATUS")

    gates = dict(result.get("frozen_hard_gates") or {})
    if float(gates.get("cross_seed_mean_tv_max", -1.0)) != EXPECTED_MEAN_TV_MAX:
        failures.append("MEAN_TV_GATE_DRIFT")
    if float(gates.get("cross_seed_p95_tv_max", -1.0)) != EXPECTED_P95_TV_MAX:
        failures.append("P95_TV_GATE_DRIFT")
    if gates.get("all_local_training_gates_required") is not True:
        failures.append("LOCAL_TRAINING_GATE_NOT_REQUIRED")

    summary = dict(result.get("summary") or {})
    if int(summary.get("training_cells", -1)) != EXPECTED_TRAINING_CELLS:
        failures.append("TRAINING_CELL_COUNT")
    if int(summary.get("cross_seed_rows", -1)) != EXPECTED_CROSS_SEED_ROWS:
        failures.append("CROSS_SEED_ROW_COUNT")

    if result.get("representation_winner") is not None:
        failures.append("STABILITY_RESULT_ILLEGALLY_SELECTS_REPRESENTATION")
    for field in (
        "selection_rule_changed",
        "changes_frozen_thresholds",
        "production_training_authorized",
        "ready_for_tables",
    ):
        if result.get(field) is not False:
            failures.append(f"FORBIDDEN_TRUE_OR_MISSING:{field}")

    chance = dict(result.get("chance_coverage") or {})
    if int(chance.get("multiplier", -1)) != 4:
        failures.append("CHANCE_MULTIPLIER_DRIFT")
    if int(chance.get("roots_per_iteration", -1)) != 256:
        failures.append("ROOTS_PER_ITERATION_DRIFT")
    if int(chance.get("iterations", -1)) != 3:
        failures.append("ITERATION_COUNT_DRIFT")
    if int(chance.get("roots_per_seed", -1)) != 768:
        failures.append("ROOTS_PER_SEED_DRIFT")
    if chance.get("production_deck_seed_semantics_preserved") is not True:
        failures.append("DECK_SEED_SEMANTICS_NOT_PRESERVED")
    if list(chance.get("independent_training_seeds") or []) != [1342191342, 1801739323]:
        failures.append("TRAINING_SEED_IDENTITY_DRIFT")

    training_pass_count = int(summary.get("training_gate_pass_count", -1))
    cross_pass_count = int(summary.get("cross_seed_gate_pass_count", -1))
    all_training = summary.get("all_local_training_gates_pass") is True
    all_cross = summary.get("all_cross_seed_gates_pass") is True
    readmission = summary.get("stability_readmission_pass") is True

    if status == "STABILITY_PASS":
        if training_pass_count != EXPECTED_TRAINING_CELLS or not all_training:
            failures.append("PASS_WITHOUT_ALL_TRAINING_GATES")
        if cross_pass_count != EXPECTED_CROSS_SEED_ROWS or not all_cross:
            failures.append("PASS_WITHOUT_ALL_CROSS_SEED_GATES")
        if not readmission:
            failures.append("PASS_WITHOUT_READMISSION_PASS")
    elif status == "STABILITY_BLOCKED":
        if readmission:
            failures.append("BLOCKED_WITH_READMISSION_PASS")
        if all_training and all_cross:
            failures.append("BLOCKED_DESPITE_ALL_REQUIRED_GATES_PASSING")

    valid = not failures
    strategic_unlocked = bool(valid and status == "STABILITY_PASS")
    return {
        "schema": "SPINCORE_R7_5_3C_POST_X4_FRONTIER_GUARD_V1",
        "status": "VALID" if valid else "INVALID",
        "source_status": status,
        "failures": failures,
        "may_run_complete_frozen_phase2_strategic_evaluation": strategic_unlocked,
        "may_select_h2_or_h3_now": False,
        "may_run_r7_5_4_now": False,
        "may_authorize_h4": False,
        "may_authorize_r8": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
        "next_frontier": (
            "COMPLETE_FROZEN_PHASE2_STRATEGIC_EVALUATION"
            if strategic_unlocked
            else (
                "FINAL_WINNER_INDEPENDENT_REMEDIATION_OR_CLOSE_R7_5_3_BLOCKED"
                if valid and status == "STABILITY_BLOCKED"
                else "STOP_INVALID_EVIDENCE"
            )
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-closed post-x4 R7.5.3C transition guard")
    ap.add_argument("--stability-result", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    result = json.loads(args.stability_result.read_text(encoding="utf-8"))
    verdict = evaluate_frontier(result)
    text = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if verdict["status"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
