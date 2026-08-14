from __future__ import annotations

import argparse
import json
from pathlib import Path

from spincore.r7_5_action_320_contract import execution_plan_from_160_result
from spincore.r7_5_action_eval_assemble_320 import (
    aggregate_r7_5_4a_320,
    discover_candidate_cells_320,
    discover_cross_seed_reports_320,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-160-result", required=True)
    parser.add_argument("--candidate-evidence-dir", required=True)
    parser.add_argument("--cross-seed-dir", required=True)
    parser.add_argument("--training-execution-sha", required=True)
    parser.add_argument("--evaluator-sha", required=True)
    parser.add_argument("--training-run-id", required=True, type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    parent = json.loads(Path(args.parent_160_result).read_text(encoding="utf-8"))
    plan = execution_plan_from_160_result(parent)
    result = aggregate_r7_5_4a_320(
        parent_160_result=parent,
        candidate_cells=discover_candidate_cells_320(
            args.candidate_evidence_dir,
            expected_execution_sha=args.training_execution_sha,
        ),
        cross_seed_reports=discover_cross_seed_reports_320(
            args.cross_seed_dir,
            plan=plan,
            expected_execution_sha=args.training_execution_sha,
        ),
        training_execution_sha=args.training_execution_sha,
        evaluator_sha=args.evaluator_sha,
        training_run_id=args.training_run_id,
        exact_counts=True,
    )
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection = result["selection"]
    print(json.dumps({
        "status": selection.get("status"),
        "survivors": selection.get("survivors"),
        "selected_candidate": selection.get("selected_candidate"),
        "next_level": selection.get("next_level"),
        "ready_for_tables": result["ready_for_tables"],
    }, sort_keys=True), flush=True)
    if selection.get("status") != "PASS_LEVEL":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
