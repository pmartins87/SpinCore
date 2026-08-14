from __future__ import annotations

import argparse
import json
from pathlib import Path

from spincore.r7_5_action_eval_assemble import (
    aggregate_r7_5_4a_160,
    discover_candidate_cells,
    discover_cross_seed_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-evidence-dir", required=True)
    parser.add_argument("--cross-seed-dir", required=True)
    parser.add_argument("--evaluator-sha", required=True)
    parser.add_argument("--training-run-id", required=True, type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = aggregate_r7_5_4a_160(
        candidate_cells=discover_candidate_cells(args.candidate_evidence_dir),
        cross_seed_reports=discover_cross_seed_reports(args.cross_seed_dir),
        evaluator_sha=args.evaluator_sha,
        training_run_id=args.training_run_id,
        exact_counts=True,
    )
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["selection"].get("status"),
        "survivors": result["selection"].get("survivors"),
        "next_level": result["selection"].get("next_level"),
        "ready_for_tables": result["ready_for_tables"],
    }, sort_keys=True), flush=True)
    if result["selection"].get("status") != "PASS_LEVEL":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
