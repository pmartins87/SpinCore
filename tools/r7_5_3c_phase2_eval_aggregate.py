from __future__ import annotations

import argparse
import json
from pathlib import Path

from spincore.r7_5_representation_v3_phase2_eval import H2, H3, resolve_frozen_winner
from spincore.r7_5_representation_v3_phase2_eval_io import (
    load_cells,
    recompute_hard_gates,
    require_exact_inventory,
    validate_training_inventory,
)
from spincore.r7_5_representation_v3_phase2_eval_strategic import (
    common_reference_summary,
    local_deviation_summary,
    pairwise_summary,
)

RESULT_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_EVALUATION_RESULT_V1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate frozen R7.5.3C Phase2 strategic evidence")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--training-inventory", type=Path, required=True)
    parser.add_argument("--training-execution-sha", required=True)
    parser.add_argument("--heldout-execution-sha", required=True)
    parser.add_argument("--evaluator-execution-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    inventory = validate_training_inventory(args.training_inventory, args.training_execution_sha)
    heldout, commonref, pairwise = load_cells(
        args.input_root,
        evaluator_sha=args.evaluator_execution_sha,
        training_sha=args.training_execution_sha,
    )
    require_exact_inventory(heldout, commonref, pairwise)
    if any(p.get("heldout_execution_sha") != args.heldout_execution_sha for p in heldout):
        raise RuntimeError("heldout execution SHA mismatch in strategic cells")

    hard, candidate_pass = recompute_hard_gates(heldout)
    localdev = local_deviation_summary(heldout)
    pair = pairwise_summary(pairwise)
    common = common_reference_summary(commonref)
    decision = resolve_frozen_winner(
        h2_hard_gate_pass=candidate_pass[H2],
        h3_hard_gate_pass=candidate_pass[H3],
        local_deviation_direction=localdev["overall_direction"],
        pairwise_crossplay_direction=pair["overall_direction"],
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": decision["status"],
        "winner": decision.get("winner"),
        "decision": decision,
        "evaluator_execution_sha": args.evaluator_execution_sha,
        "training_execution_sha": args.training_execution_sha,
        "heldout_execution_sha": args.heldout_execution_sha,
        "training_inventory_quality_flag_from_training_workflow": bool(inventory.get("training_quality_pass")),
        "hard_gates_recomputed_from_final_reports_sentinels_and_cross_seed_policy": hard,
        "local_deviation_proxy": localdev,
        "pairwise_h2_h3_crossplay": pair,
        "common_reference_crossplay": common,
        "rng_namespace_adjudication": {
            "implementation_namespace": "SpinCore|R7.5.3C|PHASE2|REFV1",
            "logical_evaluation_freeze_keys_are_suffixes_under_namespace": True,
            "candidate_or_training_seed_in_referee_rng_key": False,
        },
        "proxy_called_exact_exploitability": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "winner": result["winner"],
        "candidate_hard_gates": candidate_pass,
        "local_deviation_direction": localdev["overall_direction"],
        "pairwise_direction": pair["overall_direction"],
        "decision": decision,
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
