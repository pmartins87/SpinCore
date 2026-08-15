from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_phase2_eval import cross_seed_policy_stability
from spincore.r7_5_representation_v3_stage_contract import EVALUATION_SEEDS, TRAINING_SEEDS

SCHEMA = "SPINCORE_R7_5_3C_UPSTREAM_RNG_FACTORIAL_RESULT_V1"
CELL_SCHEMA = "SPINCORE_R7_5_3C_UPSTREAM_RNG_FACTORIAL_FINAL_CELL_V1"
DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")
REPRESENTATIONS = (H2_FINAL, H3_FINAL)


def _read_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _rows(cell: dict, evaluation_seed: int):
    found = [x for x in cell["evaluations"] if int(x["evaluation_seed"]) == int(evaluation_seed)]
    if len(found) != 1:
        raise RuntimeError("missing or duplicate evaluation seed")
    if [int(x) for x in found[0]["policy_state_indices"]] != list(range(1024)):
        raise RuntimeError("heldout state index drift")
    return found[0]["policy_rows"]


def _summary(rows: list[dict]) -> dict:
    metrics = [x["metric"] for x in rows]
    return {
        "comparisons": len(metrics),
        "mean_of_mean_tv": sum(float(x["mean"]) for x in metrics) / len(metrics),
        "mean_of_p95_tv": sum(float(x["p95"]) for x in metrics) / len(metrics),
        "max_mean_tv": max(float(x["mean"]) for x in metrics),
        "max_p95_tv": max(float(x["p95"]) for x in metrics),
        "reference_gate_pass_count": sum(bool(x["gate_pass"]) for x in metrics),
        "all_pass_frozen_cross_seed_thresholds_reference_only": all(bool(x["gate_pass"]) for x in metrics),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-root", type=Path, required=True)
    ap.add_argument("--policy-refit-result", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cells = {}
    for path in sorted(args.cells_root.rglob("cell.json.gz")):
        payload = _read_gz(path)
        if payload.get("schema") != CELL_SCHEMA:
            continue
        key = (
            str(payload["representation"]), str(payload["domain"]),
            int(payload["sampling_seed"]), int(payload["learning_seed"]),
        )
        if key in cells:
            raise RuntimeError(f"duplicate factorial cell {key}")
        cells[key] = payload
    expected = {
        (rep, domain, int(sampling), int(learning))
        for rep in REPRESENTATIONS
        for domain in DOMAINS
        for sampling in TRAINING_SEEDS
        for learning in TRAINING_SEEDS
    }
    if set(cells) != expected:
        raise RuntimeError(f"factorial inventory mismatch missing={sorted(expected-set(cells))} extra={sorted(set(cells)-expected)}")

    seed_a, seed_b = map(int, TRAINING_SEEDS)
    learning_rows = []
    sampling_rows = []
    for rep in REPRESENTATIONS:
        for domain in DOMAINS:
            for evaluation_seed in EVALUATION_SEEDS:
                for sampling in TRAINING_SEEDS:
                    left = _rows(cells[(rep, domain, int(sampling), seed_a)], int(evaluation_seed))
                    right = _rows(cells[(rep, domain, int(sampling), seed_b)], int(evaluation_seed))
                    learning_rows.append({
                        "representation": rep,
                        "domain": domain,
                        "evaluation_seed": int(evaluation_seed),
                        "fixed_sampling_seed": int(sampling),
                        "contrast": "SAME_SAMPLING_DIFFERENT_LEARNING_MEMORY_SEED",
                        "metric": cross_seed_policy_stability(left, right),
                    })
                for learning in TRAINING_SEEDS:
                    left = _rows(cells[(rep, domain, seed_a, int(learning))], int(evaluation_seed))
                    right = _rows(cells[(rep, domain, seed_b, int(learning))], int(evaluation_seed))
                    sampling_rows.append({
                        "representation": rep,
                        "domain": domain,
                        "evaluation_seed": int(evaluation_seed),
                        "fixed_learning_seed": int(learning),
                        "contrast": "DIFFERENT_SAMPLING_SAME_LEARNING_MEMORY_SEED",
                        "metric": cross_seed_policy_stability(left, right),
                    })

    prior = json.loads(args.policy_refit_result.read_text(encoding="utf-8"))
    if prior.get("schema") != "SPINCORE_R7_5_3C_POLICY_REFIT_DECOMPOSITION_RESULT_V1":
        raise RuntimeError("wrong policy-refit decomposition baseline schema")
    prior_upstream = dict(prior["summary"]["upstream_strategy_memory_sensitivity"])
    learning_summary = _summary(learning_rows)
    sampling_summary = _summary(sampling_rows)
    result = {
        "schema": SCHEMA,
        "status": "DIAGNOSTIC_COMPLETE",
        "purpose": "Winner-independent 2x2 decomposition of upstream SPNNIV3 strategy-memory instability into deck/traversal sampling randomness versus learning/reservoir randomness, with a common fixed final AveragePolicy learner.",
        "design": {
            "representations": list(REPRESENTATIONS),
            "domains": list(DOMAINS),
            "sampling_seeds": list(map(int, TRAINING_SEEDS)),
            "learning_memory_seeds": list(map(int, TRAINING_SEEDS)),
            "training_cells": len(cells),
            "phase2_iterations": 3,
            "roots_per_iteration": 64,
            "advantage_steps_per_member_per_iteration": 4096,
            "final_policy_steps": 16384,
            "heldout_states_per_metric": 1024,
            "final_policy_learner_is_common_across_all_cells": True,
        },
        "prior_combined_upstream_baseline": prior_upstream,
        "same_sampling_different_learning_memory": learning_rows,
        "different_sampling_same_learning_memory": sampling_rows,
        "summary": {
            "learning_memory_randomness_sensitivity": learning_summary,
            "sampling_traversal_randomness_sensitivity": sampling_summary,
            "interpretation_rule": "The larger controlled contrast identifies the more important upstream instability family. SAME_SAMPLING_DIFFERENT_LEARNING_MEMORY_SEED includes advantage initialization/reset/side-ensemble, optimizer minibatch randomness, and advantage/strategy reservoir replacement randomness. DIFFERENT_SAMPLING_SAME_LEARNING_MEMORY_SEED includes exact deck schedules and collector stochastic traversal/action draws. This diagnostic does not yet split subcomponents inside either family and cannot select H2 or H3.",
        },
        "representation_winner": None,
        "selection_rule_changed": False,
        "changes_frozen_thresholds": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
