from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_phase2_eval import cross_seed_policy_stability
from spincore.r7_5_representation_v3_stage_contract import EVALUATION_SEEDS, TRAINING_SEEDS

SCHEMA = "SPINCORE_R7_5_3C_POLICY_REFIT_DECOMPOSITION_RESULT_V1"
CELL_SCHEMA = "SPINCORE_R7_5_3C_POLICY_REFIT_DECOMPOSITION_CELL_V1"


def read_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def short(rep: str) -> str:
    return "H2" if rep == H2_FINAL else "H3"


def rows_for(cell: dict, evaluation_seed: int):
    matches = [x for x in cell["evaluations"] if int(x["evaluation_seed"]) == int(evaluation_seed)]
    if len(matches) != 1:
        raise RuntimeError("missing/duplicate evaluation seed in decomposition cell")
    if [int(x) for x in matches[0]["policy_state_indices"]] != list(range(1024)):
        raise RuntimeError("policy state index drift")
    return matches[0]["policy_rows"]


def summarize(metrics: list[dict]) -> dict:
    return {
        "comparisons": len(metrics),
        "mean_of_mean_tv": sum(float(x["mean"]) for x in metrics) / len(metrics),
        "mean_of_p95_tv": sum(float(x["p95"]) for x in metrics) / len(metrics),
        "all_pass_frozen_cross_seed_thresholds_reference_only": all(bool(x["gate_pass"]) for x in metrics),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-root", type=Path, required=True)
    ap.add_argument("--blocker-evidence", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cells = {}
    for path in sorted(args.cells_root.rglob("cell.json.gz")):
        payload = read_gz(path)
        if payload.get("schema") != CELL_SCHEMA:
            continue
        key = (
            str(payload["representation"]), str(payload["domain"]),
            int(payload["source_training_seed"]), int(payload["learner_seed"]),
        )
        if key in cells:
            raise RuntimeError(f"duplicate decomposition cell {key}")
        cells[key] = payload
    expected = {
        (rep, domain, source, learner)
        for rep in (H2_FINAL, H3_FINAL)
        for domain in ("TRUE_HEADS_UP", "THREE_HANDED")
        for source in TRAINING_SEEDS
        for learner in TRAINING_SEEDS
    }
    if set(cells) != expected:
        raise RuntimeError(f"decomposition cell inventory mismatch missing={sorted(expected-set(cells))} extra={sorted(set(cells)-expected)}")

    blocker = json.loads(args.blocker_evidence.read_text(encoding="utf-8"))
    baseline = {
        (row["representation"], row["domain"], int(row["evaluation_seed"])): row
        for row in blocker["cross_seed_rows"]
    }

    same_memory = []
    cross_memory = []
    baseline_rows = []
    seed_a, seed_b = map(int, TRAINING_SEEDS)
    for rep in (H2_FINAL, H3_FINAL):
        for domain in ("TRUE_HEADS_UP", "THREE_HANDED"):
            for evaluation_seed in EVALUATION_SEEDS:
                b = baseline[(short(rep), domain, int(evaluation_seed))]
                baseline_rows.append({
                    "representation": rep, "domain": domain, "evaluation_seed": int(evaluation_seed),
                    "metric": {k: b[k] for k in ("mean", "p95", "max", "gate_pass")},
                })
                for source in TRAINING_SEEDS:
                    left = rows_for(cells[(rep, domain, int(source), seed_a)], int(evaluation_seed))
                    right = rows_for(cells[(rep, domain, int(source), seed_b)], int(evaluation_seed))
                    metric = cross_seed_policy_stability(left, right)
                    same_memory.append({
                        "representation": rep,
                        "domain": domain,
                        "evaluation_seed": int(evaluation_seed),
                        "source_training_seed": int(source),
                        "contrast": "SAME_FROZEN_STRATEGY_MEMORY_DIFFERENT_FINAL_POLICY_LEARNER_SEED",
                        "metric": metric,
                    })
                for learner in TRAINING_SEEDS:
                    left = rows_for(cells[(rep, domain, seed_a, int(learner))], int(evaluation_seed))
                    right = rows_for(cells[(rep, domain, seed_b, int(learner))], int(evaluation_seed))
                    metric = cross_seed_policy_stability(left, right)
                    cross_memory.append({
                        "representation": rep,
                        "domain": domain,
                        "evaluation_seed": int(evaluation_seed),
                        "learner_seed": int(learner),
                        "contrast": "DIFFERENT_FROZEN_STRATEGY_MEMORY_COMMON_FINAL_POLICY_LEARNER_SEED",
                        "metric": metric,
                    })

    same_metrics = [x["metric"] for x in same_memory]
    memory_metrics = [x["metric"] for x in cross_memory]
    result = {
        "schema": SCHEMA,
        "status": "DIAGNOSTIC_COMPLETE",
        "purpose": "Decompose the observed SPNNIV3 cross-seed AvgPolicy instability into final-policy learner sensitivity versus upstream frozen strategy-memory/corpus sensitivity without rerunning CFR and without changing any strategic gate.",
        "design": {
            "source_final_checkpoints": 8,
            "diagnostic_refits": 16,
            "refit_steps_each": 16384,
            "same_memory_different_learner_comparisons": len(same_memory),
            "different_memory_common_learner_comparisons": len(cross_memory),
            "heldout_states_per_comparison": 1024,
            "learner_seeds": list(map(int, TRAINING_SEEDS)),
        },
        "baseline_original_cross_seed": baseline_rows,
        "same_memory_different_learner": same_memory,
        "different_memory_common_learner": cross_memory,
        "summary": {
            "final_policy_learner_sensitivity": summarize(same_metrics),
            "upstream_strategy_memory_sensitivity": summarize(memory_metrics),
            "interpretation_rule": "If same-memory/different-learner TV is large, final AveragePolicy optimization/init/batch randomness is a material source. If different-memory/common-learner TV remains large while same-memory learner TV is small, instability is already present in the strategy memories produced upstream. This result does not by itself separate deck/traversal, reservoir, or advantage-training randomness inside the upstream path.",
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
