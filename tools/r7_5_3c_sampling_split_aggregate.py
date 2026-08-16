from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_phase2_eval import cross_seed_policy_stability
from spincore.r7_5_representation_v3_stage_contract import EVALUATION_SEEDS, TRAINING_SEEDS

SCHEMA = "SPINCORE_R7_5_3C_SAMPLING_SPLIT_RESULT_V1"
CELL_SCHEMA = "SPINCORE_R7_5_3C_SAMPLING_SPLIT_FINAL_CELL_V1"
DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")
REPRESENTATIONS = (H2_FINAL, H3_FINAL)
FIXED_LEARNING_SEED = 1801739323
DOMINANCE_RATIO = 1.20


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


def _classification(deck: dict, traversal: dict) -> dict:
    d_mean = float(deck["mean_of_mean_tv"])
    d_p95 = float(deck["mean_of_p95_tv"])
    t_mean = float(traversal["mean_of_mean_tv"])
    t_p95 = float(traversal["mean_of_p95_tv"])
    if d_mean >= DOMINANCE_RATIO * t_mean and d_p95 >= DOMINANCE_RATIO * t_p95:
        return {
            "classification": "DECK_CHANCE_DOMINANT",
            "next_action": "Freeze a winner-independent chance-coverage stabilization experiment that preserves independent chance seeds while increasing/stratifying deck coverage before full H2/H3 readmission.",
        }
    if t_mean >= DOMINANCE_RATIO * d_mean and t_p95 >= DOMINANCE_RATIO * d_p95:
        return {
            "classification": "TRAVERSAL_ACTION_SAMPLING_DOMINANT",
            "next_action": "Freeze a winner-independent variance-reduced traversal experiment, preserving unbiased policy sampling while reducing stochastic path variance, before full H2/H3 readmission.",
        }
    return {
        "classification": "BOTH_MATERIAL_OR_INTERACTION",
        "next_action": "Do not choose a subcomponent post hoc. Freeze a replicated split or a joint variance-reduction experiment with both independent chance coverage and traversal variance reduction, then reassess before full readmission.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-root", type=Path, required=True)
    ap.add_argument("--prior-evidence", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cells = {}
    for path in sorted(args.cells_root.rglob("cell.json.gz")):
        payload = _read_gz(path)
        if payload.get("schema") != CELL_SCHEMA:
            continue
        if int(payload.get("fixed_learning_seed", -1)) != FIXED_LEARNING_SEED:
            raise RuntimeError("fixed learning seed drift")
        key = (
            str(payload["representation"]), str(payload["domain"]),
            int(payload["deck_seed"]), int(payload["traversal_seed"]),
        )
        if key in cells:
            raise RuntimeError(f"duplicate sampling-split cell {key}")
        cells[key] = payload
    expected = {
        (rep, domain, int(deck), int(traversal))
        for rep in REPRESENTATIONS
        for domain in DOMAINS
        for deck in TRAINING_SEEDS
        for traversal in TRAINING_SEEDS
    }
    if set(cells) != expected:
        raise RuntimeError(f"sampling-split inventory mismatch missing={sorted(expected-set(cells))} extra={sorted(set(cells)-expected)}")

    seed_a, seed_b = map(int, TRAINING_SEEDS)
    deck_rows = []
    traversal_rows = []
    for rep in REPRESENTATIONS:
        for domain in DOMAINS:
            for evaluation_seed in EVALUATION_SEEDS:
                for traversal in TRAINING_SEEDS:
                    left = _rows(cells[(rep, domain, seed_a, int(traversal))], int(evaluation_seed))
                    right = _rows(cells[(rep, domain, seed_b, int(traversal))], int(evaluation_seed))
                    deck_rows.append({
                        "representation": rep,
                        "domain": domain,
                        "evaluation_seed": int(evaluation_seed),
                        "fixed_traversal_seed": int(traversal),
                        "contrast": "DIFFERENT_DECK_SAME_TRAVERSAL",
                        "metric": cross_seed_policy_stability(left, right),
                    })
                for deck in TRAINING_SEEDS:
                    left = _rows(cells[(rep, domain, int(deck), seed_a)], int(evaluation_seed))
                    right = _rows(cells[(rep, domain, int(deck), seed_b)], int(evaluation_seed))
                    traversal_rows.append({
                        "representation": rep,
                        "domain": domain,
                        "evaluation_seed": int(evaluation_seed),
                        "fixed_deck_seed": int(deck),
                        "contrast": "SAME_DECK_DIFFERENT_TRAVERSAL",
                        "metric": cross_seed_policy_stability(left, right),
                    })

    prior = json.loads(args.prior_evidence.read_text(encoding="utf-8"))
    if prior.get("schema") != "SPINCORE_R7_5_3C_UPSTREAM_RNG_FACTORIAL_EVIDENCE_V1":
        raise RuntimeError("wrong upstream factorial evidence schema")
    if prior.get("result_file_sha256") != "4a6a470e0e510b410a709d6aa2c130d55367c27cb3a542c3315d8e9f246d54d3":
        raise RuntimeError("upstream factorial evidence hash drift")

    deck_summary = _summary(deck_rows)
    traversal_summary = _summary(traversal_rows)
    classification = _classification(deck_summary, traversal_summary)
    result = {
        "schema": SCHEMA,
        "status": "DIAGNOSTIC_COMPLETE",
        "purpose": "Winner-independent split of the dominant upstream sampling/traversal instability into exact deck/chance schedule randomness versus collector stochastic traversal/action sampling randomness.",
        "design": {
            "representations": list(REPRESENTATIONS),
            "domains": list(DOMAINS),
            "deck_seeds": list(map(int, TRAINING_SEEDS)),
            "traversal_seeds": list(map(int, TRAINING_SEEDS)),
            "fixed_learning_seed": FIXED_LEARNING_SEED,
            "fixed_learning_seed_selection_namespace": "SpinCore|R7.5.3C|SAMPLING-SPLIT|FIXED-LEARNING-SEED-INDEX",
            "fixed_learning_seed_selection_index": 1,
            "training_cells": len(cells),
            "phase2_iterations": 3,
            "roots_per_iteration": 64,
            "advantage_steps_per_member_per_iteration": 4096,
            "final_policy_steps": 16384,
            "heldout_states_per_metric": 1024,
            "final_policy_learner_is_common_across_all_cells": True,
        },
        "prior_sampling_traversal_baseline": prior["sampling_traversal_randomness_sensitivity"],
        "different_deck_same_traversal": deck_rows,
        "same_deck_different_traversal": traversal_rows,
        "summary": {
            "deck_chance_sensitivity": deck_summary,
            "traversal_action_sampling_sensitivity": traversal_summary,
            "dominance_ratio_precommitted": DOMINANCE_RATIO,
            **classification,
            "interpretation_rule": "Deck is dominant only if both aggregate mean-TV and p95-TV are at least 1.20x traversal. Traversal is dominant only under the symmetric condition. Otherwise classify both material/interaction. Thresholds 0.15/0.35 remain diagnostic references only and are not relaxed admission gates.",
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
