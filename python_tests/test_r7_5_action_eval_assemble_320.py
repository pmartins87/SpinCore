from __future__ import annotations

from spincore.r7_5_action_320_contract import (
    CONTROL,
    EXPECTED_PARENT_EVALUATOR_SHA,
    EXPECTED_PARENT_TRAINING_SHA,
    REFEREE,
    execution_plan_from_160_result,
)
from spincore.r7_5_action_eval_assemble_320 import (
    CROSS_SEED_SCHEMA,
    RESULT_SCHEMA_320,
    aggregate_r7_5_4a_320,
)
from spincore.r7_5_action_stage_contract import (
    PAIRED_EVALUATION_SEEDS,
    POSTFLOP_TRAINING_SEEDS,
    SELECTED_REPRESENTATION,
)
from spincore.r7_5_eval_artifacts import CANDIDATE_CELL_SCHEMA, CandidateCellEvidence

EXECUTION_SHA_320 = "1234567890abcdef1234567890abcdef12345678"


def _parent_result(survivors):
    survivors = tuple(sorted(survivors))
    mandatory = tuple(sorted(set(survivors) | {CONTROL, REFEREE}))
    control_only = tuple(sorted(value for value in mandatory if value not in survivors and value != REFEREE))
    return {
        "schema": "SPINCORE_R7_5_4A_160_RESULT_V1",
        "root_level": 160,
        "training_execution_sha": EXPECTED_PARENT_TRAINING_SHA,
        "evaluator_sha": EXPECTED_PARENT_EVALUATOR_SHA,
        "r7_5_4a_postflop_selected": False,
        "r7_5_4a_postflop_selected_candidate": None,
        "production_training_authorized": False,
        "ready_for_tables": False,
        "selection": {
            "status": "PASS_LEVEL",
            "root_level": 160,
            "survivors": list(survivors),
            "selected_candidate": None,
            "next_level": 320,
            "mandatory_next_level_execution": list(mandatory),
            "control_only_noneligible": list(control_only),
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
    }


def _seed_report(candidate: str, domain: str, seed: int, *, cost: float = 10.0):
    return {
        "candidate_id": candidate,
        "domain": domain,
        "training_seed": int(seed),
        "selected_representation": SELECTED_REPRESENTATION,
        "iterations": 5,
        "roots_per_iteration": 64,
        "roots": 320,
        "nodes_per_root": float(cost),
        "tree_seconds_per_root": float(cost),
        "effective_unique_aggressive_branches_per_decision": float(cost),
        "full_training_seconds_per_root": float(cost),
        "peak_rss_bytes": 1000,
        "advantage_gate_pass": True,
        "policy_gate_pass": True,
        "strategic_selection_permitted_at_160": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def _cross_report(candidate: str, domain: str, *, cost: float = 10.0):
    return {
        "schema": CROSS_SEED_SCHEMA,
        "execution_sha": EXECUTION_SHA_320,
        "candidate_id": candidate,
        "domain": domain,
        "seed_reports": [
            _seed_report(candidate, domain, seed, cost=cost)
            for seed in POSTFLOP_TRAINING_SEEDS
        ],
        "mean_tv": 0.01,
        "p95_tv": 0.02,
        "gate_pass": True,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def _cell(candidate: str, domain: str, training_seed: int, evaluation_seed: int, *, omission: float, crossplay: float):
    return CandidateCellEvidence(
        schema=CANDIDATE_CELL_SCHEMA,
        execution_sha=EXECUTION_SHA_320,
        candidate_id=candidate,
        domain=domain,
        training_seed=int(training_seed),
        evaluation_seed=int(evaluation_seed),
        omission_samples=(float(omission),),
        crossplay_samples=(float(crossplay),),
        omission_summary={},
        crossplay_mean=float(crossplay),
    )


def _matrix(parent, scores, costs=None):
    plan = execution_plan_from_160_result(parent)
    costs = dict(costs or {})
    cells = []
    reports = []
    for candidate in plan.execution_ids:
        for domain in ("TRUE_HEADS_UP", "THREE_HANDED"):
            reports.append(_cross_report(candidate, domain, cost=float(costs.get(candidate, 10.0))))
            if candidate == REFEREE:
                continue
            omission, crossplay = scores[candidate]
            for training_seed in POSTFLOP_TRAINING_SEEDS:
                for evaluation_seed in PAIRED_EVALUATION_SEEDS:
                    cells.append(
                        _cell(
                            candidate,
                            domain,
                            training_seed,
                            evaluation_seed,
                            omission=omission,
                            crossplay=crossplay,
                        )
                    )
    return cells, reports


def test_control_only_cannot_resurrect_even_when_its_metrics_are_better() -> None:
    survivor = "PF2_33_50_75_100_AI"
    parent = _parent_result((survivor,))
    cells, reports = _matrix(
        parent,
        scores={
            CONTROL: (0.0, 1.0),
            survivor: (0.02, -1.0),
        },
        costs={CONTROL: 1.0, survivor: 100.0, REFEREE: 1.0},
    )
    result = aggregate_r7_5_4a_320(
        parent_160_result=parent,
        candidate_cells=cells,
        cross_seed_reports=reports,
        training_execution_sha=EXECUTION_SHA_320,
        evaluator_sha="eval-320",
        training_run_id=320001,
        exact_counts=False,
    )
    assert result["schema"] == RESULT_SCHEMA_320
    assert result["selection"]["selected_candidate"] == survivor
    assert result["r7_5_4a_postflop_selected_candidate"] == survivor
    assert result["candidate_summaries"][CONTROL]["strategically_eligible_at_320"] is False
    assert result["production_training_authorized"] is False
    assert result["ready_for_tables"] is False


def test_two_equivalent_survivors_escalate_to_640_without_fallback() -> None:
    a = "PF2_33_50_75_100_AI"
    b = "PF4_CRUSHER_COMPACT_40_66_100_AI"
    parent = _parent_result((a, b))
    cells, reports = _matrix(
        parent,
        scores={CONTROL: (0.5, -0.5), a: (0.01, 0.01), b: (0.01, 0.01)},
        costs={CONTROL: 20.0, a: 10.0, b: 10.0, REFEREE: 10.0},
    )
    result = aggregate_r7_5_4a_320(
        parent_160_result=parent,
        candidate_cells=cells,
        cross_seed_reports=reports,
        training_execution_sha=EXECUTION_SHA_320,
        evaluator_sha="eval-320",
        training_run_id=320002,
        exact_counts=False,
    )
    assert result["selection"]["selected_candidate"] is None
    assert result["selection"]["next_level"] == 640
    assert result["selection"]["fallback_used"] is False
    assert set(result["selection"]["survivors"]) == {a, b}
    assert result["r7_5_4a_postflop_selected"] is False


def test_320_report_matrix_rejects_embedded_160_root_reports() -> None:
    survivor = "PF2_33_50_75_100_AI"
    parent = _parent_result((survivor,))
    cells, reports = _matrix(parent, scores={CONTROL: (0.1, 0.0), survivor: (0.1, 0.0)})
    reports[0]["seed_reports"][0]["roots"] = 160
    try:
        aggregate_r7_5_4a_320(
            parent_160_result=parent,
            candidate_cells=cells,
            cross_seed_reports=reports,
            training_execution_sha=EXECUTION_SHA_320,
            evaluator_sha="eval-320",
            training_run_id=320003,
            exact_counts=False,
        )
    except ValueError as exc:
        assert "root-level" in str(exc)
    else:
        raise AssertionError("320 assembler accepted a 160-root embedded seed report")
