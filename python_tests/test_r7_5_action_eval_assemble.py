from __future__ import annotations

import pytest

from spincore.r7_5_action_eval_assemble import (
    ALL_CANDIDATES,
    DOMAINS,
    ELIGIBLE_CANDIDATES,
    REFEREE,
    aggregate_r7_5_4a_160,
    assemble_selection_evidence,
)
from spincore.r7_5_action_stage_contract import PAIRED_EVALUATION_SEEDS, POSTFLOP_TRAINING_SEEDS
from spincore.r7_5_eval_artifacts import (
    CANDIDATE_CELL_SCHEMA,
    EXPECTED_EXECUTION_SHA,
    CandidateCellEvidence,
)


def _seed_report(candidate: str, domain: str, seed: int, cost: float = 1.0):
    return {
        "candidate_id": candidate,
        "domain": domain,
        "training_seed": int(seed),
        "selected_representation": "C0_V1_FROZEN_CONTROL",
        "iterations": 5,
        "roots_per_iteration": 32,
        "roots": 160,
        "nodes_per_root": 10.0 * cost,
        "tree_seconds_per_root": 1.0 * cost,
        "effective_unique_aggressive_branches_per_decision": 2.0 * cost,
        "peak_rss_bytes": int(1000 * cost),
        "full_training_seconds_per_root": 3.0 * cost,
        "advantage_gate_pass": True,
        "policy_gate_pass": True,
        "strategic_selection_permitted_at_160": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def _cross_reports():
    out = []
    for candidate in ALL_CANDIDATES:
        for domain in DOMAINS:
            out.append({
                "schema": "SPINCORE_R7_5_4A_CROSS_SEED_POLICY_STABILITY_V1",
                "execution_sha": EXPECTED_EXECUTION_SHA,
                "candidate_id": candidate,
                "domain": domain,
                "mean_tv": 0.0,
                "p95_tv": 0.0,
                "gate_pass": True,
                "seed_reports": [
                    _seed_report(candidate, domain, seed)
                    for seed in POSTFLOP_TRAINING_SEEDS
                ],
                "production_training_authorized": False,
                "ready_for_tables": False,
            })
    return tuple(out)


def _candidate_cells():
    out = []
    for candidate_index, candidate in enumerate(ELIGIBLE_CANDIDATES):
        for domain in DOMAINS:
            for training_index, training_seed in enumerate(POSTFLOP_TRAINING_SEEDS):
                for evaluation_index, evaluation_seed in enumerate(PAIRED_EVALUATION_SEEDS):
                    # The value encodes canonical seed/eval order so the test can
                    # prove the assembler does not depend on filesystem order.
                    value = float(training_index * 10 + evaluation_index) / 1000.0
                    out.append(CandidateCellEvidence(
                        schema=CANDIDATE_CELL_SCHEMA,
                        execution_sha=EXPECTED_EXECUTION_SHA,
                        candidate_id=candidate,
                        domain=domain,
                        training_seed=int(training_seed),
                        evaluation_seed=int(evaluation_seed),
                        omission_samples=(value,),
                        crossplay_samples=(0.0,),
                        omission_summary={"overall": {"mean": value}},
                        crossplay_mean=0.0,
                    ))
    return tuple(reversed(out))


def test_assembly_uses_frozen_seed_then_evaluation_order_and_dense_zero_pairing() -> None:
    evidence, summaries = assemble_selection_evidence(
        candidate_cells=_candidate_cells(),
        cross_seed_reports=_cross_reports(),
        exact_counts=False,
    )
    expected = tuple(
        float(training_index * 10 + evaluation_index) / 1000.0
        for training_index, _seed in enumerate(POSTFLOP_TRAINING_SEEDS)
        for evaluation_index, _eval in enumerate(PAIRED_EVALUATION_SEEDS)
    )
    assert evidence["PF0_CONTROL_33_75_AI"].domains["TRUE_HEADS_UP"].omission_samples == expected
    assert evidence[REFEREE].domains["TRUE_HEADS_UP"].omission_samples == (0.0,) * len(expected)
    assert evidence[REFEREE].domains["TRUE_HEADS_UP"].crossplay_samples == (0.0,) * len(expected)
    assert summaries[REFEREE]["learning_gate_pass_both_domains"] is True


def test_160_result_can_prune_or_escalate_but_never_final_select_or_authorize_tables() -> None:
    result = aggregate_r7_5_4a_160(
        candidate_cells=_candidate_cells(),
        cross_seed_reports=_cross_reports(),
        evaluator_sha="evaluator-test-sha",
        training_run_id=12345,
        exact_counts=False,
    )
    assert result["selection"]["status"] == "PASS_LEVEL"
    assert result["selection"]["selected_candidate"] is None
    assert result["selection"]["next_level"] == 320
    assert result["r7_5_4a_postflop_selected"] is False
    assert result["production_training_authorized"] is False
    assert result["ready_for_tables"] is False


def test_missing_candidate_cell_fails_closed() -> None:
    with pytest.raises(ValueError, match="matrix mismatch"):
        assemble_selection_evidence(
            candidate_cells=_candidate_cells()[:-1],
            cross_seed_reports=_cross_reports(),
            exact_counts=False,
        )
