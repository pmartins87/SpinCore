from __future__ import annotations

import pytest

from spincore.r7_5_action_evidence import conservative_domain_cost, learning_eligibility
from spincore.r7_5_action_stage_contract import POSTFLOP_TRAINING_SEEDS


def _report(seed: int, *, nodes: float, seconds: float, branches: float, full: float, pass_fit: bool = True):
    return {
        "candidate_id": "PF0_CONTROL_33_75_AI",
        "domain": "TRUE_HEADS_UP",
        "training_seed": int(seed),
        "selected_representation": "C0_V1_FROZEN_CONTROL",
        "iterations": 5,
        "roots_per_iteration": 32,
        "roots": 160,
        "nodes_per_root": float(nodes),
        "tree_seconds_per_root": float(seconds),
        "effective_unique_aggressive_branches_per_decision": float(branches),
        "peak_rss_bytes": 1000 + int(seed) % 100,
        "full_training_seconds_per_root": float(full),
        "advantage_gate_pass": bool(pass_fit),
        "policy_gate_pass": bool(pass_fit),
        "strategic_selection_permitted_at_160": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def _rows():
    a, b, c = POSTFLOP_TRAINING_SEEDS
    return (
        _report(a, nodes=10, seconds=1.0, branches=2.0, full=3.0),
        _report(b, nodes=50, seconds=1.2, branches=2.1, full=3.2),
        _report(c, nodes=12, seconds=9.0, branches=7.5, full=11.0),
    )


def test_domain_cost_uses_worst_seed_for_every_frozen_cost_metric() -> None:
    cost = conservative_domain_cost(
        _rows(), candidate_id="PF0_CONTROL_33_75_AI", domain="TRUE_HEADS_UP"
    )
    assert cost.nodes_per_root == 50.0
    assert cost.tree_seconds_per_root == 9.0
    assert cost.effective_branches_per_decision == 7.5
    assert cost.full_training_seconds_per_root == 11.0
    assert cost.per_seed_learning_gates_pass is True


def test_one_failed_seed_makes_learning_eligibility_fail() -> None:
    rows = list(_rows())
    rows[1] = {**rows[1], "policy_gate_pass": False}
    cross = {"candidate_id": "PF0_CONTROL_33_75_AI", "domain": "TRUE_HEADS_UP", "gate_pass": True}
    assert not learning_eligibility(
        rows,
        candidate_id="PF0_CONTROL_33_75_AI",
        domain="TRUE_HEADS_UP",
        cross_seed_report=cross,
    )


def test_missing_or_duplicate_seed_fails_closed() -> None:
    rows = list(_rows())
    with pytest.raises(ValueError):
        conservative_domain_cost(rows[:2], candidate_id="PF0_CONTROL_33_75_AI", domain="TRUE_HEADS_UP")
    rows[2] = dict(rows[1])
    with pytest.raises(ValueError):
        conservative_domain_cost(rows, candidate_id="PF0_CONTROL_33_75_AI", domain="TRUE_HEADS_UP")
