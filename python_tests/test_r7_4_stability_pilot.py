from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = _load("r7_4_stability_pilot_worker_test", TOOLS / "r7_4_stability_pilot_worker.py")
orchestrator = _load("run_r7_4_stability_pilot_test", TOOLS / "run_r7_4_stability_pilot.py")


def _freeze():
    return {
        "schema": "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1",
        "evidence_sha256": "a" * 64,
        "algorithm_seeds": [20260829, 20260807],
        "behavior_semantic_id": "SPINCORE_R7_3_UNCERTAINTY_POLICY_MIXTURE_V1",
        "evidence_commit_sha": "b" * 40,
        "source_head_sha": "c" * 40,
        "thread_environment_contract": "SOURCE_WORKFLOW_NO_EXPLICIT_THREAD_OVERRIDE",
    }


def test_r7_4_heldout_seed_derivation_is_deterministic_and_disjoint():
    f = _freeze()
    first = worker._heldout_seeds(f)
    second = worker._heldout_seeds(f)
    assert first == second
    assert len(first) == 2 and len(set(first)) == 2
    assert all(seed > 0 for seed in first)
    assert not set(first) & set(f["algorithm_seeds"])


def test_r7_4_seed_derivation_changes_when_frozen_evidence_changes():
    a = _freeze()
    b = dict(a)
    b["evidence_sha256"] = "d" * 64
    assert worker._heldout_seeds(a) != worker._heldout_seeds(b)


def test_r7_4_scenario_cycles_cover_domain_positions_and_preserve_chips():
    hu = worker._scenario_cycle("TRUE_HEADS_UP")
    three = worker._scenario_cycle("THREE_HANDED")
    assert len(hu) == 6
    assert len(three) == 15
    assert {ep.dealer_id for ep in hu} == {1, 2}
    assert {ep.dealer_id for ep in three} == {0, 1, 2}
    assert all(sum(ep.stacks) == 1500 for ep in hu + three)
    assert all(ep.game_is_hu for ep in hu)
    assert all(not ep.game_is_hu for ep in three)
    assert all(ep.dead_players == (0,) for ep in hu)
    assert all(ep.dead_players == () for ep in three)


def test_r7_4_scenario_cycle_rejects_unknown_domain():
    with pytest.raises(ValueError):
        worker._scenario_cycle("OTHER")


def test_r7_4_orchestrator_requires_structural_preflight_and_exact_accepted_source():
    f = _freeze()
    acceptance = {
        "schema": "SPINCORE_R7_3_FROZEN_CANDIDATE_640_ACCEPTANCE_V1",
        "r7_3_640_acceptance_pass": True,
        "r7_3_ready_to_advance_to_r7_4": True,
        "per_seed_fit_pass": True,
        "iterations": 5,
        "roots_per_iteration": 128,
        "roots_per_seed": 640,
        "source_head_sha": f["source_head_sha"],
        "durability_evidence_commit_sha": f["evidence_commit_sha"],
        "thread_environment_contract": f["thread_environment_contract"],
        "thread_environment_overrides_injected_by_certifier": False,
        "acceptance_gate_changed": False,
    }
    preflight = {
        "schema": orchestrator.PREFLIGHT_SCHEMA,
        "r7_4_structural_preflight_pass": True,
        "exact_accepted_solver_source_used": True,
        "preflight_worker_executed_from_accepted_worktree_overlay": True,
        "r7_3_source_head_sha": f["source_head_sha"],
        "r7_3_durability_evidence_commit_sha": f["evidence_commit_sha"],
        "ready_for_tables": False,
    }
    assert orchestrator._validate_prerequisites(f, acceptance, preflight) == f["source_head_sha"]

    bad = dict(preflight)
    bad["r7_4_structural_preflight_pass"] = False
    with pytest.raises(ValueError):
        orchestrator._validate_prerequisites(f, acceptance, bad)
