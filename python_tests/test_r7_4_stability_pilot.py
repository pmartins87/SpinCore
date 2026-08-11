from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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
screen_mod = _load("summarize_r7_4_heldout_screen_test", TOOLS / "summarize_r7_4_heldout_screen.py")
finalizer = _load("finalize_r7_4_gate_test", TOOLS / "finalize_r7_4_gate.py")


def _freeze():
    return {
        "schema": "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1",
        "evidence_r7_3_pass": True,
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


def test_r7_4_runtime_statistics_are_seed_local_not_registry_indexed():
    behavior = SimpleNamespace(
        calls=10,
        epsilon_sum=2.5,
        epsilon_max=0.5,
        disagreement_sum=1.25,
        raw_epsilon_max=0.75,
        cap_hit_calls=2,
        epsilon_ge_010_calls=7,
        epsilon_ge_025_calls=3,
    )
    row = worker._runtime_statistics_for_behavior(behavior, 123456)
    assert row["algorithm_seed"] == 123456
    assert row["fitted_behavior_calls"] == 10
    assert row["mean_epsilon"] == pytest.approx(0.25)
    assert row["mean_disagreement"] == pytest.approx(0.125)
    assert row["cap_hit_fraction"] == pytest.approx(0.2)
    assert row["epsilon_ge_0_10_fraction"] == pytest.approx(0.7)
    assert row["epsilon_ge_0_25_fraction"] == pytest.approx(0.3)


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


def _domain_evidence(freeze: dict, *, roots_per_iteration: int, passed: bool = True) -> dict:
    return {
        "schema": screen_mod.DOMAIN_SCHEMA,
        "domain": "THREE_HANDED",
        "accepted_r7_3_source_head_sha": freeze["source_head_sha"],
        "accepted_r7_3_evidence_commit_sha": freeze["evidence_commit_sha"],
        "accepted_r7_3_behavior_semantic_id": freeze["behavior_semantic_id"],
        "exact_accepted_algorithm_source_used": True,
        "pilot_worker_executed_from_accepted_worktree_overlay": True,
        "thread_environment_overrides_injected_by_r7_4_orchestrator": False,
        "r7_3_640_acceptance_passed_first": True,
        "r7_4_structural_preflight_passed_first": True,
        "iterations": 5,
        "roots_per_iteration": roots_per_iteration,
        "roots_per_seed": 5 * roots_per_iteration,
        "exact_opponent_levels": 2,
        "deck_formula": "seed*1000003 + global_root*97 + iteration",
        "extra_members_perturb_primary_rng": False,
        "r7_3_selection_seeds_reused": False,
        "scenario_coverage_pass": True,
        "acceptance_gate_changed": False,
        "algorithm_seeds": [111, 222],
        "per_seed_fit_pass": True,
        "cross_seed": {"mean_tv": 0.10, "p50_tv": 0.08, "p95_tv": 0.25, "max_tv": 0.5},
        "r7_4_domain_stability_pass": passed,
        "ready_for_tables": False,
    }


def test_r7_4_domain_validator_requires_exact_confirmation_scale_and_provenance():
    f = _freeze()
    row = _domain_evidence(f, roots_per_iteration=128)
    screen_mod._validate_domain(row, domain="THREE_HANDED", roots_per_iteration=128, freeze=f)
    bad = dict(row)
    bad["roots_per_seed"] = 320
    with pytest.raises(ValueError):
        screen_mod._validate_domain(bad, domain="THREE_HANDED", roots_per_iteration=128, freeze=f)


def test_r7_4_finalizer_is_finite_and_requires_three_handed_640_pass(tmp_path, monkeypatch):
    f = _freeze()
    freeze_path = tmp_path / "freeze.json"
    screen_path = tmp_path / "screen.json"
    three_path = tmp_path / "three.json"
    out_path = tmp_path / "final.json"
    freeze_path.write_text(__import__("json").dumps(f) + "\n", encoding="utf-8")
    screen = {
        "schema": finalizer.SCREEN_SCHEMA,
        "source_head_sha": f["source_head_sha"],
        "durability_evidence_commit_sha": f["evidence_commit_sha"],
        "behavior_semantic_id": f["behavior_semantic_id"],
        "heldout_algorithm_seeds": [111, 222],
        "r7_4_heldout_screen_pass": True,
        "hu": {"pass": True},
        "three_handed_screen": {"pass": True},
        "ready_for_tables": False,
    }
    screen_path.write_text(__import__("json").dumps(screen) + "\n", encoding="utf-8")
    three = _domain_evidence(f, roots_per_iteration=128, passed=True)
    three_path.write_text(__import__("json").dumps(three) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "finalize_r7_4_gate.py",
        "--freeze", str(freeze_path),
        "--screen", str(screen_path),
        "--three-handed-confirmation", str(three_path),
        "--out", str(out_path),
    ])
    assert finalizer.main() == 0
    final = __import__("json").loads(out_path.read_text(encoding="utf-8"))
    assert final["r7_4_pass"] is True
    assert final["r7_4_ready_to_advance_to_r8"] is True
    assert final["ready_for_tables"] is False

    three["r7_4_domain_stability_pass"] = False
    three_path.write_text(__import__("json").dumps(three) + "\n", encoding="utf-8")
    assert finalizer.main() == 0
    final = __import__("json").loads(out_path.read_text(encoding="utf-8"))
    assert final["r7_4_pass"] is False
    assert final["r7_4_ready_to_advance_to_r8"] is False
