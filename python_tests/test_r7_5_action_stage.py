from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spincore.r7_5_action_stage import (
    ActionStageConfig,
    finalize_stage_seed,
    load_stage_runtime,
    new_stage_runtime,
    run_one_stage_iteration,
    save_stage_runtime,
)
from spincore.solver import SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"
CANDIDATE = "PF0_CONTROL_33_75_AI"
DOMAIN = "TRUE_HEADS_UP"
SEED = 1737995611
EXECUTION_SHA = "mechanical-test-sha"


def _config() -> ActionStageConfig:
    return ActionStageConfig(
        roots_per_iteration=1,
        total_iterations=2,
        exact_opponent_levels=0,
        reservoir_capacity=4096,
        advantage_steps=1,
        policy_steps=1,
        batch_size=8,
        learning_rate=0.001,
        ensemble_size=4,
        audit_size=64,
        epsilon_scale=1.75,
        epsilon_cap=0.5,
    )


def _state_dict_equal(a, b) -> bool:
    return set(a) == set(b) and all(torch.equal(a[key], b[key]) for key in a)


def _new(solver, config):
    return new_stage_runtime(
        ROOT,
        solver=solver,
        candidate_id=CANDIDATE,
        domain=DOMAIN,
        training_seed=SEED,
        config=config,
    )


def test_stage_resume_reproduces_uninterrupted_model_and_rng_state(tmp_path: Path) -> None:
    solver = SolverLibrary(LIB)
    config = _config()

    # Interrupted path.
    bundle_a, session_a, behavior_a, _spec_a, state_a = _new(solver, config)
    report_a1 = run_one_stage_iteration(
        bundle=bundle_a,
        session=session_a,
        behavior=behavior_a,
        state=state_a,
        config=config,
        target_iteration=1,
    )
    checkpoint = tmp_path / "stage.pt"
    save_stage_runtime(
        checkpoint,
        bundle=bundle_a,
        behavior=behavior_a,
        state=state_a,
        config=config,
        execution_sha=EXECUTION_SHA,
    )
    bundle_a2, session_a2, behavior_a2, _spec_a2, state_a2 = load_stage_runtime(
        checkpoint,
        repo_root=ROOT,
        solver=solver,
        candidate_id=CANDIDATE,
        domain=DOMAIN,
        training_seed=SEED,
        config=config,
        execution_sha=EXECUTION_SHA,
    )
    report_a2 = run_one_stage_iteration(
        bundle=bundle_a2,
        session=session_a2,
        behavior=behavior_a2,
        state=state_a2,
        config=config,
        target_iteration=2,
    )
    final_a = finalize_stage_seed(
        bundle=bundle_a2,
        behavior=behavior_a2,
        session=session_a2,
        state=state_a2,
        config=config,
    )

    # Uninterrupted path from the same exact initial state/seed.
    bundle_b, session_b, behavior_b, _spec_b, state_b = _new(solver, config)
    report_b1 = run_one_stage_iteration(
        bundle=bundle_b,
        session=session_b,
        behavior=behavior_b,
        state=state_b,
        config=config,
        target_iteration=1,
    )
    report_b2 = run_one_stage_iteration(
        bundle=bundle_b,
        session=session_b,
        behavior=behavior_b,
        state=state_b,
        config=config,
        target_iteration=2,
    )
    final_b = finalize_stage_seed(
        bundle=bundle_b,
        behavior=behavior_b,
        session=session_b,
        state=state_b,
        config=config,
    )

    assert _state_dict_equal(bundle_a2.advantage.state_dict(), bundle_b.advantage.state_dict())
    assert _state_dict_equal(bundle_a2.policy.state_dict(), bundle_b.policy.state_dict())
    assert bundle_a2.batch_rng.getstate() == bundle_b.batch_rng.getstate()
    assert bundle_a2.adv_mem.state_dict()["seen"] == bundle_b.adv_mem.state_dict()["seen"]
    assert bundle_a2.pol_mem.state_dict()["seen"] == bundle_b.pol_mem.state_dict()["seen"]
    assert bundle_a2.counters == bundle_b.counters
    for member in (0, 1, 2, 3):
        assert _state_dict_equal(
            behavior_a2.models[member].state_dict(), behavior_b.models[member].state_dict()
        )

    # Timing fields are intentionally excluded from equality. Strategic/mechanical
    # state and deterministic counters must match exactly.
    for left, right in ((report_a1, report_b1), (report_a2, report_b2)):
        assert left["roots_added"] == right["roots_added"] == 1
        assert left["nodes_added"] == right["nodes_added"]
        assert left["advantage_seen_added"] == right["advantage_seen_added"]
        assert left["strategy_seen_added"] == right["strategy_seen_added"]
        assert left["branch_geometry"] == right["branch_geometry"]
        assert left["ensemble_weighted_nrmse"] == right["ensemble_weighted_nrmse"]
    assert final_a["ensemble_advantage_weighted_nrmse"] == final_b["ensemble_advantage_weighted_nrmse"]
    assert final_a["policy_weighted_mean_tv"] == final_b["policy_weighted_mean_tv"]
    assert final_a["strategic_selection_permitted_at_160"] is False
    assert final_a["production_training_authorized"] is False
    assert final_a["ready_for_tables"] is False


def test_action_branch_telemetry_uses_post_dedup_effective_legal_set() -> None:
    solver = SolverLibrary(LIB)
    config = _config()
    bundle, session, behavior, _spec, state = _new(solver, config)
    report = run_one_stage_iteration(
        bundle=bundle,
        session=session,
        behavior=behavior,
        state=state,
        config=config,
        target_iteration=1,
    )
    geometry = report["branch_geometry"]
    assert geometry["advantage_decision_visits"] > 0
    assert geometry["effective_unique_aggressive_branches"] <= geometry["nominal_aggressive_branches"]
    assert (
        geometry["effective_unique_aggressive_branches_per_decision"]
        <= geometry["nominal_aggressive_branches_per_decision"]
    )


def test_resume_rejects_execution_sha_or_config_drift(tmp_path: Path) -> None:
    solver = SolverLibrary(LIB)
    config = _config()
    bundle, session, behavior, _spec, state = _new(solver, config)
    run_one_stage_iteration(
        bundle=bundle,
        session=session,
        behavior=behavior,
        state=state,
        config=config,
        target_iteration=1,
    )
    checkpoint = tmp_path / "stage.pt"
    save_stage_runtime(
        checkpoint,
        bundle=bundle,
        behavior=behavior,
        state=state,
        config=config,
        execution_sha=EXECUTION_SHA,
    )
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_stage_runtime(
            checkpoint,
            repo_root=ROOT,
            solver=solver,
            candidate_id=CANDIDATE,
            domain=DOMAIN,
            training_seed=SEED,
            config=config,
            execution_sha="different-sha",
        )

    drift = ActionStageConfig(
        **{**config.to_dict(), "roots_per_iteration": 2}
    )
    with pytest.raises(ValueError, match="config mismatch"):
        load_stage_runtime(
            checkpoint,
            repo_root=ROOT,
            solver=solver,
            candidate_id=CANDIDATE,
            domain=DOMAIN,
            training_seed=SEED,
            config=drift,
            execution_sha=EXECUTION_SHA,
        )
