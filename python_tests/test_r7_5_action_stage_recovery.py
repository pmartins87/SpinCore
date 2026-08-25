from __future__ import annotations

import os
from pathlib import Path

import torch

from spincore.r7_5_action_stage import (
    ActionStageConfig,
    load_stage_runtime,
    new_stage_runtime,
    run_one_stage_iteration,
    save_stage_runtime,
)
from spincore.r7_5_action_stage_recovery import (
    collect_stage_root_chunk,
    fit_collected_stage_iteration,
    load_partial_collection_runtime,
    save_partial_collection_runtime,
)
from spincore.solver import SolverLibrary


ROOT = Path(os.environ.get("SPINCORE_RECOVERY_SOURCE_ROOT", Path(__file__).resolve().parents[1])).resolve()
LIB = ROOT / "build" / "libspincore_solver_c.so"
CANDIDATE = "PF0_CONTROL_33_75_AI"
DOMAIN = "TRUE_HEADS_UP"
SEED = 1737995611
SOURCE_SHA = "mechanical-source-test-sha"
RECOVERY_SHA = "mechanical-recovery-test-sha"
PROVENANCE = {"source_training_run_id": 31804178848, "test_only": True}


def _config() -> ActionStageConfig:
    return ActionStageConfig(
        roots_per_iteration=2,
        total_iterations=2,
        exact_opponent_levels=0,
        reservoir_capacity=4096,
        advantage_steps=2,
        policy_steps=1,
        batch_size=8,
        learning_rate=0.001,
        ensemble_size=4,
        audit_size=64,
        epsilon_scale=1.75,
        epsilon_cap=0.5,
    )


def _tensor_dict_equal(left: dict, right: dict) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[key], right[key]) for key in left
    )


def _scientific_report(report: dict) -> dict:
    excluded = {
        "tree_collection_seconds",
        "tree_seconds_per_root",
        "advantage_fit_seconds",
        "peak_rss_bytes",
        "mechanical_recovery",
    }
    out = {key: value for key, value in report.items() if key not in excluded}
    out["members"] = [
        {
            key: value
            for key, value in row.items()
            if key not in {"fit_seconds"}
        }
        for row in report["members"]
    ]
    return out


def test_mid_iteration_checkpoint_is_scientifically_exact(tmp_path: Path) -> None:
    solver = SolverLibrary(LIB)
    config = _config()

    bundle, session, behavior, _spec, state = new_stage_runtime(
        ROOT,
        solver=solver,
        candidate_id=CANDIDATE,
        domain=DOMAIN,
        training_seed=SEED,
        config=config,
    )
    run_one_stage_iteration(
        bundle=bundle,
        session=session,
        behavior=behavior,
        state=state,
        config=config,
        target_iteration=1,
    )
    base = tmp_path / "base.pt"
    save_stage_runtime(
        base,
        bundle=bundle,
        behavior=behavior,
        state=state,
        config=config,
        execution_sha=SOURCE_SHA,
    )

    # Frozen monolithic control.
    direct_bundle, direct_session, direct_behavior, _spec, direct_state = load_stage_runtime(
        base,
        repo_root=ROOT,
        solver=solver,
        candidate_id=CANDIDATE,
        domain=DOMAIN,
        training_seed=SEED,
        config=config,
        execution_sha=SOURCE_SHA,
    )
    direct_report = run_one_stage_iteration(
        bundle=direct_bundle,
        session=direct_session,
        behavior=direct_behavior,
        state=direct_state,
        config=config,
        target_iteration=2,
    )
    direct_torch_rng = torch.get_rng_state().clone()

    # Recovery path: one root, serialize/reload, one root, serialize/reload, fit.
    split_bundle, split_session, split_behavior, _spec, split_state = load_stage_runtime(
        base,
        repo_root=ROOT,
        solver=solver,
        candidate_id=CANDIDATE,
        domain=DOMAIN,
        training_seed=SEED,
        config=config,
        execution_sha=SOURCE_SHA,
    )
    partial, _chunk1 = collect_stage_root_chunk(
        bundle=split_bundle,
        session=split_session,
        state=split_state,
        config=config,
        target_iteration=2,
        root_budget=1,
    )
    part1 = tmp_path / "part1.pt"
    save_partial_collection_runtime(
        part1,
        bundle=split_bundle,
        behavior=split_behavior,
        state=split_state,
        partial=partial,
        config=config,
        source_execution_sha=SOURCE_SHA,
        recovery_execution_sha=RECOVERY_SHA,
        recovery_provenance=PROVENANCE,
    )
    (
        split_bundle,
        split_session,
        split_behavior,
        _spec,
        split_state,
        partial,
        _provenance,
    ) = load_partial_collection_runtime(
        part1,
        repo_root=ROOT,
        solver=solver,
        candidate_id=CANDIDATE,
        domain=DOMAIN,
        training_seed=SEED,
        config=config,
        source_execution_sha=SOURCE_SHA,
        recovery_execution_sha=RECOVERY_SHA,
    )
    partial, _chunk2 = collect_stage_root_chunk(
        bundle=split_bundle,
        session=split_session,
        state=split_state,
        config=config,
        target_iteration=2,
        root_budget=1,
        partial=partial,
    )
    part2 = tmp_path / "part2.pt"
    save_partial_collection_runtime(
        part2,
        bundle=split_bundle,
        behavior=split_behavior,
        state=split_state,
        partial=partial,
        config=config,
        source_execution_sha=SOURCE_SHA,
        recovery_execution_sha=RECOVERY_SHA,
        recovery_provenance=PROVENANCE,
    )
    (
        split_bundle,
        split_session,
        split_behavior,
        _spec,
        split_state,
        partial,
        _provenance,
    ) = load_partial_collection_runtime(
        part2,
        repo_root=ROOT,
        solver=solver,
        candidate_id=CANDIDATE,
        domain=DOMAIN,
        training_seed=SEED,
        config=config,
        source_execution_sha=SOURCE_SHA,
        recovery_execution_sha=RECOVERY_SHA,
    )
    split_report = fit_collected_stage_iteration(
        bundle=split_bundle,
        session=split_session,
        behavior=split_behavior,
        state=split_state,
        config=config,
        target_iteration=2,
        partial=partial,
    )

    assert _tensor_dict_equal(
        direct_bundle.advantage.state_dict(), split_bundle.advantage.state_dict()
    )
    assert _tensor_dict_equal(direct_bundle.policy.state_dict(), split_bundle.policy.state_dict())
    for member in range(4):
        assert _tensor_dict_equal(
            direct_behavior.models[member].state_dict(),
            split_behavior.models[member].state_dict(),
        )
    assert direct_bundle.adv_mem.state_dict() == split_bundle.adv_mem.state_dict()
    assert direct_bundle.pol_mem.state_dict() == split_bundle.pol_mem.state_dict()
    assert direct_bundle.batch_rng.getstate() == split_bundle.batch_rng.getstate()
    assert direct_bundle.counters == split_bundle.counters
    assert behavior_stats_tuple(direct_behavior) == behavior_stats_tuple(split_behavior)
    assert direct_state["global_root"] == split_state["global_root"]
    assert direct_state["scenario_counts"] == split_state["scenario_counts"]
    assert _scientific_report(direct_report) == _scientific_report(split_report)
    assert torch.equal(direct_torch_rng, torch.get_rng_state())


def behavior_stats_tuple(behavior) -> tuple:
    return (
        behavior.calls,
        behavior.epsilon_sum,
        behavior.epsilon_max,
        behavior.disagreement_sum,
        behavior.raw_epsilon_max,
        behavior.cap_hit_calls,
        behavior.epsilon_ge_010_calls,
        behavior.epsilon_ge_025_calls,
    )
