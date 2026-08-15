from __future__ import annotations

from pathlib import Path

import math

from spincore.deep_cfr import icm_delta_utility
from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3 import (
    H2_FINAL,
    RepresentationV3DeepCFRSession,
    make_representation_v3_bundle,
)
from spincore.r7_5_representation_v3_stage import (
    Phase2V3StageConfig,
    finalize_phase2_v3_seed,
    run_one_phase2_v3_iteration,
)
from spincore.r7_5_representation_v3_uncertainty import V3UncertaintyDampedPolicyMixture
from spincore.solver import SolverLibrary


def test_phase2_v3_full_stage_mechanical_smoke() -> None:
    library = Path("build/libspincore_solver_c.so")
    assert library.exists()
    solver = SolverLibrary(library)
    spec = postflop_candidate_specs(Path("."))["PF0_CONTROL_33_75_AI"]
    bundle = make_representation_v3_bundle(
        H2_FINAL,
        1342191342,
        reservoir_capacity=2048,
        lr=0.001,
    )
    behavior = V3UncertaintyDampedPolicyMixture(
        representation=H2_FINAL,
        epsilon_scale=1.75,
        epsilon_cap=0.5,
    )
    session = RepresentationV3DeepCFRSession(
        solver_library=solver,
        bundle=bundle,
        action_spec=spec,
        terminal_utility=icm_delta_utility((0.5, 0.3, 0.2)),
    )
    session.collector.policy = behavior
    session.collector.rng = bundle.batch_rng
    scenarios = action_scenario_cycle("TRUE_HEADS_UP")
    state = {
        "schema": "SPINCORE_R7_5_3C_PHASE2_STAGE_STATE_V1",
        "representation": H2_FINAL,
        "domain": "TRUE_HEADS_UP",
        "training_seed": 1342191342,
        "action_candidate": "PF0_CONTROL_33_75_AI",
        "completed_iteration": 0,
        "global_root": 0,
        "scenario_counts": [0] * len(scenarios),
        "iteration_reports": [],
        "tree_collection_seconds_total": 0.0,
        "advantage_fit_seconds_total": 0.0,
        "policy_fit_seconds_total": 0.0,
    }
    config = Phase2V3StageConfig(
        roots_per_iteration=1,
        total_iterations=1,
        exact_opponent_levels=0,
        reservoir_capacity=2048,
        advantage_steps=1,
        policy_steps=1,
        batch_size=16,
        learning_rate=0.001,
        ensemble_size=4,
        audit_size=16,
        epsilon_scale=1.75,
        epsilon_cap=0.5,
    )
    iteration = run_one_phase2_v3_iteration(
        bundle=bundle,
        session=session,
        behavior=behavior,
        state=state,
        config=config,
        target_iteration=1,
    )
    assert iteration["roots_added"] == 1
    assert iteration["nodes_added"] > 0
    assert len(iteration["members"]) == 4
    assert math.isfinite(iteration["ensemble_weighted_nrmse"])
    assert len(behavior.models) == 4

    final = finalize_phase2_v3_seed(
        bundle=bundle,
        behavior=behavior,
        session=session,
        state=state,
        config=config,
    )
    assert final["roots"] == 1
    assert final["average_policy_optimizer_steps"] == 1
    assert math.isfinite(final["final_policy_weighted_mean_tv"])
    assert final["ready_for_tables"] is False
