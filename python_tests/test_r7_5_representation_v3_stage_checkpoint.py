from __future__ import annotations

from pathlib import Path

import torch

from spincore.r7_5_representation_v3 import H2_FINAL
from spincore.r7_5_representation_v3_stage import (
    frozen_config,
    load_phase2_v3_runtime,
    new_phase2_v3_runtime,
    save_phase2_v3_runtime,
)
from spincore.r7_5_representation_v3_stage_contract import (
    MODEL_FINGERPRINTS,
    side_member_seeds,
)
from spincore.solver import SolverLibrary
from spincore_nn.models_v3_final import make_h2_final_v3

DOMAIN = "TRUE_HEADS_UP"
SEED = 1342191342
EXECUTION_SHA = "stage-checkpoint-test-sha"


def _assert_state_dict_equal(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_phase2_stage_checkpoint_preserves_full_four_member_behavior(tmp_path) -> None:
    library = Path("build/libspincore_solver_c.so")
    assert library.exists()
    solver = SolverLibrary(library)
    config = frozen_config()
    bundle, _session, behavior, _spec, state = new_phase2_v3_runtime(
        Path("."),
        solver=solver,
        representation=H2_FINAL,
        domain=DOMAIN,
        training_seed=SEED,
        config=config,
    )

    # Construct the exact frozen iteration-1 ensemble identity. The primary is
    # checkpointed by the base V3 payload; side members live in stage extra.
    models = [bundle.advantage]
    for member in (1, 2, 3):
        init_seed, _batch_seed = side_member_seeds(SEED, 1, member)
        _, model = make_h2_final_v3(seed=init_seed)
        models.append(model)
    behavior.models = models
    behavior.calls = 17
    behavior.epsilon_sum = 1.25
    behavior.epsilon_max = 0.19
    behavior.disagreement_sum = 0.71
    behavior.raw_epsilon_max = 0.28
    behavior.cap_hit_calls = 2
    behavior.epsilon_ge_010_calls = 5
    behavior.epsilon_ge_025_calls = 1

    state["completed_iteration"] = 1
    state["global_root"] = 64
    state["scenario_counts"] = [64 // len(state["scenario_counts"])] * len(state["scenario_counts"])
    checkpoint = tmp_path / "phase2-stage.pt"
    save_phase2_v3_runtime(
        checkpoint,
        bundle=bundle,
        behavior=behavior,
        state=state,
        config=config,
        execution_sha=EXECUTION_SHA,
    )

    loaded_bundle, _loaded_session, loaded_behavior, _loaded_spec, loaded_state = load_phase2_v3_runtime(
        checkpoint,
        repo_root=Path("."),
        solver=solver,
        representation=H2_FINAL,
        domain=DOMAIN,
        training_seed=SEED,
        config=config,
        execution_sha=EXECUTION_SHA,
    )
    assert loaded_state["completed_iteration"] == 1
    assert loaded_state["global_root"] == 64
    assert loaded_behavior.stats() == behavior.stats()
    assert len(loaded_behavior.models) == 4
    _assert_state_dict_equal(bundle.advantage.state_dict(), loaded_bundle.advantage.state_dict())
    for original, restored in zip(behavior.models, loaded_behavior.models):
        _assert_state_dict_equal(original.state_dict(), restored.state_dict())
    assert MODEL_FINGERPRINTS[H2_FINAL]
