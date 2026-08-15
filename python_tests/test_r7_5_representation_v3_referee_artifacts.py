from __future__ import annotations

from pathlib import Path

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_representation_v3_referee_artifacts import (
    load_heldout_v3_artifact,
    save_heldout_v3_artifact,
)
from spincore.r7_5_representation_v3_referee_states import generate_heldout_v3_states
from spincore.solver import SolverLibrary


def test_heldout_v3_artifact_is_deterministic_and_roundtrips(tmp_path) -> None:
    library = Path("build/libspincore_solver_c.so")
    assert library.exists()
    solver = SolverLibrary(library)
    spec = postflop_candidate_specs(Path("."))["PF0_CONTROL_33_75_AI"]
    states = generate_heldout_v3_states(
        solver=solver,
        action_spec=spec,
        domain="TRUE_HEADS_UP",
        evaluation_seed=2029384436,
        count=32,
    )
    first = tmp_path / "a.json.gz"
    second = tmp_path / "different-name.json.gz"
    meta_a = save_heldout_v3_artifact(first, states, generator_execution_sha="eval-test-sha")
    meta_b = save_heldout_v3_artifact(second, states, generator_execution_sha="eval-test-sha")
    assert meta_a == meta_b
    assert first.read_bytes() == second.read_bytes()
    restored = load_heldout_v3_artifact(
        first,
        expected_domain="TRUE_HEADS_UP",
        expected_evaluation_seed=2029384436,
        expected_count=32,
    )
    assert restored == states
