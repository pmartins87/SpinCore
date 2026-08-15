from __future__ import annotations

from pathlib import Path

import torch

from spincore.r7_5_representation_v3 import H2_FINAL, make_representation_v3_bundle
from spincore.r7_5_representation_v3_checkpoint import (
    RepresentationV3Progress,
    save_representation_v3_checkpoint,
)
from spincore.r7_5_representation_v3_final_policy import (
    extract_final_v3_policy_light,
    load_finalized_v3_policy_light,
)
from spincore.r7_5_representation_v3_stage import FINAL_REPORT_SCHEMA
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    MODEL_FINGERPRINTS,
)
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_representation_v3_referee_states import effective_pf0
from spincore.solver import SolverLibrary
from spincore.solver_v3 import neural_bytes_v3

SHA = "synthetic-final-policy-test-sha"
SEED = 1342191342
DOMAIN = "TRUE_HEADS_UP"


def test_light_final_v3_policy_extract_load_and_infer(tmp_path) -> None:
    bundle = make_representation_v3_bundle(H2_FINAL, SEED, reservoir_capacity=16, lr=0.001)
    # Synthetic checkpoint tests artifact plumbing only. The final-report shape is
    # deliberately production-shaped, but no strategic claim is made.
    final_report = {
        "schema": FINAL_REPORT_SCHEMA,
        "representation": H2_FINAL,
        "domain": DOMAIN,
        "training_seed": SEED,
        "action_candidate": ACTION_CANDIDATE,
        "iterations": 3,
        "roots": 192,
        "average_policy_optimizer_steps": 16384,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    checkpoint = tmp_path / "final.pt"
    save_representation_v3_checkpoint(
        checkpoint,
        bundle,
        RepresentationV3Progress(
            iteration=3,
            global_root=192,
            advantage_optimizer_step=3 * 4096,
            policy_optimizer_step=16384,
            phase="post_policy_fit",
        ),
        domain=DOMAIN,
        action_candidate=ACTION_CANDIDATE,
        execution_sha=SHA,
        architecture_fingerprint_sha256=MODEL_FINGERPRINTS[H2_FINAL],
        extra={"final_report": final_report},
    )
    light = tmp_path / "policy-light.pt"
    metadata = extract_final_v3_policy_light(
        checkpoint,
        light,
        expected_training_execution_sha=SHA,
    )
    assert metadata["representation"] == H2_FINAL
    assert metadata["domain"] == DOMAIN
    assert metadata["training_seed"] == SEED

    torch.manual_seed(0x753C)
    rng_before = torch.get_rng_state().clone()
    policy = load_finalized_v3_policy_light(
        light,
        repo_root=Path("."),
        expected_training_execution_sha=SHA,
        expected_representation=H2_FINAL,
        expected_domain=DOMAIN,
        expected_training_seed=SEED,
    )
    assert torch.equal(rng_before, torch.get_rng_state())

    library = Path("build/libspincore_solver_c.so")
    assert library.exists()
    solver = SolverLibrary(library)
    spec = postflop_candidate_specs(Path("."))[ACTION_CANDIDATE]
    state = solver.create(action_scenario_cycle(DOMAIN)[0], 0x753C2026)
    try:
        _mask, legal, _exact = effective_pf0(state, spec)
        probabilities = policy(state, neural_bytes_v3(state), legal)
        assert len(probabilities) == 10
        assert abs(sum(probabilities) - 1.0) < 1e-7
        assert all(probabilities[action] >= 0.0 for action in legal)
        assert all(probabilities[action] == 0.0 for action in range(10) if action not in legal)
    finally:
        state.close()
