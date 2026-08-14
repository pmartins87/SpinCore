from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spincore.r7_5_action_checkpoint import ActionProgress, save_action_checkpoint
from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_action_stage import FINAL_REPORT_SCHEMA
from spincore.r7_5_action_stage_contract import POLICY_STEPS, RESERVOIR_CAPACITY
from spincore.r7_5_action_training import make_action_bundle
from spincore.r7_5_final_policy import load_finalized_action_policy
from spincore.solver import Episode, SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"
EXECUTION_SHA = "test-execution-sha"
CANDIDATE = "PF0_CONTROL_33_75_AI"
DOMAIN = "TRUE_HEADS_UP"
SEED = 1737995611


def _fake_final_checkpoint(path: Path) -> None:
    spec = postflop_candidate_specs(ROOT)[CANDIDATE]
    bundle = make_action_bundle(
        SEED,
        domain=DOMAIN,
        selected_representation="C0_V1_FROZEN_CONTROL",
        action_spec=spec,
        device="cpu",
        reservoir_capacity=RESERVOIR_CAPACITY,
        lr=0.001,
    )
    final = {
        "schema": FINAL_REPORT_SCHEMA,
        "candidate_id": CANDIDATE,
        "domain": DOMAIN,
        "training_seed": SEED,
        "selected_representation": "C0_V1_FROZEN_CONTROL",
        "roots": 160,
        "average_policy_optimizer_steps": POLICY_STEPS,
        "strategic_selection_permitted_at_160": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    save_action_checkpoint(
        path,
        bundle,
        ActionProgress(
            iteration=5,
            phase="post_policy_fit",
            root_index=32,
            advantage_optimizer_step=5 * 4096,
            policy_optimizer_step=POLICY_STEPS,
        ),
        action_phase="R7_5_4A_POSTFLOP",
        extra={"execution_sha": EXECUTION_SHA, "final_report": final},
    )


def test_final_policy_loader_is_rng_neutral_and_infers_ten_action_policy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "final.pt"
    _fake_final_checkpoint(checkpoint)

    torch.manual_seed(987654321)
    before = torch.get_rng_state().clone()
    loaded = load_finalized_action_policy(
        checkpoint,
        repo_root=ROOT,
        expected_execution_sha=EXECUTION_SHA,
        expected_candidate_id=CANDIDATE,
        expected_domain=DOMAIN,
        expected_training_seed=SEED,
    )
    after = torch.get_rng_state().clone()
    assert torch.equal(before, after)

    solver = SolverLibrary(LIB)
    state = solver.create(Episode(1500, True, 0, 10, 20, (0, 750, 750), 1, (0,)), 12345)
    try:
        active = loaded.action_spec.active_mask(0)
        legal = state.universal_legal_actions(active)
        probabilities = loaded(state, state.neural_bytes(), legal)
        assert len(probabilities) == 10
        assert sum(probabilities[action] for action in legal) == pytest.approx(1.0)
        assert all(probabilities[action] == 0.0 for action in range(10) if action not in legal)
    finally:
        state.close()


def test_final_policy_loader_rejects_provenance_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "final.pt"
    _fake_final_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="execution SHA"):
        load_finalized_action_policy(
            checkpoint,
            repo_root=ROOT,
            expected_execution_sha="wrong-sha",
        )
    with pytest.raises(ValueError, match="candidate differs"):
        load_finalized_action_policy(
            checkpoint,
            repo_root=ROOT,
            expected_execution_sha=EXECUTION_SHA,
            expected_candidate_id="PF1_33_50_75_AI",
        )
    with pytest.raises(ValueError, match="domain differs"):
        load_finalized_action_policy(
            checkpoint,
            repo_root=ROOT,
            expected_execution_sha=EXECUTION_SHA,
            expected_domain="THREE_HANDED",
        )
    with pytest.raises(ValueError, match="training seed differs"):
        load_finalized_action_policy(
            checkpoint,
            repo_root=ROOT,
            expected_execution_sha=EXECUTION_SHA,
            expected_training_seed=645939859,
        )
