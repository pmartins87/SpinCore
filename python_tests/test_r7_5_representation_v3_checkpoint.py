from __future__ import annotations

from pathlib import Path

import torch

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_representation_v3 import (
    H2_FINAL,
    RepresentationV3DeepCFRSession,
    make_representation_v3_bundle,
)
from spincore.r7_5_representation_v3_checkpoint import (
    RepresentationV3Progress,
    load_representation_v3_checkpoint,
    save_representation_v3_checkpoint,
)
from spincore.solver import Episode, SolverLibrary

FP = "test-only-fingerprint-h2"
SHA = "test-only-execution-sha"
SEED = 1342191342
CANDIDATE = "PF0_CONTROL_33_75_AI"
DOMAIN = "TRUE_HEADS_UP"


def _chip_utility(state):
    return tuple(float(value) for value in state.terminal_chip_delta())


def _assert_nested_equal(a, b) -> None:
    if torch.is_tensor(a) or torch.is_tensor(b):
        assert torch.is_tensor(a) and torch.is_tensor(b)
        assert torch.equal(a, b)
        return
    if isinstance(a, dict) or isinstance(b, dict):
        assert isinstance(a, dict) and isinstance(b, dict)
        assert set(a) == set(b)
        for key in a:
            _assert_nested_equal(a[key], b[key])
        return
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        assert type(a) is type(b)
        assert len(a) == len(b)
        for left, right in zip(a, b):
            _assert_nested_equal(left, right)
        return
    assert a == b


def _session(solver, bundle, action_spec):
    return RepresentationV3DeepCFRSession(
        solver_library=solver,
        bundle=bundle,
        action_spec=action_spec,
        terminal_utility=_chip_utility,
    )


def _episode() -> Episode:
    return Episode(
        total_chips=80,
        game_is_hu=True,
        blind_index=0,
        small_blind=10,
        big_blind=20,
        stacks=(0, 40, 40),
        dead_players=(0,),
        dealer_id=1,
    )


def _advance(session, *, iteration: int, deck_seed: int) -> None:
    report = session.collect_root(
        _episode(),
        iteration=iteration,
        exact_opponent_levels=0,
        deck_seed=deck_seed,
    )
    assert report["nodes"] > 0
    session.train_advantage(steps=2, batch_size=16)
    session.train_average_policy(steps=1, batch_size=16)


def test_v3_checkpoint_resume_is_bit_exact_and_global_rng_independent(tmp_path) -> None:
    library = Path("build/libspincore_solver_c.so")
    assert library.exists()
    solver = SolverLibrary(library)
    action_spec = postflop_candidate_specs(Path("."))[CANDIDATE]

    continuous = make_representation_v3_bundle(
        H2_FINAL, SEED, reservoir_capacity=2048, lr=0.001
    )
    staged = make_representation_v3_bundle(
        H2_FINAL, SEED, reservoir_capacity=2048, lr=0.001
    )
    continuous_session = _session(solver, continuous, action_spec)
    staged_session = _session(solver, staged, action_spec)

    _advance(continuous_session, iteration=1, deck_seed=0x753C01)
    _advance(staged_session, iteration=1, deck_seed=0x753C01)

    checkpoint = tmp_path / "h2.pt"
    progress = RepresentationV3Progress(
        iteration=1,
        global_root=1,
        advantage_optimizer_step=2,
        policy_optimizer_step=1,
        phase="iteration_complete",
    )
    save_representation_v3_checkpoint(
        checkpoint,
        staged,
        progress,
        domain=DOMAIN,
        action_candidate=CANDIDATE,
        execution_sha=SHA,
        architecture_fingerprint_sha256=FP,
        extra={"sentinel": 753},
    )

    # Deliberately move global Torch RNG after save. Loading must preserve the
    # caller's current stream rather than restoring the audit-only saved state.
    torch.manual_seed(987654321)
    _ = torch.rand(17)
    rng_before_load = torch.get_rng_state().clone()
    resumed, loaded_progress, loaded_spec, extra = load_representation_v3_checkpoint(
        checkpoint,
        repo_root=Path("."),
        expected_domain=DOMAIN,
        expected_representation=H2_FINAL,
        expected_seed=SEED,
        expected_action_candidate=CANDIDATE,
        expected_execution_sha=SHA,
        expected_architecture_fingerprint_sha256=FP,
    )
    assert torch.equal(rng_before_load, torch.get_rng_state())
    assert loaded_progress == progress
    assert loaded_spec.candidate_id == CANDIDATE
    assert extra == {"sentinel": 753}

    resumed_session = _session(solver, resumed, loaded_spec)
    _advance(continuous_session, iteration=2, deck_seed=0x753C02)
    _advance(resumed_session, iteration=2, deck_seed=0x753C02)

    _assert_nested_equal(continuous.advantage.state_dict(), resumed.advantage.state_dict())
    _assert_nested_equal(continuous.policy.state_dict(), resumed.policy.state_dict())
    _assert_nested_equal(continuous.adv_opt.state_dict(), resumed.adv_opt.state_dict())
    _assert_nested_equal(continuous.pol_opt.state_dict(), resumed.pol_opt.state_dict())
    _assert_nested_equal(continuous.adv_mem.state_dict(), resumed.adv_mem.state_dict())
    _assert_nested_equal(continuous.pol_mem.state_dict(), resumed.pol_mem.state_dict())
    assert continuous.batch_rng.getstate() == resumed.batch_rng.getstate()
    assert continuous.counters == resumed.counters
