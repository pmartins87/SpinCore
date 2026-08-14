from __future__ import annotations

import random
from pathlib import Path

import torch

from spincore.deep_cfr import icm_delta_utility
from spincore.r7_5_action_checkpoint import (
    ActionProgress,
    load_action_checkpoint,
    save_action_checkpoint,
)
from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_action_training import ActionDeepCFRSession, make_action_bundle
from spincore.solver import Episode, SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"
PAYOUT = (0.5, 0.3, 0.2)


def episode() -> Episode:
    return Episode(1500, True, 0, 10, 20, (0, 750, 750), 1, (0,))


def _tensor_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _equal_nested(a, b) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(_equal_nested(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal_nested(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return a == b
    return a == b


def _snapshot(bundle):
    return {
        "config": bundle.config.to_dict(),
        "advantage": _tensor_state(bundle.advantage),
        "policy": _tensor_state(bundle.policy),
        "adv_opt": bundle.adv_opt.state_dict(),
        "pol_opt": bundle.pol_opt.state_dict(),
        "adv_mem": bundle.adv_mem.state_dict(),
        "pol_mem": bundle.pol_mem.state_dict(),
        "batch_rng": bundle.batch_rng.getstate(),
        "counters": dict(bundle.counters),
    }


def test_action_checkpoint_roundtrip_is_exact(tmp_path: Path) -> None:
    solver = SolverLibrary(LIB)
    spec = postflop_candidate_specs(ROOT)["PF0_CONTROL_33_75_AI"]
    bundle = make_action_bundle(
        1737995611,
        domain="TRUE_HEADS_UP",
        selected_representation="C1_V2_NO_FLOP_TOKEN",
        action_spec=spec,
        reservoir_capacity=4096,
        lr=0.001,
    )
    session = ActionDeepCFRSession(
        solver_library=solver,
        bundle=bundle,
        action_spec=spec,
        terminal_utility=icm_delta_utility(PAYOUT),
        device="cpu",
    )
    session.collect_root(episode(), iteration=1, exact_opponent_levels=0, deck_seed=441199)
    session.train_advantage(steps=2, batch_size=8)
    session.train_average_policy(steps=2, batch_size=8)
    progress = ActionProgress(
        iteration=1,
        phase="post_fit",
        root_index=1,
        advantage_optimizer_step=2,
        policy_optimizer_step=2,
    )
    expected = _snapshot(bundle)
    path = tmp_path / "action.pt"
    save_action_checkpoint(
        path,
        bundle,
        progress,
        action_phase="R7_5_4A_POSTFLOP",
        extra={"sentinel": "roundtrip"},
    )

    restored, restored_progress, restored_spec, extra = load_action_checkpoint(
        path,
        repo_root=ROOT,
        device="cpu",
    )
    assert restored_progress == progress
    assert restored_spec == spec
    assert extra == {"sentinel": "roundtrip"}
    assert _equal_nested(expected, _snapshot(restored))


def test_action_checkpoint_restore_continue_matches_continuous_training(tmp_path: Path) -> None:
    solver = SolverLibrary(LIB)
    spec = postflop_candidate_specs(ROOT)["PF0_CONTROL_33_75_AI"]
    bundle = make_action_bundle(
        1737995611,
        domain="TRUE_HEADS_UP",
        selected_representation="C0_V1_FROZEN_CONTROL",
        action_spec=spec,
        reservoir_capacity=4096,
        lr=0.001,
    )
    session = ActionDeepCFRSession(
        solver_library=solver,
        bundle=bundle,
        action_spec=spec,
        terminal_utility=icm_delta_utility(PAYOUT),
        device="cpu",
    )
    session.collect_root(episode(), iteration=1, exact_opponent_levels=0, deck_seed=551177)
    session.train_advantage(steps=2, batch_size=8)
    path = tmp_path / "split.pt"
    save_action_checkpoint(
        path,
        bundle,
        ActionProgress(iteration=1, phase="adv_fit", root_index=1, advantage_optimizer_step=2),
        action_phase="R7_5_4A_POSTFLOP",
    )

    # Continuous arm.
    session.train_advantage(steps=2, batch_size=8)
    session.train_average_policy(steps=3, batch_size=8)
    continuous = _snapshot(bundle)

    # Restore arm. load_action_checkpoint restores the exact saved torch/batch RNG.
    restored, _, restored_spec, _ = load_action_checkpoint(path, repo_root=ROOT, device="cpu")
    restored_session = ActionDeepCFRSession(
        solver_library=solver,
        bundle=restored,
        action_spec=restored_spec,
        terminal_utility=icm_delta_utility(PAYOUT),
        device="cpu",
    )
    # Re-bind behavior readiness/model to restored trained advantage state.
    restored_session.behavior.ready = bool(restored.counters["advantage_ready"])
    restored_session.train_advantage(steps=2, batch_size=8)
    restored_session.train_average_policy(steps=3, batch_size=8)
    resumed = _snapshot(restored)
    assert _equal_nested(continuous, resumed)
