from __future__ import annotations

from pathlib import Path

import torch

from spincore.deep_cfr import icm_delta_utility
from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_action_training import ActionDeepCFRSession, make_action_bundle
from spincore.solver import Episode, SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"
PAYOUT = (0.5, 0.3, 0.2)


def episode() -> Episode:
    return Episode(1500, True, 0, 10, 20, (0, 750, 750), 1, (0,))


def test_action_bundle_creation_does_not_perturb_global_torch_rng() -> None:
    pf0 = postflop_candidate_specs(ROOT)["PF0_CONTROL_33_75_AI"]
    for representation in ("C0_V1_FROZEN_CONTROL", "C1_V2_NO_FLOP_TOKEN"):
        torch.manual_seed(991122)
        before = torch.get_rng_state().clone()
        bundle = make_action_bundle(
            1737995611,
            domain="TRUE_HEADS_UP",
            selected_representation=representation,
            action_spec=pf0,
            reservoir_capacity=256,
        )
        after = torch.get_rng_state().clone()
        assert torch.equal(before, after)
        assert bundle.counters["advantage_ready"] == 0
        assert bundle.action_candidate == "PF0_CONTROL_33_75_AI"
        assert bundle.config.actions == 10


def test_action_session_real_solver_smoke_collect_train_and_reset() -> None:
    solver = SolverLibrary(LIB)
    pf0 = postflop_candidate_specs(ROOT)["PF0_CONTROL_33_75_AI"]
    bundle = make_action_bundle(
        1737995611,
        domain="TRUE_HEADS_UP",
        selected_representation="C1_V2_NO_FLOP_TOKEN",
        action_spec=pf0,
        reservoir_capacity=4096,
        lr=0.001,
    )
    session = ActionDeepCFRSession(
        solver_library=solver,
        bundle=bundle,
        action_spec=pf0,
        terminal_utility=icm_delta_utility(PAYOUT),
        device="cpu",
    )
    report = session.collect_root(
        episode(), iteration=1, exact_opponent_levels=0, deck_seed=778899
    )
    assert report["nodes"] > 0
    assert report["advantage_samples"] > 0
    assert report["strategy_samples"] > 0
    assert bundle.counters["roots"] == 1
    assert bundle.adv_mem.items and bundle.pol_mem.items
    assert all(len(sample.legal) == 10 and len(sample.target) == 10 for sample in bundle.adv_mem.items)
    assert all(len(sample.legal) == 10 and len(sample.target) == 10 for sample in bundle.pol_mem.items)

    session.train_advantage(steps=1, batch_size=8)
    session.train_average_policy(steps=1, batch_size=8)
    assert bundle.counters["adv_optimizer_steps"] == 1
    assert bundle.counters["policy_optimizer_steps"] == 1
    assert bundle.counters["advantage_ready"] == 1

    config_before = bundle.config.to_dict()
    session.reset_advantage_network(init_seed=1234567, lr=0.001)
    assert bundle.config.to_dict() == config_before
    assert bundle.counters["advantage_ready"] == 0
    assert bundle.counters["advantage_resets"] == 1
