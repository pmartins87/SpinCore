from __future__ import annotations

import random
from pathlib import Path

import torch

from spincore.deep_cfr import icm_delta_utility
from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_action_fit import (
    audit_action_advantage_model,
    audit_action_policy_model,
    cross_seed_action_policy_tv,
    ensemble_action_advantage_nrmse,
    fit_independent_action_advantage_member,
)
from spincore.r7_5_action_training import ActionDeepCFRSession, make_action_bundle
from spincore.solver import Episode, SolverLibrary

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"
PAYOUT = (0.5, 0.3, 0.2)


def _episode() -> Episode:
    return Episode(1500, True, 0, 10, 20, (0, 750, 750), 1, (0,))


def _collected_bundle(seed: int = 1737995611):
    solver = SolverLibrary(LIB)
    pf0 = postflop_candidate_specs(ROOT)["PF0_CONTROL_33_75_AI"]
    bundle = make_action_bundle(
        seed,
        domain="TRUE_HEADS_UP",
        selected_representation="C0_V1_FROZEN_CONTROL",
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
    session.collect_root(_episode(), iteration=1, exact_opponent_levels=0, deck_seed=991177)
    assert bundle.adv_mem.items and bundle.pol_mem.items
    return bundle, session


def _state_dict_equal(a, b) -> bool:
    return all(torch.equal(a[key], b[key]) for key in a)


def test_independent_member_fit_is_deterministic_and_does_not_perturb_caller_rngs() -> None:
    bundle, _ = _collected_bundle()

    random.seed(112233)
    python_before = random.getstate()
    torch.manual_seed(445566)
    torch_before = torch.get_rng_state().clone()

    first, report1 = fit_independent_action_advantage_member(
        bundle.adv_mem.items,
        selected_representation="C0_V1_FROZEN_CONTROL",
        init_seed=7001,
        batch_seed=8001,
        steps=2,
        batch_size=8,
        learning_rate=0.001,
    )
    python_after = random.getstate()
    torch_after = torch.get_rng_state().clone()

    assert python_before == python_after
    assert torch.equal(torch_before, torch_after)
    assert report1["caller_rng_isolation"] is True

    second, report2 = fit_independent_action_advantage_member(
        bundle.adv_mem.items,
        selected_representation="C0_V1_FROZEN_CONTROL",
        init_seed=7001,
        batch_seed=8001,
        steps=2,
        batch_size=8,
        learning_rate=0.001,
    )
    assert report1 == report2
    assert _state_dict_equal(first.state_dict(), second.state_dict())


def test_ten_action_fit_audits_are_finite_and_mask_aware() -> None:
    bundle, session = _collected_bundle()
    session.train_advantage(steps=2, batch_size=8)
    session.train_average_policy(steps=2, batch_size=8)

    member, _ = fit_independent_action_advantage_member(
        bundle.adv_mem.items,
        selected_representation="C0_V1_FROZEN_CONTROL",
        init_seed=7002,
        batch_seed=8002,
        steps=2,
        batch_size=8,
        learning_rate=0.001,
    )
    primary_nrmse = audit_action_advantage_model(
        bundle.advantage,
        bundle.adv_mem.items,
        selected_representation="C0_V1_FROZEN_CONTROL",
        sample_size=64,
        seed=9001,
    )
    ensemble_nrmse = ensemble_action_advantage_nrmse(
        [bundle.advantage, member],
        bundle.adv_mem.items,
        selected_representation="C0_V1_FROZEN_CONTROL",
        sample_size=64,
        seed=9001,
    )
    policy_tv = audit_action_policy_model(
        bundle.policy,
        bundle.pol_mem.items,
        selected_representation="C0_V1_FROZEN_CONTROL",
        sample_size=64,
        seed=9002,
    )
    assert 0.0 <= primary_nrmse < float("inf")
    assert 0.0 <= ensemble_nrmse < float("inf")
    assert 0.0 <= policy_tv <= 1.0


def test_cross_seed_policy_tv_uses_exact_ten_action_legal_masks() -> None:
    bundle_a, session_a = _collected_bundle(1737995611)
    bundle_b, session_b = _collected_bundle(645939859)
    session_a.train_average_policy(steps=1, batch_size=8)
    session_b.train_average_policy(steps=1, batch_size=8)
    samples = bundle_a.pol_mem.items[: min(16, len(bundle_a.pol_mem.items))]
    observations = [(sample.observation, sample.legal) for sample in samples]
    result = cross_seed_action_policy_tv(
        bundle_a.policy,
        bundle_b.policy,
        observations,
        selected_representation="C0_V1_FROZEN_CONTROL",
    )
    assert 0.0 <= result["mean_tv"] <= 1.0
    assert 0.0 <= result["p50_tv"] <= 1.0
    assert 0.0 <= result["p95_tv"] <= 1.0
    assert 0.0 <= result["max_tv"] <= 1.0
