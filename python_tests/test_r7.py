from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

import torch

from spincore.deep_cfr import (
    DeepCFRDomainSession,
    ExternalSamplingCollector,
    icm_delta_utility,
    uniform_policy,
)
from spincore.r7 import *
from spincore.solver import Episode, SolverLibrary
from spincore_nn import *

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "build" / "libspincore_solver_c.so"


def bundle(seed=1):
    torch.manual_seed(seed)
    cfg = NetworkConfig(card_emb=3, cat_emb=2, hidden=16, gru_hidden=6, head_hidden=8)
    advantage = AdvantageNet(cfg)
    policy = AveragePolicyNet(cfg)
    return DomainBundle(
        "HU",
        seed,
        cfg,
        advantage,
        policy,
        torch.optim.Adam(advantage.parameters(), lr=1e-3),
        torch.optim.Adam(policy.parameters(), lr=1e-3),
        UniformReservoir(2000, seed + 11),
        UniformReservoir(2000, seed + 12),
        random.Random(seed + 13),
        {},
    )


def episode():
    return Episode(1500, True, 0, 10, 20, (0, 750, 750), 1, (0,))


def make_session(b):
    return DeepCFRDomainSession(
        solver_library=SolverLibrary(LIB),
        bundle=b,
        terminal_utility=icm_delta_utility((0.5, 0.3, 0.2)),
    )


def fill(b, roots=2):
    session = make_session(b)
    for i in range(roots):
        session.collect_root(episode(), iteration=1, deck_seed=100 + i)
    return session


def test_stratified_audit_spans_reservoir():
    x = stratified_audit_indices(10000, 10, 4)
    assert x == stratified_audit_indices(10000, 10, 4)
    assert min(x) < 1000 and max(x) >= 9000 and len(set(x)) == 10


def test_weighted_metrics_exact_zero():
    pred = torch.tensor([[0.2, 0.8], [0.1, 0.9]])
    target = pred.clone()
    mask = torch.ones_like(pred, dtype=torch.bool)
    weights = torch.tensor([1.0, 3.0])
    assert weighted_normalized_rmse(pred, target, mask, weights) == 0
    assert weighted_mean_tv(pred, target, weights) == 0


def test_zero_regret_bootstrap_is_uniform_until_advantage_fit():
    b = bundle(123)
    session = make_session(b)
    root = session.solver_library.create(episode(), 7)
    try:
        legal = root.legal_actions()
        obs = root.neural_bytes()
        p = session.behavior(root, obs, legal)
    finally:
        root.close()

    expected = tuple(1.0 / len(legal) if a in legal else 0.0 for a in range(6))
    assert p == expected
    assert session.behavior.ready is False
    assert b.counters["advantage_ready"] == 0

    session.collect_root(episode(), iteration=1, deck_seed=77)
    session.train_advantage(steps=1, batch_size=8)
    assert session.behavior.ready is True
    assert b.counters["advantage_ready"] == 1

    session.reset_advantage_network(init_seed=99)
    assert session.behavior.ready is False
    assert b.counters["advantage_ready"] == 0


def test_checkpoint_roundtrip_preserves_state(tmp_path):
    b = bundle()
    fill(b, 1)
    progress = MidIterationProgress(iteration=2, phase="train", root_index=7)
    path = tmp_path / "x.pt"
    save_checkpoint(path, b, progress, {"x": 3})
    b2, p2, extra = load_checkpoint(path)
    assert p2 == progress and extra["x"] == 3
    assert b2.counters == b.counters and b2.adv_mem.items == b.adv_mem.items
    assert b2.batch_rng.random() == b.batch_rng.random()


def test_native_own_reach_matches_python_semantics():
    library = SolverLibrary(LIB)
    root_python = library.create(episode(), 333)
    root_native = library.create(episode(), 333)
    adv_python = UniformReservoir(10000, 1)
    pol_python = UniformReservoir(10000, 2)
    adv_native = UniformReservoir(10000, 1)
    pol_native = UniformReservoir(10000, 2)
    rng_python = random.Random(9)
    rng_native = random.Random(9)
    collector = ExternalSamplingCollector(
        policy=uniform_policy,
        terminal_utility=icm_delta_utility((0.5, 0.3, 0.2)),
        rng=rng_python,
        advantage_memory=adv_python,
        strategy_memory=pol_python,
    )
    n_python = collector.collect_strategy_own_reach(
        root_python, target_player=root_python.actor, iteration=3
    )
    n_native = collect_strategy_own_reach_native(
        root_native,
        target_player=root_native.actor,
        iteration=3,
        policy=uniform_policy,
        rng=rng_native,
        strategy_memory=pol_native,
    )
    root_python.close()
    root_native.close()
    assert n_python == n_native and pol_python.items == pol_native.items


def test_fit_audit_returns_finite_metrics():
    b = bundle()
    session = fill(b, 2)
    session.train_advantage(steps=1, batch_size=8)
    session.train_average_policy(steps=1, batch_size=8)
    metrics = audit_model_fit(b, sample_size=16, seed=5)
    assert all(torch.isfinite(torch.tensor(list(metrics.values()))))


def test_cross_seed_tv_zero_for_same_model():
    b = bundle()
    fill(b, 1)
    observations = [x.observation for x in b.pol_mem.items[:10]]
    metrics = cross_seed_policy_tv(b.policy, b.policy, observations)
    assert metrics["mean_tv"] == 0 and metrics["p95_tv"] == 0


def test_advantage_reset_is_deterministic_and_preserves_memories():
    a = bundle(41)
    b = bundle(41)
    sa = fill(a, 1)
    sb = fill(b, 1)
    adv_items = list(a.adv_mem.items)
    pol_items = list(a.pol_mem.items)
    policy_state = {k: v.clone() for k, v in a.policy.state_dict().items()}

    sa.reset_advantage_network(init_seed=987654321)
    sb.reset_advantage_network(init_seed=987654321)

    assert a.adv_mem.items == adv_items
    assert a.pol_mem.items == pol_items
    assert sa.behavior.model is a.advantage
    assert sa.behavior.ready is False
    assert a.counters["advantage_ready"] == 0
    assert a.counters["advantage_resets"] == 1
    for x, y in zip(a.advantage.state_dict().values(), b.advantage.state_dict().values()):
        assert torch.equal(x, y)
    for key, value in a.policy.state_dict().items():
        assert torch.equal(value, policy_state[key])


def test_fresh_process_worker_updates_checkpoint(tmp_path):
    b = bundle()
    fill(b, 1)
    path = tmp_path / "w.pt"
    save_checkpoint(path, b, MidIterationProgress(iteration=1))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "tools" / "r7_training_worker.py"),
            "--checkpoint",
            str(path),
            "--solver",
            str(LIB),
            "--kind",
            "advantage",
            "--steps",
            "1",
            "--batch-size",
            "8",
            "--payout",
            ".5",
            ".3",
            ".2",
        ],
        env=env,
    )
    b2, _progress, extra = load_checkpoint(path)
    assert b2.counters["adv_optimizer_steps"] == 1
    assert b2.counters["advantage_ready"] == 1
    assert extra["last_worker"]["kind"] == "advantage"


def test_continuous_equals_stop_restore_continue(tmp_path):
    def run(with_restore: bool):
        b = bundle(77)
        session = fill(b, 1)
        session.train_advantage(steps=1, batch_size=8)
        session.train_average_policy(steps=1, batch_size=8)
        if with_restore:
            path = tmp_path / "resume.pt"
            save_checkpoint(
                path,
                b,
                MidIterationProgress(iteration=1, phase="collect", root_index=1),
            )
            b, _, _ = load_checkpoint(path)
            session = make_session(b)
        session.collect_root(episode(), iteration=2, deck_seed=999)
        session.train_advantage(steps=1, batch_size=8)
        session.train_average_policy(steps=1, batch_size=8)
        return b

    a = run(False)
    b = run(True)
    assert a.counters == b.counters
    assert a.adv_mem.items == b.adv_mem.items
    assert a.pol_mem.items == b.pol_mem.items
    for x, y in zip(a.advantage.state_dict().values(), b.advantage.state_dict().values()):
        assert torch.equal(x, y)
    for x, y in zip(a.policy.state_dict().values(), b.policy.state_dict().values()):
        assert torch.equal(x, y)
