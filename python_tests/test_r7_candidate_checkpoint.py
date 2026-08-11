from __future__ import annotations

import random

import pytest
import torch

from spincore.r7 import MidIterationProgress, load_checkpoint, save_checkpoint
from spincore.r7_candidate_checkpoint import (
    SCHEMA,
    pack_candidate_behavior,
    restore_candidate_behavior_models,
)
from spincore_nn import (
    AdvantageNet,
    AveragePolicyNet,
    DomainBundle,
    NetworkConfig,
    UniformReservoir,
)


def _cfg():
    return NetworkConfig(card_emb=3, cat_emb=2, hidden=16, gru_hidden=6, head_hidden=8)


def _adv(cfg, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return AdvantageNet(cfg)


def _same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    return all(torch.equal(sa[k].cpu(), sb[k].cpu()) for k in sa)


def _bundle(seed=17):
    cfg = _cfg()
    primary = _adv(cfg, seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        policy = AveragePolicyNet(cfg)
    return DomainBundle(
        "HU",
        seed,
        cfg,
        primary,
        policy,
        torch.optim.Adam(primary.parameters(), lr=1e-3),
        torch.optim.Adam(policy.parameters(), lr=1e-3),
        UniformReservoir(100, seed + 2),
        UniformReservoir(100, seed + 3),
        random.Random(seed + 4),
        {"iteration": 3, "advantage_ready": 1},
    )


def test_pack_requires_authoritative_primary_as_member_zero():
    cfg = _cfg()
    primary = _adv(cfg, 1)
    other = _adv(cfg, 2)
    with pytest.raises(ValueError, match="current_models\[0\]"):
        pack_candidate_behavior(
            kind="uncertainty_s10",
            primary_model=primary,
            current_models=[other, primary],
        )


def test_uncertainty_ensemble_roundtrip_reuses_primary():
    cfg = _cfg()
    primary = _adv(cfg, 11)
    current = [primary] + [_adv(cfg, 20 + i) for i in range(7)]
    payload = pack_candidate_behavior(
        kind="uncertainty_s10",
        primary_model=primary,
        current_models=current,
        params={"epsilon_scale": 1.0, "epsilon_cap": 0.5},
        fit_generation=4,
    )
    assert payload["schema"] == SCHEMA

    restored_primary = _adv(cfg, 999)
    restored_primary.load_state_dict(primary.state_dict())
    got_current, got_previous, meta = restore_candidate_behavior_models(
        payload,
        config=cfg,
        primary_model=restored_primary,
    )

    assert got_current[0] is restored_primary
    assert len(got_current) == 8 and got_previous == []
    assert meta["kind"] == "uncertainty_s10"
    assert meta["params"] == {"epsilon_scale": 1.0, "epsilon_cap": 0.5}
    assert meta["fit_generation"] == 4
    for expected, actual in zip(current, got_current):
        assert _same(expected, actual)


def test_temporal_ensemble_roundtrip_preserves_previous_generation():
    cfg = _cfg()
    primary = _adv(cfg, 31)
    current = [primary] + [_adv(cfg, 40 + i) for i in range(3)]
    previous = [_adv(cfg, 60 + i) for i in range(4)]
    payload = pack_candidate_behavior(
        kind="temporal_w50",
        primary_model=primary,
        current_models=current,
        previous_models=previous,
        params={"current_weight": 0.5},
        fit_generation=5,
    )

    restored_primary = _adv(cfg, 777)
    restored_primary.load_state_dict(primary.state_dict())
    got_current, got_previous, meta = restore_candidate_behavior_models(
        payload,
        config=cfg,
        primary_model=restored_primary,
    )

    assert len(got_current) == 4 and len(got_previous) == 4
    assert meta["kind"] == "temporal_w50"
    assert meta["params"] == {"current_weight": 0.5}
    for expected, actual in zip(current, got_current):
        assert _same(expected, actual)
    for expected, actual in zip(previous, got_previous):
        assert _same(expected, actual)


def test_payload_is_snapshot_not_live_model_reference():
    cfg = _cfg()
    primary = _adv(cfg, 81)
    side = _adv(cfg, 82)
    payload = pack_candidate_behavior(
        kind="mixture",
        primary_model=primary,
        current_models=[primary, side],
    )
    before = {k: v.clone() for k, v in payload["current_side_states"][0].items()}
    with torch.no_grad():
        for p in side.parameters():
            p.add_(1.0)
    for key, value in before.items():
        assert torch.equal(value, payload["current_side_states"][0][key])


def test_restore_fails_closed_on_mismatched_base_primary():
    cfg = _cfg()
    primary = _adv(cfg, 91)
    payload = pack_candidate_behavior(
        kind="mixture",
        primary_model=primary,
        current_models=[primary],
    )
    wrong = _adv(cfg, 92)
    with pytest.raises(ValueError, match="does not match base checkpoint"):
        restore_candidate_behavior_models(payload, config=cfg, primary_model=wrong)


def test_candidate_payload_survives_authoritative_checkpoint_extra(tmp_path):
    bundle = _bundle(101)
    side = [_adv(bundle.config, 110 + i) for i in range(3)]
    previous = [_adv(bundle.config, 120 + i) for i in range(4)]
    behavior_payload = pack_candidate_behavior(
        kind="temporal_w50",
        primary_model=bundle.advantage,
        current_models=[bundle.advantage] + side,
        previous_models=previous,
        params={"current_weight": 0.5},
        fit_generation=3,
    )

    path = tmp_path / "candidate.pt"
    save_checkpoint(
        path,
        bundle,
        MidIterationProgress(iteration=3, phase="collect_advantage", root_index=17),
        {"candidate_behavior": behavior_payload},
    )
    restored_bundle, progress, extra = load_checkpoint(path)
    got_current, got_previous, meta = restore_candidate_behavior_models(
        extra["candidate_behavior"],
        config=restored_bundle.config,
        primary_model=restored_bundle.advantage,
    )

    assert progress.iteration == 3 and progress.root_index == 17
    assert len(got_current) == 4 and len(got_previous) == 4
    assert got_current[0] is restored_bundle.advantage
    assert meta["kind"] == "temporal_w50"
    assert meta["fit_generation"] == 3
