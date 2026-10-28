#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import run_3h_v1_semantic_shadow_10105 as old_semantic

from spincore_nn.action_models import collate_action_observations, make_advantage_action_model
from spincore_nn.semantic_v1 import (
    SEM_DIM,
    expand_v1_state_with_zero_semantics,
    make_semantic_advantage_model,
    semantic_vector_from_observation,
)


def obs(cards):
    raw = bytearray(126)
    raw[:8] = b"SPNNIV1\x00"
    values = [int(c) + 1 for c in cards] + [0] * (7 - len(cards))
    raw[8:15] = bytes(values[:7])
    return bytes(raw)


def test_shared_semantic_vector_is_exactly_the_validated_3h_vector():
    cases = [
        obs([0, 5]),
        obs([48, 44, 4, 9, 14]),
        obs([12, 13, 0, 4, 8, 20]),
        obs([36, 40, 1, 5, 9, 13, 17]),
    ]
    for observation in cases:
        sample = type("Sample", (), {"observation": observation})()
        old = old_semantic.semantic_vector(sample)
        new = semantic_vector_from_observation(observation)
        assert new.shape == (SEM_DIM,)
        assert np.array_equal(new, old)


def test_zero_column_embedding_preserves_v1_raw_outputs():
    torch.manual_seed(17)
    _, v1 = make_advantage_action_model(
        "C0_V1_FROZEN_CONTROL", device="cpu", seed=1234
    )
    v1.eval()
    sem_state = expand_v1_state_with_zero_semantics(v1.state_dict())
    sem = make_semantic_advantage_model(
        999,
        legacy_exact_zero_mode=True,
    )
    sem.load_state_dict(sem_state)
    sem.eval()

    observations = [
        obs([0, 5]),
        obs([48, 44, 4, 9, 14]),
        obs([12, 13, 0, 4, 8, 20]),
        obs([36, 40, 1, 5, 9, 13, 17]),
    ]
    legal = [tuple([True] * 10) for _ in observations]
    batch = collate_action_observations(
        "C0_V1_FROZEN_CONTROL", observations, legal, device="cpu"
    )
    sb = dict(batch)
    sb["semantic"] = torch.tensor(
        np.asarray(
            [semantic_vector_from_observation(x) for x in observations],
            dtype=np.float32,
        )
    )
    with torch.no_grad():
        a = v1(batch)
        b = sem(sb)
    assert float(torch.max(torch.abs(a - b)).item()) <= 2e-6

    legacy_width = v1.state_dict()["body.0.weight"].shape[1]
    added = sem_state["body.0.weight"][:, legacy_width:]
    assert added.shape[1] == SEM_DIM
    assert torch.count_nonzero(added).item() == 0


def test_legacy_exact_zero_mode_rejects_nonzero_semantic_columns():
    _, v1 = make_advantage_action_model(
        "C0_V1_FROZEN_CONTROL", device="cpu", seed=7
    )
    state = expand_v1_state_with_zero_semantics(v1.state_dict())
    legacy_width = v1.state_dict()["body.0.weight"].shape[1]
    state["body.0.weight"][0, legacy_width] = 1e-6
    sem = make_semantic_advantage_model(
        8,
        legacy_exact_zero_mode=True,
    )
    sem.load_state_dict(state)
    batch = collate_action_observations(
        "C0_V1_FROZEN_CONTROL",
        [obs([0, 5])],
        [tuple([True] * 10)],
        device="cpu",
    )
    batch["semantic"] = torch.tensor(
        np.asarray([semantic_vector_from_observation(obs([0, 5]))], dtype=np.float32)
    )
    try:
        sem(batch)
    except RuntimeError as exc:
        assert "requires exact-zero semantic columns" in str(exc)
    else:
        raise AssertionError("expected exact-zero migration guard to fail")
