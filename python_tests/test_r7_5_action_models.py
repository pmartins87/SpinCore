from __future__ import annotations

import torch

from spincore_nn.action_models import (
    REPRESENTATION_FLOP_CANDIDATE,
    action_model_parameter_count,
    collate_action_observations,
    make_action_models,
    representation_wire,
)


def _v1() -> bytes:
    payload = bytearray(126)
    payload[:8] = b"SPNNIV1\x00"
    payload[87:93] = bytes((1, 1, 0, 0, 0, 0))
    return bytes(payload)


def _v2() -> bytes:
    payload = bytearray(830)
    payload[:8] = b"SPNNIV2\x00"
    payload[8] = 0
    payload[183:189] = bytes((1, 1, 0, 0, 0, 0))
    return bytes(payload)


def test_action_model_parameter_count_is_equal_within_selected_representation_family() -> None:
    assert action_model_parameter_count("C0_V1_FROZEN_CONTROL") == 152954
    v2_counts = {
        action_model_parameter_count(candidate)
        for candidate in REPRESENTATION_FLOP_CANDIDATE
        if candidate != "C0_V1_FROZEN_CONTROL"
    }
    assert v2_counts == {153738}


def test_every_possible_r7_5_3_winner_can_drive_same_ten_action_head() -> None:
    legal = (1, 1, 1, 1, 0, 0, 0, 1, 0, 1)
    for candidate in REPRESENTATION_FLOP_CANDIDATE:
        wire = representation_wire(candidate)
        observation = _v1() if wire == "SPNNIV1" else _v2()
        batch = collate_action_observations(candidate, [observation, observation], [legal, legal])
        cfg, advantage, policy = make_action_models(
            candidate,
            advantage_seed=1234,
            policy_seed=5678,
        )
        assert cfg.actions == 10
        assert advantage(batch).shape == (2, 10)
        probabilities = policy.probabilities(batch)
        assert probabilities.shape == (2, 10)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6)
        assert torch.all(probabilities[:, 4:7] == 0)
        assert torch.all(probabilities[:, 8] == 0)


def test_action_legal_mask_is_external_to_frozen_six_slot_wire() -> None:
    # Both frozen wires contain only six legacy legal bytes. R7.5.4 must never
    # reinterpret those six bytes as the ten-slot action mask; the universal
    # state-local mask is supplied explicitly by the action traversal.
    legal_a = (1, 1, 1, 0, 0, 0, 0, 0, 0, 1)
    legal_b = (1, 1, 0, 1, 1, 1, 1, 1, 1, 1)
    batch = collate_action_observations(
        "C1_V2_NO_FLOP_TOKEN",
        [_v2(), _v2()],
        [legal_a, legal_b],
    )
    assert tuple(batch["legal"][0].int().tolist()) == legal_a
    assert tuple(batch["legal"][1].int().tolist()) == legal_b
