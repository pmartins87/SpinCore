from __future__ import annotations

import torch

from spincore_nn.models_v2 import AdvantageNetV2, AveragePolicyNetV2, SemanticNetworkConfigV2


def _batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    history_len = torch.tensor([0, 2, 32][:batch_size], dtype=torch.long)
    return {
        "preflop_class_id": torch.tensor([0, 13, 168][:batch_size], dtype=torch.long),
        "flop_token": torch.tensor([0, 1, 1755][:batch_size], dtype=torch.long),
        "numeric": torch.zeros((batch_size, 24), dtype=torch.float32),
        "categorical": torch.zeros((batch_size, 72), dtype=torch.long),
        "legal": torch.tensor([[1, 1, 1, 1, 1, 1]] * batch_size, dtype=torch.bool),
        "history_categorical": torch.zeros((batch_size, 32, 4), dtype=torch.long),
        "history_numeric": torch.zeros((batch_size, 32, 4), dtype=torch.float32),
        "history_len": history_len,
    }


def test_semantic_v2_parameter_budget_is_frozen() -> None:
    cfg = SemanticNetworkConfigV2()
    net = AdvantageNetV2(cfg)
    count = sum(parameter.numel() for parameter in net.parameters())
    assert count == 153350


def test_semantic_v2_forward_shape_and_zero_history_are_valid() -> None:
    net = AdvantageNetV2()
    out = net(_batch())
    assert tuple(out.shape) == (3, 6)
    assert torch.isfinite(out).all()


def test_semantic_v2_policy_respects_legal_mask() -> None:
    batch = _batch(1)
    batch["legal"] = torch.tensor([[1, 0, 1, 0, 0, 1]], dtype=torch.bool)
    net = AveragePolicyNetV2()
    probabilities = net.probabilities(batch)
    assert tuple(probabilities.shape) == (1, 6)
    assert probabilities[0, 1].item() == 0.0
    assert probabilities[0, 3].item() == 0.0
    assert probabilities[0, 4].item() == 0.0
    assert abs(probabilities.sum().item() - 1.0) < 1.0e-6
