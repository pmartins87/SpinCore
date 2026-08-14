from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn

from spincore_nn.codec import collate_inputs, decode_spnniv1
from spincore_nn.codec_v2 import collate_inputs_v2, decode_spnniv2
from spincore_nn.models_v2 import AdvantageNetV2, AveragePolicyNetV2, SemanticNetworkConfigV2

REPRESENTATION_FLOP_CANDIDATE = {
    "C0_V1_FROZEN_CONTROL": None,
    "C1_V2_NO_FLOP_TOKEN": "NONE",
    "C2_V2_H1_CANONICAL_184": "H1",
    "C3_V2_H2_MIN_CHANGE_181": "H2",
    "C4_V2_H3_RECLUSTERED_184": "H3",
    "C5_V2_H4_EXACT_1755": "H4",
}


@dataclass(frozen=True)
class ActionNetworkConfigV1:
    card_emb: int = 16
    cat_emb: int = 8
    hidden: int = 320
    gru_hidden: int = 80
    head_hidden: int = 128
    actions: int = 10

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class _ActionNetV1(nn.Module):
    """Frozen V1 feature extractor with only the action head widened 6 -> 10."""

    def __init__(self, cfg: ActionNetworkConfigV1 | None = None):
        super().__init__()
        self.cfg = cfg or ActionNetworkConfigV1()
        c = self.cfg
        if c.actions != 10:
            raise ValueError("R7.5.4 action network must expose exactly ten outputs")
        self.card_emb = nn.Embedding(53, c.card_emb, padding_idx=0)
        self.cat_emb = nn.Embedding(32, c.cat_emb)
        self.hist_emb = nn.Embedding(64, c.cat_emb, padding_idx=0)
        self.gru = nn.GRU(c.cat_emb, c.gru_hidden, batch_first=True)
        input_dim = 7 * c.card_emb + 8 * c.cat_emb + 16 + c.gru_hidden
        self.body = nn.Sequential(
            nn.Linear(input_dim, c.hidden),
            nn.ReLU(),
            nn.Linear(c.hidden, c.head_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(c.head_hidden, c.actions)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cards = self.card_emb(batch["cards"]).flatten(1)
        categorical = self.cat_emb(batch["categorical"].clamp(0, 31)).flatten(1)
        history = self.hist_emb(batch["history"].clamp(0, 63))
        _, hidden = self.gru(history)
        x = torch.cat([cards, categorical, batch["numeric"], hidden[-1]], dim=1)
        return self.head(self.body(x))


class AdvantageActionNetV1(_ActionNetV1):
    pass


class AveragePolicyActionNetV1(_ActionNetV1):
    def probabilities(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.forward(batch).masked_fill(~batch["legal"], -1e9)
        return torch.softmax(logits, dim=-1)


def representation_wire(selected_representation: str) -> str:
    if selected_representation not in REPRESENTATION_FLOP_CANDIDATE:
        raise ValueError(f"unsupported R7.5.3 representation: {selected_representation!r}")
    return "SPNNIV1" if selected_representation == "C0_V1_FROZEN_CONTROL" else "SPNNIV2"


def make_action_models(
    selected_representation: str,
    *,
    device: str = "cpu",
    advantage_seed: int | None = None,
    policy_seed: int | None = None,
):
    wire = representation_wire(selected_representation)
    if advantage_seed is not None:
        torch.manual_seed(int(advantage_seed))
    if wire == "SPNNIV1":
        cfg = ActionNetworkConfigV1()
        advantage = AdvantageActionNetV1(cfg).to(device)
    else:
        cfg = SemanticNetworkConfigV2(actions=10)
        advantage = AdvantageNetV2(cfg).to(device)

    if policy_seed is not None:
        torch.manual_seed(int(policy_seed))
    if wire == "SPNNIV1":
        policy = AveragePolicyActionNetV1(cfg).to(device)
    else:
        policy = AveragePolicyNetV2(cfg).to(device)
    return cfg, advantage, policy


def collate_action_observations(
    selected_representation: str,
    observations: Sequence[bytes],
    legal_masks: Sequence[Sequence[int]],
    *,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    if len(observations) != len(legal_masks):
        raise ValueError("observation/legal count mismatch")
    if any(len(mask) != 10 for mask in legal_masks):
        raise ValueError("R7.5.4 legal mask requires ten actions")
    wire = representation_wire(selected_representation)
    if wire == "SPNNIV1":
        batch = collate_inputs([decode_spnniv1(value) for value in observations], device=device)
    else:
        flop_candidate = REPRESENTATION_FLOP_CANDIDATE[selected_representation]
        batch = collate_inputs_v2(
            [decode_spnniv2(value) for value in observations],
            device=device,
            flop_candidate=flop_candidate,
        )
    batch["legal"] = torch.tensor(legal_masks, dtype=torch.bool, device=device)
    return batch


def action_model_parameter_count(selected_representation: str) -> int:
    _, model, _ = make_action_models(selected_representation, advantage_seed=0, policy_seed=1)
    return int(sum(parameter.numel() for parameter in model.parameters()))
