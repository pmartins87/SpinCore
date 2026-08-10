from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NetworkConfig:
    """Neural architecture used by the recovered SpinCore Deep CFR stack.

    The generation-2 recovery initially used a much smaller placeholder network
    (24,206 parameters per model).  Recovery evidence from the pre-loss R4/R7
    run recorded approximately 152,434 parameters per model.  The closest
    architecture recovered from the preserved contract has 152,438 parameters
    per AdvantageNet/AveragePolicyNet: a residual difference of four parameters
    that is tracked explicitly rather than hidden.
    """

    card_emb: int = 16
    cat_emb: int = 8
    hidden: int = 320
    gru_hidden: int = 80
    head_hidden: int = 128

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class _Net(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.cfg = cfg
        self.card_emb = nn.Embedding(53, cfg.card_emb, padding_idx=0)
        self.cat_emb = nn.Embedding(32, cfg.cat_emb)
        self.hist_emb = nn.Embedding(64, cfg.cat_emb, padding_idx=0)
        self.gru = nn.GRU(cfg.cat_emb, cfg.gru_hidden, batch_first=True)

        input_dim = 7 * cfg.card_emb + 8 * cfg.cat_emb + 16 + cfg.gru_hidden
        self.body = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.head_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(cfg.head_hidden, 6)

    def forward(self, batch):
        cards = self.card_emb(batch["cards"]).flatten(1)
        categorical = self.cat_emb(batch["categorical"].clamp(0, 31)).flatten(1)
        history = self.hist_emb(batch["history"].clamp(0, 63))
        _, hidden = self.gru(history)
        x = torch.cat([cards, categorical, batch["numeric"], hidden[-1]], dim=1)
        return self.head(self.body(x))


class AdvantageNet(_Net):
    pass


class AveragePolicyNet(_Net):
    def probabilities(self, batch):
        logits = self.forward(batch)
        logits = logits.masked_fill(~batch["legal"], -1e9)
        return torch.softmax(logits, dim=-1)
