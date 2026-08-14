from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True)
class SemanticNetworkConfigV2:
    """R7.5 semantic representation network, frozen before learning outputs.

    Every H1/H2/H3/H4 flop candidate uses the same 1,756-slot embedding table.
    Therefore a finer flop abstraction does not receive more total model
    parameters merely because it exposes more active tokens.
    """

    preflop_vocab: int = 169
    preflop_emb: int = 12
    flop_vocab: int = 1756  # 0 padding/preflop + up to 1,755 exact classes
    flop_emb: int = 20
    categorical_vocab: int = 256
    categorical_emb: int = 4
    history_vocab: int = 257  # 0 padding; observed byte value is shifted +1
    history_emb: int = 4
    history_fields: int = 4
    history_numeric: int = 4
    history_hidden: int = 64
    main_categorical_fields: int = 72
    main_numeric_fields: int = 24
    hidden: int = 192
    head_hidden: int = 96
    actions: int = 6

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class _SemanticNetV2(nn.Module):
    def __init__(self, cfg: SemanticNetworkConfigV2 | None = None):
        super().__init__()
        self.cfg = cfg or SemanticNetworkConfigV2()
        c = self.cfg

        self.preflop_emb = nn.Embedding(c.preflop_vocab, c.preflop_emb)
        self.flop_emb = nn.Embedding(c.flop_vocab, c.flop_emb, padding_idx=0)
        self.categorical_emb = nn.Embedding(c.categorical_vocab, c.categorical_emb)
        self.history_emb = nn.Embedding(c.history_vocab, c.history_emb, padding_idx=0)

        history_input = c.history_fields * c.history_emb + c.history_numeric
        self.history_gru = nn.GRU(history_input, c.history_hidden, batch_first=True)

        input_dim = (
            c.preflop_emb
            + c.flop_emb
            + c.main_categorical_fields * c.categorical_emb
            + c.main_numeric_fields
            + c.history_hidden
        )
        self.body = nn.Sequential(
            nn.Linear(input_dim, c.hidden),
            nn.ReLU(),
            nn.Linear(c.hidden, c.head_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(c.head_hidden, c.actions)

    def _history_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raw_cat = batch["history_categorical"].clamp(0, 255)
        lengths = batch["history_len"].clamp(0, raw_cat.shape[1])
        positions = torch.arange(raw_cat.shape[1], device=raw_cat.device).unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)

        # Shift observed byte values by one so token 0 is available exclusively
        # for padding. This matters because actor_rel=0/street=0/action=0 are real.
        history_tokens = (raw_cat + 1).masked_fill(~valid.unsqueeze(-1), 0)
        history_emb = self.history_emb(history_tokens).flatten(2)
        history_numeric = batch["history_numeric"] * valid.unsqueeze(-1)
        history_input = torch.cat([history_emb, history_numeric], dim=2)

        packed_lengths = lengths.clamp(min=1).detach().cpu()
        packed = pack_padded_sequence(
            history_input,
            packed_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.history_gru(packed)
        result = hidden[-1]
        return result * (lengths > 0).unsqueeze(1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        preflop = self.preflop_emb(batch["preflop_class_id"].clamp(0, 168))
        flop = self.flop_emb(batch["flop_token"].clamp(0, self.cfg.flop_vocab - 1))
        categorical = self.categorical_emb(
            batch["categorical"].clamp(0, self.cfg.categorical_vocab - 1)
        ).flatten(1)
        history = self._history_state(batch)
        x = torch.cat([preflop, flop, categorical, batch["numeric"], history], dim=1)
        return self.head(self.body(x))


class AdvantageNetV2(_SemanticNetV2):
    pass


class AveragePolicyNetV2(_SemanticNetV2):
    def probabilities(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.forward(batch)
        logits = logits.masked_fill(~batch["legal"], -1e9)
        return torch.softmax(logits, dim=-1)
