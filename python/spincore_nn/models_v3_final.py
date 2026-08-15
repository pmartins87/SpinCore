from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from spincore_nn.card_orbit_v3 import canonical_card_orbit_key_v3
from spincore_nn.codec_v3 import collate_v3, decode_spnniv3
from spincore_nn.semantics_v3 import derive_objective_semantics_v3

UNIVERSAL_ACTION_COUNT = 10
SEMANTIC_CATEGORICAL_COUNT = 58
SEMANTIC_NUMERIC_COUNT = 8


@dataclass(frozen=True)
class V3NetworkConfig:
    rank_emb: int = 16
    card_hidden: int = 128
    history_event_hidden: int = 32
    history_gru_hidden: int = 80
    current_hidden: int = 96
    semantic_hidden: int = 96
    body_hidden: int = 320
    head_hidden: int = 128

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _legal_tensor(legal_masks, device: str) -> torch.Tensor:
    legal = torch.tensor(legal_masks, dtype=torch.bool, device=device)
    if legal.ndim != 2 or legal.shape[1] != UNIVERSAL_ACTION_COUNT:
        raise ValueError("V3 universal legal mask must have shape [batch,10]")
    if not torch.all(legal.any(dim=1)):
        raise ValueError("V3 universal legal mask contains empty row")
    return legal


def collate_v3_observations(
    payloads: list[bytes],
    legal_masks,
    *,
    with_semantics: bool,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    if not payloads:
        raise ValueError("cannot collate empty V3 observation batch")
    items = [decode_spnniv3(payload) for payload in payloads]
    batch = collate_v3(items, device=device)

    orbit = [canonical_card_orbit_key_v3(item) for item in items]
    batch["canonical_ranks"] = torch.tensor(
        [key[:7] for key in orbit], dtype=torch.long, device=device
    )
    batch["canonical_suit_relations"] = torch.tensor(
        [key[7:] for key in orbit], dtype=torch.float32, device=device
    )
    batch["legal"] = _legal_tensor(legal_masks, device)

    if with_semantics:
        semantics = [derive_objective_semantics_v3(item) for item in items]
        if any(len(sem.categorical()) != SEMANTIC_CATEGORICAL_COUNT for sem in semantics):
            raise RuntimeError("objective V3 categorical semantic schema drift")
        if any(len(sem.numeric()) != SEMANTIC_NUMERIC_COUNT for sem in semantics):
            raise RuntimeError("objective V3 numeric semantic schema drift")
        batch["semantic_categorical"] = torch.tensor(
            [sem.categorical() for sem in semantics], dtype=torch.float32, device=device
        )
        batch["semantic_numeric"] = torch.tensor(
            [sem.numeric() for sem in semantics], dtype=torch.float32, device=device
        )
    return batch


class V3UniversalNet(nn.Module):
    """Lossless-card / full-history universal-action network for Phase 2.

    H2 uses the exact SPNNIV3 state without auxiliary poker semantics.
    H3 adds objective semantics derived from that same lossless carrier.
    No physical suit label or ordered-hole/flop identity is available to the
    network. No card rank is clamped: rank tokens are exactly 0..14, where 0 is
    padding/unrevealed and 2..14 are real ranks.
    """

    def __init__(self, cfg: V3NetworkConfig, *, with_semantics: bool):
        super().__init__()
        self.cfg = cfg
        self.with_semantics = bool(with_semantics)

        self.rank_emb = nn.Embedding(15, cfg.rank_emb, padding_idx=0)
        self.card_mlp = nn.Sequential(
            nn.Linear(7 * cfg.rank_emb + 21, cfg.card_hidden),
            nn.ReLU(),
        )

        # Current public categorical state is one-hot encoded exactly in forward;
        # no clamping or shared categorical embedding can alias legal values.
        # domain2 + street4 + dealer3 + SB3 + BB3 + live4 + visible6 + 3*status3
        current_cat_dim = 2 + 4 + 3 + 3 + 3 + 4 + 6 + 9
        self.current_mlp = nn.Sequential(
            nn.Linear(current_cat_dim + 16 + 6, cfg.current_hidden),
            nn.ReLU(),
        )

        # Padding is encoded as index 0; real categorical values are shifted +1.
        self.hist_actor_emb = nn.Embedding(4, 4, padding_idx=0)
        self.hist_street_emb = nn.Embedding(5, 4, padding_idx=0)
        self.hist_action_emb = nn.Embedding(7, 6, padding_idx=0)
        self.hist_forced_emb = nn.Embedding(3, 2, padding_idx=0)
        history_raw_dim = 4 + 4 + 6 + 2 + 4
        self.history_event_mlp = nn.Sequential(
            nn.Linear(history_raw_dim, cfg.history_event_hidden),
            nn.ReLU(),
        )
        self.history_gru = nn.GRU(
            cfg.history_event_hidden,
            cfg.history_gru_hidden,
            batch_first=True,
        )

        if self.with_semantics:
            self.semantic_mlp = nn.Sequential(
                nn.Linear(
                    SEMANTIC_CATEGORICAL_COUNT + SEMANTIC_NUMERIC_COUNT,
                    cfg.semantic_hidden,
                ),
                nn.ReLU(),
            )
            semantic_dim = cfg.semantic_hidden
        else:
            self.semantic_mlp = None
            semantic_dim = 0

        body_input = cfg.card_hidden + cfg.current_hidden + cfg.history_gru_hidden + semantic_dim
        self.body = nn.Sequential(
            nn.Linear(body_input, cfg.body_hidden),
            nn.ReLU(),
            nn.Linear(cfg.body_hidden, cfg.head_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(cfg.head_hidden, UNIVERSAL_ACTION_COUNT)

    @staticmethod
    def _current_one_hot(categorical: torch.Tensor) -> torch.Tensor:
        if categorical.ndim != 2 or categorical.shape[1] != 10:
            raise ValueError("SPNNIV3 categorical tensor must have shape [batch,10]")
        domain = F.one_hot(categorical[:, 0], 2)
        street = F.one_hot(categorical[:, 1], 4)
        dealer = F.one_hot(categorical[:, 2], 3)
        sb = F.one_hot(categorical[:, 3], 3)
        bb = F.one_hot(categorical[:, 4], 3)
        live = F.one_hot(categorical[:, 5], 4)
        visible = F.one_hot(categorical[:, 6], 6)
        statuses = [F.one_hot(categorical[:, index], 3) for index in (7, 8, 9)]
        return torch.cat([domain, street, dealer, sb, bb, live, visible, *statuses], dim=1).float()

    def _history(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cat = batch["history_categorical"]
        num = batch["history_numeric"]
        lengths = batch["history_len"]
        if cat.ndim != 3 or cat.shape[2] != 4 or num.shape[:2] != cat.shape[:2]:
            raise ValueError("malformed SPNNIV3 history batch")

        steps = cat.shape[1]
        valid = torch.arange(steps, device=cat.device).unsqueeze(0) < lengths.unsqueeze(1)
        shifted = (cat + 1) * valid.unsqueeze(2).long()
        num = num * valid.unsqueeze(2).float()
        event = torch.cat(
            [
                self.hist_actor_emb(shifted[:, :, 0]),
                self.hist_street_emb(shifted[:, :, 1]),
                self.hist_action_emb(shifted[:, :, 2]),
                self.hist_forced_emb(shifted[:, :, 3]),
                num,
            ],
            dim=2,
        )
        event = self.history_event_mlp(event)

        packed = pack_padded_sequence(
            event,
            lengths.clamp_min(1).detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.history_gru(packed)
        out = hidden[-1]
        # A true zero-event sequence used one synthetic all-padding step only to
        # satisfy pack_padded_sequence. Remove GRU bias/state contribution exactly.
        out = out * (lengths > 0).unsqueeze(1).float()
        return out

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ranks = batch["canonical_ranks"]
        if torch.any((ranks < 0) | (ranks > 14)):
            raise ValueError("V3 rank outside exact embedding domain 0..14")
        card = torch.cat(
            [
                self.rank_emb(ranks).flatten(1),
                batch["canonical_suit_relations"],
            ],
            dim=1,
        )
        card = self.card_mlp(card)

        current = torch.cat(
            [
                self._current_one_hot(batch["categorical"]),
                batch["numeric"],
                batch["primitive_legal"].float(),
            ],
            dim=1,
        )
        current = self.current_mlp(current)
        history = self._history(batch)

        parts = [card, current, history]
        if self.with_semantics:
            if self.semantic_mlp is None:
                raise AssertionError("semantic module missing")
            semantics = torch.cat(
                [batch["semantic_categorical"], batch["semantic_numeric"]], dim=1
            )
            parts.append(self.semantic_mlp(semantics))
        return self.head(self.body(torch.cat(parts, dim=1)))

    def probabilities(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.forward(batch)
        logits = logits.masked_fill(~batch["legal"], -1e9)
        return torch.softmax(logits, dim=-1)


def _make_final_v3(
    *,
    with_semantics: bool,
    device: str,
    seed: int | None,
):
    # Match the established R7.3/R7.4/R7.5 isolation contract: deterministic
    # network initialization must not advance or reset the caller's live/global
    # torch RNG stream.
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(int(seed))
        cfg = V3NetworkConfig()
        model = V3UniversalNet(cfg, with_semantics=with_semantics)
    return cfg, model.to(device)


def make_h2_final_v3(*, device: str = "cpu", seed: int | None = None):
    return _make_final_v3(with_semantics=False, device=device, seed=seed)


def make_h3_final_v3(*, device: str = "cpu", seed: int | None = None):
    return _make_final_v3(with_semantics=True, device=device, seed=seed)
