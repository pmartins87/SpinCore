from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from spincore_nn.codec import DecodedInput
from spincore_nn.codec_v2 import DecodedHistoryEventV2, DecodedInputV2

# Old-V2 fields that are intentionally NOT admitted into the hybrid semantics:
#   numeric[13:16]  -> dead-seat-corrupted min/effective/SPR fields in true HU
#   categorical[33] -> physical flush-draw suit id
#   categorical[65] -> ordered hole[0]-board suit count
#   categorical[66] -> ordered hole[1]-board suit count
#   categorical[71] -> reserved
SAFE_SEMANTIC_NUMERIC_INDICES = (16, 17, 18, 19, 20, 21, 22, 23)
SAFE_SEMANTIC_CATEGORICAL_INDICES = tuple(
    list(range(10, 33)) + list(range(34, 65)) + [67, 68, 69, 70]
)

CARD_ROLE_PADDING = 0
CARD_ROLE_HOLE = 1
CARD_ROLE_FLOP = 2
CARD_ROLE_TURN = 3
CARD_ROLE_RIVER = 4
CARD_ROLES = (
    CARD_ROLE_HOLE,
    CARD_ROLE_HOLE,
    CARD_ROLE_FLOP,
    CARD_ROLE_FLOP,
    CARD_ROLE_FLOP,
    CARD_ROLE_TURN,
    CARD_ROLE_RIVER,
)


@dataclass(frozen=True)
class HybridInputV3:
    """Lossless observable-card state plus selected corrected semantics.

    `raw_cards` exists only for H0 (frozen-card control).  H1+ never expose the
    physical suit number to their card encoder.  They receive exact ranks,
    card roles/visibility and the complete pairwise same-suit equivalence
    relation.  Rank + role + that relation is lossless up to global suit
    renaming for the observable cards.
    """

    raw_cards: tuple[int, ...]
    rank_tokens: tuple[int, ...]
    role_tokens: tuple[int, ...]
    visible_mask: tuple[int, ...]
    same_suit: tuple[tuple[int, ...], ...]
    numeric: tuple[float, ...]
    categorical: tuple[int, ...]
    legal: tuple[int, ...]
    legacy_history: tuple[int, ...]
    legacy_history_len: int
    preflop_class_id: int
    semantic_numeric: tuple[float, ...]
    semantic_categorical: tuple[int, ...]
    structured_history_categorical: tuple[tuple[int, ...], ...]
    structured_history_numeric: tuple[tuple[float, ...], ...]
    structured_history_len: int


def _card_rank_token(token: int) -> int:
    token = int(token)
    if token == 0:
        return 0
    if not 1 <= token <= 52:
        raise ValueError(f"card token outside [0,52]: {token}")
    return ((token - 1) // 4) + 1


def _card_suit(token: int) -> int:
    token = int(token)
    if not 1 <= token <= 52:
        raise ValueError(f"visible card token outside [1,52]: {token}")
    return (token - 1) % 4


def _relational_cards(cards: Iterable[int]) -> tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[tuple[int, ...], ...]
]:
    cards = tuple(int(value) for value in cards)
    if len(cards) != 7:
        raise ValueError("hybrid card representation requires seven role slots")
    if cards[0] == 0 or cards[1] == 0:
        raise ValueError("both Hero private cards must be visible")

    rank_tokens = tuple(_card_rank_token(token) for token in cards)
    visible = tuple(1 if token else 0 for token in cards)
    roles = tuple(role if is_visible else CARD_ROLE_PADDING for role, is_visible in zip(CARD_ROLES, visible))

    relation: list[tuple[int, ...]] = []
    for i, left in enumerate(cards):
        row: list[int] = []
        for j, right in enumerate(cards):
            if i == j or left == 0 or right == 0:
                row.append(0)
            else:
                row.append(1 if _card_suit(left) == _card_suit(right) else 0)
        relation.append(tuple(row))
    return rank_tokens, roles, visible, tuple(relation)


def _dead_rel_v1(item: DecodedInput) -> int | None:
    """Return actor-relative absent-seat index for true HU, fail-closed later.

    The absent seat is the only non-Hero slot whose stack, street commitment and
    total commitment have all remained exactly zero from hand creation while its
    legacy status is `all_in` because zero-stack seats are initialized that way.
    A real player who has moved all-in has positive total commitment.
    """
    if int(item.categorical[0]) != 1:  # StrategyDomain::TrueHeadsUp
        return None
    candidates: list[int] = []
    for rel in (1, 2):
        status = int(item.categorical[4 + rel])
        stack = float(item.numeric[3 + rel])
        street_commitment = float(item.numeric[6 + rel])
        total_commitment = float(item.numeric[9 + rel])
        if (
            status == 2
            and stack == 0.0
            and street_commitment == 0.0
            and total_commitment == 0.0
        ):
            candidates.append(rel)
    if len(candidates) != 1:
        raise ValueError(
            f"true-HU hybrid encoding requires exactly one absent relative seat; got {candidates}"
        )
    return candidates[0]


def _remap_rel(old_rel: int, old_to_new: dict[int, int]) -> int:
    if int(old_rel) not in old_to_new:
        raise ValueError(f"relative seat outside canonical map: {old_rel}")
    return int(old_to_new[int(old_rel)])


def _remap_plus_one(value: int, old_to_new: dict[int, int]) -> int:
    value = int(value)
    if value == 0:
        return 0
    return _remap_rel(value - 1, old_to_new) + 1


def _canonicalize_true_hu(
    v1: DecodedInput,
    v2: DecodedInputV2,
) -> tuple[DecodedInput, DecodedInputV2]:
    """Canonicalize only physical empty-chair identity in true-HU domain.

    New relative layout is always `[Hero, live opponent, absent]`.  Dealer,
    blind/aggressor relative-seat categories and structured-history actors are
    remapped consistently.  Three-handed states are deliberately untouched.
    """
    dead = _dead_rel_v1(v1)
    if dead is None:
        return v1, v2
    live = 1 if dead == 2 else 2
    new_to_old = (0, live, dead)
    old_to_new = {old: new for new, old in enumerate(new_to_old)}

    numeric = list(float(value) for value in v1.numeric)
    original_numeric = tuple(numeric)
    for base in (3, 6, 9):
        for new_rel, old_rel in enumerate(new_to_old):
            numeric[base + new_rel] = original_numeric[base + old_rel]

    categorical = list(int(value) for value in v1.categorical)
    original_cat = tuple(categorical)
    categorical[2] = _remap_rel(original_cat[2], old_to_new)
    for new_rel, old_rel in enumerate(new_to_old):
        categorical[4 + new_rel] = original_cat[4 + old_rel]

    v1_new = replace(
        v1,
        numeric=tuple(numeric),
        categorical=tuple(categorical),
    )

    v2_cat = list(int(value) for value in v2.categorical)
    original_v2_cat = tuple(v2_cat)
    v2_cat[2] = _remap_rel(original_v2_cat[2], old_to_new)
    for index in (3, 4, 15, 16, 22):
        v2_cat[index] = _remap_plus_one(original_v2_cat[index], old_to_new)
    for new_rel, old_rel in enumerate(new_to_old):
        v2_cat[7 + new_rel] = original_v2_cat[7 + old_rel]

    history: list[DecodedHistoryEventV2] = []
    for index, event in enumerate(v2.history):
        if index < int(v2.history_len):
            event_cat = list(int(value) for value in event.categorical)
            event_cat[0] = _remap_rel(event_cat[0], old_to_new)
            history.append(replace(event, categorical=tuple(event_cat)))
        else:
            history.append(event)

    v2_new = replace(v2, categorical=tuple(v2_cat), history=tuple(history))
    return v1_new, v2_new


def build_hybrid_input(v1: DecodedInput, v2: DecodedInputV2) -> HybridInputV3:
    if tuple(int(value) for value in v1.legal) != tuple(int(value) for value in v2.legal):
        raise ValueError("V1/V2 legal masks disagree for paired observation")
    if int(v1.categorical[0]) != int(v2.categorical[0]):
        raise ValueError("V1/V2 domain disagreement")
    if int(v1.categorical[1]) != int(v2.categorical[1]):
        raise ValueError("V1/V2 street disagreement")

    v1, v2 = _canonicalize_true_hu(v1, v2)
    rank_tokens, roles, visible, relation = _relational_cards(v1.cards)

    semantic_numeric = tuple(float(v2.numeric[index]) for index in SAFE_SEMANTIC_NUMERIC_INDICES)
    semantic_categorical = tuple(
        int(v2.categorical[index]) for index in SAFE_SEMANTIC_CATEGORICAL_INDICES
    )

    return HybridInputV3(
        raw_cards=tuple(int(value) for value in v1.cards),
        rank_tokens=rank_tokens,
        role_tokens=roles,
        visible_mask=visible,
        same_suit=relation,
        numeric=tuple(float(value) for value in v1.numeric),
        categorical=tuple(int(value) for value in v1.categorical),
        legal=tuple(int(value) for value in v1.legal),
        legacy_history=tuple(int(value) for value in v1.history),
        legacy_history_len=int(v1.history_len),
        preflop_class_id=int(v2.preflop_class_id),
        semantic_numeric=semantic_numeric,
        semantic_categorical=semantic_categorical,
        structured_history_categorical=tuple(
            tuple(int(value) for value in event.categorical) for event in v2.history
        ),
        structured_history_numeric=tuple(
            tuple(float(value) for value in event.numeric) for event in v2.history
        ),
        structured_history_len=int(v2.history_len),
    )


def collate_hybrid_inputs(items: list[HybridInputV3], device: str = "cpu") -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("cannot collate empty hybrid batch")
    return {
        "raw_cards": torch.tensor([x.raw_cards for x in items], dtype=torch.long, device=device),
        "rank_tokens": torch.tensor([x.rank_tokens for x in items], dtype=torch.long, device=device),
        "role_tokens": torch.tensor([x.role_tokens for x in items], dtype=torch.long, device=device),
        "visible_mask": torch.tensor([x.visible_mask for x in items], dtype=torch.bool, device=device),
        "same_suit": torch.tensor([x.same_suit for x in items], dtype=torch.long, device=device),
        "numeric": torch.tensor([x.numeric for x in items], dtype=torch.float32, device=device),
        "categorical": torch.tensor([x.categorical for x in items], dtype=torch.long, device=device),
        "legal": torch.tensor([x.legal for x in items], dtype=torch.bool, device=device),
        "legacy_history": torch.tensor(
            [x.legacy_history for x in items], dtype=torch.long, device=device
        ),
        "legacy_history_len": torch.tensor(
            [x.legacy_history_len for x in items], dtype=torch.long, device=device
        ),
        "preflop_class_id": torch.tensor(
            [x.preflop_class_id for x in items], dtype=torch.long, device=device
        ),
        "semantic_numeric": torch.tensor(
            [x.semantic_numeric for x in items], dtype=torch.float32, device=device
        ),
        "semantic_categorical": torch.tensor(
            [x.semantic_categorical for x in items], dtype=torch.long, device=device
        ),
        "structured_history_categorical": torch.tensor(
            [x.structured_history_categorical for x in items], dtype=torch.long, device=device
        ),
        "structured_history_numeric": torch.tensor(
            [x.structured_history_numeric for x in items], dtype=torch.float32, device=device
        ),
        "structured_history_len": torch.tensor(
            [x.structured_history_len for x in items], dtype=torch.long, device=device
        ),
    }


@dataclass(frozen=True)
class HybridNetworkConfigV3:
    raw_card_emb: int = 16
    rank_emb: int = 16
    role_emb: int = 8
    relation_emb: int = 4
    card_node_hidden: int = 32
    current_cat_emb: int = 8
    legacy_history_emb: int = 8
    structured_history_emb: int = 4
    history_hidden: int = 80
    semantic_cat_emb: int = 4
    preflop_emb: int = 12
    hidden: int = 320
    head_hidden: int = 128
    actions: int = 6


class _RelationalCardEncoder(nn.Module):
    """Permutation-equivariant card graph with role-wise invariant pooling."""

    def __init__(self, cfg: HybridNetworkConfigV3):
        super().__init__()
        self.cfg = cfg
        self.rank_emb = nn.Embedding(14, cfg.rank_emb, padding_idx=0)
        self.role_emb = nn.Embedding(5, cfg.role_emb, padding_idx=0)
        self.relation_emb = nn.Embedding(2, cfg.relation_emb)
        node_dim = cfg.rank_emb + cfg.role_emb
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + cfg.relation_emb, cfg.card_node_hidden),
            nn.ReLU(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + cfg.card_node_hidden, cfg.card_node_hidden),
            nn.ReLU(),
        )

    @property
    def output_dim(self) -> int:
        return 4 * self.cfg.card_node_hidden

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        rank = self.rank_emb(batch["rank_tokens"].clamp(0, 13))
        role = self.role_emb(batch["role_tokens"].clamp(0, 4))
        node = torch.cat([rank, role], dim=-1)  # [B,7,D]
        visible = batch["visible_mask"]

        n = node.shape[1]
        left = node.unsqueeze(2).expand(-1, -1, n, -1)
        right = node.unsqueeze(1).expand(-1, n, -1, -1)
        relation = self.relation_emb(batch["same_suit"].clamp(0, 1))
        message = self.pair_mlp(torch.cat([left, right, relation], dim=-1))

        pair_mask = visible.unsqueeze(2) & visible.unsqueeze(1)
        eye = torch.eye(n, dtype=torch.bool, device=node.device).unsqueeze(0)
        pair_mask = pair_mask & ~eye
        message = message * pair_mask.unsqueeze(-1)
        denom = pair_mask.sum(dim=2, keepdim=True).clamp(min=1).to(message.dtype)
        aggregated = message.sum(dim=2) / denom
        updated = self.update_mlp(torch.cat([node, aggregated], dim=-1))
        updated = updated * visible.unsqueeze(-1)

        pooled: list[torch.Tensor] = []
        for role_id in (CARD_ROLE_HOLE, CARD_ROLE_FLOP, CARD_ROLE_TURN, CARD_ROLE_RIVER):
            mask = (batch["role_tokens"] == role_id) & visible
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(updated.dtype)
            value = (updated * mask.unsqueeze(-1)).sum(dim=1) / denom
            value = value * (mask.sum(dim=1, keepdim=True) > 0)
            pooled.append(value)
        return torch.cat(pooled, dim=1)


class HybridNetV3(nn.Module):
    CANDIDATES = {
        "H0_FIXED_V1",
        "H1_RELATIONAL_EXACT",
        "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY",
        "H3_HYBRID_EXACT_SEMANTIC",
        "H4_HYBRID_CAPACITY",
    }

    def __init__(self, candidate: str, cfg: HybridNetworkConfigV3 | None = None):
        super().__init__()
        if candidate not in self.CANDIDATES:
            raise ValueError(f"unknown hybrid candidate: {candidate}")
        self.candidate = candidate
        base = cfg or HybridNetworkConfigV3()
        if candidate == "H4_HYBRID_CAPACITY" and cfg is None:
            base = replace(base, hidden=448, head_hidden=192)
        self.cfg = base

        self.raw_card_emb = nn.Embedding(53, base.raw_card_emb, padding_idx=0)
        self.relational_cards = _RelationalCardEncoder(base)
        self.current_cat_emb = nn.Embedding(32, base.current_cat_emb)

        self.legacy_history_emb = nn.Embedding(64, base.legacy_history_emb, padding_idx=0)
        self.legacy_history_gru = nn.GRU(
            base.legacy_history_emb, base.history_hidden, batch_first=True
        )

        self.structured_history_emb = nn.Embedding(
            257, base.structured_history_emb, padding_idx=0
        )
        structured_input = 4 * base.structured_history_emb + 4
        self.structured_history_gru = nn.GRU(
            structured_input, base.history_hidden, batch_first=True
        )

        self.preflop_emb = nn.Embedding(169, base.preflop_emb)
        self.semantic_cat_emb = nn.Embedding(256, base.semantic_cat_emb)

        card_dim = (
            7 * base.raw_card_emb
            if candidate == "H0_FIXED_V1"
            else self.relational_cards.output_dim
        )
        history_dim = base.history_hidden
        input_dim = card_dim + 8 * base.current_cat_emb + 16 + history_dim
        if candidate in {"H3_HYBRID_EXACT_SEMANTIC", "H4_HYBRID_CAPACITY"}:
            input_dim += (
                base.preflop_emb
                + len(SAFE_SEMANTIC_CATEGORICAL_INDICES) * base.semantic_cat_emb
                + len(SAFE_SEMANTIC_NUMERIC_INDICES)
            )

        self.body = nn.Sequential(
            nn.Linear(input_dim, base.hidden),
            nn.ReLU(),
            nn.Linear(base.hidden, base.head_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(base.head_hidden, base.actions)

    @staticmethod
    def _packed_gru(
        inputs: torch.Tensor,
        lengths: torch.Tensor,
        gru: nn.GRU,
    ) -> torch.Tensor:
        lengths = lengths.clamp(0, inputs.shape[1])
        packed = pack_padded_sequence(
            inputs,
            lengths.clamp(min=1).detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = gru(packed)
        return hidden[-1] * (lengths > 0).unsqueeze(1)

    def _legacy_history_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = batch["legacy_history"].clamp(0, 63)
        emb = self.legacy_history_emb(tokens)
        return self._packed_gru(emb, batch["legacy_history_len"], self.legacy_history_gru)

    def _structured_history_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raw = batch["structured_history_categorical"].clamp(0, 255)
        lengths = batch["structured_history_len"].clamp(0, raw.shape[1])
        positions = torch.arange(raw.shape[1], device=raw.device).unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
        tokens = (raw + 1).masked_fill(~valid.unsqueeze(-1), 0)
        emb = self.structured_history_emb(tokens).flatten(2)
        numeric = batch["structured_history_numeric"] * valid.unsqueeze(-1)
        inputs = torch.cat([emb, numeric], dim=2)
        return self._packed_gru(inputs, lengths, self.structured_history_gru)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.candidate == "H0_FIXED_V1":
            cards = self.raw_card_emb(batch["raw_cards"].clamp(0, 52)).flatten(1)
        else:
            cards = self.relational_cards(batch)

        current_cat = self.current_cat_emb(batch["categorical"].clamp(0, 31)).flatten(1)
        if self.candidate in {"H0_FIXED_V1", "H1_RELATIONAL_EXACT"}:
            history = self._legacy_history_state(batch)
        else:
            history = self._structured_history_state(batch)

        pieces = [cards, current_cat, batch["numeric"], history]
        if self.candidate in {"H3_HYBRID_EXACT_SEMANTIC", "H4_HYBRID_CAPACITY"}:
            preflop = self.preflop_emb(batch["preflop_class_id"].clamp(0, 168))
            semantic_cat = self.semantic_cat_emb(
                batch["semantic_categorical"].clamp(0, 255)
            ).flatten(1)
            pieces.extend([preflop, semantic_cat, batch["semantic_numeric"]])
        return self.head(self.body(torch.cat(pieces, dim=1)))

    def probabilities(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.forward(batch)
        logits = logits.masked_fill(~batch["legal"], -1e9)
        return torch.softmax(logits, dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())
