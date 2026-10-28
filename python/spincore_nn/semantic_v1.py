from __future__ import annotations

"""Shared V1 + 30 poker-semantic features used by direct Advantage teachers.

The semantic vector is domain-agnostic: it depends only on Hero hole cards and
visible board cards encoded in the canonical SPNNIV1 observation.  It therefore
applies identically to THREE_HANDED and TRUE_HEADS_UP information states.
"""

import struct
from typing import Iterable

import numpy as np
import torch
from torch import nn

from spincore_nn.action_models import ActionNetworkConfigV1

SEM_DIM = 30

STRAIGHTS = (
    {14, 2, 3, 4, 5},
    {2, 3, 4, 5, 6},
    {3, 4, 5, 6, 7},
    {4, 5, 6, 7, 8},
    {5, 6, 7, 8, 9},
    {6, 7, 8, 9, 10},
    {7, 8, 9, 10, 11},
    {8, 9, 10, 11, 12},
    {9, 10, 11, 12, 13},
    {10, 11, 12, 13, 14},
)


def rank(card: int) -> int:
    return 2 + int(card) // 4


def suit(card: int) -> int:
    return int(card) % 4


def decode_obs(obs: bytes):
    if len(obs) != 126 or obs[:8] != b"SPNNIV1\x00":
        raise ValueError("bad SPNNIV1 observation")
    cards = tuple(int(x) for x in obs[8:15])
    numeric = tuple(float(x) for x in struct.unpack_from("<16f", obs, 15))
    categorical = tuple(int(x) for x in obs[79:87])
    hole = tuple(x - 1 for x in cards[:2] if x)
    board = tuple(x - 1 for x in cards[2:] if x)
    return hole, board, numeric, categorical


def has_straight(ranks: Iterable[int]) -> bool:
    values = set(ranks)
    return any(window.issubset(values) for window in STRAIGHTS)


def made_category(hole, board) -> int:
    cards = tuple(hole) + tuple(board)
    ranks = [rank(c) for c in cards]
    suits = [suit(c) for c in cards]
    rc = {r: ranks.count(r) for r in set(ranks)}
    sc = {s: suits.count(s) for s in set(suits)}
    for s, n in sc.items():
        if n >= 5 and has_straight(rank(c) for c in cards if suit(c) == s):
            return 8
    counts = sorted(rc.values(), reverse=True)
    if counts and counts[0] >= 4:
        return 7
    triples = sum(v >= 3 for v in rc.values())
    pairs = sum(v >= 2 for v in rc.values())
    if triples >= 1 and (pairs >= 2 or triples >= 2):
        return 6
    if any(n >= 5 for n in sc.values()):
        return 5
    if has_straight(ranks):
        return 4
    if counts and counts[0] >= 3:
        return 3
    if sum(v >= 2 for v in rc.values()) >= 2:
        return 2
    if sum(v >= 2 for v in rc.values()) >= 1:
        return 1
    return 0


def private_semantics(hole, board):
    if len(hole) != 2:
        return {
            "made": 0,
            "pair_relation": 0,
            "pocket_pair": 0,
            "overcards": 0,
            "flush_draw": 0,
            "straight_draw": 0,
            "backdoor_flush": 0,
            "backdoor_straight": 0,
            "high_card_no_draw": 0,
        }

    visible = len(board)
    made = made_category(hole, board) if visible >= 3 else 0
    board_ranks = [rank(c) for c in board]
    bcounts = {r: board_ranks.count(r) for r in set(board_ranks)}
    board_high = max(board_ranks) if board_ranks else 0
    h0, h1 = rank(hole[0]), rank(hole[1])
    pocket = int(h0 == h1)
    m0 = int(bcounts.get(h0, 0) > 0)
    m1 = int(bcounts.get(h1, 0) > 0)
    over = int(h0 > board_high) + int(h1 > board_high) if board_high else 0

    pair_rel = 0
    distinct = sorted(set(board_ranks), reverse=True)
    board_paired = any(v >= 2 for v in bcounts.values())
    if pocket:
        if m0:
            pair_rel = 3
        elif h0 > board_high:
            pair_rel = 1
        else:
            pair_rel = 2
    elif m0 and m1:
        pair_rel = 7
    elif m0 or m1:
        mr = h0 if m0 else h1
        pos = distinct.index(mr)
        pair_rel = 4 if pos == 0 else (5 if pos == 1 else 6)
    elif board_paired:
        pair_rel = 8

    cards = tuple(hole) + tuple(board)
    suits = [suit(c) for c in cards]
    combined_suits = {s: suits.count(s) for s in set(suits)}
    flush_draw = 0
    if visible < 5:
        for s, n in combined_suits.items():
            if n == 4 and (suit(hole[0]) == s or suit(hole[1]) == s):
                flush_draw = 1
                break

    backdoor_flush = 0
    if visible == 3 and not flush_draw:
        for s, n in combined_suits.items():
            if n == 3 and (suit(hole[0]) == s or suit(hole[1]) == s):
                backdoor_flush = 1
                break

    ranks = set(rank(c) for c in cards)
    straight_draw = 0
    if visible < 5 and not has_straight(ranks):
        missing = set()
        board_set = set(board_ranks)
        for window in STRAIGHTS:
            if len(window & ranks) == 4:
                hero_adds = any(
                    rank(c) in window and rank(c) not in board_set for c in hole
                )
                if hero_adds:
                    missing.update(window - ranks)
        straight_draw = int(bool(missing))

    backdoor_straight = 0
    if visible == 3 and not has_straight(ranks) and not straight_draw:
        board_set = set(board_ranks)
        for window in STRAIGHTS:
            if len(window & ranks) == 3 and any(
                rank(c) in window and rank(c) not in board_set for c in hole
            ):
                backdoor_straight = 1
                break

    max_combined_suit = max(combined_suits.values()) if combined_suits else 0
    immediate_straight_draw = any(len(window & ranks) >= 4 for window in STRAIGHTS)
    if visible >= 5:
        no_immediate_draw = True
    else:
        no_immediate_draw = max_combined_suit < 4 and not immediate_straight_draw
    high_card_no_draw = int(visible >= 3 and made == 0 and no_immediate_draw)

    return {
        "made": made,
        "pair_relation": pair_rel,
        "pocket_pair": pocket,
        "overcards": over,
        "flush_draw": flush_draw,
        "straight_draw": straight_draw,
        "backdoor_flush": backdoor_flush,
        "backdoor_straight": backdoor_straight,
        "high_card_no_draw": high_card_no_draw,
    }


def board_semantics(board):
    if not board:
        return {
            "paired": 0,
            "max_suit": 0.0,
            "straight_occ": 0.0,
            "broadway": 0.0,
        }
    ranks = [rank(c) for c in board]
    suits = [suit(c) for c in board]
    paired = int(len(set(ranks)) < len(ranks))
    max_s = max(suits.count(s) for s in set(suits))
    rset = set(ranks)
    occ = max(len(window & rset) for window in STRAIGHTS)
    broad = sum(r >= 10 for r in ranks)
    return {
        "paired": paired,
        "max_suit": max_s / 5.0,
        "straight_occ": occ / 5.0,
        "broadway": broad / 5.0,
    }


def semantic_vector_from_observation(obs: bytes) -> np.ndarray:
    hole, board, _numeric, _categorical = decode_obs(obs)
    ps = private_semantics(hole, board)
    bs = board_semantics(board)
    out = []
    out.extend(float(ps["made"] == k) for k in range(9))
    out.extend(float(ps["pair_relation"] == k) for k in range(9))
    out.extend(float(ps["overcards"] == k) for k in range(3))
    out.extend(
        [
            float(ps["pocket_pair"]),
            float(ps["flush_draw"]),
            float(ps["straight_draw"]),
            float(ps["backdoor_flush"]),
            float(ps["backdoor_straight"]),
            float(bs["paired"]),
            float(bs["max_suit"]),
            float(bs["straight_occ"]),
            float(bs["broadway"]),
        ]
    )
    arr = np.asarray(out, dtype=np.float32)
    if arr.shape != (SEM_DIM,):
        raise RuntimeError(f"semantic dim drift: {arr.shape}")
    return arr


class V1SemanticAdvantageNet(nn.Module):
    def __init__(
        self,
        cfg: ActionNetworkConfigV1 | None = None,
        *,
        legacy_exact_zero_mode: bool = False,
    ):
        super().__init__()
        self.cfg = cfg or ActionNetworkConfigV1()
        self.legacy_exact_zero_mode = bool(legacy_exact_zero_mode)
        c = self.cfg
        self.card_emb = nn.Embedding(53, c.card_emb, padding_idx=0)
        self.cat_emb = nn.Embedding(32, c.cat_emb)
        self.hist_emb = nn.Embedding(64, c.cat_emb, padding_idx=0)
        self.gru = nn.GRU(c.cat_emb, c.gru_hidden, batch_first=True)
        input_dim = 7 * c.card_emb + 8 * c.cat_emb + 16 + c.gru_hidden + SEM_DIM
        self.body = nn.Sequential(
            nn.Linear(input_dim, c.hidden),
            nn.ReLU(),
            nn.Linear(c.hidden, c.head_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(c.head_hidden, c.actions)

    def forward(self, batch):
        cards = self.card_emb(batch["cards"]).flatten(1)
        categorical = self.cat_emb(batch["categorical"].clamp(0, 31)).flatten(1)
        history = self.hist_emb(batch["history"].clamp(0, 63))
        _, hidden = self.gru(history)
        legacy_x = torch.cat(
            [
                cards,
                categorical,
                batch["numeric"],
                hidden[-1],
            ],
            dim=1,
        )
        if self.legacy_exact_zero_mode:
            first = self.body[0]
            legacy_width = legacy_x.shape[1]
            semantic_weight = first.weight[:, legacy_width:]
            if torch.count_nonzero(semantic_weight).item() != 0:
                raise RuntimeError(
                    "legacy_exact_zero_mode requires exact-zero semantic columns"
                )
            x = torch.nn.functional.linear(
                legacy_x,
                first.weight[:, :legacy_width],
                first.bias,
            )
            x = self.body[1](x)
            x = self.body[2](x)
            x = self.body[3](x)
            return self.head(x)

        x = torch.cat([legacy_x, batch["semantic"]], dim=1)
        return self.head(self.body(x))


def make_semantic_advantage_model(
    seed: int,
    cfg: ActionNetworkConfigV1 | None = None,
    *,
    legacy_exact_zero_mode: bool = False,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return V1SemanticAdvantageNet(
            cfg,
            legacy_exact_zero_mode=legacy_exact_zero_mode,
        )


def expand_v1_state_with_zero_semantics(v1_state: dict[str, torch.Tensor]):
    """Embed a V1 Advantage state in the semantic architecture.

    All legacy parameters are copied exactly. The 30 added columns of the first
    body layer are exactly zero, making semantic inputs initially inert.
    """

    model = make_semantic_advantage_model(0)
    target = model.state_dict()
    source_keys = set(v1_state)
    target_keys = set(target)
    if source_keys != target_keys:
        raise RuntimeError("V1/semantic state key mismatch")

    out = {}
    for key in sorted(target):
        src = v1_state[key].detach().cpu()
        dst = target[key].detach().cpu().clone()
        if key == "body.0.weight":
            if src.ndim != 2 or dst.ndim != 2:
                raise RuntimeError("unexpected first-layer rank")
            if dst.shape[0] != src.shape[0] or dst.shape[1] != src.shape[1] + SEM_DIM:
                raise RuntimeError(
                    f"unexpected V1/semantic first-layer shapes: {tuple(src.shape)} -> {tuple(dst.shape)}"
                )
            dst.zero_()
            dst[:, : src.shape[1]].copy_(src)
            if torch.count_nonzero(dst[:, src.shape[1] :]).item() != 0:
                raise RuntimeError("semantic columns are not exactly zero")
            out[key] = dst
        else:
            if src.shape != dst.shape:
                raise RuntimeError(f"unexpected shape drift for {key}")
            out[key] = src.clone()
    return out
