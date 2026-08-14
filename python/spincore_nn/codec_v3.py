from __future__ import annotations

from dataclasses import dataclass
import struct

import torch

MAGIC = b"SPNNIV3\0"
CARD_SLOT_COUNT = 7
SUIT_RELATION_COUNT = 21
NUMERIC_COUNT = 16
CATEGORICAL_COUNT = 10
PRIMITIVE_LEGAL_COUNT = 6
FIXED_BYTES = 120
HISTORY_EVENT_BYTES = 20

CARD_ROLE_PADDING = 0
CARD_ROLE_HOLE = 1
CARD_ROLE_FLOP = 2
CARD_ROLE_TURN = 3
CARD_ROLE_RIVER = 4
_FIXED_ROLES = (
    CARD_ROLE_HOLE,
    CARD_ROLE_HOLE,
    CARD_ROLE_FLOP,
    CARD_ROLE_FLOP,
    CARD_ROLE_FLOP,
    CARD_ROLE_TURN,
    CARD_ROLE_RIVER,
)


@dataclass(frozen=True)
class DecodedHistoryEventV3:
    categorical: tuple[int, int, int, int]
    numeric: tuple[float, float, float, float]


@dataclass(frozen=True)
class DecodedInputV3:
    categorical: tuple[int, ...]
    rank_tokens: tuple[int, ...]
    same_suit: tuple[int, ...]
    numeric: tuple[float, ...]
    primitive_legal: tuple[int, ...]
    history: tuple[DecodedHistoryEventV3, ...]

    @property
    def history_len(self) -> int:
        return len(self.history)

    @property
    def visible_mask(self) -> tuple[int, ...]:
        return tuple(1 if rank else 0 for rank in self.rank_tokens)

    @property
    def role_tokens(self) -> tuple[int, ...]:
        return tuple(
            role if rank else CARD_ROLE_PADDING
            for role, rank in zip(_FIXED_ROLES, self.rank_tokens)
        )

    def same_suit_matrix(self) -> tuple[tuple[int, ...], ...]:
        matrix = [[0] * CARD_SLOT_COUNT for _ in range(CARD_SLOT_COUNT)]
        cursor = 0
        for left in range(CARD_SLOT_COUNT):
            for right in range(left + 1, CARD_SLOT_COUNT):
                value = int(self.same_suit[cursor])
                matrix[left][right] = value
                matrix[right][left] = value
                cursor += 1
        if cursor != SUIT_RELATION_COUNT:
            raise AssertionError("SPNNIV3 suit relation reconstruction drift")
        return tuple(tuple(row) for row in matrix)


def _validate(decoded: DecodedInputV3) -> None:
    if len(decoded.categorical) != CATEGORICAL_COUNT:
        raise ValueError("SPNNIV3 categorical length mismatch")
    if len(decoded.rank_tokens) != CARD_SLOT_COUNT:
        raise ValueError("SPNNIV3 rank-token length mismatch")
    if len(decoded.same_suit) != SUIT_RELATION_COUNT:
        raise ValueError("SPNNIV3 suit-relation length mismatch")
    if len(decoded.numeric) != NUMERIC_COUNT:
        raise ValueError("SPNNIV3 numeric length mismatch")
    if len(decoded.primitive_legal) != PRIMITIVE_LEGAL_COUNT:
        raise ValueError("SPNNIV3 primitive-legal length mismatch")

    domain, street, dealer_rel, sb_rel, bb_rel, live_count, visible_board, *statuses = decoded.categorical
    if domain not in (0, 1):
        raise ValueError(f"SPNNIV3 unknown domain {domain}")
    if street not in (0, 1, 2, 3):
        raise ValueError(f"SPNNIV3 unknown street {street}")
    if dealer_rel not in (0, 1, 2) or sb_rel not in (0, 1, 2) or bb_rel not in (0, 1, 2):
        raise ValueError("SPNNIV3 relative position outside [0,2]")
    if live_count != (2 if domain == 1 else 3):
        raise ValueError("SPNNIV3 live_count/domain mismatch")
    expected_visible = (0, 3, 4, 5)[street]
    if visible_board != expected_visible:
        raise ValueError(
            f"SPNNIV3 visible board/street mismatch: street={street} visible={visible_board}"
        )
    if any(status not in (0, 1, 2) for status in statuses):
        raise ValueError("SPNNIV3 unknown player status")

    if decoded.rank_tokens[0] == 0 or decoded.rank_tokens[1] == 0:
        raise ValueError("SPNNIV3 missing Hero private rank")
    for index, rank in enumerate(decoded.rank_tokens):
        should_be_visible = index < 2 or (2 <= index < 2 + visible_board)
        if should_be_visible:
            if rank < 2 or rank > 14:
                raise ValueError(f"SPNNIV3 invalid visible rank token {rank} at slot {index}")
        elif rank != 0:
            raise ValueError(f"SPNNIV3 unrevealed slot {index} contains rank {rank}")

    if any(value not in (0, 1) for value in decoded.same_suit):
        raise ValueError("SPNNIV3 same-suit relation is not binary")
    matrix = decoded.same_suit_matrix()
    visible = decoded.visible_mask
    for left in range(CARD_SLOT_COUNT):
        for right in range(CARD_SLOT_COUNT):
            if (not visible[left] or not visible[right]) and matrix[left][right] != 0:
                raise ValueError("SPNNIV3 suit relation references unrevealed card")

    if any(value not in (0, 1) for value in decoded.primitive_legal):
        raise ValueError("SPNNIV3 primitive legal mask is not binary")

    if domain == 1:
        # True-HU canonical contract: [Hero, live opponent, absent].
        if statuses[2] != 2:
            raise ValueError("SPNNIV3 true-HU absent seat is not canonical rel2/all-in marker")
        for numeric_index in (5, 8, 11):  # rel2 stack/street/total commitment
            if decoded.numeric[numeric_index] != 0.0:
                raise ValueError("SPNNIV3 true-HU absent seat has nonzero chip geometry")

    for event in decoded.history:
        actor_rel, event_street, action_type, forced = event.categorical
        if actor_rel not in (0, 1, 2):
            raise ValueError("SPNNIV3 history actor outside [0,2]")
        if domain == 1 and actor_rel == 2:
            raise ValueError("SPNNIV3 absent true-HU seat appears in public history")
        if event_street not in (0, 1, 2, 3):
            raise ValueError("SPNNIV3 history street outside [0,3]")
        if action_type not in (0, 1, 2, 3, 4, 5):
            raise ValueError("SPNNIV3 history exact action type outside [0,5]")
        if forced not in (0, 1):
            raise ValueError("SPNNIV3 history forced flag is not binary")


def decode_spnniv3(payload: bytes | bytearray | memoryview) -> DecodedInputV3:
    data = bytes(payload)
    if len(data) < FIXED_BYTES:
        raise ValueError(f"SPNNIV3 payload too short: {len(data)}")
    if data[:8] != MAGIC:
        raise ValueError(f"SPNNIV3 bad magic: {data[:8]!r}")

    offset = 8
    categorical = tuple(int(value) for value in data[offset : offset + CATEGORICAL_COUNT])
    offset += CATEGORICAL_COUNT
    rank_tokens = tuple(int(value) for value in data[offset : offset + CARD_SLOT_COUNT])
    offset += CARD_SLOT_COUNT
    same_suit = tuple(int(value) for value in data[offset : offset + SUIT_RELATION_COUNT])
    offset += SUIT_RELATION_COUNT
    numeric = tuple(struct.unpack_from("<16f", data, offset))
    offset += 4 * NUMERIC_COUNT
    primitive_legal = tuple(int(value) for value in data[offset : offset + PRIMITIVE_LEGAL_COUNT])
    offset += PRIMITIVE_LEGAL_COUNT
    (history_len,) = struct.unpack_from("<I", data, offset)
    offset += 4

    expected = FIXED_BYTES + int(history_len) * HISTORY_EVENT_BYTES
    if len(data) != expected:
        raise ValueError(
            f"SPNNIV3 byte length mismatch: got={len(data)} expected={expected} history={history_len}"
        )

    history: list[DecodedHistoryEventV3] = []
    for _ in range(int(history_len)):
        event_cat = tuple(int(value) for value in data[offset : offset + 4])
        offset += 4
        event_num = tuple(struct.unpack_from("<4f", data, offset))
        offset += 16
        history.append(DecodedHistoryEventV3(event_cat, event_num))

    if offset != len(data):
        raise AssertionError("SPNNIV3 decoder did not consume entire payload")

    decoded = DecodedInputV3(
        categorical=categorical,
        rank_tokens=rank_tokens,
        same_suit=same_suit,
        numeric=numeric,
        primitive_legal=primitive_legal,
        history=tuple(history),
    )
    _validate(decoded)
    return decoded


def collate_v3(items: list[DecodedInputV3], device: str = "cpu") -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("cannot collate empty SPNNIV3 batch")

    max_history = max(1, max(item.history_len for item in items))
    history_cat = torch.zeros((len(items), max_history, 4), dtype=torch.long, device=device)
    history_num = torch.zeros((len(items), max_history, 4), dtype=torch.float32, device=device)
    history_len = torch.tensor([item.history_len for item in items], dtype=torch.long, device=device)
    for row, item in enumerate(items):
        for index, event in enumerate(item.history):
            history_cat[row, index] = torch.tensor(event.categorical, dtype=torch.long, device=device)
            history_num[row, index] = torch.tensor(event.numeric, dtype=torch.float32, device=device)

    return {
        "categorical": torch.tensor([x.categorical for x in items], dtype=torch.long, device=device),
        "rank_tokens": torch.tensor([x.rank_tokens for x in items], dtype=torch.long, device=device),
        "same_suit": torch.tensor(
            [x.same_suit_matrix() for x in items], dtype=torch.long, device=device
        ),
        "visible_mask": torch.tensor([x.visible_mask for x in items], dtype=torch.bool, device=device),
        "role_tokens": torch.tensor([x.role_tokens for x in items], dtype=torch.long, device=device),
        "numeric": torch.tensor([x.numeric for x in items], dtype=torch.float32, device=device),
        "primitive_legal": torch.tensor(
            [x.primitive_legal for x in items], dtype=torch.bool, device=device
        ),
        "history_categorical": history_cat,
        "history_numeric": history_num,
        "history_len": history_len,
    }
