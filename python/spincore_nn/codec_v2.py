from __future__ import annotations

from dataclasses import dataclass
import struct

import torch

NUMERIC_COUNT = 24
CATEGORICAL_COUNT = 72
HISTORY_CAPACITY = 32
HISTORY_CAT_COUNT = 4
HISTORY_NUMERIC_COUNT = 4
SIZE = 830


@dataclass(frozen=True)
class DecodedHistoryEventV2:
    categorical: tuple[int, ...]
    numeric: tuple[float, ...]


@dataclass(frozen=True)
class DecodedInputV2:
    preflop_class_id: int
    canonical_flop_signature: tuple[int, ...]
    numeric: tuple[float, ...]
    categorical: tuple[int, ...]
    legal: tuple[int, ...]
    history: tuple[DecodedHistoryEventV2, ...]
    history_len: int


def decode_spnniv2(payload: bytes) -> DecodedInputV2:
    if len(payload) != SIZE or payload[:8] != b"SPNNIV2\x00":
        raise ValueError("bad SPNNIV2 payload")

    p = 8
    preflop_class_id = int(payload[p])
    p += 1
    if preflop_class_id > 168:
        raise ValueError("bad preflop class id")

    canonical_flop_signature = tuple(int(x) for x in payload[p : p + 6])
    p += 6

    numeric = tuple(float(x) for x in struct.unpack_from("<24f", payload, p))
    p += NUMERIC_COUNT * 4
    categorical = tuple(int(x) for x in payload[p : p + CATEGORICAL_COUNT])
    p += CATEGORICAL_COUNT
    legal = tuple(int(x) for x in payload[p : p + 6])
    p += 6
    history_len = int(payload[p])
    p += 1
    if history_len > HISTORY_CAPACITY:
        raise ValueError("bad V2 history length")

    history: list[DecodedHistoryEventV2] = []
    for _ in range(HISTORY_CAPACITY):
        event_cat = tuple(int(x) for x in payload[p : p + HISTORY_CAT_COUNT])
        p += HISTORY_CAT_COUNT
        event_numeric = tuple(
            float(x)
            for x in struct.unpack_from(f"<{HISTORY_NUMERIC_COUNT}f", payload, p)
        )
        p += HISTORY_NUMERIC_COUNT * 4
        history.append(DecodedHistoryEventV2(event_cat, event_numeric))

    if p != SIZE:
        raise ValueError("SPNNIV2 decoder size mismatch")

    return DecodedInputV2(
        preflop_class_id=preflop_class_id,
        canonical_flop_signature=canonical_flop_signature,
        numeric=numeric,
        categorical=categorical,
        legal=legal,
        history=tuple(history),
        history_len=history_len,
    )


def collate_inputs_v2(
    items: list[DecodedInputV2],
    device: str = "cpu",
    *,
    flop_candidate: str | None = None,
) -> dict[str, torch.Tensor]:
    batch = {
        "preflop_class_id": torch.tensor(
            [item.preflop_class_id for item in items], dtype=torch.long, device=device
        ),
        "canonical_flop_signature": torch.tensor(
            [item.canonical_flop_signature for item in items], dtype=torch.long, device=device
        ),
        "numeric": torch.tensor(
            [item.numeric for item in items], dtype=torch.float32, device=device
        ),
        "categorical": torch.tensor(
            [item.categorical for item in items], dtype=torch.long, device=device
        ),
        "legal": torch.tensor(
            [item.legal for item in items], dtype=torch.bool, device=device
        ),
        "history_categorical": torch.tensor(
            [[event.categorical for event in item.history] for item in items],
            dtype=torch.long,
            device=device,
        ),
        "history_numeric": torch.tensor(
            [[event.numeric for event in item.history] for item in items],
            dtype=torch.float32,
            device=device,
        ),
        "history_len": torch.tensor(
            [item.history_len for item in items], dtype=torch.long, device=device
        ),
    }
    if flop_candidate is not None:
        from spincore.flop_candidate_tokens import flop_token

        batch["flop_token"] = torch.tensor(
            [flop_token(flop_candidate, item.canonical_flop_signature) for item in items],
            dtype=torch.long,
            device=device,
        )
    return batch
