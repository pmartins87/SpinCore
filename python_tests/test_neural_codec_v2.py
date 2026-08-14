from __future__ import annotations

import struct

from spincore_nn.codec_v2 import SIZE, collate_inputs_v2, decode_spnniv2


def _payload() -> bytes:
    out = bytearray(b"SPNNIV2\x00")
    out.append(13)
    out.extend([2, 0, 11, 0, 12, 1])
    out.extend(struct.pack("<24f", *[float(i) for i in range(24)]))
    out.extend([i for i in range(72)])
    out.extend([1, 1, 0, 1, 0, 1])
    out.append(2)
    for i in range(32):
        out.extend([i % 3, i % 4, i % 6, 1 if i < 2 else 0])
        out.extend(struct.pack("<4f", float(i), float(i + 1), float(i + 2), float(i + 3)))
    assert len(out) == SIZE
    return bytes(out)


def test_decode_spnniv2_contract() -> None:
    decoded = decode_spnniv2(_payload())
    assert decoded.preflop_class_id == 13
    assert decoded.canonical_flop_signature == (2, 0, 11, 0, 12, 1)
    assert decoded.numeric[0] == 0.0
    assert decoded.numeric[23] == 23.0
    assert decoded.categorical[10] == 10
    assert decoded.legal == (1, 1, 0, 1, 0, 1)
    assert decoded.history_len == 2
    assert decoded.history[0].categorical == (0, 0, 0, 1)
    assert decoded.history[1].numeric == (1.0, 2.0, 3.0, 4.0)


def test_collate_spnniv2_shapes() -> None:
    decoded = decode_spnniv2(_payload())
    batch = collate_inputs_v2([decoded, decoded])
    assert tuple(batch["preflop_class_id"].shape) == (2,)
    assert tuple(batch["canonical_flop_signature"].shape) == (2, 6)
    assert tuple(batch["numeric"].shape) == (2, 24)
    assert tuple(batch["categorical"].shape) == (2, 72)
    assert tuple(batch["legal"].shape) == (2, 6)
    assert tuple(batch["history_categorical"].shape) == (2, 32, 4)
    assert tuple(batch["history_numeric"].shape) == (2, 32, 4)
    assert tuple(batch["history_len"].shape) == (2,)
