from __future__ import annotations

from spincore.card_symmetry_v1 import canonicalize_v1_input, encode_spnniv1
from spincore_nn.codec import DecodedInput, decode_spnniv1


def test_spnniv1_reencode_roundtrip_preserves_all_noncard_fields() -> None:
    item = DecodedInput(
        cards=(49, 46, 45, 25, 2, 20, 0),
        numeric=tuple(float(i) for i in range(16)),
        categorical=tuple(range(8)),
        legal=(1, 1, 0, 1, 0, 1),
        history=tuple((i * 3) % 64 for i in range(32)),
        history_len=17,
    )
    canonical = canonicalize_v1_input(item)
    payload = encode_spnniv1(canonical)
    assert len(payload) == 126
    decoded = decode_spnniv1(payload)
    assert decoded == canonical
    assert decoded.numeric == item.numeric
    assert decoded.categorical == item.categorical
    assert decoded.legal == item.legal
    assert decoded.history == item.history
    assert decoded.history_len == item.history_len
