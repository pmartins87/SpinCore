from __future__ import annotations

import base64
import hashlib
import zlib
from functools import lru_cache

from spincore.flop184_recluster import build_h3_mapping, exact_iso_flop_keys

H1_TOKEN_BYTES_SHA256 = "b4938e63343c56f313104ac7adba23335c447f5ae97d492bdca9955c5029738a"
H2_TOKEN_BYTES_SHA256 = "01e15c60c35a775089a9d306c2fcf404c2e2539f7a742aaa2d2818dbedcc1123"
H3_MAPPING_SHA256 = "2c83cf993bcc4003223d184bd6f5584720b23cf04b95e6db69f84b09a86a64d0"

# Token arrays are indexed by sorted exact suit-isomorphic flop key. They were
# generated from the recovered 22,100-row historical map before any R7.5.3
# learning output. Token IDs use lexical historical-representative order.
_H1_B64 = "eNptkolXElEYxY+atpillWkWaqaGZmZOuJQVBmpiRgtiUVQWKa6BppRlKWYLLqWiqIEJGOm4MBiM/Hl97z2GGcwzw5l3uPfO990fBOMLyXUQX3X9F/rR52BhfIJYnLAn9dCe1MN5icejTx2PPlYUFXUy9kRJUtIRkSgtVlIoPppKpVKJqlOqY6qoqBOqpBZRi0SlSA6lUkKpmBhhyooyKTgTUxnOCMdc+i/wn51faYd3p9N+LT0d7tMZGZlx6qy9A1f2tcHBmq5J12S0qq8OSNuk6mw7cZw5RxxdZ+LUDtA71NkhXRDvgoMDoiC9Aml548Z+7UqWTHb2QI621SEbyZF+2q/d1FZuwKPkfO5fbWXuCPI8hjx2bWshjHyO4O7gFQR8Q7UStywtys+/GCspKS6GiherJFZEXQlISlX5lRJVMQGCfikucEkQSKtC/JTwI0XYD+32cuTd6bSX3SwHOpcrKjIJnZq267Vqaznwq+D4aezEEeaHHKB3hHU+jkUHiQI/DfArwPwe8fwKCL8CAb8C7eNzAn5XCT+FSWF68xZdhF5d/5t+RbJYnPxU2VCtu5OXclcuB2K36uvvi0TNVRKT+KiS0gE6OfS7HfqzIq8y7L0X9ioxZN5pb2jSAwyo0lir7tw78ICQ0FPliIQm1DTS0dWIWQEJDU+CiA9rMCYS5Ul0ynoePe/Vtq7IRnoFJG4JSUAeuxAJDfIhEi9MOzgkPFF+eKprr9a9lMubOQh38hJNCIGO4oqlqOwfnunbm/TvBMXKDAYopqfwdiNtUoPGHunoIg5cjOggXueKfQaRK2aAYt19xtXOnp7Xz3uNras9qFifcdMIxfqMqJgRFQMPKoZd20YII581+AIX42uhZeGGaTAEr/Lx/eAgWdZsgGU+Sgcp+64OgR4WH4I4NGTlpCEqtOzwMFrWvDocsew9wbIoj13bRggP42VnLDOWxcUZyze44Dw6Oztls42OjXV+nZwzGOBgmaVs5jF2csIwMcbOgDjOiXCw8JLXAxo9hRWb2Y++7P7y3fvD7IcHBOgf5skJP3r5zwXiom3mBRy2BCN2IAvMud2/AgHYB89wTwTY2dAMosAM9GX39Lx3HmZMz88tLdHzZic3g7jwDOSzBPkJXg+I9C+XCwxut9/FBlgIe5eW/HNO57rL5XU6XawfPOM/F4iLdrsXWOSz/PZ5FpkgzdD+9T9rvjXWAw+IwAkiyx6G+b3sWd/a8m2xDMP4mKAPGej1AHrLGotn7RJcZMhuwX+e0IQz"
_H2_B64 = "eNptkolXElEYxY+ZaaWGpWklamaGS2aOmJYlJmqS0oJYFG1SoaagKWa5YbbgHgIaGIKRjhaDwcj/1/feY2AkzsCZd7j3zvfdHxM8WkauZHzdHrk0gr7JZUdTRKKUg1mpB7OOFQlOHsg+eSC9PC7uTMKpqrS040Lh6QRxmehEFpVFCRTZinRFXNwpRdpr4WuxQpYRSmWGUvHx/JQZZTJxJr4+nOGPqfwvIIi2R1aK8kY77TdycuBzNjc375AyP3H0WlI3HMw5qhxVbpfy+qikW6IssBPHuZKQ49whpQP0N8qCkM6L98HBAVGQ+kDa+H3zsHozv6HhwpHz6i5Hw/R5ycfD6j/q+t9wq7pY+FddXziNPI8hj117aggjnyMYG7yMgG9vkuOWV8qLiy8niKsqKqDi5UaxGVGXA5IriuJ6saKCAEH/FBeo5AVONyJ+/9lTYz0ceaOd9upbNUDnam1tHqHT3F3XojTXAL9ajp/KThxhfsgB+puwHokn9YHoIFHgpwJ+pZjfowi/UsKvlMevVP24hMfvOuEnM8gMw+/QRejdHhkekWWIRBlP5e1NmjtFmXelUiDW2tZ2Xyh81Sg2iE7IKQ28fFLo1xZ6WZFXHvbeC3vlQI3vtLc/1wIMqNLRouxNHH1ASGipGkRCFWoa5ejArICEKkKCiA+bEQkziUZI9AKJF4Pqrs2G6UEeiVY+CchjFyKhQj5EotMQxSHliXzsqaanSfNSKn3FQbhTJDAgBBqKKyZQ2MeeaXuea9/z1q7W6aCYlsLbTXdLdCp7TAcuRnQQ67hin0DkiumgWP+Qfqt3YODti0F919YAKjak/6OHYkN6VEyPioEHFcOuPT2Ekc8c7MTFIrXQsvCBaTAErzL+YWKCLGvUwTLjkgnKHtPB08PiQxAnJ82cNEmFlp2aQssat6b2LXuPtyzKY9eeHsJTeNll07JpbW3Z9BUuOM9YLEtW68zsbO+XRZtOBweThbIaZ9nFed38LLsM4hwnwsEUkbwe0OglrFiNfvRj/+cF74LRDzcI0AvGxXk/evj3VeKircZVHDYF9+1AFrC53T8CAdgHz3DPB1hLaAZRYAb6sf/bincFZnxbsTmd9IrRyc0gLjwD+UzByASvB0T6h8sFBrfb72IDLIS9Tqfftr6+43J519ddrB88c99XiYt2u1dZ5DP99HnWmCDN0P6dX9u+bdYDN4jACSIbHob5ueHZ2d317bIMw/iYoA8Z6J0Aeso2i2fFCK4xZLfgP41+e1Q="


def _decode_tokens(blob: str, expected_sha256: str, expected_max: int) -> bytes:
    raw = zlib.decompress(base64.b64decode(blob))
    if len(raw) != 1755:
        raise ValueError(f"legacy token table length mismatch: {len(raw)}")
    if max(raw) != expected_max or min(raw) < 1:
        raise ValueError("legacy token table range mismatch")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("legacy token table SHA256 mismatch")
    return raw


@lru_cache(maxsize=1)
def exact_key_texts() -> tuple[str, ...]:
    def text(key):
        ranks = "23456789TJQKA"
        suits = "shdc"
        return "".join(ranks[rank - 2] + suits[suit] for rank, suit in key)
    return tuple(text(key) for key in exact_iso_flop_keys())


@lru_cache(maxsize=None)
def candidate_token_table(candidate: str) -> dict[str, int]:
    candidate = str(candidate).upper()
    keys = exact_key_texts()
    if candidate == "H1":
        raw = _decode_tokens(_H1_B64, H1_TOKEN_BYTES_SHA256, 184)
        return {key: int(token) for key, token in zip(keys, raw)}
    if candidate == "H2":
        raw = _decode_tokens(_H2_B64, H2_TOKEN_BYTES_SHA256, 181)
        return {key: int(token) for key, token in zip(keys, raw)}
    if candidate == "H3":
        mapping = build_h3_mapping()
        representatives = sorted(set(mapping.values()))
        rep_to_token = {representative: index + 1 for index, representative in enumerate(representatives)}
        return {key: rep_to_token[mapping[key]] for key in keys}
    if candidate == "H4":
        return {key: index + 1 for index, key in enumerate(keys)}
    raise ValueError(f"unsupported flop candidate: {candidate!r}")


def signature_to_exact_text(signature: tuple[int, ...] | list[int]) -> str:
    if len(signature) != 6:
        raise ValueError("canonical flop signature must have six bytes")
    ranks = "23456789TJQKA"
    suits = "shdc"
    parts: list[str] = []
    for index in range(0, 6, 2):
        rank = int(signature[index])
        suit = int(signature[index + 1])
        if rank < 2 or rank > 14 or suit < 0 or suit > 3:
            raise ValueError("invalid canonical flop signature")
        parts.append(ranks[rank - 2] + suits[suit])
    return "".join(parts)


def flop_token(candidate: str, signature: tuple[int, ...] | list[int]) -> int:
    # All-zero signature denotes preflop and intentionally maps to padding token 0.
    if all(int(value) == 0 for value in signature):
        return 0
    exact = signature_to_exact_text(signature)
    try:
        return candidate_token_table(candidate)[exact]
    except KeyError as exc:
        raise ValueError(f"unknown exact flop key {exact!r}") from exc
