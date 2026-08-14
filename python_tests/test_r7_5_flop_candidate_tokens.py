from __future__ import annotations

from spincore.flop_candidate_tokens import (
    H1_TOKEN_BYTES_SHA256,
    H2_TOKEN_BYTES_SHA256,
    candidate_token_table,
    flop_token,
)


def test_h1_h2_compact_tables_are_complete_and_frozen() -> None:
    assert H1_TOKEN_BYTES_SHA256 == "b4938e63343c56f313104ac7adba23335c447f5ae97d492bdca9955c5029738a"
    assert H2_TOKEN_BYTES_SHA256 == "01e15c60c35a775089a9d306c2fcf404c2e2539f7a742aaa2d2818dbedcc1123"

    h1 = candidate_token_table("H1")
    h2 = candidate_token_table("H2")
    assert len(h1) == 1755
    assert len(h2) == 1755
    assert len(set(h1.values())) == 184
    assert len(set(h2.values())) == 181
    assert min(h1.values()) == 1 and max(h1.values()) == 184
    assert min(h2.values()) == 1 and max(h2.values()) == 181


def test_h3_h4_tables_share_same_exact_key_domain() -> None:
    h1 = candidate_token_table("H1")
    h3 = candidate_token_table("H3")
    h4 = candidate_token_table("H4")
    assert set(h1) == set(h3) == set(h4)
    assert len(set(h3.values())) == 184
    assert len(set(h4.values())) == 1755


def test_flop_token_uses_zero_only_for_preflop_padding() -> None:
    assert flop_token("H1", (0, 0, 0, 0, 0, 0)) == 0
    # Exact canonical Qs/Jh/2h example: 2s Js Qh.
    signature = (2, 0, 11, 0, 12, 1)
    for candidate in ("H1", "H2", "H3", "H4"):
        assert flop_token(candidate, signature) >= 1
