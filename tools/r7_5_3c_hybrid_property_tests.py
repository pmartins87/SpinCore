from __future__ import annotations

from dataclasses import replace
from itertools import permutations

import torch

from spincore_nn.codec import DecodedInput
from spincore_nn.codec_v2 import DecodedHistoryEventV2, DecodedInputV2
from spincore_nn.hybrid_v3 import (
    SAFE_SEMANTIC_CATEGORICAL_INDICES,
    SAFE_SEMANTIC_NUMERIC_INDICES,
    HybridNetV3,
    build_hybrid_input,
    collate_hybrid_inputs,
    count_parameters,
)


def _token(rank_index: int, suit: int) -> int:
    return int(rank_index) * 4 + int(suit) + 1


def _permute_suits(cards: tuple[int, ...], permutation: tuple[int, int, int, int]) -> tuple[int, ...]:
    out = []
    for token in cards:
        if token == 0:
            out.append(0)
            continue
        card_id = token - 1
        rank = card_id // 4
        suit = card_id % 4
        out.append(_token(rank, permutation[suit]))
    return tuple(out)


def _v1(
    cards: tuple[int, ...],
    *,
    domain: int = 0,
    street: int = 1,
    dealer_rel: int = 0,
    stacks=(20.0, 15.0, 10.0),
    street_commitments=(0.0, 1.0, 1.0),
    total_commitments=(1.0, 2.0, 2.0),
    statuses=(0, 0, 0),
    history=(1, 2, 3),
    history_len: int | None = None,
) -> DecodedInput:
    if history_len is None:
        history_len = len(history)
    history32 = tuple(history) + (0,) * (32 - len(history))
    numeric = (
        5.0,
        1.0,
        1.0,
        *tuple(float(x) for x in stacks),
        *tuple(float(x) for x in street_commitments),
        *tuple(float(x) for x in total_commitments),
        0.5,
        1.0,
        0.0,
        2.0 if domain == 1 else 3.0,
    )
    visible_board = sum(1 for token in cards[2:] if token)
    categorical = (
        domain,
        street,
        dealer_rel,
        2 if domain == 1 else 3,
        *statuses,
        visible_board,
    )
    return DecodedInput(
        cards=cards,
        numeric=tuple(numeric),
        categorical=tuple(categorical),
        legal=(1, 1, 0, 1, 1, 1),
        history=history32,
        history_len=history_len,
    )


def _v2(
    v1: DecodedInput,
    *,
    history_actor: int = 1,
    forced: int = 0,
    paid: float = 1.0,
) -> DecodedInputV2:
    numeric = [0.0] * 24
    numeric[0:13] = list(v1.numeric[0:13])
    numeric[13:16] = [9.0, 9.0, 1.8]  # intentionally not trusted by hybrid
    numeric[16:24] = [0.2, 2.0, 4.0, 1.0, 2.0, 0.25, 0.5, 4.0]

    cat = [0] * 72
    cat[0] = int(v1.categorical[0])
    cat[1] = int(v1.categorical[1])
    cat[2] = int(v1.categorical[2])
    cat[3] = 1
    cat[4] = 2
    cat[5] = int(v1.categorical[3])
    cat[6] = int(v1.categorical[7])
    cat[7:10] = list(v1.categorical[4:7])
    cat[10] = 2
    cat[11] = 3
    cat[12] = 1
    cat[13] = 1
    cat[14] = 1
    cat[15] = 2
    cat[16] = 2
    cat[17] = 1
    cat[18] = 2
    cat[19] = 1
    cat[20] = 1
    cat[21] = 0
    cat[22] = 2
    cat[23] = 1
    cat[24] = 0
    cat[25] = 0
    cat[26] = 0
    cat[27:33] = [1, 2, 0, 1, 2, 1]
    cat[33] = 4  # physical suit id: deliberately excluded
    cat[34:65] = [(index * 3) % 7 for index in range(31)]
    cat[65] = 3  # ordered-hole suit counts: deliberately excluded
    cat[66] = 1
    cat[67:71] = [1, 1, 0, 0]

    history = [DecodedHistoryEventV2((0, 0, 0, 0), (0.0, 0.0, 0.0, 0.0)) for _ in range(32)]
    history[0] = DecodedHistoryEventV2(
        (history_actor, 0, 3, forced),
        (paid, paid, 1.5, 1.5 + paid),
    )
    return DecodedInputV2(
        preflop_class_id=42,
        canonical_flop_signature=(1, 2, 3, 0, 1, 2),
        numeric=tuple(numeric),
        categorical=tuple(cat),
        legal=v1.legal,
        history=tuple(history),
        history_len=1,
    )


def _assert_close(a: torch.Tensor, b: torch.Tensor, *, label: str) -> None:
    if not torch.allclose(a, b, atol=1e-6, rtol=0.0):
        raise AssertionError(f"{label}: tensors differ; max={torch.max(torch.abs(a-b)).item()}")


def test_relational_card_invariance() -> None:
    # Hero Ah/Ks; flop Qh/8s/2d; turn Jc; river 3h using zero-based rank indices.
    cards = (
        _token(12, 2),
        _token(11, 0),
        _token(10, 2),
        _token(6, 0),
        _token(0, 1),
        _token(9, 3),
        _token(1, 2),
    )
    base_v1 = _v1(cards)
    base = build_hybrid_input(base_v1, _v2(base_v1))

    torch.manual_seed(20260814)
    model = HybridNetV3("H1_RELATIONAL_EXACT").eval()
    base_out = model(collate_hybrid_inputs([base]))

    # All physical suit renamings.
    for perm in permutations(range(4)):
        changed_v1 = replace(base_v1, cards=_permute_suits(cards, tuple(perm)))
        changed = build_hybrid_input(changed_v1, _v2(changed_v1))
        if changed.rank_tokens != base.rank_tokens or changed.same_suit != base.same_suit:
            raise AssertionError("label-free rank/suit relation changed under suit permutation")
        _assert_close(
            base_out,
            model(collate_hybrid_inputs([changed])),
            label=f"suit permutation {perm}",
        )

    # Hero private-card order.
    swapped_cards = (cards[1], cards[0], *cards[2:])
    swapped_v1 = replace(base_v1, cards=swapped_cards)
    swapped = build_hybrid_input(swapped_v1, _v2(swapped_v1))
    _assert_close(base_out, model(collate_hybrid_inputs([swapped])), label="hole swap")

    # All six flop permutations.
    flop = cards[2:5]
    for permuted in permutations(flop):
        changed_cards = cards[:2] + tuple(permuted) + cards[5:]
        changed_v1 = replace(base_v1, cards=changed_cards)
        changed = build_hybrid_input(changed_v1, _v2(changed_v1))
        _assert_close(
            base_out,
            model(collate_hybrid_inputs([changed])),
            label=f"flop permutation {permuted}",
        )

    # Chronology is not quotiented: turn/river role-aligned exact input changes.
    tr_cards = cards[:5] + (cards[6], cards[5])
    tr_v1 = replace(base_v1, cards=tr_cards)
    tr = build_hybrid_input(tr_v1, _v2(tr_v1))
    if tr.rank_tokens == base.rank_tokens and tr.same_suit == base.same_suit:
        raise AssertionError("turn/river swap was incorrectly collapsed")

    # Private/public role is not quotiented.
    pp = list(cards)
    pp[0], pp[2] = pp[2], pp[0]
    pp_v1 = replace(base_v1, cards=tuple(pp))
    pp_item = build_hybrid_input(pp_v1, _v2(pp_v1))
    if pp_item.rank_tokens == base.rank_tokens and pp_item.same_suit == base.same_suit:
        raise AssertionError("private/public card swap was incorrectly collapsed")


def test_true_hu_dead_seat_canonicalization() -> None:
    cards = (_token(12, 0), _token(11, 1), _token(8, 0), _token(4, 1), _token(1, 2), 0, 0)

    # Physical presentation A: live villain is rel1, dead chair rel2.
    a1 = _v1(
        cards,
        domain=1,
        dealer_rel=0,
        stacks=(20.0, 15.0, 0.0),
        street_commitments=(0.0, 1.0, 0.0),
        total_commitments=(1.0, 2.0, 0.0),
        statuses=(0, 0, 2),
    )
    a2 = _v2(a1, history_actor=1)
    a2_cat = list(a2.categorical)
    a2_cat[3], a2_cat[4] = 1, 2
    a2_cat[15] = a2_cat[16] = a2_cat[22] = 2
    a2 = replace(a2, categorical=tuple(a2_cat))

    # Physical presentation B: same poker state but dead chair is rel1 and villain rel2.
    b1 = _v1(
        cards,
        domain=1,
        dealer_rel=0,
        stacks=(20.0, 0.0, 15.0),
        street_commitments=(0.0, 0.0, 1.0),
        total_commitments=(1.0, 0.0, 2.0),
        statuses=(0, 2, 0),
    )
    b2 = _v2(b1, history_actor=2)
    b2_cat = list(b2.categorical)
    b2_cat[3], b2_cat[4] = 1, 3
    b2_cat[15] = b2_cat[16] = b2_cat[22] = 3
    b2 = replace(b2, categorical=tuple(b2_cat))

    ha = build_hybrid_input(a1, a2)
    hb = build_hybrid_input(b1, b2)
    if ha.numeric != hb.numeric or ha.categorical != hb.categorical:
        raise AssertionError("true-HU current state still depends on physical dead chair")
    if ha.structured_history_categorical != hb.structured_history_categorical:
        raise AssertionError("true-HU structured history still depends on physical dead chair")
    if ha.semantic_categorical != hb.semantic_categorical:
        raise AssertionError("true-HU semantic aggressor/position fields still depend on dead chair")


def test_buggy_old_v2_fields_are_excluded() -> None:
    cards = (_token(12, 0), _token(11, 1), _token(8, 0), _token(4, 1), _token(1, 2), 0, 0)
    v1 = _v1(cards)
    v2a = _v2(v1)
    v2b_num = list(v2a.numeric)
    v2b_num[13:16] = [0.0, 0.0, 0.0]
    v2b_cat = list(v2a.categorical)
    v2b_cat[33] = 1
    v2b_cat[65] = 0
    v2b_cat[66] = 4
    v2b = replace(v2a, numeric=tuple(v2b_num), categorical=tuple(v2b_cat))

    a = build_hybrid_input(v1, v2a)
    b = build_hybrid_input(v1, v2b)
    if a.semantic_numeric != b.semantic_numeric:
        raise AssertionError("dead-seat-corrupted V2 numeric fields leaked into hybrid semantics")
    if a.semantic_categorical != b.semantic_categorical:
        raise AssertionError("absolute/ordered suit fields leaked into hybrid semantics")
    if 33 in SAFE_SEMANTIC_CATEGORICAL_INDICES or 65 in SAFE_SEMANTIC_CATEGORICAL_INDICES or 66 in SAFE_SEMANTIC_CATEGORICAL_INDICES:
        raise AssertionError("known bad suit category was admitted")
    if any(index in SAFE_SEMANTIC_NUMERIC_INDICES for index in (13, 14, 15)):
        raise AssertionError("known bad HU effective-stack numeric field was admitted")


def test_padding_is_neutral() -> None:
    cards = (_token(12, 0), _token(11, 1), 0, 0, 0, 0, 0)
    v1 = _v1(cards, street=0, history=(1, 2), history_len=2)
    v2 = _v2(v1)
    base = build_hybrid_input(v1, v2)

    dirty_legacy = list(base.legacy_history)
    for index in range(2, 32):
        dirty_legacy[index] = (index % 31) + 1
    dirty = replace(base, legacy_history=tuple(dirty_legacy))

    torch.manual_seed(1234)
    h0 = HybridNetV3("H0_FIXED_V1").eval()
    _assert_close(
        h0(collate_hybrid_inputs([base])),
        h0(collate_hybrid_inputs([dirty])),
        label="legacy right-padding",
    )

    dirty_cat = [list(row) for row in base.structured_history_categorical]
    dirty_num = [list(row) for row in base.structured_history_numeric]
    for index in range(base.structured_history_len, 32):
        dirty_cat[index] = [2, 3, 4, 1]
        dirty_num[index] = [9.0, 8.0, 7.0, 6.0]
    dirty2 = replace(
        base,
        structured_history_categorical=tuple(tuple(row) for row in dirty_cat),
        structured_history_numeric=tuple(tuple(row) for row in dirty_num),
    )
    torch.manual_seed(5678)
    h2 = HybridNetV3("H2_RELATIONAL_EXACT_STRUCTURED_HISTORY").eval()
    _assert_close(
        h2(collate_hybrid_inputs([base])),
        h2(collate_hybrid_inputs([dirty2])),
        label="structured right-padding",
    )


def test_structured_history_keeps_material_differences() -> None:
    cards = (_token(12, 0), _token(11, 1), _token(8, 0), _token(4, 1), _token(1, 2), 0, 0)
    v1 = _v1(cards)
    small = build_hybrid_input(v1, _v2(v1, forced=0, paid=1.0))
    large = build_hybrid_input(v1, _v2(v1, forced=0, paid=2.0))
    forced = build_hybrid_input(v1, _v2(v1, forced=1, paid=1.0))
    if small.structured_history_numeric == large.structured_history_numeric:
        raise AssertionError("different sizing collapsed in structured history")
    if small.structured_history_categorical == forced.structured_history_categorical:
        raise AssertionError("forced and voluntary actions collapsed in structured history")


def test_parameter_inventory() -> None:
    counts = {candidate: count_parameters(HybridNetV3(candidate)) for candidate in sorted(HybridNetV3.CANDIDATES)}
    if counts["H4_HYBRID_CAPACITY"] > 500_000:
        raise AssertionError(f"H4 exceeds frozen parameter cap: {counts}")
    print("parameter_counts", counts)


def main() -> int:
    tests = [
        test_relational_card_invariance,
        test_true_hu_dead_seat_canonicalization,
        test_buggy_old_v2_fields_are_excluded,
        test_padding_is_neutral,
        test_structured_history_keeps_material_differences,
        test_parameter_inventory,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("R7.5.3C hybrid property preflight PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
