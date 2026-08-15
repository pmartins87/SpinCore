from __future__ import annotations

import struct
from itertools import permutations

import torch

from spincore_nn.codec_v3 import DecodedHistoryEventV3, DecodedInputV3, MAGIC
from spincore_nn.models_v3_final import (
    collate_v3_observations,
    make_h2_final_v3,
    make_h3_final_v3,
)

Card = tuple[int, int]
LEGAL = ((1, 1, 1, 1, 0, 1, 0, 0, 0, 1),)


def _relations(cards: list[Card | None]) -> tuple[int, ...]:
    out: list[int] = []
    for left in range(7):
        for right in range(left + 1, 7):
            a, b = cards[left], cards[right]
            out.append(int(a is not None and b is not None and a[1] == b[1]))
    return tuple(out)


def _item(
    hole: tuple[Card, Card],
    board: tuple[Card, ...],
    *,
    history: tuple[DecodedHistoryEventV3, ...] = (),
) -> DecodedInputV3:
    street = {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]
    cards: list[Card | None] = [hole[0], hole[1], None, None, None, None, None]
    for index, card in enumerate(board):
        cards[index + 2] = card
    return DecodedInputV3(
        categorical=(0, street, 0, 1, 2, 3, len(board), 0, 0, 0),
        rank_tokens=tuple(card[0] if card is not None else 0 for card in cards),
        same_suit=_relations(cards),
        numeric=(
            6.0, 0.5, 2.0,
            18.0, 15.0, 21.0,
            2.0, 2.0, 2.0,
            4.0, 4.0, 4.0,
            0.5, 3.0, 4.0, 22.0,
        ),
        primitive_legal=(1, 0, 1, 0, 1, 1),
        history=history,
    )


def _serialize(item: DecodedInputV3) -> bytes:
    out = bytearray(MAGIC)
    out.extend(item.categorical)
    out.extend(item.rank_tokens)
    out.extend(item.same_suit)
    out.extend(struct.pack("<16f", *item.numeric))
    out.extend(item.primitive_legal)
    out.extend(struct.pack("<I", len(item.history)))
    for event in item.history:
        out.extend(event.categorical)
        out.extend(struct.pack("<4f", *event.numeric))
    return bytes(out)


def _history(count: int) -> tuple[DecodedHistoryEventV3, ...]:
    events = []
    pot = 1.5
    commitments = [0.0, 0.5, 1.0]
    for index in range(count):
        actor = index % 3
        street = min(3, index // 12)
        action = 1 if index % 4 == 0 else (2 if index % 4 == 1 else 4)
        paid = 0.0 if action == 1 else 0.25
        commitments[actor] += paid
        before = pot
        pot += paid
        events.append(
            DecodedHistoryEventV3(
                (actor, street, action, 0),
                (paid, commitments[actor], before, pot),
            )
        )
    return tuple(events)


def test_h3_full_model_is_invariant_to_suit_hole_and_flop_symmetries() -> None:
    torch.set_num_threads(1)
    _, model = make_h3_final_v3(seed=7532026)
    model.eval()
    hole = ((14, 0), (13, 1))
    flop = ((12, 0), (8, 1), (2, 2))
    history = _history(7)
    baseline_payload = _serialize(_item(hole, flop, history=history))
    with torch.no_grad():
        baseline = model(
            collate_v3_observations(
                [baseline_payload], LEGAL, with_semantics=True
            )
        )

    variants = []
    variants.append(_item((hole[1], hole[0]), flop, history=history))
    variants.extend(_item(hole, tuple(f), history=history) for f in permutations(flop))
    for suit_perm in permutations(range(4)):
        transformed_hole = tuple((rank, suit_perm[suit]) for rank, suit in hole)
        transformed_flop = tuple((rank, suit_perm[suit]) for rank, suit in flop)
        variants.append(_item(transformed_hole, transformed_flop, history=history))

    with torch.no_grad():
        for variant in variants:
            actual = model(
                collate_v3_observations(
                    [_serialize(variant)], LEGAL, with_semantics=True
                )
            )
            assert torch.equal(actual, baseline), (actual, baseline)


def test_v3_history_padding_is_neutral_and_history_exceeds_32() -> None:
    torch.set_num_threads(1)
    _, model = make_h2_final_v3(seed=111)
    model.eval()
    hole = ((14, 0), (13, 1))
    flop = ((12, 0), (8, 1), (2, 2))
    short = _serialize(_item(hole, flop, history=_history(5)))
    long = _serialize(_item(hole, flop, history=_history(40)))

    single = collate_v3_observations([short], LEGAL, with_semantics=False)
    mixed = collate_v3_observations(
        [short, long],
        (LEGAL[0], LEGAL[0]),
        with_semantics=False,
    )
    assert int(single["history_len"][0]) == 5
    assert tuple(int(x) for x in mixed["history_len"]) == (5, 40)
    assert mixed["history_categorical"].shape[1] == 40

    with torch.no_grad():
        a = model(single)[0]
        b = model(mixed)[0]
    assert torch.allclose(a, b, atol=1e-7, rtol=0.0), (a, b)


def test_rank_domain_keeps_king_and_ace_distinct_without_clamp() -> None:
    _, model = make_h2_final_v3(seed=222)
    assert model.rank_emb.num_embeddings == 15
    assert not torch.equal(model.rank_emb.weight[13], model.rank_emb.weight[14])

    ace = _serialize(_item(((14, 0), (12, 1)), ((9, 2), (7, 3), (2, 0))))
    king = _serialize(_item(((13, 0), (12, 1)), ((9, 2), (7, 3), (2, 0))))
    a = collate_v3_observations([ace], LEGAL, with_semantics=False)
    k = collate_v3_observations([king], LEGAL, with_semantics=False)
    assert not torch.equal(a["canonical_ranks"], k["canonical_ranks"])


def test_h3_adds_semantic_capacity_without_removing_exact_channels() -> None:
    _, h2 = make_h2_final_v3(seed=333)
    _, h3 = make_h3_final_v3(seed=333)
    p2 = sum(parameter.numel() for parameter in h2.parameters())
    p3 = sum(parameter.numel() for parameter in h3.parameters())
    assert p3 > p2
    assert p2 < 400_000
    assert p3 < 500_000


def test_v3_model_factories_do_not_advance_global_torch_rng() -> None:
    torch.manual_seed(0x753C)
    state_before = torch.random.get_rng_state().clone()
    _, h2a = make_h2_final_v3(seed=123456)
    state_after_h2 = torch.random.get_rng_state().clone()
    _, h3a = make_h3_final_v3(seed=654321)
    state_after_h3 = torch.random.get_rng_state().clone()
    assert torch.equal(state_before, state_after_h2)
    assert torch.equal(state_before, state_after_h3)

    # Same isolated seed remains bit-identical regardless of caller RNG state.
    torch.manual_seed(999999)
    _, h2b = make_h2_final_v3(seed=123456)
    _, h3b = make_h3_final_v3(seed=654321)
    for a, b in zip(h2a.parameters(), h2b.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(h3a.parameters(), h3b.parameters()):
        assert torch.equal(a, b)
