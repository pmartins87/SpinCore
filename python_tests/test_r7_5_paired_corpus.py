from __future__ import annotations

from spincore.r7_5_paired_corpus import (
    BottomHashCorpus,
    PairedSample,
    immutable_sample_identity,
    is_train_sample,
    split_items,
)


def _sample(index: int) -> PairedSample:
    v1 = bytearray(126)
    v1[:8] = b"SPNNIV1\x00"
    v1[8:16] = int(index).to_bytes(8, "little", signed=False)
    v2 = bytearray(830)
    v2[:8] = b"SPNNIV2\x00"
    v2[8:16] = int(index).to_bytes(8, "little", signed=False)
    return PairedSample(
        kind="advantage" if index % 2 == 0 else "strategy",
        domain="TRUE_HEADS_UP",
        corpus_seed=1202035427,
        observation_v1=bytes(v1),
        observation_v2=bytes(v2),
        legal=(1, 1, 0, 1, 0, 1),
        target=(float(index), 0.0, 0.0, 1.0, 0.0, -1.0),
        weight=2.0,
        iteration=2,
    )


def test_bottom_hash_retention_is_insertion_order_independent() -> None:
    samples = [_sample(index) for index in range(50)]
    forward = BottomHashCorpus[PairedSample](10)
    reverse = BottomHashCorpus[PairedSample](10)
    for sample in samples:
        forward.add(sample)
    for sample in reversed(samples):
        reverse.add(sample)

    assert [immutable_sample_identity(x) for x in forward.items] == [
        immutable_sample_identity(x) for x in reverse.items
    ]
    assert forward.state_summary() == reverse.state_summary()
    assert forward.state_summary()["seen"] == 50
    assert forward.state_summary()["kept"] == 10


def test_bottom_hash_preserves_exact_duplicates_without_heap_comparison_failure() -> None:
    sample = _sample(8)
    corpus = BottomHashCorpus[PairedSample](3)
    for _ in range(5):
        corpus.add(sample)

    assert corpus.seen == 5
    assert len(corpus.items) == 3
    assert all(item == sample for item in corpus.items)
    identities = [immutable_sample_identity(item) for item in corpus.items]
    assert identities == [immutable_sample_identity(sample)] * 3


def test_hash_split_is_deterministic_and_disjoint() -> None:
    samples = [_sample(index) for index in range(200)]
    train_a, heldout_a = split_items(samples, split_seed=1925930899)
    train_b, heldout_b = split_items(reversed(samples), split_seed=1925930899)
    assert [immutable_sample_identity(x) for x in train_a] == [
        immutable_sample_identity(x) for x in train_b
    ]
    assert [immutable_sample_identity(x) for x in heldout_a] == [
        immutable_sample_identity(x) for x in heldout_b
    ]
    assert set(map(immutable_sample_identity, train_a)).isdisjoint(
        set(map(immutable_sample_identity, heldout_a))
    )
    assert len(train_a) + len(heldout_a) == len(samples)
    assert all(is_train_sample(sample, 1925930899) for sample in train_a)
    assert all(not is_train_sample(sample, 1925930899) for sample in heldout_a)
