from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from spincore_nn.lean_batch import vectorized_batch
from spincore_nn.reservoir import AdvantageSample, UniformReservoir

import lt3_3h_semantic_parallel_fit as par


def _obs(i: int) -> bytes:
    b = bytearray(126)
    b[:8] = b"SPNNIV1\x00"
    for j in range(7):
        b[8 + j] = (i + j) % 52 + 1
    struct.pack_into(
        "<16f",
        b,
        15,
        *[float(i * 0.01 + j * 0.125) for j in range(16)],
    )
    for j in range(8):
        b[79 + j] = (i + 3 * j) % 17
    b[93] = 4
    for j in range(32):
        b[94 + j] = ((i + j) % 20) + 1 if j < 4 else 0
    return bytes(b)


def _sample(i: int) -> AdvantageSample:
    legal = tuple(1 if j in (0, 1, 2, 5, 9) else 0 for j in range(10))
    target = tuple(float((i + 1) * (j + 1)) / 100.0 for j in range(10))
    return AdvantageSample(
        observation=_obs(i),
        legal=legal,
        target=target,
        weight=float(1 + (i % 5)),
        iteration=10000 + i,
    )


def _fixture():
    memory = UniformReservoir[AdvantageSample](capacity=64, seed=77)
    for i in range(64):
        memory.add(_sample(i))
    semantic = np.asarray(
        [
            [float((i + 2 * j) % 7) for j in range(par.SEM_DIM)]
            for i in range(64)
        ],
        dtype=np.float32,
    )
    protected = {3, 11, 29, 41}
    train_pool = [i for i in range(64) if i not in protected]
    member_meta = [
        {"init_seed": 1000 + i, "batch_seed": 2000 + i}
        for i in range(8)
    ]
    return memory, semantic, train_pool, member_meta


def test_semantic_packed_batch_matches_canonical_vectorized_batch(tmp_path):
    memory, semantic, train_pool, _ = _fixture()
    packed, meta = par.PackedSemanticAdvantageReservoir.build(
        memory, semantic, train_pool, tmp_path / "pack"
    )
    try:
        assert meta["count"] == 64
        assert meta["train_count"] == 60
        positions = [0, 7, 13, 31, 45, 59]
        idx = [train_pool[p] for p in positions]
        samples = [memory.items[i] for i in idx]

        expected_batch, expected_target, expected_weights = vectorized_batch(
            samples, "cpu"
        )
        expected_batch["semantic"] = torch.tensor(
            np.asarray(semantic[idx], dtype=np.float32),
            dtype=torch.float32,
        )
        actual_batch, actual_target, actual_weights = packed.batch_from_positions(
            positions
        )

        assert set(actual_batch) == set(expected_batch)
        for key in expected_batch:
            assert torch.equal(actual_batch[key], expected_batch[key]), key
        assert torch.equal(actual_target, expected_target)
        assert torch.equal(actual_weights, expected_weights)
    finally:
        packed.close()


def test_semantic_observer_composes_with_existing_semantic_observer(tmp_path):
    memory, semantic, train_pool, _ = _fixture()
    packed, _ = par.PackedSemanticAdvantageReservoir.build(
        memory, semantic, train_pool, tmp_path / "pack"
    )
    touched = []

    def prior(index, sample):
        touched.append(int(index))
        semantic[int(index)] = np.float32(9.0)

    memory.set_write_observer(prior)
    packed.bind_authoritative(
        memory,
        semantic,
        semantic_fn=lambda _sample: np.zeros(par.SEM_DIM, dtype=np.float32),
    )
    try:
        # Force a direct observer invocation; Algorithm-R replacement selection
        # is independently covered by reservoir tests.
        replacement = _sample(999)
        memory._write_observer(5, replacement)
        assert touched == [5]
        assert np.all(semantic[5] == np.float32(9.0))
        assert np.all(
            np.asarray(packed.semantic[5], dtype=np.float32)
            == np.float32(9.0)
        )
        assert bytes(np.asarray(packed.observations[5], dtype=np.uint8)) == replacement.observation
    finally:
        packed.close()


def test_parallel_semantic_fit_is_tensor_exact_to_sequential_reference(tmp_path):
    memory, semantic, train_pool, member_meta = _fixture()
    packed, _ = par.PackedSemanticAdvantageReservoir.build(
        memory, semantic, train_pool, tmp_path / "pack"
    )
    contract = par.make_fit_contract(
        member_meta,
        learning_rate=3e-4,
        batch_size=16,
        member_steps=5,
    )

    reference = par.fit_members_sequential_reference(
        memory,
        semantic,
        train_pool,
        contract,
        members=[0, 1],
        threads=1,
    )
    try:
        with par.ParallelSemanticEnsembleFitter(
            manifest_path=packed.manifest_path,
            contract=contract,
            concurrency=2,
            threads_per_member=1,
        ) as fitter:
            candidate = fitter.fit([0, 1])
        par.assert_fit_rows_exact(reference, candidate)
        assert [par.state_digest(x["state"]) for x in reference] == [
            par.state_digest(x["state"]) for x in candidate
        ]
    finally:
        packed.close()


def test_sequential_reference_matches_canonical_pilot_fit_ensemble(monkeypatch):
    memory, semantic, train_pool, member_meta = _fixture()
    contract = par.make_fit_contract(
        member_meta,
        learning_rate=3e-4,
        batch_size=16,
        member_steps=5,
    )

    import run_3h_semantic_online_pilot_10105 as pilot

    monkeypatch.setattr(pilot, "MEMBER_STEPS", 5)
    torch.set_num_threads(1)
    _models, canonical_states, canonical_meta, _wall = pilot.fit_ensemble(
        memory,
        semantic,
        train_pool,
        member_meta,
        lr=3e-4,
        batch_size=16,
        iteration=12345,
    )
    reference = par.fit_members_sequential_reference(
        memory,
        semantic,
        train_pool,
        contract,
        threads=1,
    )
    assert len(canonical_states) == len(reference) == 8
    for member in range(8):
        assert par.tensor_states_equal(
            canonical_states[member], reference[member]["state"]
        )
        assert canonical_meta[member]["init_seed"] == reference[member]["init_seed"]
        assert canonical_meta[member]["batch_seed"] == reference[member]["batch_seed"]
        assert canonical_meta[member]["steps"] == reference[member]["steps"]
        assert canonical_meta[member]["loss_last"] == reference[member]["loss_last"]
