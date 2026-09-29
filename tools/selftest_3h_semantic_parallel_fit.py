#!/usr/bin/env python3
from __future__ import annotations

"""Dependency-free local smoke/parity test for the semantic parallel fitter.

PROJECT_CONTRACT_IDS:
GOV-023,PERF-002,PERF-013,PERF-020,PERF-022,PERF-024,TRAIN-020,MODEL-020

This intentionally uses only the project runtime dependencies already required
by the trainer.  It exists because the production WSL virtualenv is not required
to contain pytest.  Full regression remains in GitHub CI; the real Ryzen gate
still performs the authoritative 1600-step serial-vs-parallel parity test on the
actual 2M reservoir.
"""

import struct
import sys
import tempfile
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


def main() -> int:
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

    with tempfile.TemporaryDirectory(prefix="spincore_semantic_parallel_") as td:
        packed, _meta = par.PackedSemanticAdvantageReservoir.build(
            memory,
            semantic,
            train_pool,
            Path(td) / "pack",
        )
        try:
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
            if set(actual_batch) != set(expected_batch):
                raise AssertionError("packed batch field-set mismatch")
            for key in expected_batch:
                if not torch.equal(actual_batch[key], expected_batch[key]):
                    raise AssertionError(f"packed batch mismatch: {key}")
            if not torch.equal(actual_target, expected_target):
                raise AssertionError("packed target mismatch")
            if not torch.equal(actual_weights, expected_weights):
                raise AssertionError("packed weights mismatch")

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
            with par.ParallelSemanticEnsembleFitter(
                manifest_path=packed.manifest_path,
                contract=contract,
                concurrency=2,
                threads_per_member=1,
            ) as fitter:
                candidate = fitter.fit([0, 1])
            par.assert_fit_rows_exact(reference, candidate)
            if [par.state_digest(x["state"]) for x in reference] != [
                par.state_digest(x["state"]) for x in candidate
            ]:
                raise AssertionError("semantic parallel state digest mismatch")
        finally:
            packed.close()

    print("SEMANTIC_PARALLEL_LOCAL_SELFTEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
