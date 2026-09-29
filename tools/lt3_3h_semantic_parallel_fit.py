#!/usr/bin/env python3
from __future__ import annotations

"""Exact-parity process-parallel fitter for the 3H semantic Advantage ensemble.

PROJECT_CONTRACT_IDS:
PERF-001,PERF-002,PERF-010,PERF-011,PERF-012,PERF-013,PERF-014,PERF-015,
PERF-016,PERF-017,PERF-019,PERF-020,PERF-022,PERF-024,
TRAIN-020,MODEL-020,RNG-001,RNG-002,RNG-003,CKPT-004,ART-015

The authoritative Python UniformReservoir remains the source of truth.  A compact
mmap mirror is built once, workers open it read-only, and the parent keeps the
mirror synchronized through the reservoir's observational write hook.  Member
initialization seeds, batch seeds, sample positions, sample order, model
architecture, Adam defaults and training steps are unchanged.

The implementation intentionally preserves the canonical sequential member
semantics while allowing independent members to fit concurrently.
"""

import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import resource
import sys
import time
from typing import Any, Callable, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import run_3h_v1_semantic_shadow_10105 as shadow
from spincore_nn.training import train_step

PACKED_SCHEMA = "SPINCORE_3H_SEMANTIC_PACKED_ADV_RESERVOIR_V1"
SEM_DIM = int(shadow.SEM_DIM)
ACTION_SLOTS = 10
OBS_BYTES = 126

_PACKED = None


def _clone_state(model) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def state_digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8") + b"\0")
        tensor = state[key].detach().cpu().contiguous()
        h.update(str(tensor.dtype).encode("ascii") + b"\0")
        h.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def tensor_states_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> bool:
    if set(left) != set(right):
        return False
    return all(torch.equal(left[key].cpu(), right[key].cpu()) for key in left)


def _files(root: Path) -> dict[str, Path]:
    return {
        "observations": root / "observations.u8",
        "legal": root / "legal.u8",
        "targets": root / "targets.f32",
        "weights": root / "weights.f32",
        "semantic": root / "semantic.f32",
        "train_indices": root / "train_indices.i32",
    }


class PackedSemanticAdvantageReservoir:
    """Read-mostly mmap mirror of the saturated 3H Advantage reservoir."""

    def __init__(self, manifest_path: Path, *, mode: str = "r"):
        self.manifest_path = Path(manifest_path).resolve()
        meta = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if meta.get("schema") != PACKED_SCHEMA:
            raise RuntimeError("wrong semantic packed-reservoir schema")
        if int(meta.get("observation_bytes", -1)) != OBS_BYTES:
            raise RuntimeError("semantic packed observation width drift")
        if int(meta.get("action_slots", -1)) != ACTION_SLOTS:
            raise RuntimeError("semantic packed action-slot drift")
        if int(meta.get("semantic_dim", -1)) != SEM_DIM:
            raise RuntimeError("semantic packed dimension drift")

        self.meta = meta
        self.count = int(meta["count"])
        self.train_count = int(meta["train_count"])
        self.root = self.manifest_path.parent
        files = _files(self.root)

        self.observations = np.memmap(
            files["observations"],
            dtype=np.uint8,
            mode=mode,
            shape=(self.count, OBS_BYTES),
        )
        self.legal = np.memmap(
            files["legal"],
            dtype=np.uint8,
            mode=mode,
            shape=(self.count, ACTION_SLOTS),
        )
        self.targets = np.memmap(
            files["targets"],
            dtype=np.float32,
            mode=mode,
            shape=(self.count, ACTION_SLOTS),
        )
        self.weights = np.memmap(
            files["weights"],
            dtype=np.float32,
            mode=mode,
            shape=(self.count,),
        )
        self.semantic = np.memmap(
            files["semantic"],
            dtype=np.float32,
            mode=mode,
            shape=(self.count, SEM_DIM),
        )
        self.train_indices = np.memmap(
            files["train_indices"],
            dtype=np.int32,
            mode=mode,
            shape=(self.train_count,),
        )
        self.write_updates = 0

    @classmethod
    def build(
        cls,
        memory,
        semantic_rows: np.ndarray,
        train_pool: list[int],
        root: Path,
        *,
        chunk_size: int = 8192,
    ):
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        count = len(memory.items)
        if count <= 0:
            raise ValueError("cannot pack empty semantic reservoir")
        if count != int(memory.capacity):
            raise RuntimeError(
                "semantic persistent mmap requires a saturated reservoir"
            )
        semantic_rows = np.asarray(semantic_rows, dtype=np.float32)
        if semantic_rows.shape != (count, SEM_DIM):
            raise RuntimeError(
                f"semantic row shape drift: {semantic_rows.shape} != {(count, SEM_DIM)}"
            )
        if not train_pool:
            raise ValueError("empty semantic train pool")
        if len(set(int(x) for x in train_pool)) != len(train_pool):
            raise RuntimeError("semantic train-pool contains duplicate indices")
        if min(train_pool) < 0 or max(train_pool) >= count:
            raise RuntimeError("semantic train-pool index out of range")

        files = _files(root)
        obs = np.memmap(
            files["observations"], dtype=np.uint8, mode="w+", shape=(count, OBS_BYTES)
        )
        legal = np.memmap(
            files["legal"], dtype=np.uint8, mode="w+", shape=(count, ACTION_SLOTS)
        )
        targets = np.memmap(
            files["targets"], dtype=np.float32, mode="w+", shape=(count, ACTION_SLOTS)
        )
        weights = np.memmap(
            files["weights"], dtype=np.float32, mode="w+", shape=(count,)
        )
        semantic = np.memmap(
            files["semantic"], dtype=np.float32, mode="w+", shape=(count, SEM_DIM)
        )
        train_indices = np.memmap(
            files["train_indices"],
            dtype=np.int32,
            mode="w+",
            shape=(len(train_pool),),
        )

        started = time.perf_counter()
        for start in range(0, count, int(chunk_size)):
            end = min(count, start + int(chunk_size))
            chunk = memory.items[start:end]
            observations = [sample.observation for sample in chunk]
            if any(
                len(value) != OBS_BYTES or value[:8] != b"SPNNIV1\x00"
                for value in observations
            ):
                raise RuntimeError("bad SPNNIV1 observation in semantic reservoir")
            if any(
                len(sample.legal) != ACTION_SLOTS
                or len(sample.target) != ACTION_SLOTS
                for sample in chunk
            ):
                raise RuntimeError("semantic reservoir action-width drift")
            raw = np.frombuffer(
                b"".join(observations), dtype=np.uint8
            ).reshape(-1, OBS_BYTES)
            obs[start:end] = raw
            legal[start:end] = np.asarray(
                [sample.legal for sample in chunk], dtype=np.uint8
            )
            targets[start:end] = np.asarray(
                [sample.target for sample in chunk], dtype=np.float32
            )
            weights[start:end] = np.asarray(
                [sample.weight for sample in chunk], dtype=np.float32
            )
            semantic[start:end] = semantic_rows[start:end]

        train_indices[:] = np.asarray(train_pool, dtype=np.int32)
        for mm in (obs, legal, targets, weights, semantic, train_indices):
            mm.flush()
        build_seconds = float(time.perf_counter() - started)
        del obs, legal, targets, weights, semantic, train_indices

        meta = {
            "schema": PACKED_SCHEMA,
            "count": int(count),
            "capacity": int(memory.capacity),
            "seen_at_build": int(memory.seen),
            "train_count": int(len(train_pool)),
            "observation_bytes": OBS_BYTES,
            "action_slots": ACTION_SLOTS,
            "semantic_dim": SEM_DIM,
            "build_seconds": build_seconds,
            "files": {name: path.name for name, path in files.items()},
            "bytes": int(sum(path.stat().st_size for path in files.values())),
        }
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return cls(manifest, mode="r+"), meta

    def update(self, index: int, sample, semantic_row) -> None:
        i = int(index)
        if not 0 <= i < self.count:
            raise IndexError("semantic packed reservoir index out of range")
        obs = sample.observation
        if len(obs) != OBS_BYTES or obs[:8] != b"SPNNIV1\x00":
            raise ValueError("bad SPNNIV1 observation")
        if len(sample.legal) != ACTION_SLOTS or len(sample.target) != ACTION_SLOTS:
            raise ValueError("semantic sample action-width drift")
        row = np.asarray(semantic_row, dtype=np.float32)
        if row.shape != (SEM_DIM,):
            raise ValueError("semantic update-row dimension drift")
        self.observations[i] = np.frombuffer(obs, dtype=np.uint8)
        self.legal[i] = np.asarray(sample.legal, dtype=np.uint8)
        self.targets[i] = np.asarray(sample.target, dtype=np.float32)
        self.weights[i] = np.float32(sample.weight)
        self.semantic[i] = row
        self.write_updates += 1

    def bind_authoritative(
        self,
        memory,
        semantic_rows: np.ndarray,
        *,
        semantic_fn: Callable[[Any], np.ndarray],
    ) -> Callable[[int, Any], None] | None:
        """Keep mmap synchronized without changing Algorithm-R semantics.

        The existing observer (installed by semantic precompute) is preserved and
        runs first.  That lets the authoritative in-memory semantic row update
        before the mmap row is copied.
        """
        if len(memory.items) != self.count or int(memory.capacity) != self.count:
            raise RuntimeError("semantic packed/authoritative shape drift")
        previous = getattr(memory, "_write_observer", None)

        def observer(index: int, sample) -> None:
            if previous is not None:
                previous(int(index), sample)
                row = semantic_rows[int(index)]
            else:
                row = semantic_fn(sample)
                semantic_rows[int(index)] = row
            self.update(int(index), sample, row)

        memory.set_write_observer(observer)
        return previous

    def batch_from_positions(self, positions: Iterable[int]):
        pos = np.asarray(list(positions), dtype=np.int64)
        if pos.ndim != 1 or pos.size == 0:
            raise ValueError("empty semantic packed batch")
        if np.any(pos < 0) or np.any(pos >= self.train_count):
            raise IndexError("semantic train position out of range")
        idx = np.asarray(self.train_indices[pos], dtype=np.int64)
        raw = np.asarray(self.observations[idx], dtype=np.uint8)
        if np.any(raw[:, 93] > 32):
            raise ValueError("bad history length in semantic packed reservoir")
        numeric = raw[:, 15:79].copy().view("<f4").reshape(-1, 16)

        def tensor(values, dtype):
            return torch.from_numpy(
                np.array(values, dtype=dtype, copy=True, order="C")
            )

        batch = {
            "cards": tensor(raw[:, 8:15], np.int64),
            "numeric": tensor(numeric, np.float32),
            "categorical": tensor(raw[:, 79:87], np.int64),
            "legal": tensor(self.legal[idx], np.bool_),
            "history_len": tensor(raw[:, 93], np.int64),
            "history": tensor(raw[:, 94:126], np.int64),
            "semantic": tensor(self.semantic[idx], np.float32),
        }
        target = tensor(self.targets[idx], np.float32)
        weights = tensor(self.weights[idx], np.float32)
        return batch, target, weights

    def flush(self) -> None:
        for mm in (
            self.observations,
            self.legal,
            self.targets,
            self.weights,
            self.semantic,
            self.train_indices,
        ):
            mm.flush()

    def close(self) -> None:
        self.flush()
        del (
            self.observations,
            self.legal,
            self.targets,
            self.weights,
            self.semantic,
            self.train_indices,
        )


def make_fit_contract(
    member_meta,
    *,
    learning_rate: float,
    batch_size: int,
    member_steps: int,
) -> dict[str, Any]:
    rows = [
        {
            "member": int(i),
            "init_seed": int(row["init_seed"]),
            "batch_seed": int(row["batch_seed"]),
        }
        for i, row in enumerate(member_meta)
    ]
    if len(rows) != 8:
        raise RuntimeError("semantic ensemble requires eight members")
    if int(member_steps) <= 0 or int(batch_size) <= 0:
        raise ValueError("positive semantic member_steps/batch_size required")
    return {
        "ensemble_size": 8,
        "member_steps": int(member_steps),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "member_meta": rows,
    }


def _fit_member_with_batch_source(
    *,
    member: int,
    contract: dict[str, Any],
    batch_source,
) -> dict[str, Any]:
    member = int(member)
    rows = list(contract.get("member_meta") or [])
    if int(contract.get("ensemble_size", -1)) != 8 or len(rows) != 8:
        raise RuntimeError("semantic ensemble contract drift")
    if not 0 <= member < 8:
        raise ValueError("semantic ensemble member out of range")
    meta = rows[member]
    if int(meta.get("member", -1)) != member:
        raise RuntimeError("semantic member metadata ordering drift")
    init_seed = int(meta["init_seed"])
    batch_seed = int(meta["batch_seed"])
    steps = int(contract["member_steps"])
    lr = float(contract["learning_rate"])
    batch_size = min(int(contract["batch_size"]), int(batch_source.train_count))

    model = shadow.make_semantic_model(init_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(batch_seed)
    population = range(int(batch_source.train_count))
    losses = []
    started = time.perf_counter()
    for _ in range(steps):
        positions = rng.sample(population, batch_size)
        batch, target, weights = batch_source.batch_from_positions(positions)
        losses.append(
            train_step(model, optimizer, batch, target, weights, "advantage")
        )
    elapsed = float(time.perf_counter() - started)
    model.eval()
    return {
        "member": member,
        "init_seed": init_seed,
        "batch_seed": batch_seed,
        "steps": steps,
        "fit_seconds": elapsed,
        "losses": [float(x) for x in losses],
        "loss_last": float(losses[-1]),
        "state": _clone_state(model),
        "maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "pid": int(os.getpid()),
    }


def _worker_init(manifest_path: str, threads: int) -> None:
    global _PACKED
    threads = int(threads)
    if threads <= 0:
        raise ValueError("positive threads_per_member required")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _PACKED = PackedSemanticAdvantageReservoir(Path(manifest_path), mode="r")


def _worker_ping(_x) -> dict[str, int]:
    if _PACKED is None:
        raise RuntimeError("semantic packed worker not initialized")
    time.sleep(0.05)
    return {
        "pid": int(os.getpid()),
        "count": int(_PACKED.count),
        "train_count": int(_PACKED.train_count),
        "maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _worker_fit(task: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _PACKED is None:
        raise RuntimeError("semantic packed worker not initialized")
    member, contract = task
    return _fit_member_with_batch_source(
        member=int(member),
        contract=contract,
        batch_source=_PACKED,
    )


class ParallelSemanticEnsembleFitter:
    """Persistent process pool for independent semantic ensemble members."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        contract: dict[str, Any],
        concurrency: int,
        threads_per_member: int,
    ):
        if int(concurrency) <= 0 or int(threads_per_member) <= 0:
            raise ValueError("positive concurrency/threads required")
        if int(concurrency) * int(threads_per_member) > 32:
            raise ValueError("semantic fitter refuses >32 declared worker threads")
        self.manifest_path = Path(manifest_path).resolve()
        self.contract = copy.deepcopy(contract)
        self.concurrency = int(concurrency)
        self.threads_per_member = int(threads_per_member)

        ctx = mp.get_context("spawn")
        names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        old = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ[name] = str(self.threads_per_member)
        started = time.perf_counter()
        try:
            self.pool = ProcessPoolExecutor(
                max_workers=self.concurrency,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(str(self.manifest_path), self.threads_per_member),
            )
            self.worker_pings = list(
                self.pool.map(_worker_ping, range(self.concurrency))
            )
        finally:
            for name, value in old.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.startup_seconds = float(time.perf_counter() - started)

    def fit(self, members: Iterable[int] | None = None) -> list[dict[str, Any]]:
        selected = list(range(8) if members is None else [int(x) for x in members])
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("semantic fit member selection invalid")
        if any(member < 0 or member >= 8 for member in selected):
            raise ValueError("semantic fit member out of range")
        started = time.perf_counter()
        tasks = [(member, self.contract) for member in selected]
        rows = list(self.pool.map(_worker_fit, tasks, chunksize=1))
        rows.sort(key=lambda row: int(row["member"]))
        wall = float(time.perf_counter() - started)
        for row in rows:
            row["parallel_wall_seconds"] = wall
            row["concurrency"] = self.concurrency
            row["threads_per_member"] = self.threads_per_member
        return rows

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=False)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class _InMemoryBatchSource:
    """Canonical sequential batch source used by parity/benchmark tooling."""

    def __init__(self, memory, semantic_rows, train_pool):
        self.memory = memory
        self.semantic_rows = np.asarray(semantic_rows, dtype=np.float32)
        self.train_pool = list(int(x) for x in train_pool)
        self.train_count = len(self.train_pool)

    def batch_from_positions(self, positions):
        from spincore_nn.lean_batch import vectorized_batch

        pos = list(int(x) for x in positions)
        idx = [self.train_pool[p] for p in pos]
        samples = [self.memory.items[i] for i in idx]
        batch, target, weights = vectorized_batch(samples, "cpu")
        batch["semantic"] = torch.tensor(
            np.asarray(self.semantic_rows[idx], dtype=np.float32),
            dtype=torch.float32,
        )
        return batch, target, weights


def fit_members_sequential_reference(
    memory,
    semantic_rows,
    train_pool,
    contract: dict[str, Any],
    *,
    members: Iterable[int] | None = None,
    threads: int = 8,
) -> list[dict[str, Any]]:
    """Reference fitter matching the canonical per-member training loop."""
    torch.set_num_threads(int(threads))
    source = _InMemoryBatchSource(memory, semantic_rows, train_pool)
    selected = list(range(8) if members is None else [int(x) for x in members])
    rows = [
        _fit_member_with_batch_source(
            member=member,
            contract=contract,
            batch_source=source,
        )
        for member in selected
    ]
    rows.sort(key=lambda row: int(row["member"]))
    return rows


def assert_fit_rows_exact(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> None:
    if [int(x["member"]) for x in reference] != [
        int(x["member"]) for x in candidate
    ]:
        raise AssertionError("semantic fit member ordering mismatch")
    for left, right in zip(reference, candidate):
        for key in ("init_seed", "batch_seed", "steps"):
            if int(left[key]) != int(right[key]):
                raise AssertionError(
                    f"semantic member {left['member']} metadata mismatch: {key}"
                )
        if len(left["losses"]) != len(right["losses"]):
            raise AssertionError("semantic loss-vector length mismatch")
        for step, (a, b) in enumerate(zip(left["losses"], right["losses"])):
            if float(a) != float(b):
                raise AssertionError(
                    f"semantic member {left['member']} loss mismatch at step {step}: {a} != {b}"
                )
        if not tensor_states_equal(left["state"], right["state"]):
            raise AssertionError(
                f"semantic member {left['member']} final tensor state mismatch"
            )


if __name__ == "__main__":
    raise SystemExit(
        "This module is imported by guarded benchmark/resume runners; "
        "do not run it directly."
    )
