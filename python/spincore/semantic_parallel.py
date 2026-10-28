from __future__ import annotations

"""Persistent process-parallel root collection for V1+semantic ENS8 teachers.

This is intentionally separate from lean_parallel.py so the frozen V1/HU
parallel path remains byte-identical.  It supports ordinary partial-exact roots
only (no experimental HU board averaging).
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from pathlib import Path
import random
import time
from typing import Iterable

import numpy as np
import torch

from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_parallel import RootJob, RootResult
from spincore.lean_solver_actions import LeanLegacyActionCollector
from spincore.lean_training_scope import LeanTrainingScope
from spincore.r7_5_action_cfr import ActionAdvantageSample
from spincore.solver import SolverLibrary
from spincore_nn.action_models import collate_action_observations
from spincore_nn.semantic_v1 import (
    make_semantic_advantage_model,
    semantic_vector_from_observation,
)

_WORKER_SOLVER_PATH: str | None = None


class _ListSink:
    def __init__(self) -> None:
        self.items: list[ActionAdvantageSample] = []
        self.seen = 0

    def add(self, item: ActionAdvantageSample) -> None:
        self.items.append(item)
        self.seen += 1


class SemanticEnsemblePolicy:
    def __init__(self, models):
        self.models = list(models)
        if len(self.models) != 8:
            raise ValueError("semantic root behavior requires eight ensemble members")

    def __call__(self, _state, observation: bytes, legal: tuple[int, ...]):
        legal = tuple(int(x) for x in legal)
        mask = [False] * 10
        for action in legal:
            mask[action] = True
        batch = collate_action_observations(
            "C0_V1_FROZEN_CONTROL",
            [observation],
            [tuple(mask)],
            device="cpu",
        )
        batch["semantic"] = torch.tensor(
            np.asarray([semantic_vector_from_observation(observation)], dtype=np.float32),
            dtype=torch.float32,
        )
        with torch.no_grad():
            raw = (
                torch.stack([model(batch)[0] for model in self.models], dim=0)
                .mean(dim=0)
                .cpu()
                .tolist()
            )
        return lean_regret_matching_policy(raw, legal)


def _worker_init(solver_path: str) -> None:
    global _WORKER_SOLVER_PATH
    _WORKER_SOLVER_PATH = str(solver_path)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _build_models(states):
    models = []
    for index, state in enumerate(states):
        model = make_semantic_advantage_model(index)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models


def _collect_chunk(
    domain: str,
    iteration: int,
    exact_opponent_levels: int,
    jobs: tuple[RootJob, ...],
    ensemble_model_states: tuple[dict, ...],
) -> tuple[RootResult, ...]:
    if _WORKER_SOLVER_PATH is None:
        raise RuntimeError("semantic parallel worker was not initialized")
    if len(ensemble_model_states) != 8:
        raise RuntimeError("semantic parallel root ensemble-size drift")

    solver = SolverLibrary(Path(_WORKER_SOLVER_PATH))
    models = _build_models(ensemble_model_states)
    behavior = SemanticEnsemblePolicy(models)
    sink = _ListSink()
    dummy = _ListSink()
    collector = LeanLegacyActionCollector(
        action_spec=FIRST_RELEASE_ACTION_SPEC,
        selected_representation="C0_V1_FROZEN_CONTROL",
        policy=behavior,
        terminal_utility=LeanTrainingScope().terminal_utility,
        rng=random.Random(0),
        advantage_memory=sink,
        strategy_memory=dummy,
    )

    out: list[RootResult] = []
    for job in jobs:
        before = len(sink.items)
        collector.rng = random.Random(int(job.policy_seed))
        nodes = 0
        live = [index for index, stack in enumerate(job.episode.stacks) if stack > 0]
        for player in live:
            root = solver.create(job.episode, int(job.deck_seed))
            try:
                result = collector.collect_advantage_partial_exact(
                    root,
                    traverser=int(player),
                    iteration=int(iteration),
                    exact_opponent_levels=int(exact_opponent_levels),
                )
            finally:
                root.close()
            nodes += int(result.nodes)
        samples = tuple(sink.items[before:])
        out.append(RootResult(int(job.root_index), int(nodes), samples))
    return tuple(out)


def _chunked(values: list[RootJob], chunks: int) -> list[tuple[RootJob, ...]]:
    if not values:
        return []
    chunks = max(1, min(int(chunks), len(values)))
    base, extra = divmod(len(values), chunks)
    out = []
    start = 0
    for index in range(chunks):
        size = base + (1 if index < extra else 0)
        out.append(tuple(values[start : start + size]))
        start += size
    return out


class SemanticParallelRootExecutor:
    def __init__(self, solver_path: str | Path, workers: int):
        workers = int(workers)
        if workers <= 1:
            raise ValueError("SemanticParallelRootExecutor requires workers > 1")
        self.workers = workers
        self.solver_path = str(Path(solver_path).resolve())
        ctx = mp.get_context("spawn")
        self.pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(self.solver_path,),
        )

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=False)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def collect(
        self,
        *,
        domain: str,
        bundle,
        iteration: int,
        exact_opponent_levels: int,
        jobs: Iterable[RootJob],
        ensemble_model_states: tuple[dict, ...],
    ) -> dict[str, float | int]:
        values = list(jobs)
        if not values:
            return {
                "roots": 0,
                "nodes": 0,
                "samples": 0,
                "seconds": 0.0,
                "ensemble_size": 8,
            }
        if len(ensemble_model_states) != 8:
            raise RuntimeError("semantic parallel collect ensemble-size drift")

        chunks = _chunked(values, self.workers)
        started = time.perf_counter()
        futures = [
            self.pool.submit(
                _collect_chunk,
                str(domain),
                int(iteration),
                int(exact_opponent_levels),
                chunk,
                ensemble_model_states,
            )
            for chunk in chunks
        ]
        results: list[RootResult] = []
        for future in as_completed(futures):
            results.extend(future.result())
        results.sort(key=lambda item: item.root_index)

        nodes = 0
        samples = 0
        for result in results:
            nodes += int(result.nodes)
            for sample in result.samples:
                bundle.adv_mem.add(sample)
                samples += 1

        counters = bundle.counters
        counters["iteration"] = max(int(counters["iteration"]), int(iteration))
        counters["roots"] += len(results)
        counters["nodes"] += nodes
        counters["advantage_samples"] += samples
        return {
            "roots": len(results),
            "nodes": nodes,
            "samples": samples,
            "seconds": float(time.perf_counter() - started),
            "ensemble_size": 8,
        }
