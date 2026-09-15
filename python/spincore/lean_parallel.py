from __future__ import annotations

"""Ryzen-oriented parallel root collection for the lean SpinCore trainer.

Only independent Deep-CFR advantage roots are parallelized. The mathematical
collector, action resolver, terminal utility, representation and reservoir
semantics are unchanged. Workers use one Torch thread each; the parent merges
samples in deterministic root order into the authoritative reservoir.

This mirrors the proven DeepPot pattern on the same 32-thread Ryzen: many
independent CPU workers, one thread per worker, leaving one logical CPU for the
parent/OS.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import random
import time
from typing import Iterable

from .r7_5_action_cfr import ActionAdvantageSample
from .solver import Episode

_WORKER_SOLVER_PATH: str | None = None


class _ListSink:
    def __init__(self) -> None:
        self.items: list[ActionAdvantageSample] = []
        self.seen = 0

    def add(self, item: ActionAdvantageSample) -> None:
        self.items.append(item)
        self.seen += 1


@dataclass(frozen=True)
class RootJob:
    root_index: int
    episode: Episode
    deck_seed: int
    policy_seed: int


@dataclass(frozen=True)
class RootResult:
    root_index: int
    nodes: int
    samples: tuple[ActionAdvantageSample, ...]


def _worker_init(solver_path: str) -> None:
    global _WORKER_SOLVER_PATH
    _WORKER_SOLVER_PATH = str(solver_path)
    # Root traversal is latency-bound on many tiny network forwards. One thread
    # per process avoids BLAS/OpenMP oversubscription when 20-31 roots run in
    # parallel on a 32-thread Ryzen.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _collect_chunk(
    domain: str,
    bundle_seed: int,
    model_state: dict,
    model_ready: bool,
    iteration: int,
    exact_opponent_levels: int,
    jobs: tuple[RootJob, ...],
) -> tuple[RootResult, ...]:
    if _WORKER_SOLVER_PATH is None:
        raise RuntimeError("parallel worker was not initialized")

    from spincore.lean_action_policy import LeanNeuralActionAdvantagePolicy
    from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
    from spincore.lean_solver_actions import LeanLegacyActionCollector
    from spincore.lean_training_scope import LeanTrainingScope
    from spincore.solver import SolverLibrary
    from spincore_nn.action_models import make_advantage_action_model

    solver = SolverLibrary(Path(_WORKER_SOLVER_PATH))
    _, model = make_advantage_action_model(
        "C0_V1_FROZEN_CONTROL",
        device="cpu",
        seed=int(bundle_seed) & 0x7FFFFFFF,
    )
    model.load_state_dict(model_state)
    model.eval()
    behavior = LeanNeuralActionAdvantagePolicy(
        model,
        selected_representation="C0_V1_FROZEN_CONTROL",
        device="cpu",
        ready=bool(model_ready),
    )
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
    out: list[tuple[RootJob, ...]] = []
    start = 0
    for index in range(chunks):
        size = base + (1 if index < extra else 0)
        out.append(tuple(values[start : start + size]))
        start += size
    return out


class ParallelRootExecutor:
    """Persistent process pool for one whole training run."""

    def __init__(self, solver_path: str | Path, workers: int):
        workers = int(workers)
        if workers <= 1:
            raise ValueError("ParallelRootExecutor requires workers > 1")
        self.workers = workers
        self.solver_path = str(Path(solver_path).resolve())
        # spawn is slower to start than fork but avoids inheriting live Torch
        # thread pools / ctypes solver state. The pool persists across all
        # iterations, so startup cost is paid once.
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
    ) -> dict[str, float | int]:
        values = list(jobs)
        if not values:
            return {"roots": 0, "nodes": 0, "samples": 0, "seconds": 0.0}

        # Freeze a small CPU copy of the current fitted advantage model. The
        # parent does not mutate it until every worker finishes this iteration.
        model_state = {
            key: value.detach().cpu().clone()
            for key, value in bundle.advantage.state_dict().items()
        }
        ready = bool(bundle.counters.get("advantage_ready", 0))
        chunks = _chunked(values, self.workers)
        started = time.perf_counter()
        futures = [
            self.pool.submit(
                _collect_chunk,
                str(domain),
                int(bundle.seed),
                model_state,
                ready,
                int(iteration),
                int(exact_opponent_levels),
                chunk,
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
        }


def recommended_ryzen_workers(logical_cpus: int | None = None) -> int:
    logical = int(logical_cpus or os.cpu_count() or 1)
    if logical <= 2:
        return 1
    # Same policy that proved effective in DeepPot on the user's 32-thread
    # Ryzen: leave one logical CPU for the parent/OS.
    return max(1, logical - 1)
