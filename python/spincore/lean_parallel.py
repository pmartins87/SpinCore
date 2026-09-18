from __future__ import annotations

"""Ryzen-oriented parallel root collection for the lean SpinCore trainer.

Only independent Deep-CFR advantage roots are parallelized. The mathematical
collector, action resolver, terminal utility, representation and reservoir
semantics are unchanged by default. Workers use one Torch thread each; the
parent merges samples in deterministic root order into the authoritative
reservoir.

An opt-in HU-preflop board-averaging experiment is also supported. When
`hu_preflop_board_average_k > 1`, only TRUE_HEADS_UP roots are affected:
the original root deal supplies the fixed hole cards and canonical board, K-1
additional future boards are drawn conditional on those same hole cards, the
same external-sampling RNG stream is replayed for every board, and only
preflop Advantage targets are averaged. Postflop samples are retained from the
canonical original board only. Sample count/order therefore stay aligned with
one ordinary root while target-generation node cost increases.
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


def _sample_street(sample: ActionAdvantageSample) -> int:
    from spincore_nn.codec import decode_spnniv1

    return int(decode_spnniv1(sample.observation).categorical[1])


def _board_seed(deck_seed: int, board_index: int) -> int:
    return (
        int(deck_seed)
        ^ 0xB04D4A5E7C15
        ^ (int(board_index) * 0x9E3779B97F4A7C15)
    ) & ((1 << 63) - 1)


def _draw_board(holes, *, seed: int) -> tuple[int, int, int, int, int]:
    used = {
        int(card)
        for row in holes
        for card in row
        if int(card) >= 0
    }
    remaining = [card for card in range(52) if card not in used]
    if len(remaining) < 5:
        raise RuntimeError("not enough cards for HU board resampling")
    board = tuple(int(x) for x in random.Random(int(seed)).sample(remaining, 5))
    return board  # type: ignore[return-value]


def _average_preflop_targets(
    per_board: list[tuple[ActionAdvantageSample, ...]],
) -> tuple[ActionAdvantageSample, ...]:
    if not per_board:
        raise ValueError("board-averaging requires at least one sample stream")
    base = list(per_board[0])
    preflop = [
        [sample for sample in stream if _sample_street(sample) == 0]
        for stream in per_board
    ]
    expected = len(preflop[0])
    if any(len(stream) != expected for stream in preflop):
        raise RuntimeError("HU board averaging changed preflop sample count across boards")

    replacements: list[ActionAdvantageSample] = []
    for pos in range(expected):
        anchor = preflop[0][pos]
        for stream in preflop[1:]:
            other = stream[pos]
            if (
                other.observation != anchor.observation
                or tuple(other.legal) != tuple(anchor.legal)
                or float(other.weight) != float(anchor.weight)
                or int(other.iteration) != int(anchor.iteration)
            ):
                raise RuntimeError(
                    "HU board averaging changed preflop information-state identity"
                )
        target = tuple(
            sum(float(stream[pos].target[action]) for stream in preflop)
            / float(len(preflop))
            for action in range(len(anchor.target))
        )
        replacements.append(
            ActionAdvantageSample(
                observation=anchor.observation,
                legal=tuple(anchor.legal),
                target=target,
                weight=float(anchor.weight),
                iteration=int(anchor.iteration),
            )
        )

    # Preserve the exact canonical-board sample order/count. Only the target of
    # preflop samples is replaced; postflop samples stay bit-for-bit from board 0.
    out: list[ActionAdvantageSample] = []
    pre_index = 0
    for sample in base:
        if _sample_street(sample) == 0:
            out.append(replacements[pre_index])
            pre_index += 1
        else:
            out.append(sample)
    if pre_index != expected:
        raise RuntimeError("HU board averaging reconstruction drift")
    return tuple(out)


def _collect_hu_board_averaged_root(
    *,
    solver,
    collector,
    job: RootJob,
    iteration: int,
    exact_opponent_levels: int,
    board_average_k: int,
) -> tuple[int, tuple[ActionAdvantageSample, ...]]:
    if int(board_average_k) <= 1:
        raise ValueError("board_average_k must be > 1 in averaged-root path")
    if int(exact_opponent_levels) != 0:
        raise ValueError("HU preflop board averaging is admitted only with exact_opponent_levels=0")
    if not job.episode.game_is_hu:
        raise ValueError("HU preflop board averaging received non-HU episode")
    if not solver.explicit_deal_available:
        raise RuntimeError("HU board averaging requires explicit-deal solver API")

    canonical = solver.create(job.episode, int(job.deck_seed))
    try:
        snapshot = canonical.deal_snapshot()
    finally:
        canonical.close()
    if int(snapshot.visible_board_count) != 0:
        raise RuntimeError("root deal snapshot unexpectedly has visible board cards")

    boards = [tuple(int(x) for x in snapshot.board)]
    for board_index in range(1, int(board_average_k)):
        boards.append(
            _draw_board(
                snapshot.holes,
                seed=_board_seed(int(job.deck_seed), int(board_index)),
            )
        )

    live = [index for index, stack in enumerate(job.episode.stacks) if stack > 0]
    total_nodes = 0
    root_samples: list[ActionAdvantageSample] = []

    for player in live:
        rng_before = collector.rng.getstate()
        rng_after_canonical = None
        streams: list[tuple[ActionAdvantageSample, ...]] = []

        for board_index, board in enumerate(boards):
            # Replay the same external-sampling stream for every board. This
            # holds preflop opponent-action sampling fixed while isolating future
            # board chance. After all variants, restore the RNG progression from
            # the canonical first board so later traversers/roots consume the
            # same stochastic stream they would have consumed at K=1.
            collector.rng.setstate(rng_before)
            temp = _ListSink()
            old_memory = collector.advantage_memory
            collector.advantage_memory = temp
            root = solver.create_with_deal(job.episode, snapshot.holes, board)
            try:
                result = collector.collect_advantage_partial_exact(
                    root,
                    traverser=int(player),
                    iteration=int(iteration),
                    exact_opponent_levels=0,
                )
            finally:
                root.close()
                collector.advantage_memory = old_memory

            if int(result.samples_added) != len(temp.items):
                raise RuntimeError("HU board averaging sample-accounting drift")
            if board_index == 0:
                rng_after_canonical = collector.rng.getstate()
            streams.append(tuple(temp.items))
            total_nodes += int(result.nodes)

        if rng_after_canonical is None:
            raise RuntimeError("HU board averaging canonical RNG state missing")
        collector.rng.setstate(rng_after_canonical)
        root_samples.extend(_average_preflop_targets(streams))

    return int(total_nodes), tuple(root_samples)


def _collect_chunk(
    domain: str,
    bundle_seed: int,
    model_state: dict,
    model_ready: bool,
    iteration: int,
    exact_opponent_levels: int,
    hu_preflop_board_average_k: int,
    jobs: tuple[RootJob, ...],
) -> tuple[RootResult, ...]:
    if _WORKER_SOLVER_PATH is None:
        raise RuntimeError("parallel worker was not initialized")
    if int(hu_preflop_board_average_k) <= 0:
        raise ValueError("hu_preflop_board_average_k must be positive")

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
        use_board_averaging = (
            str(domain) == "TRUE_HEADS_UP"
            and bool(job.episode.game_is_hu)
            and int(hu_preflop_board_average_k) > 1
        )

        if use_board_averaging:
            nodes, averaged = _collect_hu_board_averaged_root(
                solver=solver,
                collector=collector,
                job=job,
                iteration=int(iteration),
                exact_opponent_levels=int(exact_opponent_levels),
                board_average_k=int(hu_preflop_board_average_k),
            )
            for sample in averaged:
                sink.add(sample)
        else:
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
        hu_preflop_board_average_k: int = 1,
    ) -> dict[str, float | int]:
        values = list(jobs)
        if not values:
            return {"roots": 0, "nodes": 0, "samples": 0, "seconds": 0.0}
        if int(hu_preflop_board_average_k) <= 0:
            raise ValueError("hu_preflop_board_average_k must be positive")
        if int(hu_preflop_board_average_k) > 1 and int(exact_opponent_levels) != 0:
            raise ValueError(
                "HU preflop board averaging requires exact_opponent_levels=0"
            )

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
                int(hu_preflop_board_average_k),
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
            "hu_preflop_board_average_k": int(hu_preflop_board_average_k),
        }


def recommended_ryzen_workers(logical_cpus: int | None = None) -> int:
    logical = int(logical_cpus or os.cpu_count() or 1)
    if logical <= 2:
        return 1
    # Same policy that proved effective in DeepPot on the user's 32-thread
    # Ryzen: leave one logical CPU for the parent/OS.
    return max(1, logical - 1)
