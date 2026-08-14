from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from r7_5_paired_corpus_worker import PairedPartialExactCollector
from run_r7_3_partial_exact_advantage_screen import PartialExactAdvantageCollector
from spincore.deep_cfr import icm_delta_utility, uniform_policy
from spincore.r7_5_paired_corpus import BottomHashCorpus, PairedSample
from spincore.solver import Episode, SolverLibrary

LIB = ROOT / "build" / "libspincore_solver_c.so"
PAYOUT = (0.5, 0.3, 0.2)


class ListMemory:
    def __init__(self) -> None:
        self.items = []

    def add(self, sample) -> None:
        self.items.append(sample)


def episode() -> Episode:
    return Episode(1500, True, 0, 10, 20, (0, 750, 750), 1, (0,))


def _base_tuple(sample):
    return (
        sample.observation,
        tuple(sample.legal),
        tuple(float(x) for x in sample.target),
        float(sample.weight),
        int(sample.iteration),
    )


def _pair_tuple(sample: PairedSample):
    return (
        sample.observation_v1,
        tuple(sample.legal),
        tuple(float(x) for x in sample.target),
        float(sample.weight),
        int(sample.iteration),
    )


def test_paired_advantage_collection_is_exact_r7_4_semantics_plus_v2_bytes() -> None:
    solver = SolverLibrary(LIB)
    terminal = icm_delta_utility(PAYOUT)
    reference_rng = random.Random(99117)
    paired_rng = random.Random(99117)
    reference_adv = ListMemory()
    paired_adv_base = ListMemory()
    dummy_strategy_a = ListMemory()
    dummy_strategy_b = ListMemory()
    paired_adv = BottomHashCorpus[PairedSample](10000)
    paired_strategy = BottomHashCorpus[PairedSample](10000)

    reference = PartialExactAdvantageCollector(
        policy=uniform_policy,
        terminal_utility=terminal,
        rng=reference_rng,
        advantage_memory=reference_adv,
        strategy_memory=dummy_strategy_a,
    )
    paired = PairedPartialExactCollector(
        policy=uniform_policy,
        terminal_utility=terminal,
        rng=paired_rng,
        advantage_memory=paired_adv_base,
        strategy_memory=dummy_strategy_b,
        paired_advantage=paired_adv,
        paired_strategy=paired_strategy,
        domain="TRUE_HEADS_UP",
        corpus_seed=1202035427,
    )

    root_a = solver.create(episode(), 78123)
    try:
        result_a = reference.collect_advantage_partial_exact(
            root_a, traverser=1, iteration=2, exact_opponent_levels=2
        )
    finally:
        root_a.close()
    root_b = solver.create(episode(), 78123)
    try:
        result_b = paired.collect_advantage_partial_exact(
            root_b, traverser=1, iteration=2, exact_opponent_levels=2
        )
    finally:
        root_b.close()

    assert result_a == result_b
    assert reference_rng.getstate() == paired_rng.getstate()
    assert reference_adv.items == paired_adv_base.items
    assert paired_adv.seen == len(reference_adv.items)
    assert sorted(_pair_tuple(sample) for sample in paired_adv.items) == sorted(
        _base_tuple(sample) for sample in reference_adv.items
    )
    assert all(len(sample.observation_v2) == 830 for sample in paired_adv.items)
    assert all(sample.observation_v2.startswith(b"SPNNIV2\x00") for sample in paired_adv.items)


def test_paired_strategy_collection_preserves_reference_rng_and_samples() -> None:
    solver = SolverLibrary(LIB)
    terminal = icm_delta_utility(PAYOUT)
    reference_rng = random.Random(44821)
    paired_rng = random.Random(44821)
    dummy_adv_a = ListMemory()
    dummy_adv_b = ListMemory()
    reference_strategy = ListMemory()
    paired_strategy_base = ListMemory()
    paired_adv = BottomHashCorpus[PairedSample](10000)
    paired_strategy = BottomHashCorpus[PairedSample](10000)

    reference = PartialExactAdvantageCollector(
        policy=uniform_policy,
        terminal_utility=terminal,
        rng=reference_rng,
        advantage_memory=dummy_adv_a,
        strategy_memory=reference_strategy,
    )
    paired = PairedPartialExactCollector(
        policy=uniform_policy,
        terminal_utility=terminal,
        rng=paired_rng,
        advantage_memory=dummy_adv_b,
        strategy_memory=paired_strategy_base,
        paired_advantage=paired_adv,
        paired_strategy=paired_strategy,
        domain="TRUE_HEADS_UP",
        corpus_seed=1202035427,
    )

    root_a = solver.create(episode(), 99231)
    try:
        count_a = reference.collect_strategy_own_reach(root_a, target_player=1, iteration=2)
    finally:
        root_a.close()
    root_b = solver.create(episode(), 99231)
    try:
        count_b = paired.collect_strategy_own_reach(root_b, target_player=1, iteration=2)
    finally:
        root_b.close()

    assert count_a == count_b
    assert reference_rng.getstate() == paired_rng.getstate()
    assert reference_strategy.items == paired_strategy_base.items
    assert paired_strategy.seen == len(reference_strategy.items)
    assert sorted(_pair_tuple(sample) for sample in paired_strategy.items) == sorted(
        _base_tuple(sample) for sample in reference_strategy.items
    )
    assert all(len(sample.observation_v2) == 830 for sample in paired_strategy.items)
