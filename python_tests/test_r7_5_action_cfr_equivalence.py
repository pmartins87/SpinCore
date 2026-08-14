from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_r7_3_partial_exact_advantage_screen import PartialExactAdvantageCollector
from spincore.deep_cfr import icm_delta_utility, uniform_policy as old_uniform_policy
from spincore.r7_5_action_cfr import UniversalPartialExactCollector, uniform_policy as new_uniform_policy
from spincore.r7_5_action_contract import postflop_candidate_specs
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


def test_pf0_partial_exact_advantage_is_same_exact_control_tree_and_rng() -> None:
    solver = SolverLibrary(LIB)
    terminal = icm_delta_utility(PAYOUT)
    old_rng = random.Random(90311)
    new_rng = random.Random(90311)
    old_adv = ListMemory()
    new_adv = ListMemory()
    old_strategy = ListMemory()
    new_strategy = ListMemory()

    old = PartialExactAdvantageCollector(
        policy=old_uniform_policy,
        terminal_utility=terminal,
        rng=old_rng,
        advantage_memory=old_adv,
        strategy_memory=old_strategy,
    )
    pf0 = postflop_candidate_specs(ROOT)["PF0_CONTROL_33_75_AI"]
    new = UniversalPartialExactCollector(
        action_spec=pf0,
        selected_representation="C0_V1_FROZEN_CONTROL",
        policy=new_uniform_policy,
        terminal_utility=terminal,
        rng=new_rng,
        advantage_memory=new_adv,
        strategy_memory=new_strategy,
    )

    root_old = solver.create(episode(), 551122)
    try:
        result_old = old.collect_advantage_partial_exact(
            root_old, traverser=1, iteration=2, exact_opponent_levels=2
        )
    finally:
        root_old.close()
    root_new = solver.create(episode(), 551122)
    try:
        result_new = new.collect_advantage_partial_exact(
            root_new, traverser=1, iteration=2, exact_opponent_levels=2
        )
    finally:
        root_new.close()

    assert result_new.utility == result_old.utility
    assert result_new.nodes == result_old.nodes
    assert result_new.samples_added == result_old.samples_added
    assert len(new_adv.items) == len(old_adv.items)
    assert new_rng.getstate() == old_rng.getstate()
    # The selected representation is V1, so state features are exactly the
    # same bytes; only legal/target vectors are widened from six to ten slots.
    assert [sample.observation for sample in new_adv.items] == [
        sample.observation for sample in old_adv.items
    ]
    assert [sample.weight for sample in new_adv.items] == [sample.weight for sample in old_adv.items]
    assert [sample.iteration for sample in new_adv.items] == [sample.iteration for sample in old_adv.items]


def test_pf0_strategy_collection_is_same_exact_control_tree_and_rng() -> None:
    solver = SolverLibrary(LIB)
    terminal = icm_delta_utility(PAYOUT)
    old_rng = random.Random(177013)
    new_rng = random.Random(177013)
    old_adv = ListMemory()
    new_adv = ListMemory()
    old_strategy = ListMemory()
    new_strategy = ListMemory()

    old = PartialExactAdvantageCollector(
        policy=old_uniform_policy,
        terminal_utility=terminal,
        rng=old_rng,
        advantage_memory=old_adv,
        strategy_memory=old_strategy,
    )
    pf0 = postflop_candidate_specs(ROOT)["PF0_CONTROL_33_75_AI"]
    new = UniversalPartialExactCollector(
        action_spec=pf0,
        selected_representation="C0_V1_FROZEN_CONTROL",
        policy=new_uniform_policy,
        terminal_utility=terminal,
        rng=new_rng,
        advantage_memory=new_adv,
        strategy_memory=new_strategy,
    )

    root_old = solver.create(episode(), 771199)
    try:
        count_old = old.collect_strategy_own_reach(root_old, target_player=1, iteration=2)
    finally:
        root_old.close()
    root_new = solver.create(episode(), 771199)
    try:
        count_new = new.collect_strategy_own_reach(root_new, target_player=1, iteration=2)
    finally:
        root_new.close()

    assert count_new == count_old
    assert len(new_strategy.items) == len(old_strategy.items)
    assert new_rng.getstate() == old_rng.getstate()
    assert [sample.observation for sample in new_strategy.items] == [
        sample.observation for sample in old_strategy.items
    ]
    assert [sample.weight for sample in new_strategy.items] == [
        sample.weight for sample in old_strategy.items
    ]
