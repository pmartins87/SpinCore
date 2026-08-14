from __future__ import annotations

import random

from spincore.deep_cfr import (
    ExternalSamplingCollector,
    regret_matching_policy,
    uniform_policy,
    _validate_policy,
)


class Memory:
    def __init__(self):
        self.items = []

    def add(self, sample):
        self.items.append(sample)


class ToyState:
    def __init__(self, *, actor=None, children=None, utility=(0.0, 0.0, 0.0), label=""):
        self._actor = actor
        self._children = dict(children or {})
        self.utility = tuple(float(x) for x in utility)
        self.label = label
        self.closed = False

    @property
    def terminal(self):
        return self._actor is None

    @property
    def actor(self):
        if self.terminal:
            raise RuntimeError("terminal has no actor")
        return int(self._actor)

    def legal_actions(self):
        return tuple(sorted(self._children))

    def neural_bytes(self):
        return self.label.encode("ascii") or b"toy"

    def child(self, action):
        return self._children[int(action)]

    def close(self):
        self.closed = True


def terminal_utility(state):
    return state.utility


def test_regret_matching_mechanics():
    p = regret_matching_policy((-4.0, 2.0, 6.0, 0.0, 0.0, 0.0), (0, 1, 2))
    assert p == (0.0, 0.25, 0.75, 0.0, 0.0, 0.0), p
    q = regret_matching_policy((-1.0, 0.0, -3.0, 0.0, 0.0, 0.0), (0, 1, 2))
    assert q == (1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0), q


def test_policy_validation_fail_closed():
    legal = (0, 1)
    assert _validate_policy((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), legal) == (
        0.5, 0.5, 0.0, 0.0, 0.0, 0.0
    )
    try:
        _validate_policy((0.5, 0.4, 0.1, 0.0, 0.0, 0.0), legal)
    except ValueError as exc:
        assert "illegal action has mass" in str(exc)
    else:
        raise AssertionError("illegal probability mass was silently accepted")


def test_traverser_advantage_exact_tree():
    # Uniform root policy over two actions, utilities +1 and -1 to traverser 0.
    # Therefore node value = 0 and instantaneous advantages are +1 and -1.
    plus = ToyState(utility=(1.0, -1.0, 0.0), label="plus")
    minus = ToyState(utility=(-1.0, 1.0, 0.0), label="minus")
    root = ToyState(actor=0, children={0: plus, 1: minus}, label="root")
    adv = Memory()
    strat = Memory()
    collector = ExternalSamplingCollector(
        policy=uniform_policy,
        terminal_utility=terminal_utility,
        rng=random.Random(7),
        advantage_memory=adv,
        strategy_memory=strat,
    )
    result = collector.collect_advantage(root, traverser=0, iteration=3)
    assert result.utility == 0.0, result
    assert result.nodes == 3, result
    assert result.samples_added == 1, result
    assert len(adv.items) == 1
    sample = adv.items[0]
    assert sample.legal == (1, 1, 0, 0, 0, 0), sample
    assert sample.target == (1.0, -1.0, 0.0, 0.0, 0.0, 0.0), sample.target
    assert sample.weight == 3.0 and sample.iteration == 3


def test_opponent_external_sampling_uses_one_branch():
    # Traverser 0 sits below opponent 1. With deterministic policy [1,0], only
    # action 0 may be entered; action 1 would produce the opposite utility.
    good = ToyState(actor=0, children={0: ToyState(utility=(2.0, -2.0, 0.0))}, label="good")
    bad = ToyState(actor=0, children={0: ToyState(utility=(-9.0, 9.0, 0.0))}, label="bad")
    root = ToyState(actor=1, children={0: good, 1: bad}, label="opp-root")

    def policy(_state, _obs, legal):
        out = [0.0] * 6
        out[legal[0]] = 1.0
        return tuple(out)

    adv = Memory()
    collector = ExternalSamplingCollector(
        policy=policy,
        terminal_utility=terminal_utility,
        rng=random.Random(1),
        advantage_memory=adv,
        strategy_memory=Memory(),
    )
    result = collector.collect_advantage(root, traverser=0, iteration=1)
    assert result.utility == 2.0, result
    # root + traverser node + one terminal; bad branch must not be traversed.
    assert result.nodes == 3, result
    assert len(adv.items) == 1


def test_strategy_collection_own_reach_shape():
    # At target-player node one action is sampled after recording the strategy.
    target_root = ToyState(
        actor=0,
        children={
            0: ToyState(utility=(0.0, 0.0, 0.0)),
            1: ToyState(utility=(0.0, 0.0, 0.0)),
        },
        label="target",
    )
    memory = Memory()
    collector = ExternalSamplingCollector(
        policy=uniform_policy,
        terminal_utility=terminal_utility,
        rng=random.Random(4),
        advantage_memory=Memory(),
        strategy_memory=memory,
    )
    added = collector.collect_strategy_own_reach(target_root, target_player=0, iteration=5)
    assert added == 1
    assert len(memory.items) == 1
    assert memory.items[0].weight == 5.0
    assert memory.items[0].target == (0.5, 0.5, 0.0, 0.0, 0.0, 0.0)

    # At a non-target node, enumerate all actions; each reaches one target node.
    left = ToyState(actor=0, children={0: ToyState(utility=(0.0, 0.0, 0.0))}, label="left")
    right = ToyState(actor=0, children={0: ToyState(utility=(0.0, 0.0, 0.0))}, label="right")
    opponent_root = ToyState(actor=1, children={0: left, 1: right}, label="opp")
    memory2 = Memory()
    collector2 = ExternalSamplingCollector(
        policy=uniform_policy,
        terminal_utility=terminal_utility,
        rng=random.Random(4),
        advantage_memory=Memory(),
        strategy_memory=memory2,
    )
    added2 = collector2.collect_strategy_own_reach(opponent_root, target_player=0, iteration=2)
    assert added2 == 2
    assert len(memory2.items) == 2


def main() -> int:
    tests = [
        test_regret_matching_mechanics,
        test_policy_validation_fail_closed,
        test_traverser_advantage_exact_tree,
        test_opponent_external_sampling_uses_one_branch,
        test_strategy_collection_own_reach_shape,
    ]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print("R7.5.3C analytic Deep CFR toy checks PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
