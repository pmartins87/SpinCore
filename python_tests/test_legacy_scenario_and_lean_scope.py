from __future__ import annotations

from spincore.legacy_scenario import BLIND_LEVELS, LegacyScenarioSampler
from spincore.lean_training_scope import (
    FIRST_RELEASE_POLICY_FAMILY,
    FIRST_RELEASE_TOTAL_CHIPS,
    LeanTrainingScope,
    constant_scaled_chip_delta_utility,
)


def test_legacy_scenario_sampler_covers_real_blind_ladder_and_domains():
    sampler = LegacyScenarioSampler(seed=20260910)
    seen_hu = set()
    seen_3h = set()

    for _ in range(256):
        hu = sampler.sample_episode(force_domain="TRUE_HEADS_UP")
        three = sampler.sample_episode(force_domain="THREE_HANDED")

        assert hu.game_is_hu and sum(1 for x in hu.stacks if x == 0) == 1
        assert sum(hu.stacks) == 1500 and hu.dealer_id not in hu.dead_players
        assert not three.game_is_hu and all(x > 0 for x in three.stacks)
        assert sum(three.stacks) == 1500 and three.dead_players == ()
        assert (hu.small_blind, hu.big_blind) in BLIND_LEVELS
        assert (three.small_blind, three.big_blind) in BLIND_LEVELS

        seen_hu.add((hu.small_blind, hu.big_blind))
        seen_3h.add((three.small_blind, three.big_blind))

    # This is deliberately a smoke guard, not a statistical certification test.
    assert len(seen_hu) >= 4
    assert len(seen_3h) >= 3


def test_first_release_uses_one_wta_policy_family_for_every_payout():
    scope = LeanTrainingScope()
    assert scope.policy_family_for_payout((1.0, 0.0, 0.0)) == FIRST_RELEASE_POLICY_FAMILY
    assert scope.policy_family_for_payout((0.7, 0.3, 0.0)) == FIRST_RELEASE_POLICY_FAMILY
    assert scope.policy_family_for_payout((0.5, 0.3, 0.2)) == FIRST_RELEASE_POLICY_FAMILY


def test_first_release_chip_ev_scale_is_global_not_blind_dependent():
    class Terminal:
        @staticmethod
        def terminal_chip_delta():
            return (300.0, -100.0, -200.0)

    assert constant_scaled_chip_delta_utility(Terminal()) == (
        300.0 / FIRST_RELEASE_TOTAL_CHIPS,
        -100.0 / FIRST_RELEASE_TOTAL_CHIPS,
        -200.0 / FIRST_RELEASE_TOTAL_CHIPS,
    )
    assert LeanTrainingScope().terminal_utility is constant_scaled_chip_delta_utility
