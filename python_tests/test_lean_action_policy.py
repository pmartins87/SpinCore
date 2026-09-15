from __future__ import annotations

from spincore.lean_action_policy import lean_regret_matching_policy


def test_lean_action_policy_preserves_ranking_when_all_legal_advantages_nonpositive():
    advantages = [-9.0, -3.0, -8.0, -1.0, -4.0, -6.0, -2.0, -7.0, -5.0, -10.0]
    legal = (0, 1, 3, 6)
    policy = lean_regret_matching_policy(advantages, legal)
    assert policy[3] > policy[6] > policy[1] > policy[0]
    assert abs(sum(policy) - 1.0) < 1e-12
    assert all(policy[i] == 0.0 for i in range(10) if i not in legal)
