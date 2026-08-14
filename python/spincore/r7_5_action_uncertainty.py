from __future__ import annotations

import math
from typing import Sequence

from spincore.r7_5_action_cfr import (
    NUM_ACTIONS,
    legal_mask,
    regret_matching_policy,
    uniform_policy,
)
from spincore_nn.action_models import collate_action_observations


def _normalize_legal(policy: Sequence[float], legal: tuple[int, ...], action_count: int) -> tuple[float, ...]:
    legal_set = set(legal)
    out = [0.0] * int(action_count)
    total = 0.0
    for action in legal:
        value = max(0.0, float(policy[action]))
        out[action] = value
        total += value
    if total <= 0.0:
        for action in legal:
            out[action] = 1.0 / len(legal)
    else:
        for action in legal:
            out[action] /= total
    if any(out[action] != 0.0 for action in range(action_count) if action not in legal_set):
        raise AssertionError("illegal mass survived normalization")
    return tuple(out)


def uncertainty_damped_policy_from_advantages(
    member_advantages: Sequence[Sequence[float]],
    legal: tuple[int, ...],
    *,
    action_count: int,
    epsilon_scale: float = 1.75,
    epsilon_cap: float = 0.5,
) -> tuple[tuple[float, ...], dict[str, float | bool]]:
    """Dimension-generic form of the accepted R7.3 uncertainty mixture.

    Each ensemble member is converted independently through regret matching.
    Their arithmetic mean is the exploitation policy. Ensemble disagreement is
    the arithmetic mean total-variation distance between each member policy and
    that mean. The final behavior damps the mean policy toward uniform legal
    play by epsilon=min(cap, scale*disagreement).
    """
    if not member_advantages:
        raise ValueError("uncertainty damping requires at least one ensemble member")
    if not legal:
        raise ValueError("uncertainty damping requires nonempty legal set")
    if action_count <= 0:
        raise ValueError("positive action_count required")
    if any(len(row) != action_count for row in member_advantages):
        raise ValueError("ensemble action width mismatch")
    if epsilon_scale < 0.0 or epsilon_cap < 0.0:
        raise ValueError("nonnegative uncertainty parameters required")

    def rm(row: Sequence[float]) -> tuple[float, ...]:
        # The accepted regret matcher is dimension-independent; duplicate its
        # algebra here so this pure helper can also be regression-tested at six
        # actions without importing the ten-action state collector.
        out = [0.0] * action_count
        total = sum(max(0.0, float(row[action])) for action in legal)
        if total <= 0.0:
            for action in legal:
                out[action] = 1.0 / len(legal)
        else:
            for action in legal:
                out[action] = max(0.0, float(row[action])) / total
        return tuple(out)

    policies = [rm(row) for row in member_advantages]
    mean = [0.0] * action_count
    for policy in policies:
        for action in legal:
            mean[action] += float(policy[action]) / len(policies)
    mean_policy = _normalize_legal(mean, legal, action_count)

    disagreement = 0.0
    for policy in policies:
        disagreement += 0.5 * sum(
            abs(float(policy[action]) - float(mean_policy[action])) for action in legal
        )
    disagreement /= len(policies)

    raw_epsilon = float(epsilon_scale) * float(disagreement)
    epsilon = min(float(epsilon_cap), raw_epsilon)
    uniform = [0.0] * action_count
    for action in legal:
        uniform[action] = 1.0 / len(legal)
    mixed = [0.0] * action_count
    for action in legal:
        mixed[action] = (
            (1.0 - epsilon) * float(mean_policy[action])
            + epsilon * float(uniform[action])
        )
    result = _normalize_legal(mixed, legal, action_count)
    return result, {
        "disagreement": float(disagreement),
        "raw_epsilon": float(raw_epsilon),
        "epsilon": float(epsilon),
        "cap_hit": bool(raw_epsilon > float(epsilon_cap)),
    }


class ActionUncertaintyDampedPolicyMixture:
    def __init__(
        self,
        *,
        selected_representation: str,
        device: str = "cpu",
        epsilon_scale: float = 1.75,
        epsilon_cap: float = 0.5,
    ):
        self.selected_representation = str(selected_representation)
        self.device = device
        self.epsilon_scale = float(epsilon_scale)
        self.epsilon_cap = float(epsilon_cap)
        self.models = []
        self.calls = 0
        self.epsilon_sum = 0.0
        self.epsilon_max = 0.0
        self.disagreement_sum = 0.0
        self.raw_epsilon_max = 0.0
        self.cap_hit_calls = 0
        self.epsilon_ge_010_calls = 0
        self.epsilon_ge_025_calls = 0

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]):
        if not self.models:
            return uniform_policy(state, observation, legal)

        import torch

        batch = collate_action_observations(
            self.selected_representation,
            [observation],
            [legal_mask(legal)],
            device=self.device,
        )
        rows = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                rows.append(model(batch)[0].detach().cpu().tolist())
        policy, stats = uncertainty_damped_policy_from_advantages(
            rows,
            legal,
            action_count=NUM_ACTIONS,
            epsilon_scale=self.epsilon_scale,
            epsilon_cap=self.epsilon_cap,
        )
        epsilon = float(stats["epsilon"])
        raw_epsilon = float(stats["raw_epsilon"])
        disagreement = float(stats["disagreement"])
        self.calls += 1
        self.epsilon_sum += epsilon
        self.epsilon_max = max(self.epsilon_max, epsilon)
        self.disagreement_sum += disagreement
        self.raw_epsilon_max = max(self.raw_epsilon_max, raw_epsilon)
        if bool(stats["cap_hit"]):
            self.cap_hit_calls += 1
        if epsilon >= 0.10:
            self.epsilon_ge_010_calls += 1
        if epsilon >= 0.25:
            self.epsilon_ge_025_calls += 1
        return policy
