from __future__ import annotations

"""Lean action-policy adapter for the first functional SpinCore.

The historical R7.5 universal-action module still contains the old uniform
fallback when every fitted legal advantage is non-positive.  DeepSpin had
already identified that fallback as a source of gross actions.  Do not mutate
the historical module/freeze; the functional path uses this corrected adapter.
"""

import math
from typing import Sequence

from spincore.r7_5_action_cfr import NUM_ACTIONS, uniform_policy
from spincore.r7_5_action_cfr import legal_mask as universal_legal_mask
from spincore_nn.action_models import collate_action_observations


def lean_regret_matching_policy(
    advantages: Sequence[float], legal: tuple[int, ...]
) -> tuple[float, ...]:
    if len(advantages) != NUM_ACTIONS:
        raise ValueError("ten universal advantages required")
    if not legal:
        raise ValueError("empty legal set")

    out = [0.0] * NUM_ACTIONS
    positive_total = sum(max(0.0, float(advantages[action])) for action in legal)
    if positive_total > 0.0:
        for action in legal:
            out[action] = max(0.0, float(advantages[action])) / positive_total
        return tuple(out)

    raw = [float(advantages[action]) for action in legal]
    if any(not math.isfinite(value) for value in raw):
        raise ValueError("nonfinite advantage")
    maximum = max(raw)
    weights = [math.exp(max(-60.0, min(60.0, value - maximum))) for value in raw]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        for action in legal:
            out[action] = 1.0 / len(legal)
        return tuple(out)
    for action, weight in zip(legal, weights):
        out[action] = weight / total
    return tuple(out)


class LeanNeuralActionAdvantagePolicy:
    """Ten-output universal model with the repaired legacy fallback semantics."""

    def __init__(
        self,
        model,
        *,
        selected_representation: str = "C0_V1_FROZEN_CONTROL",
        device: str = "cpu",
        ready: bool = True,
    ):
        self.model = model
        self.selected_representation = str(selected_representation)
        self.device = device
        self.ready = bool(ready)

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]):
        if not self.ready:
            return uniform_policy(state, observation, legal)

        import torch

        batch = collate_action_observations(
            self.selected_representation,
            [observation],
            [universal_legal_mask(legal)],
            device=self.device,
        )
        self.model.eval()
        with torch.no_grad():
            raw = self.model(batch)[0].detach().cpu().tolist()
        return lean_regret_matching_policy(raw, legal)
