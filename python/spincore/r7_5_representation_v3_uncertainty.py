from __future__ import annotations

import torch

from spincore.r7_5_action_cfr import NUM_ACTIONS, legal_mask, uniform_policy
from spincore.r7_5_action_uncertainty import uncertainty_damped_policy_from_advantages
from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore_nn.models_v3_final import collate_v3_observations


class V3UncertaintyDampedPolicyMixture:
    """SPNNIV3 inference wrapper over the accepted R7.3/R7.5.4 algebra.

    The probability algebra is not duplicated here: it is delegated to the
    already regression-tested dimension-generic helper
    `uncertainty_damped_policy_from_advantages`. Only observation collation is
    representation-specific.
    """

    def __init__(
        self,
        *,
        representation: str,
        device: str = "cpu",
        epsilon_scale: float = 1.75,
        epsilon_cap: float = 0.5,
    ):
        if representation not in (H2_FINAL, H3_FINAL):
            raise ValueError("unsupported final SPNNIV3 representation")
        self.representation = str(representation)
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

    @property
    def with_semantics(self) -> bool:
        return self.representation == H3_FINAL

    def stats(self) -> dict[str, float | int]:
        return {
            "calls": int(self.calls),
            "epsilon_sum": float(self.epsilon_sum),
            "epsilon_max": float(self.epsilon_max),
            "disagreement_sum": float(self.disagreement_sum),
            "raw_epsilon_max": float(self.raw_epsilon_max),
            "cap_hit_calls": int(self.cap_hit_calls),
            "epsilon_ge_010_calls": int(self.epsilon_ge_010_calls),
            "epsilon_ge_025_calls": int(self.epsilon_ge_025_calls),
        }

    def restore_stats(self, values: dict) -> None:
        allowed = set(self.stats())
        if set(values) - allowed:
            raise ValueError("unknown V3 uncertainty statistic")
        for key in allowed:
            if key in values:
                setattr(self, key, values[key])

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]):
        if not self.models:
            return uniform_policy(state, observation, legal)
        batch = collate_v3_observations(
            [observation],
            [legal_mask(legal)],
            with_semantics=self.with_semantics,
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
