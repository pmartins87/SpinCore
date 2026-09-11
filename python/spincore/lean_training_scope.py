from __future__ import annotations

"""Lean first-release training policy for SpinCore.

The first strong functional SpinCore model is intentionally one WTA-trained
policy family, not one full training run per payout vector.  This keeps the
state space focused on the poker problem that dominates the real game volume.

Multi-place payout specialization remains an optional later improvement.  The
solver keeps ICM support, but it is not allowed to multiply first-release
training cost before the WTA policy is strong and usable.
"""

from dataclasses import dataclass
from typing import Sequence

from .deep_cfr import chip_delta_utility

PRIMARY_UTILITY_ID = "CHIP_EV_WTA_V1"
PRIMARY_PAYOUT_VECTOR = (1.0, 0.0, 0.0)
FIRST_RELEASE_POLICY_FAMILY = "WTA_SHARED_ACROSS_PAYOUTS_V1"
MULTIPAY_SPECIALIZATION_ENABLED = False


@dataclass(frozen=True)
class LeanTrainingScope:
    """Strategic scope, deliberately excluding presentation/economic labels."""

    utility_id: str = PRIMARY_UTILITY_ID
    training_payout: tuple[float, float, float] = PRIMARY_PAYOUT_VECTOR
    reuse_wta_policy_for_multipay: bool = True
    multipay_specialization_enabled: bool = MULTIPAY_SPECIALIZATION_ENABLED

    def __post_init__(self) -> None:
        if self.utility_id != PRIMARY_UTILITY_ID:
            raise ValueError("first-release SpinCore scope is WTA chip-EV only")
        if tuple(float(x) for x in self.training_payout) != PRIMARY_PAYOUT_VECTOR:
            raise ValueError("first-release training payout must be winner-take-all")
        if not self.reuse_wta_policy_for_multipay:
            raise ValueError("first release intentionally shares the WTA policy across payout variants")
        if self.multipay_specialization_enabled:
            raise ValueError("multi-place payout specialization is deferred from first release")

    @property
    def terminal_utility(self):
        return chip_delta_utility

    def policy_family_for_payout(self, payout_by_place: Sequence[float]) -> str:
        """Return one policy identity for every payout in the first release.

        The payout is validated as metadata so a malformed runtime state still
        fails early, but it does not select another trained network yet.
        """
        payout = tuple(float(x) for x in payout_by_place)
        if len(payout) != 3:
            raise ValueError("3-max payout vector must have three places")
        if any(x < 0.0 for x in payout) or sum(payout) <= 0.0:
            raise ValueError("invalid payout vector")
        return FIRST_RELEASE_POLICY_FAMILY


def first_release_terminal_utility(state):
    """Raw chip delta: the single first-release Deep CFR objective."""
    return chip_delta_utility(state)
