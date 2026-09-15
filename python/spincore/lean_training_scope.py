from __future__ import annotations

"""Lean first-release training policy for SpinCore.

The first strong functional SpinCore model is intentionally one WTA-trained
policy family, not one full training run per payout vector.  This keeps the
state space focused on the poker problem that dominates the real game volume.

Multi-place payout specialization remains an optional later improvement.  The
solver keeps ICM support, but it is not allowed to multiply first-release
training cost before the WTA policy is strong and usable.

Utility scaling is deliberately *constant across blind levels*.  Legacy
DeepSpin divided terminal chip delta by the current BB.  That leaves each
individual state's action ordering unchanged, but it silently changes the
relative target magnitude seen by one shared neural approximator across blind
levels.  For the first functional SpinCore we preserve chip EV itself and only
apply the global 1/1500 scale dictated by the fixed tournament chip supply.
This bounds terminal targets without reweighting 10/20 versus 100/200.
"""

from dataclasses import dataclass
from typing import Sequence

from .deep_cfr import chip_delta_utility

PRIMARY_UTILITY_ID = "CHIP_EV_WTA_V1"
PRIMARY_PAYOUT_VECTOR = (1.0, 0.0, 0.0)
FIRST_RELEASE_POLICY_FAMILY = "WTA_SHARED_ACROSS_PAYOUTS_V1"
MULTIPAY_SPECIALIZATION_ENABLED = False
FIRST_RELEASE_TOTAL_CHIPS = 1500.0
UTILITY_SCALE_ID = "TOTAL_CHIPS_CONSTANT_1500_V1"


def constant_scaled_chip_delta_utility(state) -> tuple[float, float, float]:
    """Chip EV expressed as fraction of the fixed 1500-chip tournament supply.

    This is a *global positive constant* transformation of chip EV.  It cannot
    change action ordering, regret-matching ratios, or the WTA equilibrium in an
    exactly fitted model.  Unlike legacy ``chip_delta / current_bb``, it also
    does not give different blind levels different target scales merely because
    the blind changed.
    """
    return tuple(float(x) / FIRST_RELEASE_TOTAL_CHIPS for x in chip_delta_utility(state))


@dataclass(frozen=True)
class LeanTrainingScope:
    """Strategic scope, deliberately excluding presentation/economic labels."""

    utility_id: str = PRIMARY_UTILITY_ID
    utility_scale_id: str = UTILITY_SCALE_ID
    training_payout: tuple[float, float, float] = PRIMARY_PAYOUT_VECTOR
    reuse_wta_policy_for_multipay: bool = True
    multipay_specialization_enabled: bool = MULTIPAY_SPECIALIZATION_ENABLED

    def __post_init__(self) -> None:
        if self.utility_id != PRIMARY_UTILITY_ID:
            raise ValueError("first-release SpinCore scope is WTA chip-EV only")
        if self.utility_scale_id != UTILITY_SCALE_ID:
            raise ValueError("first-release utility scale must be constant across blind levels")
        if tuple(float(x) for x in self.training_payout) != PRIMARY_PAYOUT_VECTOR:
            raise ValueError("first-release training payout must be winner-take-all")
        if not self.reuse_wta_policy_for_multipay:
            raise ValueError("first release intentionally shares the WTA policy across payout variants")
        if self.multipay_specialization_enabled:
            raise ValueError("multi-place payout specialization is deferred from first release")

    @property
    def terminal_utility(self):
        return constant_scaled_chip_delta_utility

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
    """WTA chip EV with one global numeric scale, independent of blind level."""
    return constant_scaled_chip_delta_utility(state)
