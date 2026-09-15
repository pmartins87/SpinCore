from __future__ import annotations

"""First functional SpinCore action scope.

Start from the mature DeepSpin seven-action vocabulary instead of inventing a
new abstraction before the full real SpinGo distribution is working.  The
current solver already exposes a ten-slot universal action vocabulary; this
module simply activates the seven legacy-equivalent slots and leaves the other
three dormant.

This is intentionally a *baseline*, not a claim that seven actions are globally
optimal.  We prune or expand it only after the functional WTA agent exists and
only when the expected strategic gain justifies the extra branching cost.
"""

from .r7_5_action_contract import ActionCandidateSpec, universal_mask

FIRST_RELEASE_ACTION_SCOPE_ID = "LEGACY_7_ACTION_BASELINE_V1"
FIRST_RELEASE_ACTION_NAMES = (
    "FOLD",
    "CHECK_CALL",
    "POT_33",
    "POT_50",
    "POT_75",
    "POT_100",
    "ALL_IN",
)
FIRST_RELEASE_ACTION_MASK = universal_mask(FIRST_RELEASE_ACTION_NAMES)

# Preserve the legacy action vocabulary on every street for the first functional
# baseline.  Preflop-specific simplification is a possible later compute saving,
# but is not introduced without evidence because it would change strategy.
FIRST_RELEASE_ACTION_SPEC = ActionCandidateSpec(
    candidate_id=FIRST_RELEASE_ACTION_SCOPE_ID,
    preflop_mask=FIRST_RELEASE_ACTION_MASK,
    postflop_mask=FIRST_RELEASE_ACTION_MASK,
    eligible_to_win=True,
    phase="LEAN_FUNCTIONAL_V1",
)


def active_action_names() -> tuple[str, ...]:
    return FIRST_RELEASE_ACTION_NAMES
