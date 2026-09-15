from __future__ import annotations

"""First functional SpinCore action scope.

Preserve DeepSpin's mature seven *labels* without pretending those labels always
mean literal pot fractions.  The lean C++ resolver reproduces the historical
context semantics:

- preflop POT_33 label = 2 BB normal open;
- preflop POT_75 label = isolation to 2.5 BB + 1 BB per extra limper;
- preflop POT_50 label = 3-bet to 5 BB + 2 BB per caller after the raise;
- limp-vs-isolation and multi-raise shove/fold restrictions are retained;
- postflop 33/50/75/100 are true pot-after-call fractions with legacy pruning
  and 60% near-all-in collapse.

The ten-slot network carrier is retained only as infrastructure.  MIN_RAISE,
POT_40 and POT_66 are dormant in this first functional baseline.
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

FIRST_RELEASE_ACTION_SPEC = ActionCandidateSpec(
    candidate_id=FIRST_RELEASE_ACTION_SCOPE_ID,
    preflop_mask=FIRST_RELEASE_ACTION_MASK,
    postflop_mask=FIRST_RELEASE_ACTION_MASK,
    eligible_to_win=True,
    phase="LEAN_FUNCTIONAL_V1",
)


def active_action_names() -> tuple[str, ...]:
    return FIRST_RELEASE_ACTION_NAMES
