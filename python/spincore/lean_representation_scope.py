from __future__ import annotations

"""First-release SpinCore neural representation policy.

The legacy 292-float observation remains reference/diagnostic knowledge, not the
first-release training boundary.  The first functional SpinCore uses the compact
SPNNIV1 exact-state-derived representation unless a concrete strategic failure
justifies adding a specific feature.
"""

FIRST_RELEASE_REPRESENTATION_ID = "SPNNIV1_COMPACT_EXACT_STATE"
LEGACY_292_REINTRODUCTION_ENABLED = False
ALLOW_UNPROVEN_FEATURE_EXPANSION = False


def require_first_release_representation(representation_id: str) -> str:
    value = str(representation_id)
    if value != FIRST_RELEASE_REPRESENTATION_ID:
        raise ValueError(
            "first-release SpinCore representation is frozen to compact SPNNIV1; "
            "feature expansion requires a concrete strategic/correctness reason"
        )
    return value
