from __future__ import annotations

from spincore.lean_action_scope import (
    FIRST_RELEASE_ACTION_MASK,
    FIRST_RELEASE_ACTION_NAMES,
    FIRST_RELEASE_ACTION_SPEC,
)
from spincore.r7_5_action_contract import mask_names


def test_first_release_action_scope_matches_legacy_seven_actions():
    assert mask_names(FIRST_RELEASE_ACTION_MASK) == FIRST_RELEASE_ACTION_NAMES
    assert FIRST_RELEASE_ACTION_SPEC.preflop_mask == FIRST_RELEASE_ACTION_MASK
    assert FIRST_RELEASE_ACTION_SPEC.postflop_mask == FIRST_RELEASE_ACTION_MASK
