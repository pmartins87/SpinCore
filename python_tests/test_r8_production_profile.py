from __future__ import annotations

import copy

import pytest

from spincore.production_profile import (
    BlindLevel,
    ProductionEvidence,
    ProductionProfile,
    require_unique_policy_identities,
)


def _profile(*, multiplier: int = 2, starting: int = 500, payout=(1.0, 0.0, 0.0)) -> ProductionProfile:
    # Synthetic values are used only to test identity/validation mechanics.
    # They are not asserted to be a real GGPoker production profile.
    return ProductionProfile(
        platform="GGPOKER",
        game_family="SPIN_AND_GOLD_3MAX_TEST_FIXTURE",
        table_size=3,
        multiplier=multiplier,
        starting_chips_per_player=starting,
        blind_levels=(BlindLevel(10, 20, 0, 180), BlindLevel(15, 30, 0, 180)),
        payout_by_place=tuple(payout),
        tournament_fee_fraction=0.07,
        ruleset_id="SPINRULESET-4",
        action_abstraction_id="TEST_ACTION_ABSTRACTION",
        utility_model_id="ICM_EXACT_V1_EXPLICIT_PAYOUT_DELTA",
        learning_profile_id="SPINCORE_R7_3_UNCERTAINTY_POLICY_MIXTURE_V1",
        evidence=(
            ProductionEvidence(
                source_kind="OFFICIAL_WEB",
                locator="https://example.invalid/test-fixture-not-production",
                observed_at_utc="2026-08-12T00:00:00Z",
                note="unit-test fixture only",
            ),
        ),
    )


def test_profile_id_is_stable_and_domain_policy_ids_are_separate():
    a = _profile()
    b = ProductionProfile.from_dict(a.to_dict())
    assert a.profile_id == b.profile_id
    assert a.policy_id("TRUE_HEADS_UP") != a.policy_id("THREE_HANDED")
    assert a.policy_id("TRUE_HEADS_UP") == b.policy_id("TRUE_HEADS_UP")


def test_economic_or_structural_change_forces_new_profile_identity():
    base = _profile()
    variants = [
        _profile(multiplier=3),
        _profile(starting=750),
        _profile(payout=(0.8, 0.2, 0.0)),
    ]
    assert all(row.profile_id != base.profile_id for row in variants)


def test_provenance_timestamp_does_not_change_semantic_identity():
    a = _profile()
    row = a.to_dict()
    row["evidence"][0]["observed_at_utc"] = "2026-08-13T00:00:00Z"
    row["profile_id"] = a.profile_id
    b = ProductionProfile.from_dict(row)
    assert b.profile_id == a.profile_id


def test_identity_hash_tampering_fails_closed():
    row = _profile().to_dict()
    row["starting_chips_per_player"] += 1
    with pytest.raises(ValueError, match="identity hash mismatch"):
        ProductionProfile.from_dict(row)


def test_incomplete_or_non_first_party_profile_cannot_be_constructed():
    with pytest.raises(ValueError, match="blind structure"):
        p = _profile()
        ProductionProfile(
            platform=p.platform,
            game_family=p.game_family,
            table_size=p.table_size,
            multiplier=p.multiplier,
            starting_chips_per_player=p.starting_chips_per_player,
            blind_levels=(),
            payout_by_place=p.payout_by_place,
            tournament_fee_fraction=p.tournament_fee_fraction,
            ruleset_id=p.ruleset_id,
            action_abstraction_id=p.action_abstraction_id,
            utility_model_id=p.utility_model_id,
            learning_profile_id=p.learning_profile_id,
            evidence=p.evidence,
        )
    with pytest.raises(ValueError, match="first-party"):
        ProductionEvidence("COMMUNITY_WIKI", "somewhere", "2026-08-12T00:00:00Z")


def test_r8_profile_never_claims_table_readiness():
    row = _profile().to_dict()
    assert row["ready_for_tables"] is False
    row["ready_for_tables"] = True
    with pytest.raises(ValueError, match="cannot authorize table use"):
        ProductionProfile.from_dict(row)


def test_policy_identity_registry_keeps_profile_and_domain_disjoint():
    profiles = [_profile(multiplier=2), _profile(multiplier=3)]
    registry = require_unique_policy_identities(profiles, ("TRUE_HEADS_UP", "THREE_HANDED"))
    assert len(registry) == 4
    assert len(set(registry)) == 4
