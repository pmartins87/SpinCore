from __future__ import annotations

from dataclasses import replace

import pytest

from spincore.simulator_profile import (
    HandsBlindLevel,
    OfficialRuleReference,
    SimulatorPresentation,
    UniversalThreeMaxSimulatorProfile,
    require_stake_invariant_policy_identity,
)


def _reference() -> OfficialRuleReference:
    return OfficialRuleReference(
        locator="https://ggpoker.com/poker-games/spin-gold/",
        observed_at_utc="2026-08-17T00:00:00Z",
        proven_fields=(
            "table_size",
            "buy_in_catalog",
            "multiplier_dependent_structure",
            "payout_structure",
            "deck_and_shuffle",
            "tournament_fee_fraction",
            "same_hand_elimination_tiebreak",
            "no_make_a_deal",
        ),
    )


def _profile(
    *,
    buy_in_minor_units: int = 500,
    currency: str = "USD",
    multiplier: int = 2,
    fee: float = 0.07,
    starting: int = 500,
    payout=(1.0, 0.0, 0.0),
) -> UniversalThreeMaxSimulatorProfile:
    return UniversalThreeMaxSimulatorProfile(
        starting_chips_per_player=starting,
        blind_levels=(
            HandsBlindLevel(10, 20, 0, 8),
            HandsBlindLevel(15, 30, 0, 8),
        ),
        payout_share_by_place=tuple(payout),
        ruleset_id="SPINRULESET-4",
        action_abstraction_id="SPINCORE_ACTION_ABSTRACTION_V1",
        utility_model_id="ICM_EXACT_V1_EXPLICIT_PAYOUT_DELTA",
        learning_profile_id="SPINCORE_R7_3_UNCERTAINTY_POLICY_MIXTURE_V1",
        presentation=SimulatorPresentation(
            currency=currency,
            nominal_buy_in_minor_units=buy_in_minor_units,
            displayed_multiplier=multiplier,
            tournament_fee_fraction=fee,
        ),
        official_references=(_reference(),),
    )


def test_round_trip_is_stable_and_hu_is_not_three_handed():
    profile = _profile()
    restored = UniversalThreeMaxSimulatorProfile.from_dict(profile.to_dict())
    assert restored.profile_id == profile.profile_id
    assert profile.profile_id.startswith("spinsim-v1:")
    assert profile.policy_id("TRUE_HEADS_UP") != profile.policy_id("THREE_HANDED")


def test_all_official_nominal_stakes_share_one_policy_when_rules_match():
    official_buy_ins = (25, 100, 300, 500, 1000, 2000, 5000, 10000, 20000)
    profiles = [_profile(buy_in_minor_units=value) for value in official_buy_ins]
    policy_id = require_stake_invariant_policy_identity(profiles, "THREE_HANDED")
    assert all(row.policy_id("THREE_HANDED") == policy_id for row in profiles)


def test_presentation_and_accounting_metadata_never_change_strategy_identity():
    base = _profile()
    variants = (
        _profile(buy_in_minor_units=20000),
        _profile(currency="PLAY_CHIPS"),
        _profile(multiplier=200000),
        _profile(fee=0.0),
        replace(base, presentation=replace(base.presentation, skin_reference="CUSTOM_GG_STYLE")),
    )
    assert all(row.profile_id == base.profile_id for row in variants)
    assert all(row.policy_id("THREE_HANDED") == base.policy_id("THREE_HANDED") for row in variants)


def test_effective_rules_force_new_strategy_identity():
    base = _profile()
    changed_blinds = replace(
        base,
        blind_levels=(HandsBlindLevel(10, 20, 0, 10), HandsBlindLevel(15, 30, 0, 10)),
    )
    variants = (
        _profile(starting=750),
        _profile(payout=(0.8, 0.2, 0.0)),
        changed_blinds,
    )
    assert all(row.profile_id != base.profile_id for row in variants)


def test_only_hands_based_progression_is_accepted():
    with pytest.raises(ValueError, match="completed hands"):
        replace(_profile(), blind_progression_basis="ELAPSED_SECONDS")
    with pytest.raises(ValueError, match="hands_per_level"):
        HandsBlindLevel(10, 20, 0, 0)


def test_official_provenance_is_first_party_and_complete():
    with pytest.raises(ValueError, match="GGPoker first-party"):
        replace(_reference(), locator="https://example.com/spin-gold")
    with pytest.raises(ValueError, match="official references do not cover"):
        replace(_profile(), official_references=(replace(_reference(), proven_fields=("table_size",)),))


def test_tampering_and_release_authorization_fail_closed():
    row = _profile().to_dict()
    row["starting_chips_per_player"] += 1
    with pytest.raises(ValueError, match="identity hash mismatch"):
        UniversalThreeMaxSimulatorProfile.from_dict(row)

    row = _profile().to_dict()
    row["real_money_client_integration_authorized"] = True
    with pytest.raises(ValueError, match="real-money client"):
        UniversalThreeMaxSimulatorProfile.from_dict(row)

    row = _profile().to_dict()
    row["ready_for_simulator_tables"] = True
    with pytest.raises(ValueError, match="simulator table release"):
        UniversalThreeMaxSimulatorProfile.from_dict(row)


def test_profile_is_exactly_three_max_with_official_deck_rules():
    with pytest.raises(ValueError, match="exactly 3 players"):
        replace(_profile(), table_size=6)
    with pytest.raises(ValueError, match="52-card deck"):
        replace(_profile(), deck_size=54)
    with pytest.raises(ValueError, match="52-card deck"):
        replace(_profile(), shuffle_after_each_hand=False)
