from __future__ import annotations

import pytest

from spincore.production_evidence_packet import CapturedPayouts, SelectedStateEvidencePacket
from spincore.production_profile import BlindLevel, ProductionEvidence
from spincore.production_profile_builder import ProductionStrategyIdentity, build_profile_from_bound_evidence


def _selected() -> SelectedStateEvidencePacket:
    return SelectedStateEvidencePacket(
        table_size=3,
        currency="USD",
        buy_in_minor_units=500,
        multiplier=100,
        starting_chips_per_player=750,
        blind_levels=(BlindLevel(10, 20, 0, 180), BlindLevel(15, 30, 0, 180)),
        payouts=CapturedPayouts(40000, 7500, 2500),
        source_kind="OFFICIAL_CLIENT_CAPTURE",
        source_locator="ggpoker-client-capture://" + "a" * 64,
        captured_at_utc="2026-08-12T23:30:00Z",
        capture_sha256="a" * 64,
        capture_size_bytes=1000,
        capture_note="synthetic unit-test fixture only",
    )


def _fee_evidence() -> ProductionEvidence:
    return ProductionEvidence(
        source_kind="OFFICIAL_WEB",
        locator="https://ggpoker.com/poker-games/spin-gold/",
        observed_at_utc="2026-08-12T23:30:00Z",
        scope="GLOBAL_GAME",
        proven_fields=("tournament_fee_fraction",),
        note="unit-test provenance fixture",
    )


def _strategy() -> ProductionStrategyIdentity:
    return ProductionStrategyIdentity(
        game_family="SPIN_AND_GOLD_3MAX",
        ruleset_id="SPINRULESET-4",
        action_abstraction_id="SPINCORE_ACTION_ABSTRACTION_V1",
        utility_model_id="ICM_EXACT_V1_EXPLICIT_PAYOUT_DELTA",
        learning_profile_id="SPINCORE_R7_3_UNCERTAINTY_POLICY_MIXTURE_V1",
    )


def test_builder_produces_complete_v3_profile_from_separated_evidence():
    profile = build_profile_from_bound_evidence(
        selected_state=_selected(),
        tournament_fee_fraction=0.07,
        tournament_fee_evidence=_fee_evidence(),
        strategy=_strategy(),
    )
    assert profile.table_size == 3
    assert profile.buy_in_minor_units == 500
    assert profile.multiplier == 100
    assert profile.starting_chips_per_player == 750
    assert profile.payout_share_by_place == pytest.approx((0.8, 0.15, 0.05), abs=1e-15)
    assert profile.tournament_fee_fraction == 0.07
    assert profile.profile_id.startswith("spinprofile-v3:")
    assert profile.to_dict()["ready_for_tables"] is False


def test_builder_does_not_infer_prize_pool_from_buyin_times_multiplier():
    state = _selected()
    profile = build_profile_from_bound_evidence(
        selected_state=state,
        tournament_fee_fraction=0.07,
        tournament_fee_evidence=_fee_evidence(),
        strategy=_strategy(),
    )
    # Captured payout pool is 50,000 minor units.  The builder uses only its
    # normalized shape; it never asserts buy-in*multiplier as a prize-pool rule.
    assert sum((state.payouts.first_minor_units, state.payouts.second_minor_units, state.payouts.third_minor_units)) == 50000
    assert sum(profile.payout_share_by_place) == pytest.approx(1.0, abs=1e-15)


def test_selected_state_evidence_cannot_substitute_for_global_fee_evidence():
    wrong = _selected().to_production_evidence()
    with pytest.raises(ValueError, match="GLOBAL_GAME"):
        build_profile_from_bound_evidence(
            selected_state=_selected(),
            tournament_fee_fraction=0.07,
            tournament_fee_evidence=wrong,
            strategy=_strategy(),
        )


def test_generic_global_evidence_cannot_claim_extra_fields_in_builder():
    wrong = ProductionEvidence(
        source_kind="OFFICIAL_WEB",
        locator="https://ggpoker.com/poker-games/spin-gold/",
        observed_at_utc="2026-08-12T23:30:00Z",
        scope="GLOBAL_GAME",
        proven_fields=("tournament_fee_fraction", "currency"),
        note="unit-test fixture",
    )
    with pytest.raises(ValueError, match="prove only tournament_fee_fraction"):
        build_profile_from_bound_evidence(
            selected_state=_selected(),
            tournament_fee_fraction=0.07,
            tournament_fee_evidence=wrong,
            strategy=_strategy(),
        )
