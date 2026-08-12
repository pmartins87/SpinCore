from __future__ import annotations

import copy

import pytest

from spincore.production_evidence_packet import (
    CapturedPayouts,
    SelectedStateEvidencePacket,
    require_unique_selected_states,
)
from spincore.production_profile import BlindLevel


def _packet(*, buy_in: int = 500, multiplier: int = 2, payouts=(1000, 0, 0), capture_sha="a" * 64):
    return SelectedStateEvidencePacket(
        table_size=3,
        currency="USD",
        buy_in_minor_units=buy_in,
        multiplier=multiplier,
        starting_chips_per_player=500,
        blind_levels=(
            BlindLevel(10, 20, 0, 180),
            BlindLevel(15, 30, 0, 180),
        ),
        payouts=CapturedPayouts(*payouts),
        source_kind="OFFICIAL_CLIENT_CAPTURE",
        source_locator=f"ggpoker-client-capture://{capture_sha}",
        captured_at_utc="2026-08-12T23:30:00Z",
        capture_sha256=capture_sha,
        capture_size_bytes=123456,
        capture_note="synthetic unit-test metadata; not production evidence",
    )


def test_packet_round_trip_and_v3_binding():
    packet = _packet()
    restored = SelectedStateEvidencePacket.from_dict(packet.to_dict())
    assert restored == packet
    assert restored.packet_id == packet.packet_id
    evidence = packet.to_production_evidence()
    assert evidence.scope == "SELECTED_PROFILE_STATE"
    assert (evidence.bound_table_size, evidence.bound_buy_in_minor_units, evidence.bound_multiplier) == (3, 500, 2)
    assert set(evidence.proven_fields) == {
        "table_size",
        "buy_in_minor_units",
        "multiplier",
        "starting_chips_per_player",
        "blind_levels",
        "payout_share_by_place",
    }


def test_payout_shares_come_from_captured_amounts_not_multiplier_assumption():
    packet = _packet(multiplier=100, payouts=(80000, 15000, 5000))
    assert packet.payouts.normalized_shares == pytest.approx((0.8, 0.15, 0.05), abs=1e-15)


def test_packet_identity_changes_with_selected_state_or_capture_bytes():
    base = _packet()
    assert _packet(buy_in=1000).packet_id != base.packet_id
    assert _packet(multiplier=3).packet_id != base.packet_id
    assert _packet(capture_sha="b" * 64).packet_id != base.packet_id


def test_tampered_normalized_payouts_fail_closed():
    row = _packet(payouts=(800, 200, 0)).to_dict()
    row["payout_share_by_place"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="normalization mismatch"):
        SelectedStateEvidencePacket.from_dict(row)


def test_tampered_packet_identity_fails_closed():
    row = _packet().to_dict()
    row["starting_chips_per_player"] = 750
    with pytest.raises(ValueError, match="identity mismatch"):
        SelectedStateEvidencePacket.from_dict(row)


def test_only_first_party_client_or_rule_document_can_claim_selected_state_packet():
    row = _packet().to_dict()
    row["source_kind"] = "OFFICIAL_WEB"
    row.pop("packet_id")
    with pytest.raises(ValueError, match="official client/rule-document"):
        SelectedStateEvidencePacket.from_dict(row)


def test_conflicting_authoritative_packets_for_same_state_are_rejected():
    a = _packet(capture_sha="a" * 64)
    b = _packet(capture_sha="b" * 64)
    with pytest.raises(ValueError, match="conflicting selected-state"):
        require_unique_selected_states((a, b))
    registry = require_unique_selected_states((a, a))
    assert registry[(3, 500, 2)] == a.packet_id


def test_packet_never_authorizes_table_use():
    row = _packet().to_dict()
    row["ready_for_tables"] = True
    with pytest.raises(ValueError, match="cannot authorize table use"):
        SelectedStateEvidencePacket.from_dict(row)
