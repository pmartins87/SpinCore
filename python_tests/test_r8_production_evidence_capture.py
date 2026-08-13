from __future__ import annotations

import hashlib

import pytest

from spincore.production_evidence_capture import build_selected_state_packet_from_capture


def _spec():
    return {
        "table_size": 3,
        "currency": "USD",
        "buy_in_minor_units": 500,
        "multiplier": 10,
        "starting_chips_per_player": 500,
        "blind_levels": [
            {"small_blind": 10, "big_blind": 20, "ante": 0, "duration_seconds": 180},
            {"small_blind": 15, "big_blind": 30, "ante": 0, "duration_seconds": 180},
        ],
        "payouts_minor_units": [4000, 1000, 0],
        "source_kind": "OFFICIAL_CLIENT_CAPTURE",
        "source_locator": "ggpoker-client-capture://unit-test",
        "captured_at_utc": "2026-08-13T03:00:00Z",
        "capture_note": "synthetic unit test only",
    }


def test_capture_bytes_are_bound_by_sha_and_size(tmp_path):
    capture = tmp_path / "capture.bin"
    capture.write_bytes(b"exact-captured-bytes")
    packet = build_selected_state_packet_from_capture(capture_path=capture, spec=_spec())
    assert packet.capture_sha256 == hashlib.sha256(b"exact-captured-bytes").hexdigest()
    assert packet.capture_size_bytes == len(b"exact-captured-bytes")
    assert packet.buy_in_minor_units == 500
    assert packet.multiplier == 10
    assert packet.payouts.normalized_shares == pytest.approx((0.8, 0.2, 0.0), abs=1e-15)
    assert packet.to_production_evidence().scope == "SELECTED_PROFILE_STATE"


def test_different_capture_bytes_change_packet_identity(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"capture-a")
    b.write_bytes(b"capture-b")
    pa = build_selected_state_packet_from_capture(capture_path=a, spec=_spec())
    pb = build_selected_state_packet_from_capture(capture_path=b, spec=_spec())
    assert pa.capture_sha256 != pb.capture_sha256
    assert pa.packet_id != pb.packet_id


def test_web_evidence_cannot_be_promoted_by_capture_builder(tmp_path):
    capture = tmp_path / "page.html"
    capture.write_bytes(b"indexed web page")
    spec = _spec()
    spec["source_kind"] = "OFFICIAL_WEB"
    with pytest.raises(ValueError, match="official client/rule-document"):
        build_selected_state_packet_from_capture(capture_path=capture, spec=spec)


def test_incomplete_blinds_fail_before_packet_creation(tmp_path):
    capture = tmp_path / "capture.bin"
    capture.write_bytes(b"x")
    spec = _spec()
    spec["blind_levels"] = []
    with pytest.raises(ValueError, match="complete blind_levels"):
        build_selected_state_packet_from_capture(capture_path=capture, spec=spec)


def test_missing_or_empty_capture_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_selected_state_packet_from_capture(capture_path=tmp_path / "missing.bin", spec=_spec())
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="capture file is empty"):
        build_selected_state_packet_from_capture(capture_path=empty, spec=_spec())
