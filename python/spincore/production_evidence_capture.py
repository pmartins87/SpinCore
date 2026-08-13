from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from .production_evidence_packet import CapturedPayouts, SelectedStateEvidencePacket
from .production_profile import BlindLevel


def sha256_file(path: Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_selected_state_packet_from_capture(
    *,
    capture_path: Path,
    spec: Mapping,
) -> SelectedStateEvidencePacket:
    """Bind explicit selected-state facts to the exact captured evidence bytes.

    This function intentionally performs no OCR and derives no poker rule from
    the image/document. The caller supplies the facts visible in the capture;
    the packet binds those facts to the capture SHA-256 and byte size, and the
    existing SelectedStateEvidencePacket contract performs fail-closed semantic
    validation.
    """

    capture_path = Path(capture_path)
    if not capture_path.is_file():
        raise FileNotFoundError(f"capture file not found: {capture_path}")
    size = int(capture_path.stat().st_size)
    if size <= 0:
        raise ValueError("capture file is empty")

    payout_values = list(spec.get("payouts_minor_units") or [])
    if len(payout_values) != 3:
        raise ValueError("spec payouts_minor_units must contain exactly three places")
    blind_rows = list(spec.get("blind_levels") or [])
    if not blind_rows:
        raise ValueError("spec must include the complete blind_levels sequence")

    return SelectedStateEvidencePacket(
        table_size=int(spec["table_size"]),
        currency=str(spec["currency"]),
        buy_in_minor_units=int(spec["buy_in_minor_units"]),
        multiplier=int(spec["multiplier"]),
        starting_chips_per_player=int(spec["starting_chips_per_player"]),
        blind_levels=tuple(BlindLevel(**dict(row)) for row in blind_rows),
        payouts=CapturedPayouts(*(int(x) for x in payout_values)),
        source_kind=str(spec["source_kind"]),
        source_locator=str(spec["source_locator"]),
        captured_at_utc=str(spec["captured_at_utc"]),
        capture_sha256=sha256_file(capture_path),
        capture_size_bytes=size,
        capture_note=str(spec.get("capture_note", "")),
    )
