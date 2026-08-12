from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Iterable

from .production_profile import BlindLevel, ProductionEvidence


SCHEMA = "SPINCORE_R8_SELECTED_STATE_EVIDENCE_PACKET_V1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CapturedPayouts:
    """Raw client-displayed payout amounts for one selected Spin state.

    Values are intentionally stored as integer minor units. Production shares
    are derived from the captured amounts themselves; no assumption such as
    `buy_in * multiplier == prize_pool` is made here.
    """

    first_minor_units: int
    second_minor_units: int
    third_minor_units: int

    def __post_init__(self) -> None:
        rows = (self.first_minor_units, self.second_minor_units, self.third_minor_units)
        if any(int(x) < 0 for x in rows):
            raise ValueError("captured payouts cannot be negative")
        if self.first_minor_units <= 0:
            raise ValueError("captured first-place payout must be positive")
        if not (self.first_minor_units >= self.second_minor_units >= self.third_minor_units):
            raise ValueError("captured payouts must be non-increasing by place")
        if sum(rows) <= 0:
            raise ValueError("captured prize pool must be positive")

    @property
    def normalized_shares(self) -> tuple[float, float, float]:
        values = (
            int(self.first_minor_units),
            int(self.second_minor_units),
            int(self.third_minor_units),
        )
        total = float(sum(values))
        # Keep every share as the direct ratio of an exact captured integer.
        # Do not derive the final place as 1-a-b: for legitimate zero payouts,
        # binary floating subtraction can create a tiny negative value.
        return tuple(float(x) / total for x in values)


@dataclass(frozen=True)
class SelectedStateEvidencePacket:
    """Auditable evidence for exactly one GGPoker 3-Max buy-in/multiplier state."""

    table_size: int
    currency: str
    buy_in_minor_units: int
    multiplier: int
    starting_chips_per_player: int
    blind_levels: tuple[BlindLevel, ...]
    payouts: CapturedPayouts
    source_kind: str
    source_locator: str
    captured_at_utc: str
    capture_sha256: str
    capture_size_bytes: int
    capture_note: str = ""

    def __post_init__(self) -> None:
        if self.table_size != 3:
            raise ValueError("selected-state packet must be 3-Max")
        if self.currency != "USD":
            raise ValueError("selected-state packet currently requires USD")
        if self.buy_in_minor_units <= 0:
            raise ValueError("selected-state buy-in must be positive")
        if self.multiplier <= 0:
            raise ValueError("selected-state multiplier must be positive")
        if self.starting_chips_per_player <= 0:
            raise ValueError("selected-state starting chips must be positive")
        if not self.blind_levels:
            raise ValueError("selected-state complete blind structure is required")
        if self.source_kind not in ("OFFICIAL_CLIENT_CAPTURE", "OFFICIAL_RULE_DOCUMENT"):
            raise ValueError("state packet requires official client/rule-document evidence")
        if not self.source_locator.strip():
            raise ValueError("selected-state source locator is required")
        if not self.captured_at_utc.strip():
            raise ValueError("selected-state capture time is required")
        if not _SHA256_RE.fullmatch(self.capture_sha256):
            raise ValueError("capture_sha256 must be 64 lowercase hex characters")
        if self.capture_size_bytes <= 0:
            raise ValueError("captured evidence must contain bytes")
        shares = self.payouts.normalized_shares
        if any(not math.isfinite(x) or x < 0.0 for x in shares):
            raise ValueError("normalized captured payout shares are invalid")
        if not math.isclose(sum(shares), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("normalized captured payout shares do not sum to one")

    @property
    def packet_id(self) -> str:
        canonical = json.dumps(self.semantic_payload(), sort_keys=True, separators=(",", ":")).encode()
        return "spinevidence-v1:" + hashlib.sha256(canonical).hexdigest()

    def semantic_payload(self) -> dict:
        return {
            "schema": SCHEMA,
            "table_size": self.table_size,
            "currency": self.currency,
            "buy_in_minor_units": self.buy_in_minor_units,
            "multiplier": self.multiplier,
            "starting_chips_per_player": self.starting_chips_per_player,
            "blind_levels": [
                {
                    "small_blind": int(row.small_blind),
                    "big_blind": int(row.big_blind),
                    "ante": int(row.ante),
                    "duration_seconds": row.duration_seconds,
                }
                for row in self.blind_levels
            ],
            "payouts_minor_units": [
                self.payouts.first_minor_units,
                self.payouts.second_minor_units,
                self.payouts.third_minor_units,
            ],
            "payout_share_by_place": list(self.payouts.normalized_shares),
            "source_kind": self.source_kind,
            "source_locator": self.source_locator,
            "captured_at_utc": self.captured_at_utc,
            "capture_sha256": self.capture_sha256,
            "capture_size_bytes": self.capture_size_bytes,
        }

    def to_production_evidence(self) -> ProductionEvidence:
        return ProductionEvidence(
            source_kind=self.source_kind,
            locator=self.source_locator,
            observed_at_utc=self.captured_at_utc,
            scope="SELECTED_PROFILE_STATE",
            proven_fields=(
                "table_size",
                "buy_in_minor_units",
                "multiplier",
                "starting_chips_per_player",
                "blind_levels",
                "payout_share_by_place",
            ),
            bound_table_size=self.table_size,
            bound_buy_in_minor_units=self.buy_in_minor_units,
            bound_multiplier=self.multiplier,
            note=(
                f"selected-state capture sha256={self.capture_sha256} "
                f"size={self.capture_size_bytes}; {self.capture_note}"
            ).strip(),
        )

    def to_dict(self) -> dict:
        out = self.semantic_payload()
        out["packet_id"] = self.packet_id
        out["capture_note"] = self.capture_note
        out["ready_for_tables"] = False
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "SelectedStateEvidencePacket":
        if data.get("schema") != SCHEMA:
            raise ValueError("wrong selected-state evidence packet schema")
        payout_values = list(data["payouts_minor_units"])
        if len(payout_values) != 3:
            raise ValueError("selected-state payout vector must contain exactly three places")
        obj = cls(
            table_size=int(data["table_size"]),
            currency=str(data["currency"]),
            buy_in_minor_units=int(data["buy_in_minor_units"]),
            multiplier=int(data["multiplier"]),
            starting_chips_per_player=int(data["starting_chips_per_player"]),
            blind_levels=tuple(BlindLevel(**row) for row in data["blind_levels"]),
            payouts=CapturedPayouts(*(int(x) for x in payout_values)),
            source_kind=str(data["source_kind"]),
            source_locator=str(data["source_locator"]),
            captured_at_utc=str(data["captured_at_utc"]),
            capture_sha256=str(data["capture_sha256"]),
            capture_size_bytes=int(data["capture_size_bytes"]),
            capture_note=str(data.get("capture_note", "")),
        )
        expected_shares = tuple(float(x) for x in data.get("payout_share_by_place", ()))
        if expected_shares and expected_shares != obj.payouts.normalized_shares:
            raise ValueError("captured payout-share normalization mismatch")
        if str(data.get("packet_id", obj.packet_id)) != obj.packet_id:
            raise ValueError("selected-state evidence packet identity mismatch")
        if data.get("ready_for_tables", False) is not False:
            raise ValueError("R8 evidence packet cannot authorize table use")
        return obj


def require_unique_selected_states(
    packets: Iterable[SelectedStateEvidencePacket],
) -> dict[tuple[int, int, int], str]:
    """Reject two different captures claiming authority for the same state."""
    out: dict[tuple[int, int, int], str] = {}
    for packet in packets:
        state = (packet.table_size, packet.buy_in_minor_units, packet.multiplier)
        previous = out.setdefault(state, packet.packet_id)
        if previous != packet.packet_id:
            raise ValueError(
                "conflicting selected-state evidence packets for "
                f"table/buy-in/multiplier={state}"
            )
    return out
