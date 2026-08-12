from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable


PROFILE_SCHEMA = "SPINCORE_R8_PRODUCTION_PROFILE_V1"
SUPPORTED_DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")


@dataclass(frozen=True)
class BlindLevel:
    small_blind: int
    big_blind: int
    ante: int = 0
    duration_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.small_blind <= 0 or self.big_blind <= 0:
            raise ValueError("blinds must be positive")
        if self.big_blind < self.small_blind:
            raise ValueError("big blind cannot be smaller than small blind")
        if self.ante < 0:
            raise ValueError("ante cannot be negative")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("blind level duration must be positive when specified")


@dataclass(frozen=True)
class ProductionEvidence:
    source_kind: str
    locator: str
    observed_at_utc: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.source_kind not in ("OFFICIAL_WEB", "OFFICIAL_CLIENT_CAPTURE", "OFFICIAL_RULE_DOCUMENT"):
            raise ValueError("production evidence must be an approved first-party source kind")
        if not self.locator.strip():
            raise ValueError("production evidence locator is required")
        if not self.observed_at_utc.strip():
            raise ValueError("production evidence observation time is required")


@dataclass(frozen=True)
class ProductionProfile:
    """Exact game/economic identity used to select a production policy.

    R7 pilot constants must never be promoted into this object merely because
    they allowed a validation run to execute.  Construction is intentionally
    fail-closed: a production profile contains no optional economic/structural
    fields and must cite first-party evidence.
    """

    platform: str
    game_family: str
    table_size: int
    multiplier: int
    starting_chips_per_player: int
    blind_levels: tuple[BlindLevel, ...]
    payout_by_place: tuple[float, ...]
    tournament_fee_fraction: float
    ruleset_id: str
    action_abstraction_id: str
    utility_model_id: str
    learning_profile_id: str
    evidence: tuple[ProductionEvidence, ...]

    def __post_init__(self) -> None:
        if self.platform != "GGPOKER":
            raise ValueError("R8 production profile currently supports GGPOKER only")
        if not self.game_family.strip():
            raise ValueError("game_family is required")
        if self.table_size != 3:
            raise ValueError("current SpinCore production path requires 3-max profile")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")
        if self.starting_chips_per_player <= 0:
            raise ValueError("starting chips must be positive")
        if not self.blind_levels:
            raise ValueError("complete blind structure is required")
        if len(self.payout_by_place) != self.table_size:
            raise ValueError("payout vector must have one entry per finishing place")
        payouts = tuple(float(x) for x in self.payout_by_place)
        if any(x < 0.0 for x in payouts) or payouts[0] <= 0.0:
            raise ValueError("payouts must be non-negative with positive first prize")
        if any(payouts[i] < payouts[i + 1] for i in range(len(payouts) - 1)):
            raise ValueError("payouts must be non-increasing by finishing place")
        if not (0.0 <= float(self.tournament_fee_fraction) < 1.0):
            raise ValueError("tournament fee fraction must be in [0,1)")
        for value, name in (
            (self.ruleset_id, "ruleset_id"),
            (self.action_abstraction_id, "action_abstraction_id"),
            (self.utility_model_id, "utility_model_id"),
            (self.learning_profile_id, "learning_profile_id"),
        ):
            if not value.strip():
                raise ValueError(f"{name} is required")
        if not self.evidence:
            raise ValueError("first-party production evidence is mandatory")

    def semantic_payload(self) -> dict:
        """Canonical policy-selection identity; provenance is deliberately separate."""
        return {
            "schema": PROFILE_SCHEMA,
            "platform": self.platform,
            "game_family": self.game_family,
            "table_size": self.table_size,
            "multiplier": self.multiplier,
            "starting_chips_per_player": self.starting_chips_per_player,
            "blind_levels": [asdict(x) for x in self.blind_levels],
            "payout_by_place": [float(x) for x in self.payout_by_place],
            "tournament_fee_fraction": float(self.tournament_fee_fraction),
            "ruleset_id": self.ruleset_id,
            "action_abstraction_id": self.action_abstraction_id,
            "utility_model_id": self.utility_model_id,
            "learning_profile_id": self.learning_profile_id,
        }

    @property
    def profile_id(self) -> str:
        raw = json.dumps(self.semantic_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        return "spinprofile-v1:" + hashlib.sha256(raw).hexdigest()

    def policy_id(self, domain: str) -> str:
        if domain not in SUPPORTED_DOMAINS:
            raise ValueError("unsupported strategy domain")
        raw = f"{self.profile_id}|{domain}".encode()
        return "spinpolicy-v1:" + hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict:
        out = self.semantic_payload()
        out["profile_id"] = self.profile_id
        out["evidence"] = [asdict(x) for x in self.evidence]
        out["ready_for_tables"] = False
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "ProductionProfile":
        if data.get("schema") != PROFILE_SCHEMA:
            raise ValueError("wrong production profile schema")
        obj = cls(
            platform=str(data["platform"]),
            game_family=str(data["game_family"]),
            table_size=int(data["table_size"]),
            multiplier=int(data["multiplier"]),
            starting_chips_per_player=int(data["starting_chips_per_player"]),
            blind_levels=tuple(BlindLevel(**row) for row in data["blind_levels"]),
            payout_by_place=tuple(float(x) for x in data["payout_by_place"]),
            tournament_fee_fraction=float(data["tournament_fee_fraction"]),
            ruleset_id=str(data["ruleset_id"]),
            action_abstraction_id=str(data["action_abstraction_id"]),
            utility_model_id=str(data["utility_model_id"]),
            learning_profile_id=str(data["learning_profile_id"]),
            evidence=tuple(ProductionEvidence(**row) for row in data["evidence"]),
        )
        if str(data.get("profile_id", obj.profile_id)) != obj.profile_id:
            raise ValueError("production profile identity hash mismatch")
        if data.get("ready_for_tables", False) is not False:
            raise ValueError("R8 profile cannot authorize table use")
        return obj


def require_unique_policy_identities(profiles: Iterable[ProductionProfile], domains: Iterable[str]) -> dict[str, tuple[str, str]]:
    """Return policy-id -> (profile-id, domain), rejecting any accidental collision."""
    out: dict[str, tuple[str, str]] = {}
    for profile in profiles:
        for domain in domains:
            policy_id = profile.policy_id(domain)
            identity = (profile.profile_id, domain)
            previous = out.setdefault(policy_id, identity)
            if previous != identity:
                raise RuntimeError("production policy identity collision")
    return out
