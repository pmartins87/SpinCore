"""Historical exact selected-state profile retained for audit compatibility.\n\nThe active SpinCore product target is the offline, stake-invariant simulator\nprofile in ``spincore.simulator_profile``.  This legacy module must not make\nnominal buy-in or currency select policy for that active target.\n"""\n\nfrom __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable
from urllib.parse import urlparse


PROFILE_SCHEMA = "SPINCORE_R8_PRODUCTION_PROFILE_V3"
SUPPORTED_DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")
_ALLOWED_WEB_HOSTS = ("ggpoker.com",)
_EVIDENCE_SCOPES = ("GLOBAL_GAME", "SELECTED_PROFILE_STATE")
_ALLOWED_PROVEN_FIELDS = frozenset(
    {
        "table_size",
        "currency",
        "buy_in_minor_units",
        "multiplier",
        "starting_chips_per_player",
        "blind_levels",
        "payout_share_by_place",
        "tournament_fee_fraction",
    }
)
_STATE_BOUND_FIELDS = frozenset(
    {
        "table_size",
        "buy_in_minor_units",
        "multiplier",
        "starting_chips_per_player",
        "blind_levels",
        "payout_share_by_place",
    }
)
_REQUIRED_PROFILE_EVIDENCE_FIELDS = frozenset(
    {
        "table_size",
        "buy_in_minor_units",
        "multiplier",
        "starting_chips_per_player",
        "blind_levels",
        "payout_share_by_place",
        "tournament_fee_fraction",
    }
)


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
    """First-party evidence with an explicit semantic scope.

    A dynamic official URL is provenance, not proof that the values rendered by
    a crawler belong to a particular buy-in/multiplier.  V3 therefore requires
    every evidence item to declare what fields it proves and, for any
    profile-state-dependent fact, the exact selected 3-max state it was
    captured from.
    """

    source_kind: str
    locator: str
    observed_at_utc: str
    scope: str
    proven_fields: tuple[str, ...]
    bound_table_size: int | None = None
    bound_buy_in_minor_units: int | None = None
    bound_multiplier: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.source_kind not in ("OFFICIAL_WEB", "OFFICIAL_CLIENT_CAPTURE", "OFFICIAL_RULE_DOCUMENT"):
            raise ValueError("production evidence must be an approved first-party source kind")
        locator = self.locator.strip()
        if not locator:
            raise ValueError("production evidence locator is required")
        if not self.observed_at_utc.strip():
            raise ValueError("production evidence observation time is required")
        if self.source_kind == "OFFICIAL_WEB":
            parsed = urlparse(locator)
            host = (parsed.hostname or "").lower()
            if parsed.scheme != "https" or not any(host == root or host.endswith("." + root) for root in _ALLOWED_WEB_HOSTS):
                raise ValueError("OFFICIAL_WEB evidence must be an HTTPS GGPoker first-party URL")

        if self.scope not in _EVIDENCE_SCOPES:
            raise ValueError("production evidence scope must be GLOBAL_GAME or SELECTED_PROFILE_STATE")
        fields = tuple(str(x) for x in self.proven_fields)
        if not fields:
            raise ValueError("production evidence must name at least one proven field")
        if len(set(fields)) != len(fields):
            raise ValueError("production evidence proven_fields cannot contain duplicates")
        unknown = set(fields) - _ALLOWED_PROVEN_FIELDS
        if unknown:
            raise ValueError(f"production evidence contains unsupported proven fields: {sorted(unknown)}")

        bindings = (self.bound_table_size, self.bound_buy_in_minor_units, self.bound_multiplier)
        if self.scope == "GLOBAL_GAME":
            if any(x is not None for x in bindings):
                raise ValueError("GLOBAL_GAME evidence cannot carry selected-state bindings")
            forbidden = set(fields) & _STATE_BOUND_FIELDS
            if forbidden:
                raise ValueError(
                    "GLOBAL_GAME evidence cannot prove selected-state fields: "
                    + ", ".join(sorted(forbidden))
                )
        else:
            if self.bound_table_size is None or self.bound_table_size <= 0:
                raise ValueError("SELECTED_PROFILE_STATE evidence requires bound_table_size")
            if self.bound_buy_in_minor_units is None or self.bound_buy_in_minor_units <= 0:
                raise ValueError("SELECTED_PROFILE_STATE evidence requires bound_buy_in_minor_units")
            if self.bound_multiplier is None or self.bound_multiplier <= 0:
                raise ValueError("SELECTED_PROFILE_STATE evidence requires bound_multiplier")


@dataclass(frozen=True)
class ProductionProfile:
    """Exact game/economic identity used to select a production policy.

    R7 pilot constants must never be promoted into this object merely because
    they allowed a validation run to execute. Construction is intentionally
    fail-closed: a production profile contains no optional economic/structural
    fields and must cite first-party evidence that is explicitly bound to the
    selected state for every state-dependent production constant.

    `payout_share_by_place` is normalized to the total tournament prize pool.
    This avoids accidentally feeding raw buy-in multiples into the learning
    utility while still binding the profile to the payout shape that affects
    ICM. `buy_in_minor_units` and `currency` remain part of profile identity
    because the official Spin & Gold prize/multiplier menu can vary by buy-in.
    """

    platform: str
    game_family: str
    table_size: int
    currency: str
    buy_in_minor_units: int
    multiplier: int
    starting_chips_per_player: int
    blind_levels: tuple[BlindLevel, ...]
    payout_share_by_place: tuple[float, ...]
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
        if self.currency != "USD":
            raise ValueError("current GGPoker Spin & Gold production profile requires USD")
        if self.buy_in_minor_units <= 0:
            raise ValueError("buy-in minor units must be positive")
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")
        if self.starting_chips_per_player <= 0:
            raise ValueError("starting chips must be positive")
        if not self.blind_levels:
            raise ValueError("complete blind structure is required")
        if len(self.payout_share_by_place) != self.table_size:
            raise ValueError("payout-share vector must have one entry per finishing place")
        payouts = tuple(float(x) for x in self.payout_share_by_place)
        if any(not math.isfinite(x) or x < 0.0 for x in payouts) or payouts[0] <= 0.0:
            raise ValueError("payout shares must be finite/non-negative with positive first prize")
        if any(payouts[i] < payouts[i + 1] for i in range(len(payouts) - 1)):
            raise ValueError("payout shares must be non-increasing by finishing place")
        if not math.isclose(sum(payouts), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("payout shares must sum to exactly one prize pool within 1e-12")
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
        self._validate_evidence_binding()

    def _validate_evidence_binding(self) -> None:
        covered: set[str] = set()
        expected_binding = (self.table_size, self.buy_in_minor_units, self.multiplier)
        for evidence in self.evidence:
            fields = set(evidence.proven_fields)
            if evidence.scope == "SELECTED_PROFILE_STATE":
                actual_binding = (
                    evidence.bound_table_size,
                    evidence.bound_buy_in_minor_units,
                    evidence.bound_multiplier,
                )
                if actual_binding != expected_binding:
                    raise ValueError(
                        "selected-state production evidence binding does not match profile "
                        f"(expected table/buy-in/multiplier={expected_binding}, got {actual_binding})"
                    )
            covered.update(fields)

        missing = _REQUIRED_PROFILE_EVIDENCE_FIELDS - covered
        if missing:
            raise ValueError(
                "production evidence does not prove all required profile fields: "
                + ", ".join(sorted(missing))
            )

    def semantic_payload(self) -> dict:
        """Canonical policy-selection identity; provenance timestamps are separate."""
        return {
            "schema": PROFILE_SCHEMA,
            "platform": self.platform,
            "game_family": self.game_family,
            "table_size": self.table_size,
            "currency": self.currency,
            "buy_in_minor_units": self.buy_in_minor_units,
            "multiplier": self.multiplier,
            "starting_chips_per_player": self.starting_chips_per_player,
            "blind_levels": [asdict(x) for x in self.blind_levels],
            "payout_share_by_place": [float(x) for x in self.payout_share_by_place],
            "tournament_fee_fraction": float(self.tournament_fee_fraction),
            "ruleset_id": self.ruleset_id,
            "action_abstraction_id": self.action_abstraction_id,
            "utility_model_id": self.utility_model_id,
            "learning_profile_id": self.learning_profile_id,
        }

    @property
    def profile_id(self) -> str:
        raw = json.dumps(self.semantic_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        return "spinprofile-v3:" + hashlib.sha256(raw).hexdigest()

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
            currency=str(data["currency"]),
            buy_in_minor_units=int(data["buy_in_minor_units"]),
            multiplier=int(data["multiplier"]),
            starting_chips_per_player=int(data["starting_chips_per_player"]),
            blind_levels=tuple(BlindLevel(**row) for row in data["blind_levels"]),
            payout_share_by_place=tuple(float(x) for x in data["payout_share_by_place"]),
            tournament_fee_fraction=float(data["tournament_fee_fraction"]),
            ruleset_id=str(data["ruleset_id"]),
            action_abstraction_id=str(data["action_abstraction_id"]),
            utility_model_id=str(data["utility_model_id"]),
            learning_profile_id=str(data["learning_profile_id"]),
            evidence=tuple(
                ProductionEvidence(
                    source_kind=str(row["source_kind"]),
                    locator=str(row["locator"]),
                    observed_at_utc=str(row["observed_at_utc"]),
                    scope=str(row["scope"]),
                    proven_fields=tuple(str(x) for x in row["proven_fields"]),
                    bound_table_size=(None if row.get("bound_table_size") is None else int(row["bound_table_size"])),
                    bound_buy_in_minor_units=(None if row.get("bound_buy_in_minor_units") is None else int(row["bound_buy_in_minor_units"])),
                    bound_multiplier=(None if row.get("bound_multiplier") is None else int(row["bound_multiplier"])),
                    note=str(row.get("note", "")),
                )
                for row in data["evidence"]
            ),
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
