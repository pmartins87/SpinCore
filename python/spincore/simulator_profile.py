from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable
from urllib.parse import urlparse


SIMULATOR_PROFILE_SCHEMA = "SPINCORE_R8_UNIVERSAL_3MAX_SIMULATOR_PROFILE_V1"
SUPPORTED_DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")
PRODUCT_TARGET = "OFFLINE_3MAX_SIMULATOR_NOT_REAL_MONEY_CLIENT"
REFERENCE_GAME_FAMILY = "GGPOKER_SPIN_AND_GOLD_3MAX_RULE_REFERENCE"
_ALLOWED_WEB_HOSTS = ("ggpoker.com",)
_ALLOWED_PROVEN_FIELDS = frozenset(
    {
        "table_size",
        "buy_in_catalog",
        "multiplier_dependent_structure",
        "payout_structure",
        "deck_and_shuffle",
        "tournament_fee_fraction",
        "same_hand_elimination_tiebreak",
        "no_make_a_deal",
        "default_starting_stack",
        "time_based_blind_cadence",
    }
)
_REQUIRED_OFFICIAL_FIELDS = frozenset(
    {
        "table_size",
        "buy_in_catalog",
        "multiplier_dependent_structure",
        "payout_structure",
        "deck_and_shuffle",
        "tournament_fee_fraction",
        "same_hand_elimination_tiebreak",
        "no_make_a_deal",
    }
)


@dataclass(frozen=True)
class HandsBlindLevel:
    small_blind: int
    big_blind: int
    ante: int = 0
    hands_per_level: int = 1

    def __post_init__(self) -> None:
        if self.small_blind <= 0 or self.big_blind <= 0:
            raise ValueError("blinds must be positive")
        if self.big_blind < self.small_blind:
            raise ValueError("big blind cannot be smaller than small blind")
        if self.ante < 0:
            raise ValueError("ante cannot be negative")
        if self.hands_per_level <= 0:
            raise ValueError("hands_per_level must be positive")


@dataclass(frozen=True)
class OfficialRuleReference:
    locator: str
    observed_at_utc: str
    proven_fields: tuple[str, ...]
    note: str = ""

    def __post_init__(self) -> None:
        parsed = urlparse(self.locator.strip())
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not any(
            host == root or host.endswith("." + root) for root in _ALLOWED_WEB_HOSTS
        ):
            raise ValueError("official rule reference must be an HTTPS GGPoker first-party URL")
        if not self.observed_at_utc.strip():
            raise ValueError("official rule reference observation time is required")
        if not self.proven_fields:
            raise ValueError("official rule reference must prove at least one field")
        if len(set(self.proven_fields)) != len(self.proven_fields):
            raise ValueError("official rule reference fields cannot contain duplicates")
        unknown = set(self.proven_fields) - _ALLOWED_PROVEN_FIELDS
        if unknown:
            raise ValueError(f"unsupported official rule fields: {sorted(unknown)}")


@dataclass(frozen=True)
class SimulatorPresentation:
    """Display/accounting metadata that must never select a strategy policy."""

    currency: str
    nominal_buy_in_minor_units: int
    displayed_multiplier: int
    tournament_fee_fraction: float
    skin_reference: str = "GGPOKER_SPIN_AND_GOLD_STYLE"

    def __post_init__(self) -> None:
        if not self.currency.strip():
            raise ValueError("presentation currency is required")
        if self.nominal_buy_in_minor_units <= 0:
            raise ValueError("nominal buy-in must be positive")
        if self.displayed_multiplier <= 0:
            raise ValueError("displayed multiplier must be positive")
        if not (0.0 <= float(self.tournament_fee_fraction) < 1.0):
            raise ValueError("tournament fee fraction must be in [0,1)")
        if not self.skin_reference.strip():
            raise ValueError("skin reference is required")


@dataclass(frozen=True)
class UniversalThreeMaxSimulatorProfile:
    """Stake-invariant strategic identity for the offline SpinCore simulator.

    Nominal buy-in, currency, displayed multiplier, rake and skin are retained
    for presentation/accounting but excluded from policy identity.  A label
    change can never select a different policy.  A multiplier affects strategy
    only through the effective stack, blind schedule or payout vector encoded
    below.
    """

    starting_chips_per_player: int
    blind_levels: tuple[HandsBlindLevel, ...]
    payout_share_by_place: tuple[float, ...]
    ruleset_id: str
    action_abstraction_id: str
    utility_model_id: str
    learning_profile_id: str
    presentation: SimulatorPresentation
    official_references: tuple[OfficialRuleReference, ...]
    same_hand_elimination_tiebreak: str = "START_OF_HAND_STACK_THEN_LEFT_OF_BUTTON"
    make_a_deal_allowed: bool = False
    deck_size: int = 52
    shuffle_after_each_hand: bool = True
    table_size: int = 3
    blind_progression_basis: str = "COMPLETED_HANDS"

    def __post_init__(self) -> None:
        if self.table_size != 3:
            raise ValueError("universal SpinCore simulator profile requires exactly 3 players")
        if self.deck_size != 52 or self.shuffle_after_each_hand is not True:
            raise ValueError("official reference requires a 52-card deck shuffled after every hand")
        if self.blind_progression_basis != "COMPLETED_HANDS":
            raise ValueError("simulator blind progression must be based on completed hands")
        if self.starting_chips_per_player <= 0:
            raise ValueError("starting chips must be positive")
        if not self.blind_levels:
            raise ValueError("a complete hands-based blind schedule is required")
        payouts = tuple(float(x) for x in self.payout_share_by_place)
        if len(payouts) != self.table_size:
            raise ValueError("payout vector must contain first, second and third place")
        if any(not math.isfinite(x) or x < 0.0 for x in payouts) or payouts[0] <= 0.0:
            raise ValueError("payout shares must be finite/non-negative with positive first prize")
        if any(payouts[i] < payouts[i + 1] for i in range(len(payouts) - 1)):
            raise ValueError("payout shares must be non-increasing by finishing place")
        if not math.isclose(sum(payouts), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("payout shares must sum to one within 1e-12")
        if self.same_hand_elimination_tiebreak != "START_OF_HAND_STACK_THEN_LEFT_OF_BUTTON":
            raise ValueError("unsupported same-hand elimination tiebreak")
        if self.make_a_deal_allowed is not False:
            raise ValueError("official 3-max reference does not allow Make a Deal")
        for value, name in (
            (self.ruleset_id, "ruleset_id"),
            (self.action_abstraction_id, "action_abstraction_id"),
            (self.utility_model_id, "utility_model_id"),
            (self.learning_profile_id, "learning_profile_id"),
        ):
            if not value.strip():
                raise ValueError(f"{name} is required")
        covered = {field for row in self.official_references for field in row.proven_fields}
        missing = _REQUIRED_OFFICIAL_FIELDS - covered
        if missing:
            raise ValueError("official references do not cover: " + ", ".join(sorted(missing)))

    def semantic_payload(self) -> dict:
        """Only effective game/strategy semantics participate in identity."""
        return {
            "schema": SIMULATOR_PROFILE_SCHEMA,
            "product_target": PRODUCT_TARGET,
            "reference_game_family": REFERENCE_GAME_FAMILY,
            "table_size": self.table_size,
            "deck_size": self.deck_size,
            "shuffle_after_each_hand": self.shuffle_after_each_hand,
            "starting_chips_per_player": self.starting_chips_per_player,
            "blind_progression_basis": self.blind_progression_basis,
            "blind_levels": [asdict(x) for x in self.blind_levels],
            "payout_share_by_place": [float(x) for x in self.payout_share_by_place],
            "same_hand_elimination_tiebreak": self.same_hand_elimination_tiebreak,
            "make_a_deal_allowed": self.make_a_deal_allowed,
            "ruleset_id": self.ruleset_id,
            "action_abstraction_id": self.action_abstraction_id,
            "utility_model_id": self.utility_model_id,
            "learning_profile_id": self.learning_profile_id,
        }

    @property
    def profile_id(self) -> str:
        raw = json.dumps(
            self.semantic_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
        return "spinsim-v1:" + hashlib.sha256(raw).hexdigest()

    def policy_id(self, domain: str) -> str:
        if domain not in SUPPORTED_DOMAINS:
            raise ValueError("unsupported strategy domain")
        raw = f"{self.profile_id}|{domain}".encode()
        return "spinpolicy-sim-v1:" + hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict:
        out = self.semantic_payload()
        out["profile_id"] = self.profile_id
        out["presentation"] = asdict(self.presentation)
        out["official_references"] = [asdict(x) for x in self.official_references]
        out["identity_excludes"] = [
            "currency",
            "nominal_buy_in_minor_units",
            "displayed_multiplier",
            "tournament_fee_fraction",
            "skin_reference",
        ]
        out["real_money_client_integration_authorized"] = False
        out["ready_for_simulator_tables"] = False
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "UniversalThreeMaxSimulatorProfile":
        if data.get("schema") != SIMULATOR_PROFILE_SCHEMA:
            raise ValueError("wrong simulator profile schema")
        if data.get("real_money_client_integration_authorized", False) is not False:
            raise ValueError("simulator profile cannot authorize a real-money client")
        if data.get("ready_for_simulator_tables", False) is not False:
            raise ValueError("R8 contract cannot authorize simulator table release")
        presentation = data["presentation"]
        obj = cls(
            starting_chips_per_player=int(data["starting_chips_per_player"]),
            blind_levels=tuple(HandsBlindLevel(**row) for row in data["blind_levels"]),
            payout_share_by_place=tuple(float(x) for x in data["payout_share_by_place"]),
            ruleset_id=str(data["ruleset_id"]),
            action_abstraction_id=str(data["action_abstraction_id"]),
            utility_model_id=str(data["utility_model_id"]),
            learning_profile_id=str(data["learning_profile_id"]),
            presentation=SimulatorPresentation(
                currency=str(presentation["currency"]),
                nominal_buy_in_minor_units=int(presentation["nominal_buy_in_minor_units"]),
                displayed_multiplier=int(presentation["displayed_multiplier"]),
                tournament_fee_fraction=float(presentation["tournament_fee_fraction"]),
                skin_reference=str(presentation.get("skin_reference", "GGPOKER_SPIN_AND_GOLD_STYLE")),
            ),
            official_references=tuple(
                OfficialRuleReference(
                    locator=str(row["locator"]),
                    observed_at_utc=str(row["observed_at_utc"]),
                    proven_fields=tuple(str(x) for x in row["proven_fields"]),
                    note=str(row.get("note", "")),
                )
                for row in data["official_references"]
            ),
            same_hand_elimination_tiebreak=str(data["same_hand_elimination_tiebreak"]),
            make_a_deal_allowed=bool(data["make_a_deal_allowed"]),
            deck_size=int(data["deck_size"]),
            shuffle_after_each_hand=bool(data["shuffle_after_each_hand"]),
            table_size=int(data["table_size"]),
            blind_progression_basis=str(data["blind_progression_basis"]),
        )
        if str(data.get("profile_id", obj.profile_id)) != obj.profile_id:
            raise ValueError("simulator profile identity hash mismatch")
        return obj


def require_stake_invariant_policy_identity(
    profiles: Iterable[UniversalThreeMaxSimulatorProfile], domain: str
) -> str:
    """Fail if presentation-only stake variants select different policies."""
    identities = {profile.policy_id(domain) for profile in profiles}
    if len(identities) != 1:
        raise ValueError("stake variants changed strategic policy identity")
    return next(iter(identities))
