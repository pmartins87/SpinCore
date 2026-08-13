from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Sequence


SCHEMA = "SPINCORE_STRATEGIC_ACTION_SENTINELS_V1"
NUM_ACTIONS = 6
_SUPPORTED_DOMAINS = {"TRUE_HEADS_UP", "THREE_HANDED"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: str, name: str) -> str:
    value = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class SentinelActionObservation:
    """One deterministic policy observation at a named canonical/extreme state."""

    sentinel_id: str
    profile_id: str
    domain: str
    model_sha256: str
    observation_sha256: str
    legal_actions: tuple[int, ...]
    policy: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.sentinel_id.strip():
            raise ValueError("sentinel_id is required")
        if not self.profile_id.strip():
            raise ValueError("profile_id is required")
        if self.domain not in _SUPPORTED_DOMAINS:
            raise ValueError("unsupported sentinel domain")
        object.__setattr__(self, "model_sha256", _require_sha256(self.model_sha256, "model_sha256"))
        object.__setattr__(
            self,
            "observation_sha256",
            _require_sha256(self.observation_sha256, "observation_sha256"),
        )
        legal = tuple(int(a) for a in self.legal_actions)
        if not legal or len(set(legal)) != len(legal):
            raise ValueError("legal_actions must be non-empty and unique")
        if any(a < 0 or a >= NUM_ACTIONS for a in legal):
            raise ValueError("legal action outside frozen six-action abstraction")
        if tuple(sorted(legal)) != legal:
            raise ValueError("legal_actions must be sorted for canonical fingerprints")
        object.__setattr__(self, "legal_actions", legal)

        policy = tuple(float(x) for x in self.policy)
        if len(policy) != NUM_ACTIONS:
            raise ValueError("sentinel policy must contain six action probabilities")
        if any(not math.isfinite(x) or x < 0.0 or x > 1.0 for x in policy):
            raise ValueError("sentinel policy contains invalid probability")
        legal_set = set(legal)
        if any(policy[a] != 0.0 for a in range(NUM_ACTIONS) if a not in legal_set):
            raise ValueError("sentinel policy assigns mass to an illegal action")
        legal_total = sum(policy[a] for a in legal)
        if not math.isclose(legal_total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("sentinel legal policy mass must sum to one")
        object.__setattr__(self, "policy", policy)

    @property
    def action_fingerprint(self) -> str:
        payload = {
            "sentinel_id": self.sentinel_id,
            "profile_id": self.profile_id,
            "domain": self.domain,
            "model_sha256": self.model_sha256,
            "observation_sha256": self.observation_sha256,
            "legal_actions": list(self.legal_actions),
            # float.hex preserves the exact Python binary floating-point value.
            "policy_hex": [x.hex() for x in self.policy],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class IntegrityExpectation:
    sentinel_id: str
    expected_action_fingerprint: str

    def __post_init__(self) -> None:
        if not self.sentinel_id.strip():
            raise ValueError("sentinel_id is required")
        object.__setattr__(
            self,
            "expected_action_fingerprint",
            _require_sha256(self.expected_action_fingerprint, "expected_action_fingerprint"),
        )


@dataclass(frozen=True)
class PlausibilityRule:
    """A precommitted action-level plausibility bound.

    Bounds are intentionally data, not hard-coded poker opinions. They must be
    frozen only after the exact production profile/payout semantics are known.
    """

    sentinel_id: str
    action: int
    rationale: str
    min_probability: float | None = None
    max_probability: float | None = None

    def __post_init__(self) -> None:
        if not self.sentinel_id.strip():
            raise ValueError("sentinel_id is required")
        if self.action < 0 or self.action >= NUM_ACTIONS:
            raise ValueError("plausibility action outside frozen abstraction")
        if not self.rationale.strip():
            raise ValueError("plausibility rationale is required")
        if self.min_probability is None and self.max_probability is None:
            raise ValueError("plausibility rule needs at least one bound")
        for name, value in (
            ("min_probability", self.min_probability),
            ("max_probability", self.max_probability),
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0 or value > 1.0):
                raise ValueError(f"{name} must be finite in [0, 1]")
        if (
            self.min_probability is not None
            and self.max_probability is not None
            and self.min_probability > self.max_probability
        ):
            raise ValueError("plausibility minimum exceeds maximum")


def evaluate_strategic_sentinels(
    *,
    observations: Sequence[SentinelActionObservation],
    required_sentinel_ids: Sequence[str],
    integrity_expectations: Sequence[IntegrityExpectation],
    plausibility_rules: Sequence[PlausibilityRule],
) -> dict:
    """Evaluate deterministic integrity plus precommitted strategic plausibility.

    The gate fails closed when plausibility rules are absent or do not cover
    every required sentinel. This prevents integrity-only checks from being
    mistaken for strategic validation.
    """

    required = tuple(str(x).strip() for x in required_sentinel_ids)
    if not required or any(not x for x in required) or len(set(required)) != len(required):
        raise ValueError("required_sentinel_ids must be non-empty and unique")

    by_id: dict[str, SentinelActionObservation] = {}
    for observation in observations:
        if observation.sentinel_id in by_id:
            raise ValueError(f"duplicate sentinel observation: {observation.sentinel_id}")
        by_id[observation.sentinel_id] = observation

    expected: dict[str, str] = {}
    for row in integrity_expectations:
        if row.sentinel_id in expected:
            raise ValueError(f"duplicate integrity expectation: {row.sentinel_id}")
        expected[row.sentinel_id] = row.expected_action_fingerprint

    missing_observations = [sid for sid in required if sid not in by_id]
    missing_integrity = [sid for sid in required if sid not in expected]
    integrity_rows = []
    for sid in required:
        actual = by_id[sid].action_fingerprint if sid in by_id else None
        exp = expected.get(sid)
        integrity_rows.append(
            {
                "sentinel_id": sid,
                "expected_action_fingerprint": exp,
                "actual_action_fingerprint": actual,
                "exact_match": bool(actual is not None and exp is not None and actual == exp),
            }
        )
    integrity_complete = not missing_observations and not missing_integrity
    integrity_pass = bool(integrity_complete and all(row["exact_match"] for row in integrity_rows))

    rule_ids = {rule.sentinel_id for rule in plausibility_rules}
    unknown_rule_ids = sorted(rule_ids - set(required))
    if unknown_rule_ids:
        raise ValueError(f"plausibility rules reference non-required sentinels: {unknown_rule_ids}")
    missing_plausibility = [sid for sid in required if sid not in rule_ids]
    plausibility_complete = bool(plausibility_rules and not missing_plausibility and not missing_observations)

    plausibility_rows = []
    for rule in plausibility_rules:
        obs = by_id.get(rule.sentinel_id)
        probability = obs.policy[rule.action] if obs is not None else None
        legal = bool(obs is not None and rule.action in set(obs.legal_actions))
        min_ok = bool(
            probability is not None
            and (rule.min_probability is None or probability >= rule.min_probability)
        )
        max_ok = bool(
            probability is not None
            and (rule.max_probability is None or probability <= rule.max_probability)
        )
        passed = bool(legal and min_ok and max_ok)
        plausibility_rows.append(
            {
                "sentinel_id": rule.sentinel_id,
                "action": int(rule.action),
                "probability": probability,
                "min_probability": rule.min_probability,
                "max_probability": rule.max_probability,
                "rationale": rule.rationale,
                "action_is_legal": legal,
                "pass": passed,
            }
        )
    plausibility_pass = bool(
        plausibility_complete and all(row["pass"] for row in plausibility_rows)
    )

    gate_pass = bool(integrity_pass and plausibility_pass)
    return {
        "schema": SCHEMA,
        "required_sentinel_ids": list(required),
        "missing_observations": missing_observations,
        "integrity": {
            "complete": integrity_complete,
            "pass": integrity_pass,
            "missing_expectations": missing_integrity,
            "rows": integrity_rows,
        },
        "plausibility": {
            "complete": plausibility_complete,
            "pass": plausibility_pass,
            "missing_rule_sentinels": missing_plausibility,
            "rows": plausibility_rows,
        },
        "strategic_sentinel_gate_pass": gate_pass,
        "integrity_only_can_authorize_release": False,
        "ready_for_tables": False,
    }
