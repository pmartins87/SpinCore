from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

from spincore.r7_5_representation_v3_referee_states import HeldoutV3State
from spincore.solver import ResolvedExactAction

SCHEMA = "SPINCORE_R7_5_3C_HELDOUT_V3_STATES_V1"


def _row(item: HeldoutV3State) -> dict:
    return {
        "domain": item.domain,
        "evaluation_seed": int(item.evaluation_seed),
        "state_index": int(item.state_index),
        "hand_index": int(item.hand_index),
        "scenario_index": int(item.scenario_index),
        "deck_seed": int(item.deck_seed),
        "action_path": [int(x) for x in item.action_path],
        "actor": int(item.actor),
        "observation_v3_b64": base64.b64encode(item.observation_v3).decode("ascii"),
        "active_mask": int(item.active_mask),
        "legal_slots": [int(x) for x in item.legal_slots],
        "exact_actions": [
            [int(action.action_type), int(action.amount_to)]
            for action in item.exact_actions
        ],
    }


def _item(row: dict) -> HeldoutV3State:
    return HeldoutV3State(
        domain=str(row["domain"]),
        evaluation_seed=int(row["evaluation_seed"]),
        state_index=int(row["state_index"]),
        hand_index=int(row["hand_index"]),
        scenario_index=int(row["scenario_index"]),
        deck_seed=int(row["deck_seed"]),
        action_path=tuple(int(x) for x in row["action_path"]),
        actor=int(row["actor"]),
        observation_v3=base64.b64decode(row["observation_v3_b64"], validate=True),
        active_mask=int(row["active_mask"]),
        legal_slots=tuple(int(x) for x in row["legal_slots"]),
        exact_actions=tuple(
            ResolvedExactAction(int(action_type), int(amount_to))
            for action_type, amount_to in row["exact_actions"]
        ),
    )


def state_payload_sha256(states: tuple[HeldoutV3State, ...]) -> str:
    rows = [_row(item) for item in states]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def save_heldout_v3_artifact(
    path: str | Path,
    states: tuple[HeldoutV3State, ...],
    *,
    generator_execution_sha: str,
) -> dict:
    if not states:
        raise ValueError("cannot save empty heldout artifact")
    domains = {item.domain for item in states}
    evaluation_seeds = {int(item.evaluation_seed) for item in states}
    if len(domains) != 1 or len(evaluation_seeds) != 1:
        raise ValueError("heldout artifact must contain one domain/evaluation seed")
    if [item.state_index for item in states] != list(range(len(states))):
        raise ValueError("heldout state indices must be contiguous")
    payload = {
        "schema": SCHEMA,
        "generator_execution_sha": str(generator_execution_sha),
        "domain": next(iter(domains)),
        "evaluation_seed": next(iter(evaluation_seeds)),
        "count": len(states),
        "candidate_independent": True,
        "training_seed_independent": True,
        "state_payload_sha256": state_payload_sha256(states),
        "states": [_row(item) for item in states],
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with gzip.open(path, "wb", compresslevel=6, mtime=0) as handle:
        handle.write(encoded)
    return {key: value for key, value in payload.items() if key != "states"}


def load_heldout_v3_artifact(
    path: str | Path,
    *,
    expected_domain: str,
    expected_evaluation_seed: int,
    expected_count: int = 2048,
) -> tuple[HeldoutV3State, ...]:
    with gzip.open(Path(path), "rb") as handle:
        payload = json.loads(handle.read().decode("utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("wrong heldout V3 artifact schema")
    if payload.get("domain") != str(expected_domain):
        raise ValueError("heldout V3 artifact domain mismatch")
    if int(payload.get("evaluation_seed", -1)) != int(expected_evaluation_seed):
        raise ValueError("heldout V3 artifact evaluation-seed mismatch")
    if int(payload.get("count", -1)) != int(expected_count):
        raise ValueError("heldout V3 artifact count mismatch")
    if payload.get("candidate_independent") is not True or payload.get("training_seed_independent") is not True:
        raise ValueError("heldout V3 artifact lost independence contract")
    if bool(payload.get("production_training_authorized")) or bool(payload.get("ready_for_tables")):
        raise ValueError("heldout V3 artifact illegally authorizes production/table use")
    states = tuple(_item(row) for row in payload["states"])
    if len(states) != int(expected_count):
        raise ValueError("heldout V3 artifact row count mismatch")
    if state_payload_sha256(states) != payload.get("state_payload_sha256"):
        raise ValueError("heldout V3 artifact payload hash mismatch")
    return states
