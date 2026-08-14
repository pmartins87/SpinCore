from __future__ import annotations

import json

import pytest

from spincore.r7_5_action_320_contract import validate_card_symmetry_parent_320


def _write(tmp_path, *, winner="S0_V1_FROZEN_CONTROL", status="PASS", consistent=True):
    validation = tmp_path / "validation"
    validation.mkdir(exist_ok=True)
    payload = {
        "schema": "SPINCORE_R7_5_3B_CARD_SYMMETRY_RESULT_V1",
        "status": status,
        "winner_id": winner,
        "representation_decision": {
            "existing_R7_5_4A_evidence_representation_consistent": consistent,
        },
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    (validation / "R7_5_3B_CARD_SYMMETRY_RESULT.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_320_card_symmetry_guard_requires_durable_result(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="gated by R7.5.3B"):
        validate_card_symmetry_parent_320(tmp_path)


def test_320_card_symmetry_guard_accepts_only_frozen_v1_winner(tmp_path) -> None:
    _write(tmp_path)
    result = validate_card_symmetry_parent_320(tmp_path)
    assert result["winner_id"] == "S0_V1_FROZEN_CONTROL"


def test_320_card_symmetry_guard_blocks_representation_change(tmp_path) -> None:
    _write(tmp_path, winner="S1_V1_CARD_SYMMETRY_CANON", consistent=False)
    with pytest.raises(ValueError, match="changed the representation"):
        validate_card_symmetry_parent_320(tmp_path)


def test_320_card_symmetry_guard_blocks_nonpass_or_consistency_drift(tmp_path) -> None:
    _write(tmp_path, status="BLOCKED")
    with pytest.raises(ValueError, match="requires R7.5.3B PASS"):
        validate_card_symmetry_parent_320(tmp_path)

    _write(tmp_path, consistent=False)
    with pytest.raises(ValueError, match="consistency declaration mismatch"):
        validate_card_symmetry_parent_320(tmp_path)
