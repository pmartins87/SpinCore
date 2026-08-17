#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "STATUS.json"
VERSION = ROOT / "VERSION.json"
AUDIT = ROOT / "validation/GOVERNANCE_STALENESS_AUDIT_20260816.json"


def main() -> None:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    version = json.loads(VERSION.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    expected_version = "1.12.1-recovery.32"
    expected_gate = "IN_PROGRESS_X4_CHANCE_COVERAGE_READMISSION"
    expected_candidates = [
        "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL",
        "H3_HYBRID_EXACT_SEMANTIC_FINAL",
    ]

    assert status["version"] == expected_version
    assert version["version"] == expected_version
    assert status["current_gate"] == "R7.5"
    assert status["r7_5"]["status"] == expected_gate
    assert status["r7_5"]["r7_5_3"] == expected_gate
    assert status["r7_5"]["incumbent_neural_schema"] == "SPNNIV1"
    assert status["r7_5"]["candidate_successor_neural_schema"] == "SPNNIV3"
    assert status["r7_5"]["candidate_representations"] == expected_candidates
    assert status["r7_5"]["representation_winner"] is None
    assert status["r7_5"]["candidate_universal_action_width"] == 10
    assert status["r7_5"]["complete_variable_length_history_required"] is True
    assert status["r7_5"]["production_training_authorized"] is False
    assert status["r7_5"]["production_training_blocker"] is True
    assert status["r8"]["ready_to_start_official_training"] is False
    assert status["ready_for_tables"] is False

    assert version["neural_schema"] == "SPNNIV1"
    assert version["candidate_successor_neural_schema"] == "SPNNIV3"
    assert version["candidate_successor_action_width"] == 10
    assert version["candidate_representation_winner"] is None
    assert version["production_training_authorized"] is False
    assert version["ready_for_tables"] is False

    safe = audit["safe_future_refresh_rules"]
    assert safe["production_or_incumbent_neural_schema_until_r7_5_5"] == version["neural_schema"]
    assert safe["candidate_successor_neural_schema"] == version["candidate_successor_neural_schema"]
    assert safe["r7_5_3_status"] == status["r7_5"]["r7_5_3"]
    assert safe["representation_winner"] is None
    assert safe["production_training_authorized"] is False
    assert safe["ready_for_tables"] is False

    print(
        json.dumps(
            {
                "status": "PASS",
                "incumbent_neural_schema": version["neural_schema"],
                "candidate_successor_neural_schema": version["candidate_successor_neural_schema"],
                "representation_winner": None,
                "production_training_authorized": False,
                "ready_for_tables": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
