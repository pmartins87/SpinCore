from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import r7_5_4_strategic_preflight as preflight


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _complete_tree(root: Path) -> None:
    v = root / "validation"
    _write(
        v / "R7_5_3_REPRESENTATION_ABLATION_RESULT.json",
        {
            "schema": preflight.REP_SCHEMA,
            "r7_5_3_representation_ablation_pass": True,
            "selected_candidate": "C4_V2_H3_RECLUSTERED_184",
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
    )
    _write(
        v / "R7_5_4_ACTION_STRUCTURAL_AUDIT.json",
        {
            "schema": preflight.STRUCT_SCHEMA,
            "structural_gate_pass": True,
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
    )
    _write(
        v / "R7_5_4_UNCERTAINTY_EQUIVALENCE.json",
        {
            "schema": preflight.UNCERTAINTY_SCHEMA,
            "uncertainty_equivalence_pass": True,
            "maximum_abs_difference": 0.0,
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
    )
    for name, schema in (
        ("R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT.json", preflight.PRECOMMIT_V1_SCHEMA),
        ("R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V2.json", preflight.PRECOMMIT_V2_SCHEMA),
        ("R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V3.json", preflight.PRECOMMIT_V3_SCHEMA),
        ("R7_5_4_TRAINING_IMPLEMENTATION_FREEZE.json", preflight.TRAINING_FREEZE_SCHEMA),
    ):
        _write(v / name, {"schema": schema, "ready_for_tables": False})


def test_preflight_allows_only_initial_160_when_all_durable_dependencies_pass(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is True
    assert result["selected_representation"] == "C4_V2_H3_RECLUSTERED_184"
    assert result["production_training_authorized"] is False
    assert result["ready_for_tables"] is False

    higher = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=320)
    assert higher["ready_to_start"] is False
    assert higher["checks"]["initial_postflop_level"]["pass"] is False


def test_preflight_fails_closed_when_uncertainty_evidence_is_missing(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    (tmp_path / "validation" / "R7_5_4_UNCERTAINTY_EQUIVALENCE.json").unlink()
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is False
    assert result["checks"]["uncertainty_evidence_available"]["pass"] is False


def test_preflight_fails_closed_on_ready_for_tables_contamination(tmp_path: Path) -> None:
    _complete_tree(tmp_path)
    path = tmp_path / "validation" / "R7_5_4_ACTION_STRUCTURAL_AUDIT.json"
    payload = json.loads(path.read_text())
    payload["ready_for_tables"] = True
    _write(path, payload)
    result = preflight.evaluate(tmp_path, phase="R7_5_4A_POSTFLOP", root_level=160)
    assert result["ready_to_start"] is False
    assert result["checks"]["structural_not_table_authority"]["pass"] is False
