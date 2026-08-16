#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "validation/R9_R12_SPNNIV3_MIGRATION_DEBT_AUDIT_20260816.json"
ADJUDICATION = ROOT / "validation/R9_R10_SPNNIV3_GATE_DESIGN_ADJUDICATION_20260816.md"
V3_MODELS = ROOT / "python/spincore_nn/models_v3_final.py"
LEGACY_MODELS = ROOT / "python/spincore_nn/models.py"
LEGACY_CODEC = ROOT / "python/spincore_nn/codec.py"
TOP_LEVEL = ROOT / "python/spincore_nn/__init__.py"


def assigned_int(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value = node.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, int):
                        return int(value.value)
    raise AssertionError(f"{name} not found as an integer constant in {path}")


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["schema"] == "SPINCORE_R9_R12_SPNNIV3_MIGRATION_DEBT_AUDIT_V1"
    assert audit["status"] == "AUDIT_COMPLETE_WITH_PREEXECUTION_DEBT"
    assert audit["canonical_current_representation_lineage"]["production_winner_selected"] is False
    assert audit["canonical_current_representation_lineage"]["intended_successor_schema"] == "SPNNIV3"
    assert audit["canonical_current_representation_lineage"]["universal_action_width"] == 10
    for key in (
        "r9_execution_authorized",
        "r10_execution_authorized",
        "r11_execution_authorized",
        "r12_execution_authorized",
        "ready_for_tables",
    ):
        assert audit[key] is False, f"pre-execution guard drift: {key} must remain false"

    v3_width = assigned_int(V3_MODELS, "UNIVERSAL_ACTION_COUNT")
    assert v3_width == 10, f"unexpected SPNNIV3 universal action width: {v3_width}"
    v3_text = V3_MODELS.read_text(encoding="utf-8")
    assert "shape [batch,10]" in v3_text
    assert "pack_padded_sequence" in v3_text

    legacy_models = LEGACY_MODELS.read_text(encoding="utf-8")
    legacy_codec = LEGACY_CODEC.read_text(encoding="utf-8")
    top_level = TOP_LEVEL.read_text(encoding="utf-8")
    assert "nn.Linear(cfg.head_hidden, 6)" in legacy_models
    assert "SPNNIV1" in legacy_codec and "hlen>32" in legacy_codec.replace(" ", "")
    assert '(".models", "AdvantageNet")' in top_level
    assert '(".models", "AveragePolicyNet")' in top_level

    adjudication = ADJUDICATION.read_text(encoding="utf-8")
    required = (
        "SPNNIV3",
        "universal action width is **10**",
        "No production R10 path may silently fall back to SPNNIV1",
        "not eligible as implicit production defaults",
    )
    for token in required:
        assert token in adjudication, f"missing downstream adjudication clause: {token}"

    print(
        json.dumps(
            {
                "status": "PASS",
                "spnniv3_action_width": v3_width,
                "legacy_v1_quarantined_by_adjudication": True,
                "downstream_execution_authorized": False,
                "ready_for_tables": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
