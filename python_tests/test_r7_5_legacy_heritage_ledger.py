from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "validation"


def test_every_readable_prior_attempt_source_has_explicit_heritage_disposition() -> None:
    manifest = json.loads(
        (VALIDATION / "R7_5_LEGACY_HERITAGE_SOURCE_MANIFEST_20260814.json").read_text(
            encoding="utf-8"
        )
    )
    ledger = json.loads(
        (VALIDATION / "R7_5_LEGACY_HERITAGE_INTEGRATION_LEDGER_20260814.json").read_text(
            encoding="utf-8"
        )
    )
    assert ledger["schema"] == "SPINCORE_R7_5_LEGACY_HERITAGE_INTEGRATION_LEDGER_V1"
    expected = {
        str(row["source"])
        for row in manifest["prior_attempt_archive"]["readable_sources"]
    }
    entries = ledger["entries"]
    actual = {str(row["source"]) for row in entries}
    assert len(entries) == len(actual), "duplicate legacy source in heritage ledger"
    assert actual == expected

    for row in entries:
        assert row["reviewed_full"] is True
        assert row["best_of"] and all(str(x).strip() for x in row["best_of"])
        assert row["preserve"] and all(str(x).strip() for x in row["preserve"])
        assert str(row["destination"]).strip()
        assert str(row["status"]).strip()
        assert row["current_evidence"] and all(str(x).strip() for x in row["current_evidence"])
        assert row["never_inherit_as_truth"] and all(
            str(x).strip() for x in row["never_inherit_as_truth"]
        )

    assert ledger["strategic_output"] is False
    assert ledger["production_training_authorized"] is False
    assert ledger["ready_for_tables"] is False
