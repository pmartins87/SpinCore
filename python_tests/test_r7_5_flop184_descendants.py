from __future__ import annotations

import json
from pathlib import Path

from spincore.flop184_descendants import (
    audit_descendant,
    derive_exact_class_mapping,
    load_legacy_mapping,
)


def _fixture(path: Path) -> None:
    # Three physical suit spellings of one exact flop class. The canonical
    # spelling deliberately points to A while the majority points to B.
    path.write_text(
        json.dumps(
            {
                "2s5s7h": "2s3d7s",
                "2h5h7s": "2s4s7d",
                "2d5d7s": "2s4s7d",
            }
        ),
        encoding="utf-8",
    )


def test_canonical_input_descendant_uses_canonical_physical_spelling(tmp_path: Path) -> None:
    path = tmp_path / "184Flops.json"
    _fixture(path)
    physical = load_legacy_mapping(path)
    exact = derive_exact_class_mapping(physical, mode="canonical_input")
    assert len(exact) == 1
    assert set(exact.values()) == {"2s3d7s"}


def test_majority_descendant_minimizes_historical_assignment_changes(tmp_path: Path) -> None:
    path = tmp_path / "184Flops.json"
    _fixture(path)
    physical = load_legacy_mapping(path)
    exact = derive_exact_class_mapping(physical, mode="majority_min_change")
    assert len(exact) == 1
    assert set(exact.values()) == {"2s4s7d"}


def test_both_descendants_are_suit_invariant_by_construction(tmp_path: Path) -> None:
    path = tmp_path / "184Flops.json"
    _fixture(path)

    canonical = audit_descendant(path, mode="canonical_input")
    majority = audit_descendant(path, mode="majority_min_change")

    assert canonical["suit_permutation_invariance_pass"] is True
    assert canonical["suit_isomorphic_classes_split"] == 0
    assert canonical["physical_assignments_changed_from_legacy"] == 2

    assert majority["suit_permutation_invariance_pass"] is True
    assert majority["suit_isomorphic_classes_split"] == 0
    assert majority["physical_assignments_changed_from_legacy"] == 1
