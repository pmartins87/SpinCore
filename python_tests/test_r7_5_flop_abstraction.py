from __future__ import annotations

import csv
import json
from pathlib import Path

from spincore.flop_abstraction import (
    RSCard,
    audit_legacy_184_mapping,
    audit_legacy_184_summary,
    encode_spin_card_id,
    flop53_key,
    parse_flop_text,
    reference_counts,
    suit_isomorphic_key,
)


def _ids(text: str) -> tuple[int, int, int]:
    return tuple(encode_spin_card_id(c) for c in parse_flop_text(text))


def test_reference_class_counts_are_exact() -> None:
    summary = reference_counts()
    assert summary["physical_flops"] == 22100
    assert summary["suit_isomorphic_classes"] == 1755
    assert summary["flop53_classes"] == 53


def test_user_example_qs_jh_2h_is_nsnsfd() -> None:
    assert flop53_key(_ids("QsJh2h")) == "NSNSFD"


def test_suit_permutation_and_card_order_do_not_change_exact_iso_key() -> None:
    a = _ids("QsJh2h")
    b = _ids("QdJc2c")
    c = tuple(reversed(a))
    assert suit_isomorphic_key(a) == suit_isomorphic_key(b)
    assert suit_isomorphic_key(a) == suit_isomorphic_key(c)


def test_structurally_different_flops_remain_distinct_in_exact_iso_reference() -> None:
    assert suit_isomorphic_key(_ids("QsJh2h")) != suit_isomorphic_key(_ids("QhJh2h"))
    assert suit_isomorphic_key(_ids("KsKd2h")) != suit_isomorphic_key(_ids("Kc2d2h"))


def test_legacy_184_summary_shape(tmp_path: Path) -> None:
    p = tmp_path / "classes_184_resumo.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["centroide", "rotulo", "quantidade_flops"])
        w.writeheader()
        # Synthetic shape-only fixture: 184 unique parseable centroids is not
        # practical to hand-construct here, so verify failure on an incomplete file.
        w.writerow({"centroide": "QsJh2h", "rotulo": "x", "quantidade_flops": 22100})
    audit = audit_legacy_184_summary(p)
    assert audit["rows"] == 1
    assert audit["physical_flops_claimed"] == 22100
    assert audit["summary_shape_pass"] is False


def test_mapping_audit_detects_incomplete_json(tmp_path: Path) -> None:
    p = tmp_path / "184Flops.json"
    p.write_text(json.dumps({"QsJh2h": "QsJh2h"}), encoding="utf-8")
    audit = audit_legacy_184_mapping(p)
    assert audit["normalized_physical_flops"] == 1
    assert audit["complete_22100_pass"] is False
    assert audit["missing_physical_flops"] == 22099
