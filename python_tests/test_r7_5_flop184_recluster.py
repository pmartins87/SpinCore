from __future__ import annotations

from spincore.flop184_recluster import (
    H3_FEATURE_SCHEMA,
    H3_SCHEMA,
    audit_h3_mapping,
    build_h3_mapping,
    exact_iso_flop_keys,
    hard_stratum,
    mapping_sha256,
)

EXPECTED_H3_MAPPING_SHA256 = "2c83cf993bcc4003223d184bd6f5584720b23cf04b95e6db69f84b09a86a64d0"


def test_h3_exact_reference_has_1755_classes_and_14_hard_strata() -> None:
    keys = exact_iso_flop_keys()
    assert len(keys) == 1755
    assert len({hard_stratum(key) for key in keys}) == 14


def test_h3_mapping_is_frozen_complete_and_suit_invariant() -> None:
    mapping = build_h3_mapping()
    assert len(mapping) == 1755
    assert len(set(mapping.values())) == 184
    assert mapping_sha256(mapping) == EXPECTED_H3_MAPPING_SHA256

    audit = audit_h3_mapping(mapping)
    assert audit["schema"] == H3_SCHEMA
    assert audit["feature_schema"] == H3_FEATURE_SCHEMA
    assert audit["mapping_sha256"] == EXPECTED_H3_MAPPING_SHA256
    assert audit["complete_1755_pass"] is True
    assert audit["complete_22100_pass"] is True
    assert audit["suit_permutation_invariance_pass"] is True
    assert audit["representative_count_pass"] is True
    assert audit["hard_stratum_homogeneity_pass"] is True
    assert audit["hard_stratum_mismatches"] == 0
    assert audit["physical_flops"] == 22100
    assert audit["exact_bucket_size_min"] == 1
    assert audit["exact_bucket_size_median"] == 9.0
    assert audit["exact_bucket_size_max"] == 23
    assert audit["physical_bucket_size_min"] == 4
    assert audit["physical_bucket_size_median"] == 108.0
    assert audit["physical_bucket_size_max"] == 384
