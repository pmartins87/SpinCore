from __future__ import annotations

import json
from pathlib import Path

import pytest

from spincore.r7_5_referee_rng import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_REPLICATES,
    PREFIX,
    canonical_key,
    keyed_uniform01,
    ordered_candidate_pair,
    paired_bootstrap_mean_ci,
    paired_difference,
    sample_discrete_with_uniform,
    stable_seed64,
)

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "validation" / "R7_5_4_REFEREE_EVALUATION_SEMANTIC_FREEZE_20260814.json"


def test_rng_contract_matches_frozen_referee_semantics() -> None:
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert payload["schema"] == "SPINCORE_R7_5_4_REFEREE_EVALUATION_SEMANTIC_FREEZE_V1"
    assert payload["stable_rng_derivation"]["hash"] == "SHA256"
    assert payload["stable_rng_derivation"]["candidate_id_in_rng_key"] is False
    assert payload["dense_referee_omission"]["bootstrap"]["replicates"] == BOOTSTRAP_REPLICATES == 2000
    assert payload["dense_referee_omission"]["bootstrap"]["confidence_level"] == BOOTSTRAP_CONFIDENCE == 0.95
    assert PREFIX == "SpinCore|R7.5.4A|REFV1"


def test_seed_and_uniform_are_stable_stateless_and_field_sensitive() -> None:
    fields = ("crossplay", "TRUE_HEADS_UP", 1737995611, 1817694185, 42, "deck")
    seed1 = stable_seed64(*fields)
    seed2 = stable_seed64(*fields)
    assert seed1 == seed2
    assert seed1 != stable_seed64(*fields[:-1], "policy")
    u1 = keyed_uniform01(*fields)
    u2 = keyed_uniform01(*fields)
    assert u1 == u2
    assert 0.0 < u1 < 1.0
    assert canonical_key(*fields).startswith(PREFIX + "|")


def test_candidate_identity_is_not_needed_for_poker_rng_keys() -> None:
    # Same hand/seat/decision fields necessarily produce the same uniform for any
    # candidate because candidate_id is absent from the API call by construction.
    fields = ("crossplay", "THREE_HANDED", 645939859, 1617273629, 99, 2, 4)
    assert keyed_uniform01(*fields) == keyed_uniform01(*fields)
    with pytest.raises(ValueError):
        canonical_key("bad|candidate")


def test_discrete_sampling_respects_legal_mask_and_same_uniform() -> None:
    probabilities = (0.0, 0.2, 0.0, 0.3, 0.0, 0.5)
    legal = (1, 3, 5)
    assert sample_discrete_with_uniform(probabilities, legal, 0.1) == 1
    assert sample_discrete_with_uniform(probabilities, legal, 0.3) == 3
    assert sample_discrete_with_uniform(probabilities, legal, 0.9) == 5
    assert sample_discrete_with_uniform(probabilities, legal, 0.9) == 5

    with pytest.raises(ValueError, match="illegal action"):
        sample_discrete_with_uniform((0.1, 0.9), (1,), 0.5)


def test_paired_bootstrap_is_exactly_reproducible_and_chunk_invariant() -> None:
    values = tuple((index - 50) / 1000.0 for index in range(101))
    fields = ("bootstrap", "omission", 160, "TRUE_HEADS_UP", 1737995611, 1817694185, "PF0_CONTROL_33_75_AI")
    a = paired_bootstrap_mean_ci(values, seed_fields=fields, chunk_size=17)
    b = paired_bootstrap_mean_ci(values, seed_fields=fields, chunk_size=64)
    assert a == b
    assert a["replicates"] == 2000
    assert a["confidence"] == 0.95
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]


def test_pairwise_bootstrap_pair_id_is_lexically_canonical() -> None:
    assert ordered_candidate_pair("PF4", "PF1") == "PF1::PF4"
    assert ordered_candidate_pair("PF1", "PF4") == "PF1::PF4"
    left = (0.1, 0.2, 0.3)
    right = (0.0, 0.1, 0.1)
    assert paired_difference(left, right) == pytest.approx((0.1, 0.1, 0.2))
