#!/usr/bin/env python3
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tools" / "run_post15315_shared_corpus_variance_pilot.py").read_text(encoding="utf-8")
PREREG = (ROOT / "contracts" / "post15315_shared_corpus_variance_pilot_v2.json").read_text(encoding="utf-8")
ORIGINAL_PROTOCOL_PATH = ROOT / "contracts" / "post15315_multirebuild_preregistration.json"


def test_pilot_is_research_only_and_preserves_closed_fail():
    assert '"closed_k12_failure_is_immutable": True' in SOURCE
    assert '"promotion_decision_performed": False' in SOURCE
    assert '"external_strength_benchmark_performed": False' in SOURCE
    assert '"member_selection_performed": False' in SOURCE
    assert '"member_weighting_performed": False' in SOURCE
    assert '"k_greater_than_12_performed": False' in SOURCE
    assert 'str(gate.get("k12_stability_result")) != "FAIL"' in SOURCE


def test_common_corpus_and_fresh_bank_are_frozen():
    assert 'range(153160001, 153160009)' in SOURCE
    assert 'range(153161001, 153161009)' in SOURCE
    assert 'int(bank.get("seed", -1)) != 153159501' in SOURCE
    assert 'int(bank.get("forced_three_handed_episodes", -1)) != 40000' in SOURCE
    assert 'member = 1' in SOURCE
    assert 'member_cache.cache_path(k8_dir, member)' in SOURCE
    assert 'list(payload["ordinary"]["train"])' in SOURCE
    assert 'list(payload["augmentation"]["train"]) + list(payload["augmentation"]["holdout"])' in SOURCE
    assert 'list(payload["specialist"]["train"]) + list(payload["specialist"]["holdout"])' in SOURCE
    assert 'for member in range(1, 13)' not in SOURCE.split("def load_fixed_corpus",1)[1].split("def save_pilot_member",1)[0]


def test_attribution_bands_are_frozen_before_results():
    assert 'ratio_mean <= 0.70 and ratio_p95 <= 0.70' in SOURCE
    assert 'ratio_mean >= 0.90 and ratio_p95 >= 0.90' in SOURCE
    assert 'COLLECTION_CORPUS_HETEROGENEITY_LIKELY_DOMINANT' in SOURCE
    assert 'FIT_ARCHITECTURE_DOMINANT' in SOURCE
    assert 'MIXED_VARIANCE' in SOURCE
    assert '"SPINCORE_POST15315_SHARED_CORPUS_VARIANCE_PILOT_V2"' in PREREG
    assert '"anchor_member": 1' in PREREG
    assert '"No pooling or enlargement of the fixed member-01 training corpus."' in PREREG
    assert '"PREREGISTERED_RESEARCH_ONLY_NOT_PROMOTIONAL"' in PREREG
    assert '"seed": 153159501' in PREREG
    assert '"forced_three_handed_episodes": 40000' in PREREG


def test_original_protocol_identity_keeps_git_blob_and_sha256_distinct():
    data = ORIGINAL_PROTOCOL_PATH.read_bytes()
    git_blob_sha = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
    assert git_blob_sha == "5fd7667926c6112310fe5eca61afff396ad218de"
    assert 'EXPECTED_PROTOCOL_BLOB_SHA = "5fd7667926c6112310fe5eca61afff396ad218de"' in SOURCE
    assert "if git_blob_sha1(protocol_path) != EXPECTED_PROTOCOL_BLOB_SHA:" in SOURCE
    assert "protocol_sha = sha256(protocol_path)" in SOURCE
    assert "sha256(protocol_path) != EXPECTED_PROTOCOL" not in SOURCE
    assert '"original_protocol_sha256": protocol_sha' in SOURCE
    assert '"original_protocol_git_blob_sha1": EXPECTED_PROTOCOL_BLOB_SHA' in SOURCE
