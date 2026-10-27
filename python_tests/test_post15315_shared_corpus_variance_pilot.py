#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tools" / "run_post15315_shared_corpus_variance_pilot.py").read_text(encoding="utf-8")
PREREG = (ROOT / "contracts" / "post15315_shared_corpus_variance_pilot_v1.json").read_text(encoding="utf-8")


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
    assert 'list(payload["ordinary"]["train"])' in SOURCE
    assert 'list(payload["augmentation"]["train"]) + list(payload["augmentation"]["holdout"])' in SOURCE
    assert 'list(payload["specialist"]["train"]) + list(payload["specialist"]["holdout"])' in SOURCE


def test_attribution_bands_are_frozen_before_results():
    assert 'ratio_mean <= 0.70 and ratio_p95 <= 0.70' in SOURCE
    assert 'ratio_mean >= 0.90 and ratio_p95 >= 0.90' in SOURCE
    assert 'CORPUS_SUPPORT_DOMINANT_HYPOTHESIS_SUPPORTED' in SOURCE
    assert 'FIT_ARCHITECTURE_DOMINANT' in SOURCE
    assert 'MIXED_VARIANCE' in SOURCE
    assert '"PREREGISTERED_RESEARCH_ONLY_NOT_PROMOTIONAL"' in PREREG
    assert '"seed": 153159501' in PREREG
    assert '"forced_three_handed_episodes": 40000' in PREREG
