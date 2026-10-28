#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tools" / "run_post15315_canonical_corpus_pilot.py").read_text(encoding="utf-8")
PREREG = json.loads((ROOT / "contracts" / "post15315_canonical_corpus_pilot_v1.json").read_text(encoding="utf-8"))


def test_canonical_sources_and_size_matched_control_are_frozen():
    assert PREREG["canonical_sources"]["anchor_members"] == [1, 4, 7, 10]
    fixed = PREREG["treatments"]["fixed_size_canonical"]
    assert fixed["ordinary_train_samples"] == 26798
    assert fixed["novel_strong_samples"] == 2645
    assert fixed["fullpool_train_samples"] == 29443
    assert fixed["specialist_unique_strong_samples"] == 4125
    assert "canonical_unique" in SOURCE
    assert "duplicate canonical state/legal key has teacher-target drift" in SOURCE
    assert "rank_key" in SOURCE


def test_fit_seeds_match_v2_control_and_bank_is_fresh():
    fit = PREREG["fit_design"]
    assert fit["fullpool_fit_seeds"] == list(range(153160001, 153160009))
    assert fit["specialist_fit_seeds"] == list(range(153161001, 153161009))
    bank = PREREG["fresh_confirmation_bank"]
    assert bank["seed"] == 153159503
    assert bank["forced_three_handed_episodes"] == 100000
    assert 'int(bank.get("seed", -1)) != 153159503' in SOURCE
    assert 'int(bank.get("forced_three_handed_episodes", -1)) != 100000' in SOURCE


def test_three_groups_use_unchanged_stability_evaluator():
    assert 'control_report = stability.evaluate(control_probs, legal, s72, trips, high, protocol)' in SOURCE
    assert 'fixed_report = stability.evaluate(fixed_probs, legal, s72, trips, high, protocol)' in SOURCE
    assert 'expanded_report = stability.evaluate(expanded_probs, legal, s72, trips, high, protocol)' in SOURCE
    assert 'coverage = k8stab.validate_surface_coverage(protocol, s72, trips, high)' in SOURCE
    assert '"BANK_SENSITIVITY_STOP"' in SOURCE
    assert '"STANDARDIZATION_SUFFICIENT"' in SOURCE
    assert '"EXPANSION_REQUIRED"' in SOURCE
    assert '"CANONICAL_CORPUS_INSUFFICIENT"' in SOURCE


def test_research_only_guards_are_explicit():
    assert '"closed_k12_failure_is_immutable": True' in SOURCE
    assert '"promotion_decision_performed": False' in SOURCE
    assert '"external_strength_benchmark_performed": False' in SOURCE
    assert '"member_selection_performed": False' in SOURCE
    assert '"member_weighting_performed": False' in SOURCE
    assert '"k_greater_than_12_performed": False' in SOURCE
    assert '"new_teacher_collection_performed": False' in SOURCE
    assert "No external strength benchmark." in json.dumps(PREREG)
    assert "No new teacher collection." in json.dumps(PREREG)
