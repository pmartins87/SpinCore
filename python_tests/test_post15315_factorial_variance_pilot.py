#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tools" / "run_post15315_factorial_variance_pilot.py").read_text(encoding="utf-8")
PREREG = json.loads((ROOT / "contracts" / "post15315_factorial_variance_pilot_v1.json").read_text(encoding="utf-8"))


def test_factorial_design_is_crossed_and_not_result_selected():
    assert PREREG["corpus_factor"]["anchor_members"] == [1, 4, 7, 10]
    assert "Deterministic stride-3 anchors" in PREREG["corpus_factor"]["selection_rule"]
    assert PREREG["fit_factor"]["matched_fit_seed_ids"] == [1, 2, 3, 4]
    assert PREREG["fit_factor"]["fullpool_fit_seeds"] == [153162001, 153162002, 153162003, 153162004]
    assert PREREG["fit_factor"]["specialist_fit_seeds"] == [153163001, 153163002, 153163003, 153163004]
    assert "for anchor in anchors:" in SOURCE
    assert "for j, fit_id in enumerate(fit_ids):" in SOURCE
    assert 'p.get("attribution_result") != "MIXED_VARIANCE"' in SOURCE


def test_factorial_bank_and_classification_are_frozen():
    bank = PREREG["fresh_research_bank"]
    assert bank["seed"] == 153159502
    assert bank["forced_three_handed_episodes"] == 40000
    assert 'int(bank.get("seed", -1)) != 153159502' in SOURCE
    assert 'ratio_mean >= 1.25 and ratio_p95 >= 1.25' in SOURCE
    assert 'ratio_mean <= 0.80 and ratio_p95 <= 0.80' in SOURCE
    assert '"CORPUS_CONDITIONED_LARGER"' in SOURCE
    assert '"FIT_SEED_LARGER"' in SOURCE
    assert '"MIXED_INTERACTION"' in SOURCE


def test_factorial_is_research_only_and_preserves_closed_failure():
    assert '"closed_k12_failure_is_immutable": True' in SOURCE
    assert '"promotion_decision_performed": False' in SOURCE
    assert '"external_strength_benchmark_performed": False' in SOURCE
    assert '"member_selection_performed": False' in SOURCE
    assert '"member_weighting_performed": False' in SOURCE
    assert '"k_greater_than_12_performed": False' in SOURCE
    assert "two_way_decomposition" in SOURCE
    assert "within-anchor" in SOURCE
    assert "matched fit-seed" in SOURCE
