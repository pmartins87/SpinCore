#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tools" / "diagnose_post15315_k12_instability.py").read_text(encoding="utf-8")


def test_diagnostic_is_non_promotional_and_exact_k12():
    assert 'if probs.shape[0] != 12' in SOURCE
    assert '"frozen_k12_failure_is_immutable": True' in SOURCE
    assert '"promotion_decision_performed": False' in SOURCE
    assert '"training_performed": False' in SOURCE
    assert '"member_selection_performed": False' in SOURCE
    assert '"member_weighting_performed": False' in SOURCE
    assert '"threshold_change_performed": False' in SOURCE
    assert '"seed_change_performed": False' in SOURCE
    assert '"external_strength_benchmark_performed": False' in SOURCE
    assert 'str(gate.get("k12_stability_result")) != "FAIL"' in SOURCE


def test_diagnostic_reuses_only_closed_common_banks():
    assert 'post15315_k12_stability_bank.npz' in SOURCE
    assert 'post15315_k8_stability_bank.npz' in SOURCE
    assert 'np.array_equal(legal8, legal)' in SOURCE
    assert 'max_first8_diff > 1e-6' in SOURCE
    assert 'stability._balanced_splits(12)' in SOURCE


def test_diagnostic_decomposes_instability_without_selection():
    assert '"action_dispersion"' in SOURCE
    assert '"state_dispersion_concentration"' in SOURCE
    assert '"top_15_pairwise_disagreements"' in SOURCE
    assert '"worst_10_balanced_splits_by_broad_mean_tv"' in SOURCE
    assert '"worst_10_balanced_splits_by_trips_fold_mean_delta"' in SOURCE
    assert '"k8_to_k12_ensemble_shift"' in SOURCE
    assert '"original8_vs_extension4"' in SOURCE
