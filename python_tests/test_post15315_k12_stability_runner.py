from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_post15315_rebuild_stability_k12 as k12


def test_k12_member_set_is_exactly_one_through_twelve():
    assert k12.MEMBERS == tuple(range(1, 13))
    assert k12.K12_COUNT == 12


def test_k12_stability_has_no_extension_beyond_twelve():
    passed = k12.branch_decision({"rebuild_stability_pass": True})
    failed = k12.branch_decision({"rebuild_stability_pass": False})
    assert passed["aggregate_quality_authorized"] is True
    assert failed["aggregate_quality_authorized"] is False
    assert "Do not add members" in failed["next_gate"]


def test_k12_regenerates_same_fixed_common_bank_and_keeps_thresholds_external():
    source = (ROOT / "tools" / "run_post15315_rebuild_stability_k12.py").read_text(
        encoding="utf-8"
    )
    assert "k8stab.BANK_EPISODES" in source
    assert "master_seed=k8stab.BANK_SEED" in source
    assert "stability.evaluate(" in source
    assert 'rebuild_stability_gate"]["thresholds"]' not in source
    assert "thresholds =" not in source


def test_k12_requires_k8_authorization_and_extension_pass():
    source = (ROOT / "tools" / "run_post15315_rebuild_stability_k12.py").read_text(
        encoding="utf-8"
    )
    assert "ext.validate_k8_gate(" in source
    assert "validate_extension(" in source
    assert "members_constructed" in source
    assert '"strength_benchmark_performed": False' in source
