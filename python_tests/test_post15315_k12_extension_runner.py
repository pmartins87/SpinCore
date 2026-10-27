from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_post15315_multirebuild_k12_extension as ext


def test_extension_member_set_is_exactly_preregistered_9_to_12():
    assert ext.MEMBERS == (9, 10, 11, 12)


def test_k8_fail_gate_is_mandatory_for_extension():
    source = (ROOT / "tools" / "run_post15315_multirebuild_k12_extension.py").read_text(
        encoding="utf-8"
    )
    assert "k12_extension_authorized" in source
    assert 'k8_stability_result")) != "FAIL"' in source
    assert "mechanically_clean" in source
    assert "surface_coverage_minima" in source


def test_extension_reuses_exact_k8_execution_shape_without_strength():
    source = (ROOT / "tools" / "run_post15315_multirebuild_k12_extension.py").read_text(
        encoding="utf-8"
    )
    assert '"collection_workers": 4' in source
    assert '"collection_threads_per_worker": 2' in source
    assert '"fit_threads": 8' in source
    assert '"strength_benchmark_performed": False' in source
    assert '"stability_evaluation_performed": False' in source


def test_extension_does_not_change_protocol_or_thresholds():
    source = (ROOT / "tools" / "run_post15315_multirebuild_k12_extension.py").read_text(
        encoding="utf-8"
    )
    assert "conditional_extension_rebuild_count" in source
    assert "maximum_rebuild_count" in source
    assert 'rebuild_stability_gate"]["thresholds"]' not in source
    assert "thresholds =" not in source
