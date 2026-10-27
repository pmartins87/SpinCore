from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_post15315_rebuild_stability_k8 as runner


def test_k8_stability_bank_contract_is_frozen():
    assert runner.K == 8
    assert runner.BANK_SEED == 153159001
    assert runner.BANK_EPISODES == 80000
    assert runner.TARGET_ITERATION == 15315


def test_k12_extension_only_follows_clean_k8_stability_fail():
    passed = runner.branch_decision({"rebuild_stability_pass": True})
    failed = runner.branch_decision({"rebuild_stability_pass": False})

    assert passed["k8_stability_result"] == "PASS"
    assert passed["k12_extension_authorized"] is False

    assert failed["k8_stability_result"] == "FAIL"
    assert failed["k12_extension_authorized"] is True
    assert "members 9-12" in failed["next_gate"]


def test_stability_runner_uses_common_teacher_bank_and_no_strength_path():
    source = (ROOT / "tools" / "run_post15315_rebuild_stability_k8.py").read_text(
        encoding="utf-8"
    )
    assert "collect.collect_fresh_split(" in source
    assert "master_seed=BANK_SEED" in source
    assert "BANK_EPISODES" in source
    assert "rebuild policies do not influence state generation" in source
    assert "strength_benchmark_performed" in source
    assert "DeepCrusherR8Policy" not in source
    assert "evaluate_deepcrusher" not in source


def test_stability_runner_applies_frozen_member_specialist_route():
    source = (ROOT / "tools" / "run_post15315_rebuild_stability_k8.py").read_text(
        encoding="utf-8"
    )
    assert "route_specialist" in source
    assert "surface_trips_fold" in source
    assert "np.where(use_sp, sp, fp)" in source
    assert "base_fullpool_sha256" in source
