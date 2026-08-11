from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = _load("r7_4_domain_preflight_worker_test", TOOLS / "r7_4_domain_preflight_worker.py")
orchestrator = _load("run_r7_4_domain_preflight_test", TOOLS / "run_r7_4_domain_preflight.py")


def test_r7_4_structural_preflight_exercises_hu_and_three_handed(tmp_path, monkeypatch):
    out = tmp_path / "r7_4.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "r7_4_domain_preflight_worker.py",
            "--solver",
            str(ROOT / "build" / "libspincore_solver_c.so"),
            "--out",
            str(out),
        ],
    )
    assert preflight.main() == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["schema"] == "SPINCORE_R7_4_DOMAIN_PREFLIGHT_V1"
    assert d["hu_case_count"] == 6
    assert d["three_handed_case_count"] == 9
    assert d["hu_domains"] == [1]
    assert d["three_handed_domains"] == [0]
    assert d["all_chip_zero_sum"] is True
    assert d["all_icm_zero_sum_within_1e12"] is True
    assert d["all_clone_neural_exact"] is True
    assert d["strategic_gate_defined"] is False
    assert d["ready_for_tables"] is False


def _valid_certification_pair():
    freeze = {
        "schema": orchestrator.FREEZE_SCHEMA,
        "source_head_sha": "a" * 40,
        "evidence_commit_sha": "b" * 40,
        "thread_environment_contract": orchestrator.THREAD_ENV_CONTRACT,
    }
    acceptance = {
        "schema": orchestrator.ACCEPT_SCHEMA,
        "r7_3_640_acceptance_pass": True,
        "r7_3_ready_to_advance_to_r7_4": True,
        "per_seed_fit_pass": True,
        "iterations": 5,
        "roots_per_iteration": 128,
        "roots_per_seed": 640,
        "source_head_sha": freeze["source_head_sha"],
        "durability_evidence_commit_sha": freeze["evidence_commit_sha"],
        "thread_environment_contract": orchestrator.THREAD_ENV_CONTRACT,
        "thread_environment_overrides_injected_by_certifier": False,
        "acceptance_gate_changed": False,
    }
    return freeze, acceptance


def test_r7_4_requires_corrected_exact_source_certification_contract():
    freeze, acceptance = _valid_certification_pair()
    assert orchestrator._validate_r7_3_prerequisites(freeze, acceptance) == freeze["source_head_sha"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda f, a: a.pop("thread_environment_contract"),
        lambda f, a: a.__setitem__("thread_environment_overrides_injected_by_certifier", True),
        lambda f, a: a.__setitem__("durability_evidence_commit_sha", "c" * 40),
        lambda f, a: a.__setitem__("roots_per_seed", 320),
        lambda f, a: a.__setitem__("per_seed_fit_pass", False),
        lambda f, a: a.__setitem__("acceptance_gate_changed", True),
    ],
)
def test_r7_4_rejects_legacy_or_incomplete_acceptance(mutation):
    freeze, acceptance = _valid_certification_pair()
    mutation(freeze, acceptance)
    with pytest.raises(ValueError):
        orchestrator._validate_r7_3_prerequisites(freeze, acceptance)
