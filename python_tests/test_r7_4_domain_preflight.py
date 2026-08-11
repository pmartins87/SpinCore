from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

spec = importlib.util.spec_from_file_location(
    "r7_4_domain_preflight_worker_test",
    TOOLS / "r7_4_domain_preflight_worker.py",
)
assert spec and spec.loader
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


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
