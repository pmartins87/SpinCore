from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import run_r7_3_frozen_candidate_fresh_repro as r7_repro


R7_FREEZE_SCHEMA = "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1"
RULESET_FREEZE_SCHEMA = "SPINCORE_R7_4_RULESET_EXTENSION_V1"
REPORT_SCHEMA = "SPINCORE_R7_4_HU_INVARIANCE_V1"
EVIDENCE_SCHEMA = "SPINCORE_R7_4_HU_STRUCTURAL_INVARIANCE_EVIDENCE_V1"
INVARIANCE_BASIS = "FROZEN_TRAINING_COMPONENT_BLOB_IDENTITY_PLUS_EXTENSION_REGRESSION"


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _git_bytes(*args: str, cwd: Path) -> bytes:
    return subprocess.check_output(["git", *args], cwd=cwd)


def _blob(ref: str, path: str, *, cwd: Path) -> str:
    return _git_bytes("rev-parse", f"{ref}:{path}", cwd=cwd).decode().strip()


def _validate_freezes(r7: dict, ruleset: dict, *, repo_root: Path) -> tuple[str, str, dict[str, dict]]:
    if r7.get("schema") != R7_FREEZE_SCHEMA or r7.get("evidence_r7_3_pass") is not True:
        raise ValueError("invalid R7.3 semantic freeze")
    if ruleset.get("schema") != RULESET_FREEZE_SCHEMA or ruleset.get("ruleset_schema") != "SPINRULESET-4":
        raise ValueError("invalid R7.4 ruleset extension freeze")
    base = str(r7.get("source_head_sha", ""))
    extension = str(ruleset.get("ruleset_extension_source_head_sha", ""))
    if ruleset.get("base_r7_3_source_head_sha") != base:
        raise ValueError("ruleset extension base source differs from R7.3 freeze")
    if ruleset.get("base_r7_3_evidence_commit_sha") != r7.get("evidence_commit_sha"):
        raise ValueError("ruleset extension evidence commit differs from R7.3 freeze")
    if ruleset.get("base_r7_3_evidence_sha256") != r7.get("evidence_sha256"):
        raise ValueError("ruleset extension evidence bytes differ from R7.3 freeze")
    if subprocess.run(["git", "merge-base", "--is-ancestor", base, extension], cwd=repo_root).returncode != 0:
        raise ValueError("SPINRULESET-4 source is not a descendant of selected R7.3 source")

    for path, row in dict(ruleset.get("core_ruleset_deltas") or {}).items():
        if _blob(base, path, cwd=repo_root) != row.get("base_blob_sha"):
            raise ValueError(f"base blob mismatch for {path}")
        if _blob(extension, path, cwd=repo_root) != row.get("extension_blob_sha"):
            raise ValueError(f"extension blob mismatch for {path}")

    selected_components = dict(ruleset.get("selected_training_components_byte_identical_across_sources") or {})
    if not selected_components:
        raise ValueError("ruleset freeze does not pin selected training components")
    component_rows: dict[str, dict] = {}
    for path, expected in selected_components.items():
        a = _blob(base, path, cwd=repo_root)
        b = _blob(extension, path, cwd=repo_root)
        identical = a == expected and b == expected
        component_rows[path] = {
            "expected_blob_sha": expected,
            "base_blob_sha": a,
            "extension_blob_sha": b,
            "byte_identical": identical,
        }
        if not identical:
            raise ValueError(f"selected training component changed across ruleset extension: {path}")
    return base, extension, component_rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Prove that SPINRULESET-4 preserves the selected R7.3 HU implementation without "
            "reusing the deferred historical numeric-reproducibility gate"
        )
    )
    ap.add_argument("--r7-freeze", type=Path, required=True)
    ap.add_argument("--ruleset-freeze", type=Path, required=True)
    ap.add_argument("--fresh-out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    r7 = json.loads(args.r7_freeze.read_text(encoding="utf-8"))
    ruleset = json.loads(args.ruleset_freeze.read_text(encoding="utf-8"))
    try:
        base, extension, component_rows = _validate_freezes(r7, ruleset, repo_root=repo_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Preserve immutable provenance of the selected R7.3 winner, but deliberately do
    # not compare a new run against historical floating-point evidence. That exact
    # fresh-process reproduction is tracked separately as release-blocking debt.
    evidence_path = str(r7["evidence_path"])
    evidence_commit = str(r7["evidence_commit_sha"])
    original_bytes = _git_bytes("show", f"{evidence_commit}:{evidence_path}", cwd=repo_root)
    original_sha256 = hashlib.sha256(original_bytes).hexdigest()
    if original_sha256 != r7.get("evidence_sha256"):
        raise SystemExit("immutable R7.3 evidence SHA-256 mismatch")

    args.fresh_out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    # Regression-test the exact frozen SPINRULESET-4 extension source. The later R7.4
    # held-out HU640 gate remains the physical strategy-quality test. This pre-gate
    # isolates whether the ruleset extension changed any frozen R7.3 training code.
    with tempfile.TemporaryDirectory(prefix="spincore_r7_4_hu_invariance_") as td:
        worktree = Path(td) / "ruleset4"
        _run(["git", "worktree", "add", "--detach", str(worktree), extension], cwd=repo_root)
        try:
            _run(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], cwd=worktree)
            _run(["cmake", "--build", "build", "-j2"], cwd=worktree)
            _run(["ctest", "--test-dir", "build", "--output-on-failure"], cwd=worktree)
            env = r7_repro._source_execution_env(r7, os.environ)
            env["PYTHONPATH"] = str(worktree / "python")
            _run([sys.executable, "-m", "pytest", "-q", "python_tests"], cwd=worktree, env=env)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo_root, check=False)

    component_difference_count = sum(1 for row in component_rows.values() if not row["byte_identical"])
    passed = component_difference_count == 0
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "ruleset_schema": "SPINRULESET-4",
        "invariance_basis": INVARIANCE_BASIS,
        "r7_3_behavior_semantic_id": r7["behavior_semantic_id"],
        "r7_3_base_source_head_sha": base,
        "ruleset_extension_source_head_sha": extension,
        "historical_r7_3_evidence_commit_sha": evidence_commit,
        "historical_r7_3_evidence_sha256": original_sha256,
        "selected_training_components": component_rows,
        "selected_training_component_difference_count": component_difference_count,
        "selected_training_components_byte_identical": passed,
        "ruleset_extension_regression_passed": True,
        "historical_numeric_reproduction_intentionally_not_required_here": True,
        "historical_exact_reproducibility_debt_preserved": True,
        "heldout_hu640_strategy_test_still_required": True,
        "ready_for_tables": False,
    }
    args.fresh_out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "schema": REPORT_SCHEMA,
        "ruleset_schema": "SPINRULESET-4",
        "invariance_basis": INVARIANCE_BASIS,
        "difference_scope": "SELECTED_R7_3_TRAINING_COMPONENT_BLOBS",
        "r7_3_behavior_semantic_id": r7["behavior_semantic_id"],
        "r7_3_base_source_head_sha": base,
        "ruleset_extension_source_head_sha": extension,
        "original_r7_3_evidence_commit_sha": evidence_commit,
        "original_r7_3_evidence_sha256": original_sha256,
        "fresh_ruleset4_hu_evidence_path": str(args.fresh_out),
        "selected_training_components_byte_identical": passed,
        "ruleset_extension_regression_passed_before_hu_run": True,
        "thread_environment_contract": r7["thread_environment_contract"],
        "thread_environment_overrides_injected_by_invariance_runner": False,
        "difference_count": component_difference_count,
        "differences": [],
        "historical_numeric_evidence_reproduction_evaluated": False,
        "historical_exact_reproducibility_debt_preserved": True,
        "heldout_hu640_strategy_test_still_required": True,
        "selected_r7_3_hu_evidence_invariant": passed,
        "r7_4_rules_source_accepted": passed,
        "r7_4_gate_changed": False,
        "ready_for_tables": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
