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


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _git_bytes(*args: str, cwd: Path) -> bytes:
    return subprocess.check_output(["git", *args], cwd=cwd)


def _blob(ref: str, path: str, *, cwd: Path) -> str:
    return _git_bytes("rev-parse", f"{ref}:{path}", cwd=cwd).decode().strip()


def _validate_freezes(r7: dict, ruleset: dict, *, repo_root: Path) -> tuple[str, str]:
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
    for path, expected in selected_components.items():
        a = _blob(base, path, cwd=repo_root)
        b = _blob(extension, path, cwd=repo_root)
        if a != expected or b != expected:
            raise ValueError(f"selected training component changed across ruleset extension: {path}")
    return base, extension


def main() -> int:
    ap = argparse.ArgumentParser(description="Prove that SPINRULESET-4 preserves the selected R7.3 true-HU physical evidence")
    ap.add_argument("--r7-freeze", type=Path, required=True)
    ap.add_argument("--ruleset-freeze", type=Path, required=True)
    ap.add_argument("--fresh-out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    r7 = json.loads(args.r7_freeze.read_text(encoding="utf-8"))
    ruleset = json.loads(args.ruleset_freeze.read_text(encoding="utf-8"))
    try:
        base, extension = _validate_freezes(r7, ruleset, repo_root=repo_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    evidence_path = str(r7["evidence_path"])
    evidence_commit = str(r7["evidence_commit_sha"])
    original_bytes = _git_bytes("show", f"{evidence_commit}:{evidence_path}", cwd=repo_root)
    original_sha256 = hashlib.sha256(original_bytes).hexdigest()
    if original_sha256 != r7.get("evidence_sha256"):
        raise SystemExit("immutable R7.3 evidence SHA-256 mismatch")
    original = json.loads(original_bytes.decode("utf-8"))

    args.fresh_out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
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
            temp_fresh = Path(td) / "ruleset4_hu.json"
            _run(r7_repro._runner_command(r7, temp_fresh), cwd=worktree, env=env)
            fresh = json.loads(temp_fresh.read_text(encoding="utf-8"))
            shutil.copyfile(temp_fresh, args.fresh_out)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo_root, check=False)

    diffs = r7_repro._compare(original, fresh)
    report = {
        "schema": REPORT_SCHEMA,
        "ruleset_schema": "SPINRULESET-4",
        "r7_3_behavior_semantic_id": r7["behavior_semantic_id"],
        "r7_3_base_source_head_sha": base,
        "ruleset_extension_source_head_sha": extension,
        "original_r7_3_evidence_commit_sha": evidence_commit,
        "original_r7_3_evidence_sha256": original_sha256,
        "fresh_ruleset4_hu_evidence_path": str(args.fresh_out),
        "selected_training_components_byte_identical": True,
        "ruleset_extension_regression_passed_before_hu_run": True,
        "thread_environment_contract": r7["thread_environment_contract"],
        "thread_environment_overrides_injected_by_invariance_runner": False,
        "ignored_nondeterministic_keys": sorted(r7_repro.IGNORE_KEYS),
        "numeric_tolerance": 1e-9,
        "difference_count": len(diffs),
        "differences": diffs[:200],
        "selected_r7_3_hu_evidence_invariant": len(diffs) == 0,
        "r7_4_rules_source_accepted": len(diffs) == 0,
        "r7_4_gate_changed": False,
        "ready_for_tables": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["selected_r7_3_hu_evidence_invariant"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
