from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


FREEZE_SCHEMA = "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1"
ACCEPT_SCHEMA = "SPINCORE_R7_3_FROZEN_CANDIDATE_640_ACCEPTANCE_V1"
PREFLIGHT_SCHEMA = "SPINCORE_R7_4_DOMAIN_PREFLIGHT_V1"


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run R7.4 structural HU/3H preflight on the exact accepted R7.3 source")
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    acceptance = json.loads(args.acceptance.read_text(encoding="utf-8"))
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise SystemExit("wrong R7.3 freeze schema")
    if acceptance.get("schema") != ACCEPT_SCHEMA or acceptance.get("r7_3_640_acceptance_pass") is not True:
        raise SystemExit("R7.3 640 acceptance must pass before R7.4 preflight")
    if acceptance.get("r7_3_ready_to_advance_to_r7_4") is not True:
        raise SystemExit("R7.3 acceptance did not authorize R7.4")
    source_head = str(freeze["source_head_sha"])
    if str(acceptance.get("source_head_sha")) != source_head:
        raise SystemExit("acceptance source head differs from frozen winner")

    repo_root = Path(__file__).resolve().parents[1]
    worker = repo_root / "tools" / "r7_4_domain_preflight_worker.py"
    if not worker.is_file():
        raise SystemExit("R7.4 domain preflight worker missing")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="spincore_r7_4_preflight_") as td:
        worktree = Path(td) / "source"
        temp_out = Path(td) / "preflight.json"
        _run(["git", "worktree", "add", "--detach", str(worktree), source_head], cwd=repo_root)
        try:
            _run(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], cwd=worktree)
            _run(["cmake", "--build", "build", "-j2"], cwd=worktree)
            _run(["ctest", "--test-dir", "build", "--output-on-failure"], cwd=worktree)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(worktree / "python")
            _run([
                sys.executable,
                str(worker),
                "--solver", str(worktree / "build" / "libspincore_solver_c.so"),
                "--out", str(temp_out),
            ], cwd=worktree, env=env)
            shutil.copyfile(temp_out, args.out)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo_root, check=False)

    report = json.loads(args.out.read_text(encoding="utf-8"))
    if report.get("schema") != PREFLIGHT_SCHEMA:
        raise SystemExit("wrong R7.4 preflight worker schema")
    report["r7_3_640_acceptance_passed_first"] = True
    report["r7_3_accepted_behavior_semantic_id"] = freeze["behavior_semantic_id"]
    report["r7_3_source_head_sha"] = source_head
    report["exact_accepted_source_used"] = True
    report["r7_4_structural_preflight_pass"] = bool(
        report.get("hu_domains") == [1]
        and report.get("three_handed_domains") == [0]
        and report.get("all_chip_zero_sum") is True
        and report.get("all_icm_zero_sum_within_1e12") is True
        and report.get("all_clone_neural_exact") is True
    )
    report["r7_4_strategic_pilot_pass"] = False
    report["r7_4_strategic_gate_defined"] = False
    report["ready_for_tables"] = False
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["r7_4_structural_preflight_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
