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


FREEZE_SCHEMA = "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1"
ACCEPT_SCHEMA = "SPINCORE_R7_3_FROZEN_CANDIDATE_640_ACCEPTANCE_V1"
PREFLIGHT_SCHEMA = "SPINCORE_R7_4_DOMAIN_PREFLIGHT_V1"
THREAD_ENV_CONTRACT = "SOURCE_WORKFLOW_NO_EXPLICIT_THREAD_OVERRIDE"
WORKER_REL = Path("tools/r7_4_domain_preflight_worker.py")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_r7_3_prerequisites(freeze: dict, acceptance: dict) -> str:
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise ValueError("wrong R7.3 freeze schema")
    if acceptance.get("schema") != ACCEPT_SCHEMA:
        raise ValueError("wrong R7.3 640 acceptance schema")
    if acceptance.get("r7_3_640_acceptance_pass") is not True:
        raise ValueError("R7.3 640 acceptance must pass before R7.4 preflight")
    if acceptance.get("r7_3_ready_to_advance_to_r7_4") is not True:
        raise ValueError("R7.3 acceptance did not authorize R7.4")
    if acceptance.get("per_seed_fit_pass") is not True:
        raise ValueError("R7.3 640 acceptance did not preserve fit gates")
    if int(acceptance.get("iterations", -1)) != 5:
        raise ValueError("R7.3 acceptance iteration count is not frozen 5")
    if int(acceptance.get("roots_per_iteration", -1)) != 128 or int(acceptance.get("roots_per_seed", -1)) != 640:
        raise ValueError("R7.3 acceptance is not the certified 5x128 scale")
    source_head = str(freeze.get("source_head_sha", ""))
    if not source_head or str(acceptance.get("source_head_sha")) != source_head:
        raise ValueError("acceptance source head differs from frozen winner")
    if str(acceptance.get("durability_evidence_commit_sha", "")) != str(freeze.get("evidence_commit_sha", "")):
        raise ValueError("acceptance durability evidence differs from frozen winner")
    if freeze.get("thread_environment_contract") != THREAD_ENV_CONTRACT:
        raise ValueError("frozen winner lacks the corrected exact-source thread contract")
    if acceptance.get("thread_environment_contract") != THREAD_ENV_CONTRACT:
        raise ValueError("acceptance was not produced under the corrected exact-source thread contract")
    if acceptance.get("thread_environment_overrides_injected_by_certifier") is not False:
        raise ValueError("acceptance used or does not disprove hidden certifier thread overrides")
    if acceptance.get("acceptance_gate_changed") is not False:
        raise ValueError("R7.3 acceptance gate changed")
    return source_head


def main() -> int:
    ap = argparse.ArgumentParser(description="Run R7.4 structural HU/3H preflight on the exact accepted R7.3 solver source")
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    acceptance = json.loads(args.acceptance.read_text(encoding="utf-8"))
    try:
        source_head = _validate_r7_3_prerequisites(freeze, acceptance)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    repo_root = Path(__file__).resolve().parents[1]
    worker = repo_root / WORKER_REL
    if not worker.is_file():
        raise SystemExit("R7.4 domain preflight worker missing")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="spincore_r7_4_preflight_") as td:
        worktree = Path(td) / "source"
        temp_out = Path(td) / "preflight.json"
        _run(["git", "worktree", "add", "--detach", str(worktree), source_head], cwd=repo_root)
        try:
            # R7.4's worker is new test harness code, not part of the accepted
            # R7.3 algorithm. Overlay its exact bytes into the accepted-source
            # worktree and execute it there so all spincore imports are pinned
            # to the accepted source tree rather than current-main Python.
            target_worker = worktree / WORKER_REL
            target_worker.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(worker, target_worker)

            _run(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], cwd=worktree)
            _run(["cmake", "--build", "build", "-j2"], cwd=worktree)
            _run(["ctest", "--test-dir", "build", "--output-on-failure"], cwd=worktree)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(worktree / "python")
            _run([
                sys.executable,
                str(target_worker),
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
    report["r7_3_durability_evidence_commit_sha"] = freeze["evidence_commit_sha"]
    report["r7_3_thread_environment_contract"] = THREAD_ENV_CONTRACT
    report["r7_3_acceptance_hidden_thread_overrides_rejected"] = True
    report["exact_accepted_solver_source_used"] = True
    report["preflight_worker_overlay_sha256"] = _sha256(worker)
    report["preflight_worker_executed_from_accepted_worktree_overlay"] = True
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
