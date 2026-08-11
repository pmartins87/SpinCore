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

import r7_3_certification_evidence as evidence_resolver
import run_r7_3_frozen_candidate_fresh_repro as fresh_runner


FREEZE_SCHEMA = "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1"
FRESH_SCHEMA = "SPINCORE_R7_3_FROZEN_CANDIDATE_FRESH_REPRO_V1"
RECERT_SCHEMA = "SPINCORE_R7_3_CANDIDATE_CHECKPOINT_RECERT_V1"
HELPER_REL = Path("python/spincore/r7_candidate_checkpoint.py")
WORKER_REL = Path("tools/r7_3_frozen_candidate_checkpoint_worker.py")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _overlay_targets(worktree: Path) -> tuple[Path, Path]:
    return worktree / HELPER_REL, worktree / WORKER_REL


def _validate_fresh_prerequisite(freeze: dict, fresh: dict) -> None:
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise ValueError("wrong semantic-freeze schema")
    if fresh.get("schema") != FRESH_SCHEMA or fresh.get("fresh_process_reproducible") is not True:
        raise ValueError("fresh-process reproducibility must pass before checkpoint recertification")
    if int(fresh.get("difference_count", -1)) != 0:
        raise ValueError("fresh-process report is not an exact zero-difference reproduction")
    if str(fresh.get("source_head_sha")) != str(freeze.get("source_head_sha")):
        raise ValueError("fresh reproducibility source head does not match semantic freeze")
    if str(fresh.get("original_evidence_commit_sha")) != str(freeze.get("evidence_commit_sha")):
        raise ValueError("fresh reproducibility evidence commit does not match semantic freeze")
    if str(fresh.get("original_evidence_sha256")) != str(freeze.get("evidence_sha256")):
        raise ValueError("fresh reproducibility evidence bytes do not match semantic freeze")
    if fresh.get("behavior_semantic_id") != freeze.get("behavior_semantic_id"):
        raise ValueError("fresh reproducibility behavior semantic id does not match semantic freeze")
    if fresh.get("thread_environment_contract") != freeze.get("thread_environment_contract"):
        raise ValueError("fresh reproducibility thread environment contract does not match semantic freeze")
    if fresh.get("thread_environment_overrides_injected_by_certifier") is not False:
        raise ValueError("fresh reproducibility used or does not disprove hidden certifier thread overrides")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run checkpoint/resume recertification against the exact frozen R7.3 algorithm source")
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--fresh-report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split-iteration", type=int, default=3)
    args = ap.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    try:
        fresh, fresh_origin = evidence_resolver.resolve_valid_json(
            args.fresh_report,
            validator=lambda data: _validate_fresh_prerequisite(freeze, data),
            repo_root=Path(__file__).resolve().parents[1],
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    repo_root = Path(__file__).resolve().parents[1]
    helper = repo_root / HELPER_REL
    worker = repo_root / WORKER_REL
    if not helper.is_file() or not worker.is_file():
        raise SystemExit("checkpoint certification helper/worker missing")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    source_head = str(freeze["source_head_sha"])
    with tempfile.TemporaryDirectory(prefix="spincore_r7_checkpoint_recert_") as td:
        temp = Path(td)
        worktree = temp / "source"
        checkpoint_dir = temp / "checkpoints"
        _run(["git", "worktree", "add", "--detach", str(worktree), source_head], cwd=repo_root)
        try:
            # Overlay only checkpoint-certification plumbing. The worker itself
            # must execute from worktree/tools so its behavior/training imports
            # resolve to the frozen source tree, never to current-main tools.
            target_helper, target_worker = _overlay_targets(worktree)
            target_helper.parent.mkdir(parents=True, exist_ok=True)
            target_worker.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(helper, target_helper)
            shutil.copyfile(worker, target_worker)

            _run(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], cwd=worktree)
            _run(["cmake", "--build", "build", "-j2"], cwd=worktree)
            _run(["ctest", "--test-dir", "build", "--output-on-failure"], cwd=worktree)
            env = fresh_runner._source_execution_env(freeze, os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(worktree / "python"), str(worktree / "tools")])
            _run([sys.executable, "-m", "pytest", "-q", "python_tests"], cwd=worktree, env=env)

            _run([
                sys.executable,
                str(target_worker),
                "--freeze", str(args.freeze.resolve()),
                "--solver", str(worktree / "build" / "libspincore_solver_c.so"),
                "--checkpoint-dir", str(checkpoint_dir),
                "--out", str(args.out.resolve()),
                "--split-iteration", str(int(args.split_iteration)),
            ], cwd=worktree, env=env)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo_root, check=False)

    report = json.loads(args.out.read_text(encoding="utf-8"))
    if report.get("schema") != RECERT_SCHEMA:
        raise SystemExit("worker produced wrong checkpoint recertification schema")
    report["algorithm_source_head_sha"] = source_head
    report["algorithm_source_exact_worktree"] = True
    report["thread_environment_contract"] = freeze["thread_environment_contract"]
    report["thread_environment_overrides_injected_by_certifier"] = False
    report["checkpoint_helper_overlay_sha256"] = _sha256(helper)
    report["checkpoint_worker_overlay_sha256"] = _sha256(worker)
    report["checkpoint_worker_executed_from_frozen_worktree_overlay"] = True
    report["fresh_process_reproducibility_gate_passed_first"] = True
    report["fresh_process_zero_difference_gate_passed_first"] = True
    report["validated_fresh_report_origin"] = fresh_origin
    report["validated_fresh_report_sha256"] = fresh_origin["sha256"]
    report["validated_fresh_original_evidence_sha256"] = fresh["original_evidence_sha256"]
    report["validated_fresh_behavior_semantic_id"] = fresh["behavior_semantic_id"]
    report["source_cpp_regression_passed_before_recertification"] = True
    report["source_python_regression_passed_before_recertification"] = True
    report["ready_for_640"] = False
    report["ready_for_tables"] = False
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report.get("checkpoint_resume_recertification_pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
