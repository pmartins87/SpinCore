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

import run_r7_4_domain_preflight as preflight_gate


FREEZE_SCHEMA = "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1"
ACCEPT_SCHEMA = "SPINCORE_R7_3_FROZEN_CANDIDATE_640_ACCEPTANCE_V1"
PREFLIGHT_SCHEMA = "SPINCORE_R7_4_DOMAIN_PREFLIGHT_V1"
PILOT_SCHEMA = "SPINCORE_R7_4_HELDOUT_DOMAIN_STABILITY_V1"
WORKER_REL = Path("tools/r7_4_stability_pilot_worker.py")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_prerequisites(freeze: dict, acceptance: dict, preflight: dict) -> str:
    source_head = preflight_gate._validate_r7_3_prerequisites(freeze, acceptance)
    if preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError("wrong R7.4 structural-preflight schema")
    if preflight.get("r7_4_structural_preflight_pass") is not True:
        raise ValueError("R7.4 structural preflight must pass before held-out stability pilot")
    if preflight.get("exact_accepted_solver_source_used") is not True:
        raise ValueError("R7.4 structural preflight did not use exact accepted solver source")
    if preflight.get("preflight_worker_executed_from_accepted_worktree_overlay") is not True:
        raise ValueError("R7.4 structural worker provenance is incomplete")
    if preflight.get("r7_3_source_head_sha") != source_head:
        raise ValueError("R7.4 preflight source head differs from accepted R7.3 winner")
    if preflight.get("r7_3_durability_evidence_commit_sha") != freeze.get("evidence_commit_sha"):
        raise ValueError("R7.4 preflight durability provenance differs from semantic freeze")
    if preflight.get("ready_for_tables") is not False:
        raise ValueError("R7.4 preflight unexpectedly marked table readiness")
    return source_head


def main() -> int:
    ap = argparse.ArgumentParser(description="Run R7.4 held-out HU/3H stability pilot on exact accepted R7.3 source")
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--preflight", type=Path, required=True)
    ap.add_argument("--domain", choices=("TRUE_HEADS_UP", "THREE_HANDED"), required=True)
    ap.add_argument("--roots-per-iteration", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    acceptance = json.loads(args.acceptance.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    try:
        source_head = _validate_prerequisites(freeze, acceptance, preflight)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if int(args.roots_per_iteration) <= 0:
        raise SystemExit("roots-per-iteration must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    worker = repo_root / WORKER_REL
    if not worker.is_file():
        raise SystemExit("R7.4 stability pilot worker missing")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="spincore_r7_4_stability_") as td:
        worktree = Path(td) / "source"
        temp_out = Path(td) / "pilot.json"
        _run(["git", "worktree", "add", "--detach", str(worktree), source_head], cwd=repo_root)
        try:
            target_worker = worktree / WORKER_REL
            target_worker.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(worker, target_worker)
            _run(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], cwd=worktree)
            _run(["cmake", "--build", "build", "-j2"], cwd=worktree)
            _run(["ctest", "--test-dir", "build", "--output-on-failure"], cwd=worktree)
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(worktree / "python"), str(worktree / "tools")])
            _run([sys.executable, "-m", "pytest", "-q", "python_tests"], cwd=worktree, env=env)
            _run([
                sys.executable,
                str(target_worker),
                "--freeze", str(args.freeze.resolve()),
                "--solver", str(worktree / "build" / "libspincore_solver_c.so"),
                "--domain", str(args.domain),
                "--roots-per-iteration", str(int(args.roots_per_iteration)),
                "--out", str(temp_out),
            ], cwd=worktree, env=env)
            shutil.copyfile(temp_out, args.out)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo_root, check=False)

    report = json.loads(args.out.read_text(encoding="utf-8"))
    if report.get("schema") != PILOT_SCHEMA:
        raise SystemExit("wrong R7.4 held-out pilot schema")
    report["r7_3_640_acceptance_passed_first"] = True
    report["r7_4_structural_preflight_passed_first"] = True
    report["accepted_r7_3_source_head_sha"] = source_head
    report["accepted_r7_3_evidence_commit_sha"] = freeze["evidence_commit_sha"]
    report["accepted_r7_3_behavior_semantic_id"] = freeze["behavior_semantic_id"]
    report["thread_environment_contract"] = freeze["thread_environment_contract"]
    report["thread_environment_overrides_injected_by_r7_4_orchestrator"] = False
    report["exact_accepted_algorithm_source_used"] = True
    report["pilot_worker_overlay_sha256"] = _sha256(worker)
    report["pilot_worker_executed_from_accepted_worktree_overlay"] = True
    report["ready_for_tables"] = False
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report.get("r7_4_domain_stability_pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
