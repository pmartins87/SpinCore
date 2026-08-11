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
RULESET_FREEZE_SCHEMA = "SPINCORE_R7_4_RULESET_EXTENSION_V1"
RULESET_ACCEPT_SCHEMA = "SPINCORE_R7_4_RULESET_ACCEPTANCE_V1"
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


def _validate_ruleset_prerequisites(freeze: dict, acceptance: dict, ruleset_freeze: dict, ruleset_acceptance: dict) -> str:
    r7_source = _validate_r7_3_prerequisites(freeze, acceptance)
    if ruleset_freeze.get("schema") != RULESET_FREEZE_SCHEMA or ruleset_freeze.get("ruleset_schema") != "SPINRULESET-4":
        raise ValueError("wrong R7.4 ruleset freeze")
    if ruleset_acceptance.get("schema") != RULESET_ACCEPT_SCHEMA or ruleset_acceptance.get("ruleset_schema") != "SPINRULESET-4":
        raise ValueError("wrong R7.4 ruleset acceptance")
    if ruleset_freeze.get("base_r7_3_source_head_sha") != r7_source:
        raise ValueError("SPINRULESET-4 base source differs from certified R7.3 source")
    if ruleset_freeze.get("base_r7_3_evidence_commit_sha") != freeze.get("evidence_commit_sha"):
        raise ValueError("SPINRULESET-4 base evidence differs from R7.3 freeze")
    if ruleset_freeze.get("base_r7_3_evidence_sha256") != freeze.get("evidence_sha256"):
        raise ValueError("SPINRULESET-4 base evidence bytes differ from R7.3 freeze")
    rules_source = str(ruleset_freeze.get("ruleset_extension_source_head_sha", ""))
    if not rules_source or ruleset_acceptance.get("ruleset_extension_source_head_sha") != rules_source:
        raise ValueError("R7.4 rules source is not consistently frozen/accepted")
    if ruleset_acceptance.get("base_r7_3_source_head_sha") != r7_source:
        raise ValueError("R7.4 ruleset acceptance has wrong R7.3 base")
    if ruleset_acceptance.get("base_r7_3_evidence_sha256") != freeze.get("evidence_sha256"):
        raise ValueError("R7.4 ruleset acceptance has wrong evidence bytes")
    if ruleset_acceptance.get("hu_invariance_pass") is not True:
        raise ValueError("SPINRULESET-4 HU invariance must pass before R7.4 preflight")
    if int(ruleset_acceptance.get("hu_invariance_difference_count", -1)) != 0:
        raise ValueError("SPINRULESET-4 HU invariance was not exact")
    if ruleset_acceptance.get("selected_training_components_byte_identical") is not True:
        raise ValueError("selected R7.3 training components changed in SPINRULESET-4 source")
    if ruleset_acceptance.get("r7_4_rules_source_accepted") is not True:
        raise ValueError("SPINRULESET-4 source is not accepted")
    if ruleset_acceptance.get("r7_4_gate_changed") is not False:
        raise ValueError("R7.4 ruleset acceptance changed strategic gate")
    return rules_source


def main() -> int:
    ap = argparse.ArgumentParser(description="Run R7.4 structural HU/3H preflight on the frozen HU-invariant SPINRULESET-4 source")
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--ruleset-freeze", type=Path, required=True)
    ap.add_argument("--ruleset-acceptance", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    acceptance = json.loads(args.acceptance.read_text(encoding="utf-8"))
    ruleset_freeze = json.loads(args.ruleset_freeze.read_text(encoding="utf-8"))
    ruleset_acceptance = json.loads(args.ruleset_acceptance.read_text(encoding="utf-8"))
    try:
        source_head = _validate_ruleset_prerequisites(freeze, acceptance, ruleset_freeze, ruleset_acceptance)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    repo_root = Path(__file__).resolve().parents[1]
    worker = repo_root / WORKER_REL
    if not worker.is_file():
        raise SystemExit("R7.4 domain preflight worker missing")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="spincore_r7_4_preflight_") as td:
        worktree = Path(td) / "ruleset4"
        temp_out = Path(td) / "preflight.json"
        _run(["git", "worktree", "add", "--detach", str(worktree), source_head], cwd=repo_root)
        try:
            # The strategic/rules source is the frozen SPINRULESET-4 descendant.
            # Only the R7.4 test harness is overlaid; algorithm/training imports
            # remain pinned to the accepted rules-source worktree.
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
    report["r7_3_certified_source_head_sha"] = freeze["source_head_sha"]
    report["r7_3_durability_evidence_commit_sha"] = freeze["evidence_commit_sha"]
    report["r7_3_thread_environment_contract"] = THREAD_ENV_CONTRACT
    report["r7_3_acceptance_hidden_thread_overrides_rejected"] = True
    report["r7_4_ruleset_schema"] = "SPINRULESET-4"
    report["r7_4_rules_source_head_sha"] = source_head
    report["r7_4_hu_invariance_passed_first"] = True
    report["exact_frozen_r7_4_rules_source_used"] = True
    report["preflight_worker_overlay_sha256"] = _sha256(worker)
    report["preflight_worker_executed_from_rules_worktree_overlay"] = True
    report["r7_4_structural_preflight_pass"] = bool(
        report.get("hu_domains") == [1]
        and report.get("three_handed_domains") == [0]
        and report.get("all_chip_zero_sum") is True
        and report.get("all_icm_zero_sum_within_1e12") is True
        and report.get("all_clone_neural_exact") is True
    )
    report["r7_4_strategic_pilot_pass"] = False
    report["r7_4_strategic_gate_defined"] = True
    report["ready_for_tables"] = False
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["r7_4_structural_preflight_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
