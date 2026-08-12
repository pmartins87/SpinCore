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
STRICT_ACCEPT_SCHEMA = "SPINCORE_R7_3_FROZEN_CANDIDATE_640_ACCEPTANCE_V1"
PROVISIONAL_ACCEPT_SCHEMA = "SPINCORE_R7_3_PROVISIONAL_640_STRATEGY_ACCEPTANCE_V1"
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


def _acceptance_mode(acceptance: dict) -> str:
    schema = acceptance.get("schema")
    if schema == STRICT_ACCEPT_SCHEMA:
        return "STRICT_EXACT_CERTIFICATION"
    if schema == PROVISIONAL_ACCEPT_SCHEMA:
        return "PROVISIONAL_640_STRATEGY_QUALITY"
    raise ValueError("wrong R7.3 640 prerequisite schema")


def _validate_r7_3_prerequisites(freeze: dict, acceptance: dict) -> str:
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise ValueError("wrong R7.3 freeze schema")
    if freeze.get("thread_environment_contract") != THREAD_ENV_CONTRACT:
        raise ValueError("frozen winner lacks the corrected exact-source thread contract")

    mode = _acceptance_mode(acceptance)
    source_head = str(freeze.get("source_head_sha", ""))
    if not source_head or str(acceptance.get("source_head_sha")) != source_head:
        raise ValueError("R7.3 prerequisite source head differs from frozen winner")
    if str(acceptance.get("durability_evidence_commit_sha", "")) != str(freeze.get("evidence_commit_sha", "")):
        raise ValueError("R7.3 prerequisite durability evidence differs from frozen winner")
    if acceptance.get("thread_environment_contract") != THREAD_ENV_CONTRACT:
        raise ValueError("R7.3 prerequisite was not produced under the frozen thread contract")
    if int(acceptance.get("iterations", -1)) != 5:
        raise ValueError("R7.3 prerequisite iteration count is not frozen 5")
    if int(acceptance.get("roots_per_iteration", -1)) != 128 or int(acceptance.get("roots_per_seed", -1)) != 640:
        raise ValueError("R7.3 prerequisite is not the 5x128 strategy scale")
    if acceptance.get("per_seed_fit_pass") is not True:
        raise ValueError("R7.3 prerequisite did not preserve per-seed fit gates")

    if mode == "STRICT_EXACT_CERTIFICATION":
        if acceptance.get("r7_3_640_acceptance_pass") is not True:
            raise ValueError("strict R7.3 640 acceptance must pass before R7.4 preflight")
        if acceptance.get("r7_3_ready_to_advance_to_r7_4") is not True:
            raise ValueError("strict R7.3 acceptance did not authorize R7.4")
        if acceptance.get("thread_environment_overrides_injected_by_certifier") is not False:
            raise ValueError("strict R7.3 acceptance used or does not disprove hidden thread overrides")
        if acceptance.get("acceptance_gate_changed") is not False:
            raise ValueError("strict R7.3 acceptance gate changed")
        return source_head

    # Provisional path deliberately advances engineering without relabeling the
    # unsatisfied exact-reproducibility requirement as PASS. Every strategic
    # threshold remains the frozen threshold; the exception is release-blocking.
    if str(acceptance.get("durability_evidence_sha256", "")) != str(freeze.get("evidence_sha256", "")):
        raise ValueError("provisional R7.3 prerequisite has wrong frozen evidence bytes")
    if acceptance.get("behavior_semantic_id") != freeze.get("behavior_semantic_id"):
        raise ValueError("provisional R7.3 prerequisite has wrong behavior semantic")
    if int(acceptance.get("exact_opponent_levels", -1)) != 2:
        raise ValueError("provisional R7.3 prerequisite changed exact-opponent level")
    if acceptance.get("thread_environment_overrides_injected_by_bridge") is not False:
        raise ValueError("provisional R7.3 bridge injected or does not disprove thread overrides")
    if acceptance.get("frozen_cross_seed_gates_pass") is not True:
        raise ValueError("provisional R7.3 cross-seed strategy gates did not pass")
    if acceptance.get("structural_contract_pass") is not True:
        raise ValueError("provisional R7.3 structural contract did not pass")
    if acceptance.get("strategic_acceptance_gate_changed") is not False:
        raise ValueError("provisional R7.3 strategic acceptance gate changed")
    if acceptance.get("r7_3_strategy_quality_640_pass") is not True:
        raise ValueError("provisional R7.3 640 strategy quality did not pass")
    if acceptance.get("r7_4_provisional_advance_allowed") is not True:
        raise ValueError("provisional R7.3 report did not authorize engineering advance")
    if acceptance.get("certification_sequence_exception") is not True:
        raise ValueError("provisional R7.3 exception is not explicit")
    if acceptance.get("exact_reproducibility_must_close_before_ready_for_tables") is not True:
        raise ValueError("provisional R7.3 report lost exact-repro release debt")
    if acceptance.get("r7_3_fully_certified") is not False:
        raise ValueError("provisional R7.3 report incorrectly claims full certification")
    if acceptance.get("ready_for_tables") is not False:
        raise ValueError("provisional R7.3 report cannot authorize tables")
    return source_head


def _validate_ruleset_prerequisites(freeze: dict, acceptance: dict, ruleset_freeze: dict, ruleset_acceptance: dict) -> str:
    mode = _acceptance_mode(acceptance)
    r7_source = _validate_r7_3_prerequisites(freeze, acceptance)
    if ruleset_freeze.get("schema") != RULESET_FREEZE_SCHEMA or ruleset_freeze.get("ruleset_schema") != "SPINRULESET-4":
        raise ValueError("wrong R7.4 ruleset freeze")
    if ruleset_acceptance.get("schema") != RULESET_ACCEPT_SCHEMA or ruleset_acceptance.get("ruleset_schema") != "SPINRULESET-4":
        raise ValueError("wrong R7.4 ruleset acceptance")
    if ruleset_freeze.get("base_r7_3_source_head_sha") != r7_source:
        raise ValueError("SPINRULESET-4 base source differs from frozen R7.3 source")
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
    if ruleset_acceptance.get("ready_for_tables") is not False:
        raise ValueError("R7.4 ruleset acceptance cannot authorize tables")

    recorded_mode = ruleset_acceptance.get("r7_3_prerequisite_mode")
    if mode == "PROVISIONAL_640_STRATEGY_QUALITY":
        if recorded_mode != mode:
            raise ValueError("R7.4 ruleset evidence did not preserve provisional prerequisite mode")
        if ruleset_acceptance.get("r7_3_exact_reproducibility_debt_preserved") is not True:
            raise ValueError("R7.4 ruleset evidence lost exact-reproducibility debt")
    elif recorded_mode not in (None, mode):
        raise ValueError("R7.4 ruleset evidence reports inconsistent strict prerequisite mode")
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
    prerequisite_mode = _acceptance_mode(acceptance)

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
    report["r7_3_640_strategy_prerequisite_passed_first"] = True
    report["r7_3_640_acceptance_passed_first"] = prerequisite_mode == "STRICT_EXACT_CERTIFICATION"
    report["r7_3_prerequisite_mode"] = prerequisite_mode
    report["r7_3_exact_reproducibility_debt_preserved"] = prerequisite_mode == "PROVISIONAL_640_STRATEGY_QUALITY"
    report["r7_3_accepted_behavior_semantic_id"] = freeze["behavior_semantic_id"]
    report["r7_3_frozen_source_head_sha"] = freeze["source_head_sha"]
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
