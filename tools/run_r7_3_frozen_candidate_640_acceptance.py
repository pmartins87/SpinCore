from __future__ import annotations

import argparse
import copy
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
REPORT_SCHEMA = "SPINCORE_R7_3_FROZEN_CANDIDATE_640_ACCEPTANCE_V1"


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _validate_fresh(freeze: dict, fresh: dict) -> None:
    if fresh.get("schema") != FRESH_SCHEMA or fresh.get("fresh_process_reproducible") is not True:
        raise ValueError("fresh-process reproducibility must pass first")
    if int(fresh.get("difference_count", -1)) != 0:
        raise ValueError("fresh-process reproducibility did not produce zero differences")
    if fresh.get("source_head_sha") != freeze.get("source_head_sha"):
        raise ValueError("fresh report source head differs from freeze")
    if fresh.get("original_evidence_commit_sha") != freeze.get("evidence_commit_sha"):
        raise ValueError("fresh report evidence commit differs from freeze")
    if fresh.get("original_evidence_sha256") != freeze.get("evidence_sha256"):
        raise ValueError("fresh report evidence bytes differ from freeze")
    if fresh.get("behavior_semantic_id") != freeze.get("behavior_semantic_id"):
        raise ValueError("fresh report behavior semantic differs from freeze")
    if fresh.get("thread_environment_contract") != freeze.get("thread_environment_contract"):
        raise ValueError("fresh report thread environment differs from freeze")
    if fresh.get("thread_environment_overrides_injected_by_certifier") is not False:
        raise ValueError("fresh report used or does not disprove hidden thread overrides")


def _validate_checkpoint(freeze: dict, checkpoint: dict) -> None:
    if checkpoint.get("schema") != RECERT_SCHEMA or checkpoint.get("checkpoint_resume_recertification_pass") is not True:
        raise ValueError("checkpoint/resume recertification must pass first")
    if checkpoint.get("source_head_sha") != freeze.get("source_head_sha"):
        raise ValueError("checkpoint source head differs from freeze")
    if checkpoint.get("evidence_commit_sha") != freeze.get("evidence_commit_sha"):
        raise ValueError("checkpoint evidence commit differs from freeze")
    if checkpoint.get("behavior_semantic_id") != freeze.get("behavior_semantic_id"):
        raise ValueError("checkpoint behavior semantic differs from freeze")
    if checkpoint.get("thread_environment_contract") != freeze.get("thread_environment_contract"):
        raise ValueError("checkpoint report thread environment differs from freeze")
    if checkpoint.get("thread_environment_overrides_injected_by_certifier") is not False:
        raise ValueError("checkpoint report used or does not disprove hidden thread overrides")
    if checkpoint.get("algorithm_source_exact_worktree") is not True:
        raise ValueError("checkpoint recertification did not use exact frozen source worktree")
    if checkpoint.get("checkpoint_worker_executed_from_frozen_worktree_overlay") is not True:
        raise ValueError("checkpoint worker was not executed from frozen worktree overlay")
    if checkpoint.get("fresh_process_zero_difference_gate_passed_first") is not True:
        raise ValueError("checkpoint recertification did not require zero-difference fresh gate")
    if checkpoint.get("validated_fresh_original_evidence_sha256") != freeze.get("evidence_sha256"):
        raise ValueError("checkpoint did not attest the frozen fresh evidence bytes")
    if checkpoint.get("validated_fresh_behavior_semantic_id") != freeze.get("behavior_semantic_id"):
        raise ValueError("checkpoint did not attest the frozen fresh behavior semantic")
    if checkpoint.get("acceptance_gate_changed") is not False:
        raise ValueError("checkpoint recertification changed acceptance gate")


def _validate_certification_chain(freeze: dict, fresh: dict, checkpoint: dict) -> str:
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("evidence_r7_3_pass") is not True:
        raise ValueError("invalid semantic freeze")
    _validate_fresh(freeze, fresh)
    _validate_checkpoint(freeze, checkpoint)
    return str(freeze["source_head_sha"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Run 640-root acceptance from the exact source of a fully certified R7.3 durability winner")
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--fresh-report", type=Path, required=True)
    ap.add_argument("--checkpoint-report", type=Path, required=True)
    ap.add_argument("--evidence-out", type=Path, required=True)
    ap.add_argument("--report-out", type=Path, required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("evidence_r7_3_pass") is not True:
        raise SystemExit("invalid semantic freeze")
    try:
        fresh, fresh_origin = evidence_resolver.resolve_valid_json(
            args.fresh_report,
            validator=lambda data: _validate_fresh(freeze, data),
            repo_root=repo_root,
        )
        checkpoint, checkpoint_origin = evidence_resolver.resolve_valid_json(
            args.checkpoint_report,
            validator=lambda data: _validate_checkpoint(freeze, data),
            repo_root=repo_root,
        )
        source_head = _validate_certification_chain(freeze, fresh, checkpoint)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    acceptance_freeze = copy.deepcopy(freeze)
    acceptance_freeze["execution_contract"] = dict(freeze["execution_contract"])
    acceptance_freeze["execution_contract"]["roots_per_iteration"] = 128

    args.evidence_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spincore_r7_acceptance_") as td:
        worktree = Path(td) / "source"
        _run(["git", "worktree", "add", "--detach", str(worktree), source_head], cwd=repo_root)
        try:
            _run(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], cwd=worktree)
            _run(["cmake", "--build", "build", "-j2"], cwd=worktree)
            _run(["ctest", "--test-dir", "build", "--output-on-failure"], cwd=worktree)
            env = fresh_runner._source_execution_env(freeze, os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(worktree / "python"), str(worktree / "tools")])
            _run([sys.executable, "-m", "pytest", "-q", "python_tests"], cwd=worktree, env=env)
            temp_evidence = Path(td) / "acceptance.json"
            _run(fresh_runner._runner_command(acceptance_freeze, temp_evidence), cwd=worktree, env=env)
            shutil.copyfile(temp_evidence, args.evidence_out)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo_root, check=False)

    evidence = json.loads(args.evidence_out.read_text(encoding="utf-8"))
    cross = dict(evidence.get("cross_seed") or {})
    structural_pass = bool(
        int(evidence.get("iterations", -1)) == 5
        and int(evidence.get("roots_per_iteration", -1)) == 128
        and int(evidence.get("roots_per_seed", -1)) == 640
        and int(evidence.get("exact_opponent_levels", -1)) == 2
        and evidence.get("deck_formula") == "seed*1000003 + global_root*97 + iteration"
        and evidence.get("extra_members_perturb_primary_rng") is False
        and evidence.get("acceptance_gate_changed") is False
        and evidence.get("per_seed_fit_pass") is True
    )
    cross_pass = bool(
        float(cross.get("mean_tv", float("inf"))) <= 0.15
        and float(cross.get("p95_tv", float("inf"))) <= 0.35
        and evidence.get("cross_seed_pass") is True
        and evidence.get("r7_3_pass") is True
    )
    passed = bool(structural_pass and cross_pass)
    report = {
        "schema": REPORT_SCHEMA,
        "label": freeze["label"],
        "behavior_semantic_id": freeze["behavior_semantic_id"],
        "source_head_sha": source_head,
        "durability_evidence_commit_sha": freeze["evidence_commit_sha"],
        "durability_evidence_sha256": freeze["evidence_sha256"],
        "validated_fresh_report_origin": fresh_origin,
        "validated_checkpoint_report_origin": checkpoint_origin,
        "legacy_or_stale_head_evidence_can_be_bypassed_only_via_provenance_valid_git_history": True,
        "acceptance_evidence_path": str(args.evidence_out),
        "thread_environment_contract": freeze["thread_environment_contract"],
        "thread_environment_overrides_injected_by_certifier": False,
        "fresh_process_zero_difference_gate_passed_first": True,
        "checkpoint_exact_state_gate_passed_first": True,
        "checkpoint_worker_executed_from_frozen_worktree_overlay": True,
        "iterations": 5,
        "roots_per_iteration": 128,
        "roots_per_seed": 640,
        "per_seed_fit_pass": bool(evidence.get("per_seed_fit_pass")),
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "structural_contract_pass": structural_pass,
        "frozen_cross_seed_gates_pass": cross_pass,
        "r7_3_640_acceptance_pass": passed,
        "acceptance_gate_changed": False,
        "r7_3_ready_to_advance_to_r7_4": passed,
        "ready_for_tables": False,
    }
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
