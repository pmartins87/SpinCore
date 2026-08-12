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

import run_r7_3_frozen_candidate_fresh_repro as fresh_runner


FREEZE_SCHEMA = "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1"
REPORT_SCHEMA = "SPINCORE_R7_3_PROVISIONAL_640_STRATEGY_ACCEPTANCE_V1"
EXPECTED_GATES = {
    "advantage_weighted_nrmse_max": 0.75,
    "policy_weighted_mean_tv_max": 0.12,
    "cross_seed_mean_tv_max": 0.15,
    "cross_seed_p95_tv_max": 0.35,
}
DECISION_RECORD = "validation/R7_3_EXACT_REPRO_DEFERRED_DECISION_20260812.md"


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _same_number(a, b) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def _validate_strategy_evidence(evidence: dict) -> tuple[bool, dict]:
    frozen_gates = dict(evidence.get("frozen_gates") or {})
    gates_unchanged = all(_same_number(frozen_gates.get(k), v) for k, v in EXPECTED_GATES.items())

    per_seed = list(evidence.get("per_seed") or [])
    per_seed_details = []
    per_seed_pass = len(per_seed) == 5
    for item in per_seed:
        final_fit = dict(item.get("final_fit") or {})
        adv = float(final_fit.get("ensemble_advantage_weighted_nrmse", float("inf")))
        policy = float(final_fit.get("policy_weighted_mean_tv", float("inf")))
        adv_pass = bool(
            final_fit.get("advantage_gate_pass") is True
            and adv <= EXPECTED_GATES["advantage_weighted_nrmse_max"]
        )
        policy_pass = bool(
            final_fit.get("policy_gate_pass") is True
            and policy <= EXPECTED_GATES["policy_weighted_mean_tv_max"]
        )
        row_pass = bool(adv_pass and policy_pass)
        per_seed_pass = bool(per_seed_pass and row_pass)
        per_seed_details.append({
            "seed": item.get("seed", item.get("algorithm_seed")),
            "ensemble_advantage_weighted_nrmse": adv,
            "policy_weighted_mean_tv": policy,
            "advantage_gate_pass": adv_pass,
            "policy_gate_pass": policy_pass,
            "strategy_fit_pass": row_pass,
        })

    cross = dict(evidence.get("cross_seed") or {})
    mean_tv = float(cross.get("mean_tv", float("inf")))
    p95_tv = float(cross.get("p95_tv", float("inf")))
    cross_pass = bool(
        mean_tv <= EXPECTED_GATES["cross_seed_mean_tv_max"]
        and p95_tv <= EXPECTED_GATES["cross_seed_p95_tv_max"]
        and evidence.get("cross_seed_pass") is True
    )

    structural_pass = bool(
        int(evidence.get("iterations", -1)) == 5
        and int(evidence.get("roots_per_iteration", -1)) == 128
        and int(evidence.get("roots_per_seed", -1)) == 640
        and int(evidence.get("exact_opponent_levels", -1)) == 2
        and evidence.get("deck_formula") == "seed*1000003 + global_root*97 + iteration"
        and evidence.get("extra_members_perturb_primary_rng") is False
        and evidence.get("acceptance_gate_changed") is False
        and gates_unchanged
    )

    aggregate_flags_pass = bool(
        evidence.get("per_seed_fit_pass") is True
        and evidence.get("cross_seed_pass") is True
        and evidence.get("r7_3_pass") is True
    )
    passed = bool(structural_pass and per_seed_pass and cross_pass and aggregate_flags_pass)
    detail = {
        "frozen_gates": frozen_gates,
        "strategic_gates_unchanged": gates_unchanged,
        "per_seed": per_seed_details,
        "per_seed_strategy_fit_pass": per_seed_pass,
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "cross_seed_strategy_pass": cross_pass,
        "structural_contract_pass": structural_pass,
        "aggregate_evidence_flags_pass": aggregate_flags_pass,
    }
    return passed, detail


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run the frozen R7.3 winner at 640 roots/seed and enforce unchanged strategy-quality "
            "gates while exact fresh-process reproducibility remains explicit release debt"
        )
    )
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--evidence-out", type=Path, required=True)
    ap.add_argument("--report-out", type=Path, required=True)
    ap.add_argument(
        "--exact-repro-report",
        type=Path,
        default=Path("validation/R7_3_FRESH_REPRO_LAST_REPORT.json"),
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("evidence_r7_3_pass") is not True:
        raise SystemExit("invalid R7.3 semantic freeze")
    if freeze.get("thread_environment_contract") != fresh_runner.THREAD_ENV_CONTRACT:
        raise SystemExit("unexpected frozen thread-environment contract")
    if not (repo_root / DECISION_RECORD).is_file():
        raise SystemExit("missing explicit R7.3 exact-repro deferral decision record")

    exact_report = {}
    if args.exact_repro_report.is_file():
        exact_report = json.loads(args.exact_repro_report.read_text(encoding="utf-8"))

    acceptance_freeze = copy.deepcopy(freeze)
    acceptance_freeze["execution_contract"] = dict(freeze["execution_contract"])
    acceptance_freeze["execution_contract"]["roots_per_iteration"] = 128

    source_head = str(freeze["source_head_sha"])
    args.evidence_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="spincore_r7_provisional_640_") as td:
        worktree = Path(td) / "source"
        temp_evidence = Path(td) / "strategy_640.json"
        _run(["git", "worktree", "add", "--detach", str(worktree), source_head], cwd=repo_root)
        try:
            _run(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], cwd=worktree)
            _run(["cmake", "--build", "build", "-j2"], cwd=worktree)
            _run(["ctest", "--test-dir", "build", "--output-on-failure"], cwd=worktree)
            env = fresh_runner._source_execution_env(freeze, os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(worktree / "python"), str(worktree / "tools")])
            _run([sys.executable, "-m", "pytest", "-q", "python_tests"], cwd=worktree, env=env)
            _run(fresh_runner._runner_command(acceptance_freeze, temp_evidence), cwd=worktree, env=env)
            shutil.copyfile(temp_evidence, args.evidence_out)
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=repo_root,
                check=False,
            )

    evidence = json.loads(args.evidence_out.read_text(encoding="utf-8"))
    strategy_pass, detail = _validate_strategy_evidence(evidence)

    exact_pass = bool(
        exact_report.get("fresh_process_reproducible") is True
        and int(exact_report.get("difference_count", -1)) == 0
    )
    report = {
        "schema": REPORT_SCHEMA,
        "label": freeze["label"],
        "behavior_semantic_id": freeze["behavior_semantic_id"],
        "source_head_sha": source_head,
        "durability_evidence_commit_sha": freeze["evidence_commit_sha"],
        "durability_evidence_sha256": freeze["evidence_sha256"],
        "decision_record": DECISION_RECORD,
        "strategy_evidence_path": str(args.evidence_out),
        "thread_environment_contract": freeze["thread_environment_contract"],
        "thread_environment_overrides_injected_by_bridge": False,
        "iterations": 5,
        "roots_per_iteration": 128,
        "roots_per_seed": 640,
        "exact_opponent_levels": 2,
        "strategy_gate_detail": detail,
        "per_seed_fit_pass": bool(detail["per_seed_strategy_fit_pass"]),
        "frozen_cross_seed_gates_pass": bool(detail["cross_seed_strategy_pass"]),
        "structural_contract_pass": bool(detail["structural_contract_pass"]),
        "strategic_acceptance_gate_changed": False,
        "certification_sequence_exception": True,
        "exact_reproducibility_pass": exact_pass,
        "exact_reproducibility_difference_count": exact_report.get("difference_count"),
        "exact_reproducibility_deferred": not exact_pass,
        "exact_reproducibility_must_close_before_ready_for_tables": True,
        "action_level_sentinels_required_before_ready_for_tables": True,
        "r7_3_strategy_quality_640_pass": strategy_pass,
        "r7_4_provisional_advance_allowed": strategy_pass,
        "r7_3_fully_certified": False if not exact_pass else strategy_pass,
        "ready_for_tables": False,
    }
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if strategy_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
