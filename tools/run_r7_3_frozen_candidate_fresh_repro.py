from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


FREEZE_SCHEMA = "SPINCORE_R7_3_CANDIDATE_SEMANTIC_FREEZE_V1"
REPORT_SCHEMA = "SPINCORE_R7_3_FROZEN_CANDIDATE_FRESH_REPRO_V1"
THREAD_ENV_CONTRACT = "SOURCE_WORKFLOW_NO_EXPLICIT_THREAD_OVERRIDE"
THREAD_ENV_KEYS = ("SPINCORE_TORCH_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

RUNNERS = {
    "uncertainty_damping": "tools/run_r7_3_policy_mixture_uncertainty_damping.py",
    "temporal_blend": "tools/run_r7_3_policy_mixture_temporal_blend.py",
    "policy_mixture": "tools/run_r7_3_partial_exact_policy_mixture_paired.py",
}

IGNORE_KEYS = {"generated_at_unix", "duration_seconds"}


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def _source_execution_env(freeze: dict, base_env) -> dict[str, str]:
    """Return the caller environment without inventing thread-count overrides.

    The selected evidence workflows did not explicitly set PyTorch/OpenMP/MKL
    thread variables. Exact-source certification must therefore inherit the
    runner environment exactly as that source workflow did, rather than forcing
    a new thread count such as 2. The semantic freeze records and validates this
    contract before any fresh/checkpoint/640 execution is allowed.
    """
    contract = str(freeze.get("thread_environment_contract", ""))
    if contract != THREAD_ENV_CONTRACT:
        raise ValueError(f"unsupported frozen thread environment contract: {contract!r}")
    return dict(base_env)


def _compare(a, b, path="$") -> list[dict]:
    diffs: list[dict] = []
    if isinstance(a, dict) and isinstance(b, dict):
        ak = set(a) - IGNORE_KEYS
        bk = set(b) - IGNORE_KEYS
        if ak != bk:
            diffs.append({"path": path, "kind": "KEY_SET", "left_only": sorted(ak - bk), "right_only": sorted(bk - ak)})
        for key in sorted(ak & bk):
            diffs.extend(_compare(a[key], b[key], f"{path}.{key}"))
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [{"path": path, "kind": "LIST_LENGTH", "left": len(a), "right": len(b)}]
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(_compare(x, y, f"{path}[{i}]"))
        return diffs
    if isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            diffs.append({"path": path, "kind": "BOOL", "left": a, "right": b})
        return diffs
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        x, y = float(a), float(b)
        if math.isnan(x) and math.isnan(y):
            return diffs
        delta = abs(x - y)
        if not math.isfinite(delta) or delta > 1e-9:
            diffs.append({"path": path, "kind": "NUMBER", "left": a, "right": b, "abs_delta": delta})
        return diffs
    if a != b:
        diffs.append({"path": path, "kind": "VALUE", "left": a, "right": b})
    return diffs


def _runner_command(freeze: dict, fresh_out: Path) -> list[str]:
    kind = freeze["behavior_kind"]
    ec = freeze["execution_contract"]
    cmd = [
        sys.executable,
        RUNNERS[kind],
        "--solver", "build/libspincore_solver_c.so",
        "--out", str(fresh_out),
        "--ensemble-size", str(int(freeze["ensemble_size"])),
        "--exact-opponent-levels", str(int(ec["exact_opponent_levels"])),
        "--iterations", str(int(ec["iterations"])),
        "--roots-per-iteration", str(int(ec["roots_per_iteration"])),
        "--advantage-chunk-steps", str(int(ec["advantage_chunk_steps"])),
        "--advantage-max-steps-per-iteration", str(int(ec["advantage_max_steps_per_iteration"])),
        "--advantage-fit-target", str(float(ec["advantage_fit_target"])),
        "--policy-chunk-steps", str(int(ec["policy_chunk_steps"])),
        "--policy-max-steps", str(int(ec["policy_max_steps"])),
        "--policy-fit-target", str(float(ec["policy_fit_target"])),
        "--batch-size", str(int(ec["batch_size"])),
        "--audit-size", str(int(ec["audit_size"])),
        "--cross-seed-per-seed", str(int(ec["cross_seed_per_seed"])),
        "--reservoir-capacity", str(int(ec["reservoir_capacity"])),
    ]
    params = freeze.get("params") or {}
    if kind == "uncertainty_damping":
        cmd += ["--epsilon-scale", str(float(params["epsilon_scale"])), "--epsilon-cap", str(float(params["epsilon_cap"]))]
    elif kind == "temporal_blend":
        cmd += ["--current-weight", str(float(params["current_policy_weight"]))]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-run a frozen R7.3 winner from its exact source commit and compare complete deterministic evidence")
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--fresh-out", type=Path, default=Path("validation/R7_3_FROZEN_CANDIDATE_FRESH_EVIDENCE.json"))
    ap.add_argument("--report", type=Path, default=Path("validation/R7_3_FROZEN_CANDIDATE_FRESH_REPRO.json"))
    args = ap.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise SystemExit("wrong semantic-freeze schema")
    if freeze.get("evidence_r7_3_pass") is not True:
        raise SystemExit("semantic freeze is not based on a gate-clearing candidate")
    kind = str(freeze.get("behavior_kind", ""))
    if kind not in RUNNERS:
        raise SystemExit(f"unsupported frozen behavior kind: {kind!r}")

    evidence_path = str(freeze["evidence_path"])
    evidence_commit = str(freeze.get("evidence_commit_sha", ""))
    if len(evidence_commit) < 12:
        raise SystemExit("semantic freeze does not pin evidence_commit_sha")
    original_bytes = _git_bytes("show", f"{evidence_commit}:{evidence_path}")
    expected_sha256 = str(freeze.get("evidence_sha256", ""))
    actual_sha256 = hashlib.sha256(original_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit("immutable evidence bytes do not match semantic-freeze SHA-256")
    original = json.loads(original_bytes.decode("utf-8"))
    source_head = str(freeze["source_head_sha"])

    args.fresh_out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spincore_r7_frozen_") as td:
        worktree = Path(td) / "source"
        _run(["git", "worktree", "add", "--detach", str(worktree), source_head], cwd=Path("."))
        try:
            _run(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], cwd=worktree)
            _run(["cmake", "--build", "build", "-j2"], cwd=worktree)
            _run(["ctest", "--test-dir", "build", "--output-on-failure"], cwd=worktree)
            env = _source_execution_env(freeze, os.environ)
            env["PYTHONPATH"] = str(worktree / "python")
            _run([sys.executable, "-m", "pytest", "-q", "python_tests"], cwd=worktree, env=env)
            temp_fresh = Path(td) / "fresh.json"
            _run(_runner_command(freeze, temp_fresh), cwd=worktree, env=env)
            fresh = json.loads(temp_fresh.read_text(encoding="utf-8"))
            shutil.copyfile(temp_fresh, args.fresh_out)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], check=False)

    diffs = _compare(original, fresh)
    report = {
        "schema": REPORT_SCHEMA,
        "label": freeze["label"],
        "behavior_semantic_id": freeze["behavior_semantic_id"],
        "source_head_sha": source_head,
        "original_evidence_path": evidence_path,
        "original_evidence_commit_sha": evidence_commit,
        "original_evidence_sha256": actual_sha256,
        "fresh_evidence_path": str(args.fresh_out),
        "source_cpp_regression_required": True,
        "source_python_regression_required": True,
        "thread_environment_contract": freeze["thread_environment_contract"],
        "thread_environment_overrides_injected_by_certifier": False,
        "ignored_nondeterministic_keys": sorted(IGNORE_KEYS),
        "numeric_tolerance": 1e-9,
        "difference_count": len(diffs),
        "differences": diffs[:200],
        "fresh_process_reproducible": len(diffs) == 0,
        "acceptance_gate_changed": False,
        "ready_for_640": False,
        "ready_for_tables": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["fresh_process_reproducible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
