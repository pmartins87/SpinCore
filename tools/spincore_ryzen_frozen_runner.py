from __future__ import annotations

"""Generic evidence wrapper for heavy frozen SpinCore runs on the Ryzen host.

This program deliberately does *not* know poker/CFR semantics.  It verifies the
source identity and tracked-worktree cleanliness, records the runtime, executes
one already-frozen command, and hashes the declared result artifacts.  GitHub
remains the certification/referee environment for returned artifacts.

Example (PowerShell, from an exact checked-out commit):

  python tools/spincore_ryzen_frozen_runner.py `
    --expected-commit <40-char-sha> `
    --run-name r7_5_3c_example `
    --contract validation/SOME_FROZEN_CONTRACT.json `
    --artifact runs/example/output `
    -- python tools/some_frozen_worker.py --out runs/example/output
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Iterable

SCHEMA = "SPINCORE_RYZEN_FROZEN_RUN_MANIFEST_V1"
RUNNER_VERSION = "2026-08-16.1"
THREAD_ENV_KEYS = (
    "SPINCORE_TORCH_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def safe_name(text: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(text).strip())
    out = out.strip("._-")
    if not out:
        raise ValueError("run name becomes empty after sanitization")
    return out


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def resolve_commit(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", f"{ref}^{{commit}}")


def tracked_status(repo: Path) -> str:
    # Generated build/run directories are allowed to be untracked.  Any tracked
    # modification is forbidden because it would invalidate the frozen source.
    return _git(repo, "status", "--porcelain=v1", "--untracked-files=no")


def runtime_inventory() -> dict[str, object]:
    optional: dict[str, object] = {}
    try:
        import torch  # type: ignore

        optional["torch_version"] = str(torch.__version__)
        optional["torch_num_threads"] = int(torch.get_num_threads())
        optional["torch_num_interop_threads"] = int(torch.get_num_interop_threads())
    except Exception as exc:  # pragma: no cover - environment dependent
        optional["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import numpy as np  # type: ignore

        optional["numpy_version"] = str(np.__version__)
    except Exception as exc:  # pragma: no cover - environment dependent
        optional["numpy_import_error"] = f"{type(exc).__name__}: {exc}"

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
        **optional,
    }


def parse_env(rows: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        if "=" not in row:
            raise ValueError(f"--env requires KEY=VALUE, got {row!r}")
        key, value = row.split("=", 1)
        key = key.strip()
        if not key or "\x00" in key or "=" in key:
            raise ValueError(f"invalid environment key {key!r}")
        result[key] = value
    return result


def normalize_declared_path(repo: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo / path
    return path.resolve()


def inventory_path(path: Path, *, label: str) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"declared {label} does not exist: {path}")
    if path.is_file():
        return [{
            "declared_as": label,
            "path": str(path),
            "relative_path": path.name,
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }]
    if not path.is_dir():
        raise ValueError(f"declared {label} is neither regular file nor directory: {path}")
    rows: list[dict[str, object]] = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rows.append({
            "declared_as": label,
            "path": str(item),
            "relative_path": item.relative_to(path).as_posix(),
            "size_bytes": int(item.stat().st_size),
            "sha256": sha256_file(item),
        })
    return rows


def write_manifest(path: Path, payload: dict[str, object]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def tee_process(command: list[str], *, repo: Path, env: dict[str, str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8", errors="replace", newline="") as log:
        proc = subprocess.Popen(
            command,
            cwd=str(repo),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return int(proc.wait())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a frozen SpinCore command on Ryzen with auditable evidence")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--expected-commit", required=True, help="Frozen commit/ref; HEAD must resolve exactly to it")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("ryzen_runs"))
    parser.add_argument("--contract", action="append", default=[], help="Frozen contract/evidence file to hash; repeatable")
    parser.add_argument("--artifact", action="append", default=[], help="Result file/directory to SHA-256 inventory after success; repeatable")
    parser.add_argument("--env", action="append", default=[], help="Explicit child environment KEY=VALUE; repeatable")
    parser.add_argument("--require-no-untracked", action="store_true", help="Also reject pre-existing untracked files")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    args = parser.parse_args()

    repo = args.repo_root.expanduser().resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"not a Git working tree root: {repo}")

    expected = resolve_commit(repo, str(args.expected_commit))
    head = resolve_commit(repo, "HEAD")
    if head != expected:
        raise SystemExit(f"frozen commit mismatch: HEAD={head} expected={expected}")
    dirty = tracked_status(repo)
    if dirty:
        raise SystemExit("tracked worktree is not clean; refusing frozen run:\n" + dirty)
    if args.require_no_untracked:
        any_status = _git(repo, "status", "--porcelain=v1")
        if any_status:
            raise SystemExit("worktree contains tracked/untracked changes; refusing strict frozen run:\n" + any_status)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing frozen command; place it after --")

    contracts = [normalize_declared_path(repo, raw) for raw in args.contract]
    contract_inventory: list[dict[str, object]] = []
    for path, raw in zip(contracts, args.contract):
        if not path.is_file():
            raise SystemExit(f"contract must be a regular file: {path}")
        contract_inventory.append({
            "declared_path": str(raw),
            "resolved_path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })

    child_env = dict(os.environ)
    explicit_env = parse_env(args.env)
    child_env.update(explicit_env)

    run_name = safe_name(args.run_name)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = repo / output_root
    run_dir = (output_root / f"{stamp}_{run_name}_{head[:12]}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    child_env["SPINCORE_RYZEN_RUN_DIR"] = str(run_dir)
    log_path = run_dir / "console.log"
    manifest_path = run_dir / "manifest.json"

    start_utc = utc_now()
    start_perf = time.perf_counter()
    preflight: dict[str, object] = {
        "schema": SCHEMA,
        "runner_version": RUNNER_VERSION,
        "status": "RUNNING",
        "run_name": run_name,
        "run_dir": str(run_dir),
        "repo_root": str(repo),
        "git": {
            "head_commit": head,
            "expected_commit": expected,
            "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "tracked_worktree_clean": True,
            "require_no_untracked": bool(args.require_no_untracked),
        },
        "contracts": contract_inventory,
        "command": command,
        "command_display": subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command),
        "explicit_child_environment": explicit_env,
        "runtime": runtime_inventory(),
        "started_at_utc": start_utc,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    write_manifest(manifest_path, preflight)

    print(f"[SpinCore Ryzen] frozen HEAD {head}", flush=True)
    print(f"[SpinCore Ryzen] run directory {run_dir}", flush=True)
    print(f"[SpinCore Ryzen] command: {preflight['command_display']}", flush=True)

    return_code: int | None = None
    failure: str | None = None
    try:
        return_code = tee_process(command, repo=repo, env=child_env, log_path=log_path)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.perf_counter() - start_perf
        artifact_inventory: list[dict[str, object]] = []
        artifact_error: str | None = None
        if return_code == 0:
            try:
                for raw in args.artifact:
                    target = normalize_declared_path(repo, raw)
                    artifact_inventory.extend(inventory_path(target, label=str(raw)))
            except Exception as exc:
                artifact_error = f"{type(exc).__name__}: {exc}"
                return_code = 97

        final = dict(preflight)
        final.update({
            "status": "SUCCESS" if return_code == 0 and failure is None and artifact_error is None else "FAILED",
            "finished_at_utc": utc_now(),
            "elapsed_seconds": float(elapsed),
            "return_code": return_code,
            "runner_exception": failure,
            "artifact_inventory_error": artifact_error,
            "artifacts": artifact_inventory,
            "artifact_file_count": len(artifact_inventory),
            "artifact_total_bytes": int(sum(int(row["size_bytes"]) for row in artifact_inventory)),
            "console_log": {
                "path": str(log_path),
                "exists": log_path.exists(),
                "size_bytes": int(log_path.stat().st_size) if log_path.exists() else 0,
                "sha256": sha256_file(log_path) if log_path.exists() else None,
            },
        })
        manifest_digest = write_manifest(manifest_path, final)
        print(f"[SpinCore Ryzen] manifest SHA256 {manifest_digest}", flush=True)
        print(f"[SpinCore Ryzen] status {final['status']} return_code={return_code}", flush=True)

    return int(return_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
