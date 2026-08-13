from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path


_R7_4_3H_PYTHON = "3.11.15"
_R7_4_3H_TORCH = "2.13.0+cpu"


def _required_runtime_for_label(label: str) -> tuple[str, str] | None:
    if str(label).startswith("r7.4-3h"):
        return (_R7_4_3H_PYTHON, _R7_4_3H_TORCH)
    return None


def _validate_runtime_contract(label: str, python_version: str, torch_version: str) -> None:
    required = _required_runtime_for_label(label)
    if required is None:
        return
    required_python, required_torch = required
    if python_version != required_python or torch_version != required_torch:
        raise RuntimeError(
            "frozen R7.4 3H runtime mismatch: "
            f"required python={required_python} torch={required_torch}; "
            f"observed python={python_version} torch={torch_version}"
        )


def _enforce_runtime_contract(label: str) -> None:
    required = _required_runtime_for_label(label)
    if required is None:
        return
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised as fail-closed integration behavior
        raise RuntimeError("frozen R7.4 3H runtime requires importable torch") from exc
    _validate_runtime_contract(str(label), platform.python_version(), str(torch.__version__))
    print(
        json.dumps(
            {
                "event": "runtime_contract_pass",
                "label": str(label),
                "python": platform.python_version(),
                "runtime_contract_schema": "SPINCORE_R7_4_3H_RUNTIME_CONTRACT_V1",
                "torch": str(torch.__version__),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _process_snapshot(root_pid: int) -> dict:
    """Best-effort Linux process-tree telemetry; never affects child semantics."""
    try:
        text = subprocess.check_output(
            ["ps", "-e", "-o", "pid=,ppid=,%cpu=,rss=,etime=,comm="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    rows: dict[int, dict] = {}
    children: dict[int, list[int]] = {}
    for raw in text.splitlines():
        parts = raw.strip().split(None, 5)
        if len(parts) < 6:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            cpu = float(parts[2])
            rss_kib = int(parts[3])
        except ValueError:
            continue
        row = {
            "pid": pid,
            "ppid": ppid,
            "cpu_percent": cpu,
            "rss_kib": rss_kib,
            "etime": parts[4],
            "command": parts[5],
        }
        rows[pid] = row
        children.setdefault(ppid, []).append(pid)

    selected: list[int] = []
    queue = [int(root_pid)]
    seen: set[int] = set()
    while queue:
        pid = queue.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if pid in rows:
            selected.append(pid)
        queue.extend(children.get(pid, ()))

    selected_rows = [rows[pid] for pid in selected if pid in rows]
    selected_rows.sort(key=lambda row: (-row["cpu_percent"], -row["rss_kib"], row["pid"]))
    return {
        "available": True,
        "process_count": len(selected_rows),
        "cpu_percent_sum": round(sum(float(row["cpu_percent"]) for row in selected_rows), 2),
        "rss_mib_sum": round(sum(int(row["rss_kib"]) for row in selected_rows) / 1024.0, 1),
        "top_processes": selected_rows[:8],
    }


def _emit(label: str, event: str, started: float, pid: int, returncode: int | None = None) -> None:
    payload = {
        "heartbeat_schema": "SPINCORE_LONG_JOB_HEARTBEAT_V1",
        "label": label,
        "event": event,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "child_pid": int(pid),
        "returncode": returncode,
        "process_tree": _process_snapshot(pid),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run an unchanged command while emitting observability-only heartbeats")
    ap.add_argument("--label", default="long-job")
    ap.add_argument("--interval-seconds", type=float, default=300.0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a command is required after --")
    if not (1.0 <= float(args.interval_seconds) <= 3600.0):
        raise SystemExit("interval-seconds must be between 1 and 3600")

    _enforce_runtime_contract(str(args.label))
    print("+ heartbeat child:", " ".join(command), flush=True)
    proc = subprocess.Popen(command, start_new_session=(os.name == "posix"))
    started = time.monotonic()

    def _forward(sig, _frame):
        try:
            if proc.poll() is None:
                if os.name == "posix":
                    os.killpg(proc.pid, sig)
                else:
                    proc.send_signal(sig)
        finally:
            raise SystemExit(128 + int(sig))

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _forward)

    _emit(str(args.label), "started", started, proc.pid)
    while True:
        try:
            rc = proc.wait(timeout=float(args.interval_seconds))
            _emit(str(args.label), "completed", started, proc.pid, int(rc))
            return int(rc)
        except subprocess.TimeoutExpired:
            _emit(str(args.label), "alive", started, proc.pid)


if __name__ == "__main__":
    raise SystemExit(main())
