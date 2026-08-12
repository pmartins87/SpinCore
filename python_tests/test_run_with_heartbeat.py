from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_with_heartbeat.py"


def _events(stdout: str) -> list[dict]:
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if row.get("heartbeat_schema") == "SPINCORE_LONG_JOB_HEARTBEAT_V1":
            out.append(row)
    return out


def test_heartbeat_reports_liveness_without_changing_child_exit_code():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--label",
            "test-sleep",
            "--interval-seconds",
            "1",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(1.2)",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    events = _events(proc.stdout)
    assert events[0]["event"] == "started"
    assert any(row["event"] == "alive" for row in events)
    assert events[-1]["event"] == "completed"
    assert events[-1]["returncode"] == 0


def test_heartbeat_propagates_failure_code_exactly():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--interval-seconds",
            "60",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 7
    events = _events(proc.stdout)
    assert events[-1]["event"] == "completed"
    assert events[-1]["returncode"] == 7
