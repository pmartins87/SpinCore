#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ID_RE = re.compile(r"^[A-Z]+-[0-9]{3}$")


def contract_ids() -> set[str]:
    ids: set[str] = set()
    paths = [ROOT / "PROJECT_CONTRACT.yaml", *sorted((ROOT / "contracts").glob("*.yaml"))]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        ids.update(re.findall(r"(?m)^\s*- id:\s*([A-Z]+-[0-9]{3})\s*$", text))
    return ids


def audit_coverage() -> dict[str, str]:
    path = ROOT / "contracts" / "AUDIT_STATUS.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()
    out: dict[str, str] = {}
    in_cov = False
    for line in lines:
        if line == "coverage:":
            in_cov = True
            continue
        if in_cov:
            if line and not line.startswith(" "):
                break
            m = re.match(r"^  ([A-Z]+):\s*([A-Z_]+)\s*$", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def root_status() -> str:
    text = (ROOT / "PROJECT_CONTRACT.yaml").read_text(encoding="utf-8")
    m = re.search(r"(?m)^status:\s*([A-Z_]+)\s*$", text)
    if not m:
        raise RuntimeError("PROJECT_CONTRACT root status missing")
    return m.group(1)


def git_blob(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path)], text=True, cwd=ROOT
    ).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--runner", type=Path, required=True)
    args = ap.parse_args()

    manifest_path = (ROOT / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest
    runner_path = (ROOT / args.runner).resolve() if not args.runner.is_absolute() else args.runner
    m = json.loads(manifest_path.read_text(encoding="utf-8"))

    if m.get("schema") != "SPINCORE_STAGE_MANIFEST_V1":
        raise SystemExit("STAGE_CONTRACT_FAIL wrong manifest schema")
    if Path(m.get("runner", "")).as_posix() != runner_path.relative_to(ROOT).as_posix():
        raise SystemExit("STAGE_CONTRACT_FAIL runner path mismatch")

    status = str(m.get("status"))
    if status != "READY":
        reason = str(m.get("block_reason") or status)
        print(f"STAGE_CONTRACT_BLOCKED status={status} reason={reason}")
        return 3

    known = contract_ids()
    refs = [str(x) for x in (m.get("contract_ids") or [])]
    if not refs or any(not ID_RE.fullmatch(x) for x in refs):
        raise SystemExit("STAGE_CONTRACT_FAIL invalid/empty contract_ids")
    unknown = sorted(set(refs) - known)
    if unknown:
        raise SystemExit("STAGE_CONTRACT_FAIL unknown contract IDs: " + ",".join(unknown))

    if root_status() == "BOOTSTRAP_AUDIT_REQUIRED":
        coverage = audit_coverage()
        affected = [str(x) for x in (m.get("affected_domains") or [])]
        incomplete = sorted(d for d in affected if coverage.get(d) != "COMPLETE")
        if incomplete:
            raise SystemExit(
                "STAGE_CONTRACT_FAIL affected contract domains not COMPLETE: "
                + ",".join(incomplete)
            )

    if str(m.get("duration_class")) == "LONG_GT_60M":
        pg = m.get("performance_gate") or {}
        if pg.get("status") != "PASS":
            raise SystemExit("STAGE_CONTRACT_FAIL long stage lacks PERFORMANCE_GATE_PASS")
        if not pg.get("evidence"):
            raise SystemExit("STAGE_CONTRACT_FAIL performance evidence missing")
        if not m.get("target_host_profile"):
            raise SystemExit("STAGE_CONTRACT_FAIL target_host_profile missing")

    expected_blob = str(m.get("runner_blob_sha") or "")
    if not expected_blob:
        raise SystemExit("STAGE_CONTRACT_FAIL runner_blob_sha missing")
    actual_blob = git_blob(runner_path)
    if actual_blob != expected_blob:
        raise SystemExit(
            f"STAGE_CONTRACT_FAIL runner blob drift actual={actual_blob} expected={expected_blob}"
        )

    print("STAGE_CONTRACT_PASS")
    print("stage_id=" + str(m.get("stage_id")))
    print("runner=" + runner_path.relative_to(ROOT).as_posix())
    print("contract_ids=" + ",".join(refs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
