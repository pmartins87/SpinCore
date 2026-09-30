#!/usr/bin/env python3
from __future__ import annotations

"""Dependency-free contract preflight for the postlong-10315 performance gate.

PROJECT_CONTRACT_IDS:
GOV-022,GOV-023,GOV-024,PERF-001,PERF-002,PERF-010,PERF-013,PERF-014,
PERF-015,PERF-017,PERF-025,TRAIN-021,MODEL-021,MODEL-022,MODEL-023,
RNG-001,RNG-002,RNG-003,VALID-002,VALID-020,VALID-021,VALID-022,
CKPT-001,ART-001,SRC-003

This checker intentionally uses only the Python standard library. Full YAML
contract lint remains a GitHub-CI responsibility; the local Ryzen gate verifies
the exact contract state that authorizes this performance experiment.
"""

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_IDS = {
    "GOV-022","GOV-023","GOV-024",
    "PERF-001","PERF-002","PERF-010","PERF-013","PERF-014","PERF-015",
    "PERF-017","PERF-025",
    "TRAIN-021","MODEL-021","MODEL-022","MODEL-023",
    "RNG-001","RNG-002","RNG-003",
    "VALID-002","VALID-020","VALID-021","VALID-022",
    "CKPT-001","ART-001","SRC-003",
}


def contract_ids() -> set[str]:
    ids: set[str] = set()
    paths = [ROOT / "PROJECT_CONTRACT.yaml", *sorted((ROOT / "contracts").glob("*.yaml"))]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        ids.update(re.findall(r"(?m)^\s*- id:\s*([A-Z]+-[0-9]{3})\s*$", text))
    return ids


def root_status() -> str:
    text = (ROOT / "PROJECT_CONTRACT.yaml").read_text(encoding="utf-8")
    m = re.search(r"(?m)^status:\s*([A-Z_]+)\s*$", text)
    if not m:
        raise RuntimeError("PROJECT_CONTRACT root status missing")
    return m.group(1)


def main() -> int:
    if root_status() != "COMPLETE":
        raise SystemExit("POSTLONG_PERF_PREFLIGHT_FAIL project contract is not COMPLETE")

    known = contract_ids()
    missing = sorted(REQUIRED_IDS - known)
    if missing:
        raise SystemExit(
            "POSTLONG_PERF_PREFLIGHT_FAIL missing contract IDs: " + ",".join(missing)
        )

    postlong_path = ROOT / "contracts/run_manifests/postlong_10315.json"
    postlong = json.loads(postlong_path.read_text(encoding="utf-8"))
    if postlong.get("schema") != "SPINCORE_STAGE_MANIFEST_V1":
        raise SystemExit("POSTLONG_PERF_PREFLIGHT_FAIL wrong postlong manifest schema")
    if postlong.get("stage_id") != "POSTLONG_3H_SEMANTIC_10315_REBUILD_AND_VALIDATION":
        raise SystemExit("POSTLONG_PERF_PREFLIGHT_FAIL wrong postlong stage_id")
    if postlong.get("status") != "BLOCKED_PERFORMANCE_GATE":
        raise SystemExit(
            "POSTLONG_PERF_PREFLIGHT_FAIL postlong must still be BLOCKED_PERFORMANCE_GATE"
        )
    perf = postlong.get("performance_gate") or {}
    if perf.get("status") != "PENDING":
        raise SystemExit(
            "POSTLONG_PERF_PREFLIGHT_FAIL performance gate must be PENDING before benchmark"
        )
    if perf.get("runner") != "tools/run_3h_semantic_postlong_10315_performance_gate.sh":
        raise SystemExit("POSTLONG_PERF_PREFLIGHT_FAIL performance-gate runner drift")
    if postlong.get("target_host_profile") != "RYZEN_9_9950X_32_LOGICAL_THREADS":
        raise SystemExit("POSTLONG_PERF_PREFLIGHT_FAIL target host profile drift")

    refs = set(str(x) for x in (postlong.get("contract_ids") or []))
    missing_manifest_ids = sorted(
        {
            "GOV-023","PERF-001","PERF-002","PERF-010","PERF-013",
            "PERF-014","PERF-015","PERF-017","PERF-025",
            "TRAIN-021","MODEL-021","MODEL-022","MODEL-023",
            "RNG-001","RNG-002","RNG-003",
            "VALID-002","VALID-020","VALID-021","VALID-022",
            "CKPT-001","ART-001","SRC-003",
        } - refs
    )
    if missing_manifest_ids:
        raise SystemExit(
            "POSTLONG_PERF_PREFLIGHT_FAIL manifest missing IDs: "
            + ",".join(missing_manifest_ids)
        )

    long_path = ROOT / "contracts/run_manifests/semantic_long_10115_10315_active.json"
    long_stage = json.loads(long_path.read_text(encoding="utf-8"))
    if long_stage.get("status") != "COMPLETE_PASS":
        raise SystemExit("POSTLONG_PERF_PREFLIGHT_FAIL semantic long stage not COMPLETE_PASS")
    evidence = long_stage.get("completion_evidence") or {}
    required_evidence = {
        "completed_iteration": 10315,
        "new_roots_recorded": 12800,
        "final_safety_guard_pass": True,
        "source_10105_mutated": False,
        "hu_training_performed": False,
        "final_ensemble_sha256":
            "acf86fa05d7e6ac611fa60a346758a3d838c7918937d14488810af9c95abf5ca",
    }
    for key, expected in required_evidence.items():
        if evidence.get(key) != expected:
            raise SystemExit(
                f"POSTLONG_PERF_PREFLIGHT_FAIL long evidence drift {key}: "
                f"{evidence.get(key)!r} != {expected!r}"
            )

    print("POSTLONG_10315_LOCAL_CONTRACT_PREFLIGHT_PASS")
    print("root_status=COMPLETE")
    print("postlong_status=BLOCKED_PERFORMANCE_GATE")
    print("performance_gate_status=PENDING")
    print("semantic_long_status=COMPLETE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
