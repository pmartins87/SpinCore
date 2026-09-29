#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required for project-contract validation") from exc

ROOT = Path(__file__).resolve().parents[1]
ROOT_CONTRACT = ROOT / "PROJECT_CONTRACT.yaml"
AUDIT_STATUS = ROOT / "contracts" / "AUDIT_STATUS.yaml"
RUNNER_BASELINE = ROOT / "contracts" / "RUNNER_BASELINE.yaml"

ID_RE = re.compile(r"^[A-Z]+-[0-9]{3}$")
ALLOWED_INVARIANT_STATUS = {
    "ACTIVE",
    "CONDITIONAL",
    "SUPERSEDED",
    "RETIRED",
    "EXPERIMENT_ONLY",
}
ALLOWED_ROOT_STATUS = {"BOOTSTRAP_AUDIT_REQUIRED", "COMPLETE"}


def load_yaml(path: Path):
    if not path.is_file():
        raise AssertionError(f"missing contract file: {path.relative_to(ROOT)}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError(f"contract YAML must be a mapping: {path.relative_to(ROOT)}")
    return data


def invariant_rows(root, modules):
    rows = []
    for row in root.get("bootstrap_invariants") or []:
        rows.append(("PROJECT_CONTRACT.yaml", row))
    for path, module in modules:
        for row in module.get("invariants") or []:
            rows.append((str(path.relative_to(ROOT)), row))
    return rows


def main() -> int:
    root = load_yaml(ROOT_CONTRACT)
    if root.get("schema") != "SPINCORE_PROJECT_CONTRACT_V1":
        raise AssertionError("wrong PROJECT_CONTRACT schema")
    if root.get("status") not in ALLOWED_ROOT_STATUS:
        raise AssertionError(f"invalid root contract status: {root.get('status')!r}")

    coverage = root.get("coverage_domains") or []
    required_domains = {
        str(row["id"])
        for row in coverage
        if isinstance(row, dict) and bool(row.get("required"))
    }
    if not required_domains:
        raise AssertionError("no required coverage domains declared")

    module_paths = [ROOT / str(p) for p in (root.get("contract_modules") or [])]
    if not module_paths:
        raise AssertionError("contract_modules is empty")

    modules = []
    for path in module_paths:
        module = load_yaml(path)
        if module.get("schema") != "SPINCORE_CONTRACT_MODULE_V1":
            raise AssertionError(f"wrong module schema: {path.relative_to(ROOT)}")
        if not module.get("module_id"):
            raise AssertionError(f"missing module_id: {path.relative_to(ROOT)}")
        modules.append((path, module))

    rows = invariant_rows(root, modules)
    if not rows:
        raise AssertionError("project contract contains no invariants")

    ids = []
    represented_prefixes = set()
    for source, row in rows:
        if not isinstance(row, dict):
            raise AssertionError(f"non-mapping invariant in {source}")
        iid = str(row.get("id") or "")
        if not ID_RE.fullmatch(iid):
            raise AssertionError(f"invalid invariant id {iid!r} in {source}")
        ids.append(iid)
        represented_prefixes.add(iid.split("-", 1)[0])

        status = row.get("status")
        if status not in ALLOWED_INVARIANT_STATUS:
            raise AssertionError(f"invalid status for {iid}: {status!r}")
        if not str(row.get("statement") or "").strip():
            raise AssertionError(f"missing statement for {iid}")
        if not row.get("provenance"):
            raise AssertionError(f"missing provenance for {iid}")
        if status == "SUPERSEDED" and not row.get("superseded_by"):
            raise AssertionError(f"SUPERSEDED invariant {iid} lacks superseded_by")

    duplicates = [iid for iid, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise AssertionError(f"duplicate invariant IDs: {duplicates}")

    audit = load_yaml(AUDIT_STATUS)
    if audit.get("schema") != "SPINCORE_CONTRACT_AUDIT_STATUS_V1":
        raise AssertionError("wrong audit-status schema")
    audit_coverage = audit.get("coverage") or {}
    missing_audit_domains = sorted(required_domains - set(audit_coverage))
    if missing_audit_domains:
        raise AssertionError(
            "audit status omits required domains: " + ", ".join(missing_audit_domains)
        )

    unknown_audit_domains = sorted(set(audit_coverage) - required_domains)
    if unknown_audit_domains:
        raise AssertionError(
            "audit status contains undeclared domains: " + ", ".join(unknown_audit_domains)
        )


    baseline = load_yaml(RUNNER_BASELINE)
    if baseline.get("schema") != "SPINCORE_RUNNER_BASELINE_V1":
        raise AssertionError("wrong runner-baseline schema")
    baseline_rows = baseline.get("entries") or []
    baseline_by_path = {
        str(row["path"]): str(row["blob_sha"])
        for row in baseline_rows
        if isinstance(row, dict)
    }
    all_contract_ids = set(ids)
    governed_runner_paths = sorted(
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "tools").iterdir()
        if p.is_file()
        and re.match(r"^(?:run_|benchmark_).+\.(?:py|sh)$", p.name)
    )
    missing_baseline_or_declaration = []
    for rel in governed_runner_paths:
        path = ROOT / rel
        actual = __import__("subprocess").check_output(
            ["git", "hash-object", str(path)], text=True
        ).strip()
        if baseline_by_path.get(rel) == actual:
            continue
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:120])
        if "PROJECT_CONTRACT_IDS" not in head:
            missing_baseline_or_declaration.append(rel)
            continue
        declared = set(re.findall(r"[A-Z]+-[0-9]{3}", head))
        if not declared:
            raise AssertionError(
                f"{rel} declares PROJECT_CONTRACT_IDS but contains no IDs"
            )
        unknown = sorted(declared - all_contract_ids)
        if unknown:
            raise AssertionError(
                f"{rel} references unknown contract IDs: {unknown}"
            )
    if missing_baseline_or_declaration:
        raise AssertionError(
            "new/modified runner or benchmark lacks PROJECT_CONTRACT_IDS: "
            + ", ".join(missing_baseline_or_declaration)
        )

    manifest_dir = ROOT / "contracts" / "run_manifests"
    manifest_rows = []
    if manifest_dir.is_dir():
        for manifest_path in sorted(manifest_dir.glob("*.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != "SPINCORE_STAGE_MANIFEST_V1":
                raise AssertionError(
                    f"wrong stage-manifest schema: {manifest_path.relative_to(ROOT)}"
                )
            stage_id = str(manifest.get("stage_id") or "")
            runner_rel = str(manifest.get("runner") or "")
            if not stage_id or not runner_rel:
                raise AssertionError(
                    f"incomplete stage manifest: {manifest_path.relative_to(ROOT)}"
                )
            runner = ROOT / runner_rel
            if not runner.is_file():
                raise AssertionError(
                    f"stage manifest runner missing: {runner_rel}"
                )
            refs = [str(x) for x in (manifest.get("contract_ids") or [])]
            if not refs:
                raise AssertionError(
                    f"stage manifest has no contract IDs: {stage_id}"
                )
            unknown = sorted(set(refs) - all_contract_ids)
            if unknown:
                raise AssertionError(
                    f"stage manifest {stage_id} references unknown IDs: {unknown}"
                )
            affected = [str(x) for x in (manifest.get("affected_domains") or [])]
            unknown_domains = sorted(set(affected) - required_domains)
            if unknown_domains:
                raise AssertionError(
                    f"stage manifest {stage_id} has undeclared domains: {unknown_domains}"
                )

            runner_head = "\n".join(
                runner.read_text(encoding="utf-8").splitlines()[:120]
            )
            if "PROJECT_CONTRACT_IDS" in runner_head:
                declared = set(re.findall(r"[A-Z]+-[0-9]{3}", runner_head))
                missing_from_runner = sorted(set(refs) - declared)
                if missing_from_runner:
                    raise AssertionError(
                        f"runner {runner_rel} omits manifest contract IDs: "
                        + ",".join(missing_from_runner)
                    )

            status = str(manifest.get("status") or "")
            if str(manifest.get("duration_class") or "") == "LONG_GT_60M":
                perf = manifest.get("performance_gate") or {}
                if status == "READY" and perf.get("status") != "PASS":
                    raise AssertionError(
                        f"READY long stage lacks PASS performance gate: {stage_id}"
                    )
                if not manifest.get("target_host_profile"):
                    raise AssertionError(
                        f"long stage lacks target host profile: {stage_id}"
                    )

            expected_blob = manifest.get("runner_blob_sha")
            if status == "READY":
                if not expected_blob:
                    raise AssertionError(
                        f"READY stage lacks runner_blob_sha: {stage_id}"
                    )
                actual_blob = __import__("subprocess").check_output(
                    ["git", "hash-object", str(runner)], text=True
                ).strip()
                if str(expected_blob) != actual_blob:
                    raise AssertionError(
                        f"READY stage runner blob drift: {stage_id}"
                    )
            manifest_rows.append(stage_id)

    if root.get("status") == "COMPLETE":
        incomplete = sorted(
            d for d in required_domains if audit_coverage.get(d) != "COMPLETE"
        )
        if incomplete:
            raise AssertionError(
                "root marked COMPLETE while domains remain incomplete: "
                + ", ".join(incomplete)
            )
        open_recon = [
            x.get("id")
            for x in (audit.get("known_reconciliation_items") or [])
            if x.get("status") == "OPEN"
        ]
        if open_recon:
            raise AssertionError(
                "root marked COMPLETE with open reconciliation items: "
                + ", ".join(map(str, open_recon))
            )

    sys.path.insert(0, str(ROOT / "python"))
    from spincore.legacy_scenario import LegacyScenarioConfig, BLIND_LEVELS
    from spincore.lean_action_scope import FIRST_RELEASE_ACTION_NAMES
    from spincore.lean_training_scope import (
        FIRST_RELEASE_TOTAL_CHIPS,
        PRIMARY_PAYOUT_VECTOR,
        UTILITY_SCALE_ID,
    )

    assert LegacyScenarioConfig().total_chips == 1500
    assert abs(LegacyScenarioConfig().heads_up_prob - 0.4548) < 1e-12
    assert tuple(BLIND_LEVELS) == (
        (10, 20), (15, 30), (20, 40), (30, 60), (40, 80),
        (50, 100), (60, 120), (80, 160), (100, 200),
    )
    assert tuple(FIRST_RELEASE_ACTION_NAMES) == (
        "FOLD", "CHECK_CALL", "POT_33", "POT_50",
        "POT_75", "POT_100", "ALL_IN",
    )
    assert float(FIRST_RELEASE_TOTAL_CHIPS) == 1500.0
    assert tuple(PRIMARY_PAYOUT_VECTOR) == (1.0, 0.0, 0.0)
    assert UTILITY_SCALE_ID == "TOTAL_CHIPS_CONSTANT_1500_V1"

    long_text = (ROOT / "tools" / "run_3h_semantic_long_10115_10315.py").read_text(
        encoding="utf-8"
    )
    for needle in (
        "START_ITERATION=10115",
        "TARGET_ITERATION=10315",
        "EXPECTED_ADDITIONAL_ITERATIONS=200",
        "MILESTONES=(10165,10215,10265,10315)",
    ):
        if needle not in long_text:
            raise AssertionError(f"semantic-long executable contract drift: {needle}")

    print("PROJECT_CONTRACT_LINT_PASS")
    print(f"invariants={len(ids)}")
    print(f"required_domains={len(required_domains)}")
    print(f"governed_runner_baseline={len(baseline_by_path)}")
    print(f"stage_manifests={len(manifest_rows)}")
    print("represented_prefixes=" + ",".join(sorted(represented_prefixes)))
    print(f"root_status={root.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
