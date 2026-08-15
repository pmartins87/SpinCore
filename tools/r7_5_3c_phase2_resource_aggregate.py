from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "SPINCORE_R7_5_3C_PHASE2_RESOURCE_PREFLIGHT_AGGREGATE_V1"
REPORT_SCHEMA = "SPINCORE_R7_5_3C_PHASE2_RESOURCE_PREFLIGHT_REPORT_V1"
H0 = "H0_V1_RESOURCE_CONTROL"
H2 = "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL"
H3 = "H3_HYBRID_EXACT_SEMANTIC_FINAL"
REPS = (H0, H2, H3)
DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")


def _fingerprint(representation: str, report: dict) -> str:
    material = {
        "representation": representation,
        "config": report["benchmark"]["config"],
        "parameter_count": report["benchmark"]["parameter_count"],
        "source_sha256": report["source_sha256"],
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reports = {}
    for path in sorted(args.reports_dir.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != REPORT_SCHEMA:
            continue
        key = (str(data["representation"]), str(data["domain"]))
        if key in reports:
            raise RuntimeError(f"duplicate resource report {key}")
        reports[key] = data

    expected = {(rep, domain) for rep in REPS for domain in DOMAINS}
    if set(reports) != expected:
        raise RuntimeError(
            f"resource report inventory mismatch: missing={sorted(expected-set(reports))} "
            f"extra={sorted(set(reports)-expected)}"
        )
    if any(report["execution_sha"] != args.execution_sha for report in reports.values()):
        raise RuntimeError("resource report execution SHA mismatch")
    if any(report.get("selection_authority") is not False for report in reports.values()):
        raise RuntimeError("resource report illegally gained selection authority")

    hard_failures: list[str] = []
    model_evidence = {}
    for rep in REPS:
        per_domain = [reports[(rep, domain)] for domain in DOMAINS]
        parameter_counts = {int(row["benchmark"]["parameter_count"]) for row in per_domain}
        configs = {
            json.dumps(row["benchmark"]["config"], sort_keys=True)
            for row in per_domain
        }
        source_hash_sets = {
            json.dumps(row["source_sha256"], sort_keys=True)
            for row in per_domain
        }
        if len(parameter_counts) != 1 or len(configs) != 1 or len(source_hash_sets) != 1:
            hard_failures.append(f"{rep}: model evidence differs by domain")
            continue
        parameter_count = next(iter(parameter_counts))
        if rep in (H2, H3) and parameter_count > 500_000:
            hard_failures.append(f"{rep}: parameter_count={parameter_count}>500000")
        if any(not row["benchmark"]["finite_output_audit"] for row in per_domain):
            hard_failures.append(f"{rep}: nonfinite output audit")
        for row in per_domain:
            if int(row["collection"]["states"]) != 256:
                hard_failures.append(f"{rep}/{row['domain']}: wrong state count")
            street_counts = row["collection"]["street_counts"]
            if any(int(street_counts[str(street)]) < 1 for street in range(4)):
                hard_failures.append(f"{rep}/{row['domain']}: incomplete street coverage")
        model_evidence[rep] = {
            "parameter_count": parameter_count,
            "config": per_domain[0]["benchmark"]["config"],
            "source_sha256": per_domain[0]["source_sha256"],
            "architecture_fingerprint_sha256": _fingerprint(rep, per_domain[0]),
        }

    comparisons = {}
    for domain in DOMAINS:
        control = reports[(H0, domain)]["benchmark"]
        domain_rows = {}
        for rep in (H2, H3):
            row = reports[(rep, domain)]["benchmark"]
            domain_rows[rep] = {
                "parameter_count_ratio_vs_H0": row["parameter_count"] / control["parameter_count"],
                "preprocess_ratio_vs_H0": row["preprocess_seconds_per_sample"] / max(control["preprocess_seconds_per_sample"], 1e-12),
                "model_batch1_ratio_vs_H0": row["model_batch1_seconds_per_sample"] / max(control["model_batch1_seconds_per_sample"], 1e-12),
                "online_batch1_ratio_vs_H0": row["online_batch1_seconds_per_sample"] / max(control["online_batch1_seconds_per_sample"], 1e-12),
                "model_batch64_ratio_vs_H0": row["model_batch64_seconds_per_sample"] / max(control["model_batch64_seconds_per_sample"], 1e-12),
                "peak_rss_ratio_vs_H0": row["peak_rss_bytes"] / max(control["peak_rss_bytes"], 1),
            }
        comparisons[domain] = domain_rows

    compact_reports = {
        f"{rep}|{domain}": {
            "collection": reports[(rep, domain)]["collection"],
            "observation_bytes": reports[(rep, domain)]["observation_bytes"],
            "history_length": reports[(rep, domain)]["history_length"],
            "benchmark": reports[(rep, domain)]["benchmark"],
            "runtime": reports[(rep, domain)]["runtime"],
        }
        for rep in REPS
        for domain in DOMAINS
    }

    payload = {
        "schema": SCHEMA,
        "execution_sha": str(args.execution_sha),
        "status": "PASS" if not hard_failures else "FAIL",
        "resource_preflight_pass": not hard_failures,
        "selection_authority": False,
        "hard_failures": hard_failures,
        "model_evidence": model_evidence,
        "comparisons_vs_H0": comparisons,
        "reports": compact_reports,
        "interpretation_guard": {
            "absolute_ci_time_is_ryzen_time": False,
            "resource_preflight_selects_strategy": False,
            "slower_than_H0_alone_rejects_candidate": False,
            "phase2_strategic_evidence_still_required": True
        },
        "production_training_authorized": False,
        "ready_for_tables": False
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if hard_failures:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
