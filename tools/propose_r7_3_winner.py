from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


SCHEMA = "SPINCORE_R7_3_WINNER_PROPOSAL_V1"
PROVENANCE_SCHEMA = "SPINCORE_R7_3_ACTIVE_CANDIDATE_PROVENANCE_V1"
SEEDS = [20260829, 20260807]
MEAN_GATE = 0.15
P95_GATE = 0.35


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _latest_evidence_commit(path: str) -> str:
    sha = _git("log", "-1", "--format=%H", "--", path)
    if not sha:
        raise RuntimeError(f"no evidence commit for {path}")
    return sha


def _ancestor(older: str, newer: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _schema_for(kind: str) -> str:
    return {
        "policy_mixture": "SPINCORE_R7_3_PARTIAL_EXACT_POLICY_MIXTURE_PAIRED_V1",
        "temporal_blend": "SPINCORE_R7_3_POLICY_MIXTURE_TEMPORAL_BLEND_V1",
        "uncertainty_damping": "SPINCORE_R7_3_POLICY_MIXTURE_UNCERTAINTY_DAMPING_V1",
    }[kind]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a provenance-bound proposal over the active R7.3 five-iteration frontier")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--provenance", type=Path, default=Path("validation/R7_3_ACTIVE_CANDIDATE_PROVENANCE.json"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_WINNER_PROPOSAL.json"))
    args = ap.parse_args()

    root = args.repo_root.resolve()
    prov = json.loads((root / args.provenance).read_text(encoding="utf-8"))
    if prov.get("schema") != PROVENANCE_SCHEMA:
        raise SystemExit("wrong active-candidate provenance schema")

    pending: list[dict] = []
    rows: list[dict] = []
    for item in prov.get("candidates", []):
        rel = str(item["evidence_path"])
        path = root / rel
        if not path.is_file():
            pending.append({"label": item["label"], "evidence_path": rel, "reason": "MISSING"})
            continue
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("runner_failed_before_report"):
            pending.append({"label": item["label"], "evidence_path": rel, "reason": "RUNNER_FAILED"})
            continue
        kind = str(item["behavior_kind"])
        if evidence.get("schema") != _schema_for(kind):
            raise SystemExit(f"{item['label']}: evidence schema mismatch")
        if [int(x) for x in evidence.get("algorithm_seeds", [])] != SEEDS:
            raise SystemExit(f"{item['label']}: algorithm seeds mismatch")
        if int(evidence.get("iterations", -1)) != 5 or int(evidence.get("roots_per_iteration", -1)) != 64:
            raise SystemExit(f"{item['label']}: not a 5x64 durability result")
        if int(evidence.get("roots_per_seed", -1)) != 320:
            raise SystemExit(f"{item['label']}: roots_per_seed mismatch")
        if int(evidence.get("exact_opponent_levels", -1)) != 2:
            raise SystemExit(f"{item['label']}: exact opponent level mismatch")
        if evidence.get("deck_formula") != "seed*1000003 + global_root*97 + iteration":
            raise SystemExit(f"{item['label']}: deck formula mismatch")
        if evidence.get("extra_members_perturb_primary_rng") is not False:
            raise SystemExit(f"{item['label']}: side members perturb primary RNG")
        if evidence.get("acceptance_gate_changed") is not False:
            raise SystemExit(f"{item['label']}: acceptance gate changed")

        cross = dict(evidence.get("cross_seed") or {})
        mean = float(cross.get("mean_tv", float("inf")))
        p95 = float(cross.get("p95_tv", float("inf")))
        fit = evidence.get("per_seed_fit_pass") is True
        qualifies = bool(
            fit
            and mean <= MEAN_GATE
            and p95 <= P95_GATE
            and evidence.get("cross_seed_pass") is True
            and evidence.get("r7_3_pass") is True
        )
        evidence_commit = _latest_evidence_commit(rel)
        if not _ancestor(str(item["source_head_sha"]), evidence_commit):
            raise SystemExit(f"{item['label']}: source head is not ancestor of evidence commit")
        rows.append({
            "label": item["label"],
            "behavior_kind": kind,
            "ensemble_size": int(item["ensemble_size"]),
            "params": dict(item.get("params") or {}),
            "workflow_run": int(item["workflow_run"]),
            "source_head_sha": item["source_head_sha"],
            "source_workflow_path": item["source_workflow_path"],
            "evidence_path": rel,
            "evidence_commit_sha": evidence_commit,
            "per_seed_fit_pass": fit,
            "mean_tv": mean,
            "p50_tv": float(cross.get("p50_tv", float("nan"))),
            "p95_tv": p95,
            "max_tv": float(cross.get("max_tv", float("nan"))),
            "mean_margin_to_gate": MEAN_GATE - mean,
            "p95_margin_to_gate": P95_GATE - p95,
            "full_5x64_gate_pass": qualifies,
        })

    if pending:
        print(json.dumps({
            "schema": SCHEMA,
            "complete": False,
            "pending": pending,
            "completed_labels": [r["label"] for r in rows],
            "ready_for_tables": False,
        }, indent=2, sort_keys=True))
        return 3

    gate_rows = [r for r in rows if r["full_5x64_gate_pass"]]
    gate_rows_by_complexity = sorted(
        gate_rows,
        key=lambda r: (r["ensemble_size"], r["p95_tv"], r["mean_tv"]),
    )
    gate_rows_by_margin = sorted(
        gate_rows,
        key=lambda r: (-min(r["mean_margin_to_gate"], r["p95_margin_to_gate"]), r["ensemble_size"]),
    )
    smallest_size = min((r["ensemble_size"] for r in gate_rows), default=None)
    smallest_frontier = [r for r in gate_rows_by_complexity if r["ensemble_size"] == smallest_size]

    payload = {
        "schema": SCHEMA,
        "complete": True,
        "candidate_count": len(rows),
        "frozen_gates": {"cross_seed_mean_tv_max": MEAN_GATE, "cross_seed_p95_tv_max": P95_GATE},
        "rows": sorted(rows, key=lambda r: (r["p95_tv"], r["mean_tv"])),
        "full_5x64_gate_pass_count": len(gate_rows),
        "full_5x64_gate_pass_rows_by_complexity": gate_rows_by_complexity,
        "full_5x64_gate_pass_rows_by_minimum_gate_margin": gate_rows_by_margin,
        "smallest_gate_passing_ensemble_size": smallest_size,
        "smallest_gate_passing_frontier": smallest_frontier,
        "selection_automatic": False,
        "selection_note": (
            "This report deliberately does not select a winner. It binds measured candidates to immutable source/evidence provenance and exposes the smallest passing frontier plus robustness margins. Final selection must balance minimal mechanism complexity against measured gate margin before creating SPINCORE_R7_3_WINNER_SELECTION_V1."
        ),
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
