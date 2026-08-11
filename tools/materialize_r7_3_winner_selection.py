from __future__ import annotations

import argparse
import json
from pathlib import Path


PROPOSAL_SCHEMA = "SPINCORE_R7_3_WINNER_PROPOSAL_V1"
SELECTION_SCHEMA = "SPINCORE_R7_3_WINNER_SELECTION_V1"


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize a deliberate R7.3 winner choice from a complete provenance-bound proposal")
    ap.add_argument("--proposal", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_WINNER_SELECTION.json"))
    args = ap.parse_args()

    proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
    if proposal.get("schema") != PROPOSAL_SCHEMA or proposal.get("complete") is not True:
        raise SystemExit("requires a complete SPINCORE_R7_3_WINNER_PROPOSAL_V1")
    label = str(args.label).strip()
    rows = [r for r in proposal.get("rows", []) if r.get("label") == label]
    if len(rows) != 1:
        raise SystemExit(f"winner label {label!r} is not uniquely present in proposal")
    row = rows[0]
    if row.get("full_5x64_gate_pass") is not True:
        raise SystemExit("selected row is not a full 5x64 R7.3 gate pass")
    if row.get("per_seed_fit_pass") is not True:
        raise SystemExit("selected row failed per-seed fit gates")
    if float(row.get("mean_tv", float("inf"))) > 0.15 or float(row.get("p95_tv", float("inf"))) > 0.35:
        raise SystemExit("selected row exceeds frozen cross-seed gates")

    selection = {
        "schema": SELECTION_SCHEMA,
        "label": row["label"],
        "behavior_kind": row["behavior_kind"],
        "ensemble_size": int(row["ensemble_size"]),
        "params": dict(row.get("params") or {}),
        "evidence_path": row["evidence_path"],
        "evidence_commit_sha": row["evidence_commit_sha"],
        "source_workflow_run": int(row["workflow_run"]),
        "source_head_sha": row["source_head_sha"],
        "source_workflow_path": row["source_workflow_path"],
        "measured_5x64": {
            "mean_tv": float(row["mean_tv"]),
            "p50_tv": float(row["p50_tv"]),
            "p95_tv": float(row["p95_tv"]),
            "max_tv": float(row["max_tv"]),
            "mean_margin_to_gate": float(row["mean_margin_to_gate"]),
            "p95_margin_to_gate": float(row["p95_margin_to_gate"]),
        },
        "selection_source": "DELIBERATE_LABEL_CHOICE_FROM_COMPLETE_PROVENANCE_BOUND_PROPOSAL",
        "selection_automatic": False,
        "acceptance_gate_changed": False,
        "ready_for_640": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
