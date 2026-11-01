#!/usr/bin/env python3
"""Read-only G4B: add denominators to ALREADY-published 4,676 legal-count errors.

Does not call neural networks, train, evaluate EV, create CFR roots,
modify checkpoints, or change the frozen October preregistration.
Project contracts: REP-015,TRAIN-004,MODEL-020,VALID-002,VALID-035,SRC-002.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import torch

RESUME = ROOT / "runs/3h_semantic_long_15315_17315/resume_state.pt"
TRAIN_REPORT = ROOT / "runs/3h_semantic_long_15315_17315/3h_semantic_long_15315_17315.json"
DIAGNOSTIC = ROOT / "runs/teacher_fallback_sign_error_17315/teacher_fallback_sign_error_17315.json"
PROBE_PARENT = ROOT / "runs/3h_advantage_controlled_budget_curve"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    expected_keys = (2, 3, 4, 5, 6, 7)
    report = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    train = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
    if report.get("schema") != "SPINCORE_TEACHER_FALLBACK_SIGN_ERROR_17315_V1":
        raise RuntimeError("unexpected diagnostic schema")
    if report.get("status") != "PASS" or report.get("protected_holdout_count") != 50000:
        raise RuntimeError("source diagnostic did not pass")
    if report.get("teacher_iteration") != 17315 or train.get("completed_iteration") != 17315:
        raise RuntimeError("teacher iteration drift")
    if sha256(RESUME) != train["resume_state_sha256"]:
        raise RuntimeError("immutable resume SHA256 mismatch")
    groups = {int(k): v for k, v in report["by_legal_action_count"].items()}
    if tuple(sorted(groups)) != expected_keys:
        raise RuntimeError("changed reported legal-action groups")
    if sum(int(v["count"]) for v in groups.values()) != 4676:
        raise RuntimeError("reported error count drift")

    probe_files = list(PROBE_PARENT.glob("*/3h_advantage_controlled_split_probe.pt"))
    if not probe_files:
        raise RuntimeError("protected probe not found")
    probe_path = max(probe_files, key=lambda p: p.parent.stat().st_mtime)
    # Only locally created, trusted checkpoints, as in the original diagnostics.
    resume = torch.load(RESUME, map_location="cpu", weights_only=False)
    probe = torch.load(probe_path, map_location="cpu", weights_only=False)
    items = resume["adv_mem"]["items"]
    if len(items) != 2000000 or int(probe["holdout_size"]) != 50000:
        raise RuntimeError("reservoir/holdout size drift")
    rng = random.Random(int(probe["holdout_seed"]))
    selected = sorted(rng.sample(range(len(items)), 50000))
    if len(selected) != len(set(selected)):
        raise RuntimeError("duplicated selected index")
    all_legal = Counter()
    target_positive_legal = Counter()
    for idx in selected:
        s = items[idx]
        legal = tuple(i for i, bit in enumerate(s.legal) if bool(bit))
        if not legal or len(legal) not in expected_keys:
            raise RuntimeError("unexpected legal mask")
        if not all(isinstance(a, int) for a in legal):
            raise RuntimeError("unexpected legal slot")
        n = len(legal)
        all_legal[n] += 1
        if max(float(s.target[a]) for a in legal) > 0:
            target_positive_legal[n] += 1

    if sum(all_legal.values()) != 50000:
        raise RuntimeError("holdout count mismatch")
    if sum(target_positive_legal.values()) != int(report["all_target_positive"]["count"]):
        raise RuntimeError("target-positive count mismatch with frozen source report")

    denominator_table = []
    for n in expected_keys:
        positive = target_positive_legal[n]
        errors = int(groups[n]["count"])
        if not 0 <= errors <= positive:
            raise RuntimeError(f"error counts exceed denominator for legal_count={n}")
        denominator_table.append({
            "legal_count": n,
            "holdout_rows": all_legal[n],
            "target_positive_rows": positive,
            "current_fallback_target_positive_rows": errors,
            "error_share_of_all_4676": errors / 4676,
            "error_rate_given_target_positive": errors / positive if positive else None,
            "error_rate_given_all_holdout": errors / all_legal[n] if all_legal[n] else None,
            "target_strength_mean_in_error_set": groups[n]["target_strength_mean"],
            "target_best_positive_vote_mean_in_error_set": groups[n]["target_best_positive_vote_mean"],
        })

    result = {
        "schema": "SPINCORE_G4B_TEACHER_SIGN_ERROR_LEGAL_DENOMINATORS_V1",
        "status": "PASS",
        "source_report_sha256": sha256(DIAGNOSTIC),
        "resume_sha256": train["resume_state_sha256"],
        "source_probe_path": str(probe_path),
        "protected_holdout_rows": 50000,
        "target_positive_rows": sum(target_positive_legal.values()),
        "error_rows": sum(int(v["count"]) for v in groups.values()),
        "overall_error_rate_given_target_positive": 4676 / sum(target_positive_legal.values()),
        "by_legal_action_count": denominator_table,
        "interpretation": (
            "This supplies denominators, NOT a causal estimate of the loss "
            "normalization effect. Do not use the protected holdout repeatedly "
            "to select new candidates. No production change authorized."
        ),
        "decision": "DENOMINATOR_AUDIT_ONLY_NO_TRAINING_OR_POLICY_CHANGE",
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("SPINCORE_G4B_LEGAL_DENOMINATORS_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
