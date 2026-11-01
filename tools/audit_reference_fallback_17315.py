#!/usr/bin/env python3
from __future__ import annotations

"""Read-only Deep CFR argmax fallback comparison on frozen @17315 raw-report rows.

Scientific scope: fixed-policy decision snapshots previously selected under the
current policy; no counterfactual full-hand rollout, no EV claim, no promotion.

The reference behavior is Brown et al. (2019) / OpenSpiel PyTorch Deep CFR:
when no *legal* raw Advantage is positive, use deterministic argmax(raw legal).
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.r7_5_action_contract import NAME_BY_SLOT

SCHEMA = "SPINCORE_3H_TEACHER_RAW_ADVANTAGE_DIAGNOSTIC_17315_V1"
ITERATION = 17315


def reference_policy(raw: list[float], legal: tuple[int, ...]) -> tuple[float, ...]:
    """OpenSpiel's neural Deep CFR regret matching, including argmax fallback."""
    if len(raw) != 10 or not legal:
        raise ValueError("expected ten slot values and nonempty legal actions")
    if len(set(legal)) != len(legal) or any(a < 0 or a >= 10 for a in legal):
        raise ValueError("illegal legal-slot collection")
    if any(not math.isfinite(raw[a]) for a in legal):
        raise ValueError("nonfinite Advantage")
    result = [0.0] * 10
    positive = sum(max(0.0, raw[a]) for a in legal)
    if positive > 0:
        for a in legal:
            result[a] = max(0.0, raw[a]) / positive
    else:
        # Tie breaks by the first action in legal order, as Python max does.
        result[max(legal, key=lambda a: raw[a])] = 1.0
    return tuple(result)


def self_test() -> None:
    cases = [
        ([-1.0, -0.2, -3.0] + [0.0] * 7, (0, 1, 2), 1),
        ([0.0, 0.0, 0.0] + [0.0] * 7, (0, 1, 2), 0),
        ([-1.0] * 10, (0, 9), 0),
        ([2.0, -3.0, 6.0] + [0.0] * 7, (0, 1, 2), None),
    ]
    for raw, legal, best in cases:
        ref = reference_policy(raw, legal)
        current = lean_regret_matching_policy(raw, legal)
        assert abs(sum(ref) - 1.0) < 1e-12
        assert all(ref[a] == 0.0 for a in range(10) if a not in legal)
        if best is None:
            assert max(abs(a - b) for a, b in zip(ref, current)) < 1e-12
        else:
            assert ref[best] == 1.0
            if len(legal) > 1:
                assert current[best] < 1.0, "current fitted fallback unexpectedly deterministic"
    print("G1_REFERENCE_FALLBACK_SEMANTICS_SELFTEST_PASS")


def evaluate(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA or data.get("status") != "PASS":
        raise RuntimeError("frozen raw diagnostic must have PASS and expected schema")
    if int(data.get("teacher_iteration", -1)) != ITERATION:
        raise RuntimeError("teacher iteration mismatch")
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("empty frozen diagnostic row set")
    if int(data.get("matched_count", -1)) != len(rows):
        raise RuntimeError("matched_count differs from row length")

    groups: dict[str, list[dict]] = defaultdict(list)
    mismatches = 0
    for i, row in enumerate(rows):
        detail = row.get("policy_detail") or {}
        legal = tuple(int(x) for x in detail["legal_slots"])
        names = [str(NAME_BY_SLOT[a]) for a in legal]
        raw_named = detail["mean_raw_advantages"]
        if set(raw_named) != set(names):
            raise RuntimeError(f"row {i} raw/legal names mismatch")
        raw = [0.0] * 10
        for a, name in zip(legal, names):
            raw[a] = float(raw_named[name])
        base = lean_regret_matching_policy(raw, legal)
        stored = detail["sigma"]
        diff = max(abs(base[a] - float(stored[name])) for a, name in zip(legal, names))
        if diff > 1e-9:
            mismatches += 1
        ref = reference_policy(raw, legal)
        positive = sum(max(0.0, raw[a]) for a in legal)
        expected_branch = "POSITIVE_REGRET_MATCHING" if positive > 0 else "SOFTMAX_FALLBACK"
        if detail.get("branch") != expected_branch:
            raise RuntimeError(f"row {i} branch mismatch")

        item = {
            "fallback": positive <= 0,
            "tv": 0.5 * sum(abs(base[a] - ref[a]) for a in legal),
            "allin_legal": 9 in legal,
            "base_allin": base[9] if 9 in legal else None,
            "ref_allin": ref[9] if 9 in legal else None,
            "base_fold": base[0] if 0 in legal else None,
            "ref_fold": ref[0] if 0 in legal else None,
            "selected": detail.get("selected_slot_name"),
        }
        groups["ALL_FROZEN_ROWS"].append(item)
        if row.get("flag_code") is not None:
            groups["FLAG:" + str(row["flag_code"])].append(item)
        for code in sorted(set(row.get("policy_surface_codes") or [])):
            groups["OPPORTUNITY:" + str(code)].append(item)

    if mismatches:
        raise RuntimeError(f"stored baseline policy parity mismatch in {mismatches} rows")

    summary = {}
    for key, items in sorted(groups.items()):
        fall = [r for r in items if r["fallback"]]
        allin = [r for r in items if r["allin_legal"]]
        fall_allin = [r for r in allin if r["fallback"]]
        summary[key] = {
            "rows": len(items),
            "fallback_rows": len(fall),
            "policy_changed_rows": sum(r["tv"] > 1e-10 for r in items),
            "mean_policy_tv": sum(r["tv"] for r in items) / len(items),
            "allin_legal_rows": len(allin),
            "base_mean_allin_probability": sum(r["base_allin"] for r in allin) / len(allin) if allin else None,
            "reference_mean_allin_probability": sum(r["ref_allin"] for r in allin) / len(allin) if allin else None,
            "fallback_allin_legal_rows": len(fall_allin),
            "fallback_base_mean_allin_probability": sum(r["base_allin"] for r in fall_allin) / len(fall_allin) if fall_allin else None,
            "fallback_reference_mean_allin_probability": sum(r["ref_allin"] for r in fall_allin) / len(fall_allin) if fall_allin else None,
        }
    return {
        "schema": "SPINCORE_REFERENCE_FALLBACK_FROZEN_DIAGNOSTIC_17315_V1",
        "status": "PASS",
        "teacher_iteration": ITERATION,
        "frozen_rows": len(rows),
        "baseline_policy_parity": True,
        "groups": summary,
        "decision": "NO_PRODUCTION_CHANGE_AUTHORIZED",
        "interpretation": (
            "Rows were selected under current teacher policy, including flagged "
            "and extreme-probability cases: selection bias prevents population "
            "inference. Frozen-state policy deltas do NOT establish direct-play EV, "
            "new-trajectory sanity, or benefit of replacing the fallback."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-report", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    if args.raw_report is None:
        if args.self_test:
            return 0
        ap.error("--raw-report is required unless --self-test is set")
    report = evaluate(args.raw_report)
    print(json.dumps(report, sort_keys=True, indent=2))
    print("SPINCORE_REFERENCE_FALLBACK_FROZEN_DIAGNOSTIC_17315_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
