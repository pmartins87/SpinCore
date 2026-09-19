#!/usr/bin/env python3
from __future__ import annotations

"""Read-only decomposition of LT2 Jammer FAI raw Advantage drift."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.lean_action_policy import lean_regret_matching_policy

CHIP_SCALE = 1500.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _mean_ci(values):
    xs = [float(x) for x in values]
    mean = float(statistics.fmean(xs)) if xs else 0.0
    sem = (
        float(statistics.stdev(xs) / math.sqrt(len(xs)))
        if len(xs) > 1
        else 0.0
    )
    return {
        "mean": mean,
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
        "n": len(xs),
    }


def _cluster_ci(rows, field):
    by_seed = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(float(row[field]))
    seed_means = [
        float(statistics.fmean(values))
        for _, values in sorted(by_seed.items())
    ]
    return _mean_ci(seed_means)


def _value(policy, q, legal):
    return float(
        sum(float(policy[a]) * float(q[a]) for a in legal) * CHIP_SCALE
    )


def _raw_with_center_gap(template, center, gap):
    raw = [float(x) for x in template]
    raw[0] = float(center) + 0.5 * float(gap)
    raw[1] = float(center) - 0.5 * float(gap)
    return raw


def _argmax_fallback(raw, legal):
    positive_total = sum(max(0.0, float(raw[a])) for a in legal)
    if positive_total > 0.0:
        return lean_regret_matching_policy(raw, legal)
    best = max(float(raw[a]) for a in legal)
    winners = [a for a in legal if abs(float(raw[a]) - best) <= 1e-15]
    out = [0.0] * len(raw)
    for action in winners:
        out[action] = 1.0 / len(winners)
    return tuple(out)


def main() -> int:
    args = parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if source.get("schema") != "SPINCORE_LT2_JAMMER_FAI_STRUCTURAL_INFOSET_CONFIRMATION_V1":
        raise SystemExit("wrong input schema")
    rows = list(source.get("rows") or [])
    if len(rows) != 384:
        raise SystemExit(f"expected 384 rows, got {len(rows)}")

    derived = []
    for row in rows:
        legal = tuple(int(x) for x in row["legal"])
        if legal != (0, 1):
            raise RuntimeError(f"unexpected legal set: {legal}")

        raw_a = [float(x) for x in row["stage_a"]["raw"]]
        raw_b = [float(x) for x in row["stage_b"]["raw"]]
        q = [float(x) for x in row["stage_a"]["q_canonical"]]

        center_a = (raw_a[0] + raw_a[1]) / 2.0
        center_b = (raw_b[0] + raw_b[1]) / 2.0
        gap_a = raw_a[0] - raw_a[1]
        gap_b = raw_b[0] - raw_b[1]

        raw_offset = _raw_with_center_gap(raw_a, center_b, gap_a)
        raw_gap = _raw_with_center_gap(raw_a, center_a, gap_b)

        sigma_a = lean_regret_matching_policy(raw_a, legal)
        sigma_b = lean_regret_matching_policy(raw_b, legal)
        sigma_offset = lean_regret_matching_policy(raw_offset, legal)
        sigma_gap = lean_regret_matching_policy(raw_gap, legal)
        sigma_argfb = _argmax_fallback(raw_b, legal)

        value_a = _value(sigma_a, q, legal)
        value_b = _value(sigma_b, q, legal)
        value_offset = _value(sigma_offset, q, legal)
        value_gap = _value(sigma_gap, q, legal)
        value_argfb = _value(sigma_argfb, q, legal)

        full = value_b - value_a
        recorded = float(row["policy_value_b_minus_a_chips"])
        if abs(full - recorded) > 2e-4:
            raise RuntimeError(
                f"production value reproduction drift: {full} vs {recorded}"
            )

        fallback_a = bool(row["stage_a"]["fallback_all_nonpositive"])
        fallback_b = bool(row["stage_b"]["fallback_all_nonpositive"])

        derived.append({
            "seed": int(row["seed"]),
            "anchor_index": int(row["anchor_index"]),
            "fold_shift_group": str(row["fold_shift_group"]),
            "fallback_transition": (
                f"{'F' if fallback_a else 'N'}->{'F' if fallback_b else 'N'}"
            ),
            "full_b_minus_a_chips": float(full),
            "offset_only_b_minus_a_chips": float(value_offset - value_a),
            "gap_only_b_minus_a_chips": float(value_gap - value_a),
            "nonlinear_interaction_chips": float(
                full - (value_offset - value_a) - (value_gap - value_a)
            ),
            "b_argmax_fallback_minus_a_chips": float(value_argfb - value_a),
            "argmax_fallback_recovery_vs_b_chips": float(value_argfb - value_b),
            "center_a": float(center_a),
            "center_b": float(center_b),
            "center_b_minus_a": float(center_b - center_a),
            "gap_a": float(gap_a),
            "gap_b": float(gap_b),
            "gap_b_minus_a": float(gap_b - gap_a),
        })

    groups = {}
    for group in ("B_MORE_FOLD", "B_LESS_FOLD"):
        selected = [r for r in derived if r["fold_shift_group"] == group]
        transitions = {}
        for transition in sorted({r["fallback_transition"] for r in selected}):
            subset = [
                r for r in selected
                if r["fallback_transition"] == transition
            ]
            transitions[transition] = {
                "n": len(subset),
                "full_b_minus_a_chips": _mean_ci(
                    [r["full_b_minus_a_chips"] for r in subset]
                ),
            }

        groups[group] = {
            "n": len(selected),
            "full_b_minus_a_chips": {
                "seed_cluster_ci": _cluster_ci(
                    selected, "full_b_minus_a_chips"
                )
            },
            "offset_only_b_minus_a_chips": {
                "seed_cluster_ci": _cluster_ci(
                    selected, "offset_only_b_minus_a_chips"
                )
            },
            "gap_only_b_minus_a_chips": {
                "seed_cluster_ci": _cluster_ci(
                    selected, "gap_only_b_minus_a_chips"
                )
            },
            "nonlinear_interaction_chips": {
                "seed_cluster_ci": _cluster_ci(
                    selected, "nonlinear_interaction_chips"
                )
            },
            "b_argmax_fallback_minus_a_chips": {
                "seed_cluster_ci": _cluster_ci(
                    selected, "b_argmax_fallback_minus_a_chips"
                )
            },
            "argmax_fallback_recovery_vs_b_chips": {
                "seed_cluster_ci": _cluster_ci(
                    selected, "argmax_fallback_recovery_vs_b_chips"
                )
            },
            "fallback_transitions": transitions,
        }

    report = {
        "schema": "SPINCORE_LT2_JAMMER_FAI_RAW_MARGIN_DECOMPOSITION_V1",
        "source_report": str(args.input.resolve()),
        "method": {
            "read_only": True,
            "new_training_roots": 0,
            "optimizer_steps": 0,
            "training_memory_writes": 0,
            "future_holdout_seeds_touched": False,
            "decomposition": (
                "two-legal-action raw = center + gap; exact production lean RM "
                "applied to hybrid raw outputs"
            ),
            "offset_only": (
                "Stage-B legal raw center with Stage-A legal raw gap"
            ),
            "gap_only": (
                "Stage-A legal raw center with Stage-B legal raw gap"
            ),
            "argmax_fallback_probe": (
                "only all-nonpositive Stage-B fallback replaced by deterministic "
                "raw argmax; diagnostic, not authorized intervention"
            ),
        },
        "summary": {"n": len(derived), "groups": groups},
        "rows": derived,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== LT2 JAMMER FAI RAW-MARGIN DECOMPOSITION ===")
    for group in ("B_MORE_FOLD", "B_LESS_FOLD"):
        summary = groups[group]

        def fmt(key):
            ci = summary[key]["seed_cluster_ci"]
            return (
                f"{ci['mean']:+.3f} "
                f"[{ci['ci95_low']:+.3f},{ci['ci95_high']:+.3f}]"
            )

        print(
            f"{group}: "
            f"full={fmt('full_b_minus_a_chips')} "
            f"offset_only={fmt('offset_only_b_minus_a_chips')} "
            f"gap_only={fmt('gap_only_b_minus_a_chips')} "
            f"interaction={fmt('nonlinear_interaction_chips')}"
        )
        print(
            "  argmax_fallback "
            f"B-A={fmt('b_argmax_fallback_minus_a_chips')} "
            f"recovery_vs_B={fmt('argmax_fallback_recovery_vs_b_chips')}"
        )
        print(
            "  transitions="
            + str({
                key: value["n"]
                for key, value in summary["fallback_transitions"].items()
            })
        )

    print("LT2_JAMMER_FAI_RAW_MARGIN_DECOMPOSITION_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
