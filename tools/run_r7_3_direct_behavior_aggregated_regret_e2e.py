from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import run_r7_3_direct_behavior_e2e as base
from spincore.deep_cfr import regret_matching_policy
from spincore_nn import UniformReservoir
from spincore_nn.reservoir import StrategySample


def _aggregated_surrogate_memory(adv_memory):
    """Group exact infosets, average regrets with LCFR weights, then apply RM.

    This preserves the empirical weighted conditional-regret ordering
    E_w[regret | observation, legal] -> hard regret matching, instead of
    applying regret matching independently to every noisy Advantage sample.
    It is an explicit algorithmic diagnostic, not recovered production
    semantics.
    """
    groups = {}
    for sample in adv_memory.items:
        key = (sample.observation, tuple(int(x) for x in sample.legal))
        row = groups.get(key)
        if row is None:
            row = {
                "weighted_regret": [0.0] * 6,
                "weight": 0.0,
                "max_iteration": int(sample.iteration),
                "count": 0,
            }
            groups[key] = row
        w = float(sample.weight)
        row["weight"] += w
        row["max_iteration"] = max(int(row["max_iteration"]), int(sample.iteration))
        row["count"] += 1
        for a in range(6):
            row["weighted_regret"][a] += w * float(sample.target[a])

    memory = UniformReservoir(max(len(groups), 1), 0xD1AEC7)
    memory.items = []
    duplicate_groups = 0
    max_group_count = 0
    for (observation, legal_mask), row in groups.items():
        total_w = max(float(row["weight"]), 1e-12)
        mean_regret = [x / total_w for x in row["weighted_regret"]]
        legal_actions = tuple(i for i, yes in enumerate(legal_mask) if yes)
        target = tuple(regret_matching_policy(mean_regret, legal_actions))
        memory.items.append(
            StrategySample(
                observation,
                legal_mask,
                target,
                total_w,
                int(row["max_iteration"]),
            )
        )
        if int(row["count"]) > 1:
            duplicate_groups += 1
        max_group_count = max(max_group_count, int(row["count"]))
    memory.seen = len(memory.items)
    memory._aggregation_metadata = {
        "raw_advantage_samples": len(adv_memory.items),
        "aggregated_infosets": len(memory.items),
        "compression_ratio": len(memory.items) / max(len(adv_memory.items), 1),
        "groups_with_duplicates": duplicate_groups,
        "max_group_count": max_group_count,
        "target_order": "LCFR_WEIGHTED_MEAN_REGRET_PER_EXACT_INFOSET_THEN_HARD_REGRET_MATCHING",
    }
    return memory


def main() -> int:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--out", type=Path, default=Path("validation/R7_3_DIRECT_BEHAVIOR_AGGREGATED_REGRET_E2E.json"))
    known, _ = probe.parse_known_args()

    base._surrogate_memory = _aggregated_surrogate_memory
    rc = int(base.main())

    if known.out.exists():
        payload = json.loads(known.out.read_text(encoding="utf-8"))
        if not payload.get("runner_failed_before_report"):
            payload["schema"] = "SPINCORE_R7_3_DIRECT_BEHAVIOR_AGGREGATED_REGRET_E2E_V1"
            payload["surrogate_target_order"] = (
                "LCFR_WEIGHTED_MEAN_REGRET_PER_EXACT_INFOSET_THEN_HARD_REGRET_MATCHING"
            )
            payload["source_diagnostic"] = {
                "file": "validation/R7_3_BEHAVIOR_TARGET_AGGREGATION_256.json",
                "same_memory_pairwise_mean_tv": 0.10243512317538261,
                "same_memory_pairwise_p95_tv": 0.3476518243551254,
                "raw_sample_compression_ratio": 0.9952673923331756,
            }
            payload["production_algorithm_changed"] = False
            payload["theoretical_equivalence_claimed"] = False
            payload["promotion_note"] = (
                "Diagnostic only. Exact observation/legal groups are LCFR-weighted in regret space, "
                "then hard regret matching is applied once per group before training the smooth behavior "
                "surrogate. This tests a more coherent conditional-regret target order; it is not declared "
                "equivalent to recovered Deep CFR and cannot change production semantics without versioning."
            )
            payload["ready_for_tables"] = False
            known.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
