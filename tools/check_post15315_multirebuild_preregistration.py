#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed verifier for the frozen post-15315 multi-rebuild preregistration.

This checker is intentionally standard-library only so it can run in the lean
production environment. It validates the scientific choices that must not drift
after teacher 15315 is observed.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "contracts" / "post15315_multirebuild_preregistration.json"

EXPECTED_SCHEMA = "SPINCORE_POST15315_MULTIREBUILD_PREREG_V1"
EXPECTED_STATUS = "PREREGISTERED_BLOCKED_TEACHER_15315_AND_PERFORMANCE_GATE"
PRIMARY_K = 8
MAX_K = 12
TARGET_ITERATION = 15315

EXPECTED_SHARED_SEEDS = {
    "stability": 153159001,
    "quality": 153159002,
    "specialist": (153159101, 153159102, 153159103, 153159104),
    "safety": 153159201,
    "dc1": 153159301,
    "dc2": 153159401,
}

REQUIRED_CONTRACT_IDS = {
    "TRAIN-022", "TRAIN-023", "MODEL-025", "RNG-001", "RNG-003", "RNG-013",
    "VALID-025", "VALID-031", "VALID-032", "VALID-034", "VALID-035",
    "VALID-036", "VALID-037", "VALID-038", "SAFE-001", "SAFE-010",
    "SAFE-011", "PERF-001", "PERF-002", "DC-001", "BENCH-002", "BENCH-013",
}


def _all_member_seeds(rows):
    out = []
    for row in rows:
        for key, value in row.items():
            if key.endswith("_seed"):
                out.append(int(value))
    return out


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert data["schema"] == EXPECTED_SCHEMA
    assert data["status"] == EXPECTED_STATUS
    assert int(data["teacher"]["required_iteration"]) == TARGET_ITERATION
    assert bool(data["teacher"]["no_strength_benchmark_before_teacher"]) is True

    rp = data["rebuild_plan"]
    assert int(rp["primary_rebuild_count"]) == PRIMARY_K
    assert int(rp["conditional_extension_rebuild_count"]) == 4
    assert int(rp["maximum_rebuild_count"]) == MAX_K
    members = list(rp["member_seeds"])
    assert len(members) == MAX_K
    assert [int(x["member"]) for x in members] == list(range(1, MAX_K + 1))

    member_seeds = _all_member_seeds(members)
    assert len(member_seeds) == MAX_K * 6
    assert len(set(member_seeds)) == len(member_seeds)

    ev = data["shared_fresh_evaluation"]
    shared = [
        int(ev["stability_state_bank"]["seed"]),
        int(ev["policy_quality_eval"]["seed"]),
        *[int(x) for x in ev["specialist_confirmation"]["seeds"]],
        int(ev["safety_pathology"]["seed"]),
        int(ev["fresh_dc1"]["seed"]),
        int(ev["reserved_dc2"]["seed"]),
    ]
    assert ev["stability_state_bank"]["forced_three_handed_episodes"] == 80000
    assert tuple(ev["specialist_confirmation"]["seeds"]) == EXPECTED_SHARED_SEEDS["specialist"]
    assert int(ev["stability_state_bank"]["seed"]) == EXPECTED_SHARED_SEEDS["stability"]
    assert int(ev["policy_quality_eval"]["seed"]) == EXPECTED_SHARED_SEEDS["quality"]
    assert int(ev["safety_pathology"]["seed"]) == EXPECTED_SHARED_SEEDS["safety"]
    assert int(ev["fresh_dc1"]["seed"]) == EXPECTED_SHARED_SEEDS["dc1"]
    assert int(ev["reserved_dc2"]["seed"]) == EXPECTED_SHARED_SEEDS["dc2"]
    assert len(shared) == len(set(shared))
    assert set(member_seeds).isdisjoint(shared)

    banned = {int(x) for x in data["prior_observed_or_reserved_seeds_not_reused"]}
    assert set(member_seeds).isdisjoint(banned)
    assert set(shared).isdisjoint(banned)

    agg = data["aggregation"]
    assert agg["primary_candidate"] == "UNIFORM_PROBABILITY_ENSEMBLE_OF_ALL_ADMITTED_REBUILDS"
    assert bool(agg["apply_member_route_before_aggregation"]) is True
    assert bool(agg["weight_by_fit_or_strength"]) is False
    assert bool(agg["drop_members_after_results"]) is False
    assert agg["representative_policy_selection"] == "PROHIBITED_FOR_PRIMARY_PROMOTION"
    carrier = agg["action_carrier"]
    assert carrier["schema"] == "UNIVERSAL_10_SLOT_NETWORK_CARRIER"
    assert carrier["active_legacy7_slots"] == [0, 1, 3, 5, 7, 8, 9]
    assert carrier["dormant_slots"] == [2, 4, 6]
    assert int(carrier["fold_slot"]) == 0
    assert int(carrier["all_in_slot"]) == 9
    assert carrier["action_names"] == [
        "FOLD", "CHECK_CALL", "MIN_RAISE", "POT_33", "POT_40",
        "POT_50", "POT_66", "POT_75", "POT_100", "ALL_IN",
    ]

    st = data["rebuild_stability_gate"]["thresholds"]
    assert float(st["broad_worst_split_mean_tv_max"]) == 0.03
    assert float(st["broad_worst_split_statewise_p95_tv_max"]) == 0.12
    assert float(st["leave_one_out_worst_mean_tv_max"]) == 0.01
    assert float(st["preflop_72o_allin_worst_split_mean_probability_delta_max"]) == 0.025
    assert float(st["trips_plus_fold_worst_split_mean_probability_delta_max"]) == 0.02
    assert float(st["high_card_no_draw_allin_worst_split_mean_probability_delta_max"]) == 0.025

    perf = data["performance_and_cost"]
    assert int(perf["primary_teacher_collection_episodes"]) == 3_904_000
    assert int(perf["primary_fit_steps"]) == 8_200
    assert perf["production_rebuild_execution"] == "BLOCKED_UNTIL_TARGET_HOST_PERFORMANCE_PARITY_GATE"

    ids = set(data["contract_ids"])
    missing = REQUIRED_CONTRACT_IDS - ids
    assert not missing, f"missing contract ids: {sorted(missing)}"

    ext = data["external_strength_gate"]
    assert bool(ext["requires_dc0_real_openholdem_parity_for_canonical_claim"]) is True
    assert int(ev["fresh_dc1"]["scenarios"]) == 5000
    assert int(ev["reserved_dc2"]["min_sampled_states"]) >= 100000

    print("POST15315_MULTI_REBUILD_PREREGISTRATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
