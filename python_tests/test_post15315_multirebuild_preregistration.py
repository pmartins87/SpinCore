from __future__ import annotations

import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "contracts" / "run_manifests" / "post15315_multirebuild_preregistration.json"


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_post15315_preregistration_checker_passes():
    ns = runpy.run_path(
        str(ROOT / "tools" / "check_post15315_multirebuild_preregistration.py"),
        run_name="post15315_checker_test",
    )
    assert ns["main"]() == 0


def test_post15315_primary_ensemble_and_extension_are_frozen():
    m = load_manifest()
    rp = m["rebuild_plan"]
    assert rp["primary_rebuild_count"] == 8
    assert rp["conditional_extension_rebuild_count"] == 4
    assert rp["maximum_rebuild_count"] == 12
    assert len(rp["member_seeds"]) == 12

    agg = m["aggregation"]
    assert agg["primary_candidate"] == "UNIFORM_PROBABILITY_ENSEMBLE_OF_ALL_ADMITTED_REBUILDS"
    assert agg["weight_by_fit_or_strength"] is False
    assert agg["drop_members_after_results"] is False
    assert agg["representative_policy_selection"] == "PROHIBITED_FOR_PRIMARY_PROMOTION"


def test_post15315_eval_seeds_are_disjoint_from_rebuild_and_banned_seeds():
    m = load_manifest()
    member_seeds = {
        int(value)
        for row in m["rebuild_plan"]["member_seeds"]
        for key, value in row.items()
        if key.endswith("_seed")
    }
    ev = m["shared_fresh_evaluation"]
    eval_seeds = {
        int(ev["stability_state_bank"]["seed"]),
        int(ev["policy_quality_eval"]["seed"]),
        int(ev["safety_pathology"]["seed"]),
        int(ev["fresh_dc1"]["seed"]),
        int(ev["reserved_dc2"]["seed"]),
        *[int(x) for x in ev["specialist_confirmation"]["seeds"]],
    }
    banned = {int(x) for x in m["prior_observed_or_reserved_seeds_not_reused"]}
    assert member_seeds.isdisjoint(eval_seeds)
    assert member_seeds.isdisjoint(banned)
    assert eval_seeds.isdisjoint(banned)


def test_post15315_stability_thresholds_are_literal():
    t = load_manifest()["rebuild_stability_gate"]["thresholds"]
    assert t == {
        "broad_worst_split_mean_tv_max": 0.03,
        "broad_worst_split_statewise_p95_tv_max": 0.12,
        "leave_one_out_worst_mean_tv_max": 0.01,
        "preflop_72o_allin_worst_split_mean_probability_delta_max": 0.025,
        "preflop_72o_allin_worst_split_statewise_p95_probability_delta_max": 0.12,
        "trips_plus_fold_worst_split_mean_probability_delta_max": 0.02,
        "trips_plus_fold_worst_split_statewise_p95_probability_delta_max": 0.10,
        "high_card_no_draw_allin_worst_split_mean_probability_delta_max": 0.025,
        "high_card_no_draw_allin_worst_split_statewise_p95_probability_delta_max": 0.12,
    }
