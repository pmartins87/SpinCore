from __future__ import annotations

import r7_5_arch_reset_v1plus_phase2b14_b13_residual_localization as b14


def _group(mass: float, tail: float) -> dict:
    return {
        "pilot_tv_mass_share": float(mass),
        "pilot_tail_gt_035_share_of_all_tail": float(tail),
    }


def _common_rows(root_improvements=(0.03, 0.02)) -> list[dict]:
    rows = []
    for evaluation_seed, improvement in zip((2029384436, 1150634112), root_improvements):
        rows.append({
            "evaluation_seed": evaluation_seed,
            "region": "PREFLOP_ROOT",
            "baseline_tv": 0.30,
            "pilot_tv": 0.30 - float(improvement),
        })
    return rows


def main() -> int:
    region_groups = {
        "PREFLOP_ROOT": _group(0.20, 0.20),
        "PREFLOP_CONTINUATION_1": _group(0.30, 0.30),
        "PREFLOP_CONTINUATION_2PLUS": _group(0.25, 0.25),
        "FLOP": _group(0.10, 0.10),
        "TURN": _group(0.08, 0.08),
        "RIVER": _group(0.07, 0.07),
    }
    scenario_groups = {
        "0": _group(0.10, 0.10),
        "1": _group(0.10, 0.10),
        "2": _group(0.10, 0.10),
    }
    decision = b14._route(_common_rows(), region_groups, scenario_groups)
    assert decision["classification"] == "PREFLOP_CONTINUATION_RESIDUAL_DOMINANT_AFTER_ROOT_IID64"
    assert decision["root_effect_consistent"] is True
    assert decision["next_route"] == "PRECOMMIT_POSTERIOR_WEIGHTED_PREFLOP_CONTINUATION_CHANCE_SCREEN"

    inconsistent = b14._route(_common_rows((0.03, -0.01)), region_groups, scenario_groups)
    assert inconsistent["classification"] == "PREFLOP_CONTINUATION_DOMINANT_ROOT_EFFECT_NOT_LOCALIZED"
    assert inconsistent["root_effect_consistent"] is False

    root_groups = dict(region_groups)
    root_groups["PREFLOP_ROOT"] = _group(0.55, 0.60)
    root_groups["PREFLOP_CONTINUATION_1"] = _group(0.15, 0.12)
    root_groups["PREFLOP_CONTINUATION_2PLUS"] = _group(0.10, 0.08)
    root_decision = b14._route(_common_rows(), root_groups, scenario_groups)
    assert root_decision["classification"] == "ROOT_RESIDUAL_DOMINANT_AFTER_IID64"

    assert b14.B13_RESULT_SHA256 == "6de7996282236d34adf5e8e53416fd8a443a1fbf5abc89fc807492d0cb3dbf80"
    assert b14.REPRO_TOL == 1e-12
    assert b14.DOMINANCE_MIN == 0.35
    print("R7.5 architecture-reset Phase2B14 residual-localization synthetic tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
