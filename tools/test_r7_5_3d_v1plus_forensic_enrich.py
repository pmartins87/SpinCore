from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    here = Path(__file__).resolve().parent
    target = here / "r7_5_3d_v1plus_forensic_enrich.py"
    spec = importlib.util.spec_from_file_location("v1plus_enrich", target)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load enrichment module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_row(rep: str, index: int, tv: float) -> dict:
    return {
        "representation": rep,
        "domain": "D",
        "evaluation_seed": 7,
        "state_index": index,
        "street_name": "FLOP",
        "history_len": 3,
        "action_path_len": 2,
        "pot_bb": 5.0,
        "to_call_bb": 1.0,
        "current_bet_bb": 2.0,
        "stack_geometry_bin": "(10,20]|(20,40]|(20,40]",
        "spr": 4.0,
        "forced_count": 1,
        "nonforced_count": 2,
        "unique_history_actors": 2,
        "last_actor": 1,
        "last_action_type": 4,
        "action_composition": "(0,0,1,0,1,0)",
        "v1_history_projection_sha256": f"v1-{index}",
        "structured_history_projection_sha256": f"s-{index}",
        "exact_history_sha256": f"e-{index}",
        "history_paid_over_pot_mean": 0.25,
        "history_paid_over_pot_std": 0.10,
        "history_paid_over_pot_max": 0.50,
        "history_commitment_over_pot_max": 1.0,
        "tv": tv,
    }


def main() -> int:
    m = _load_module()

    assert [m._pot_bin(x) for x in (2, 2.01, 5, 5.01, 10, 10.01, 20, 20.01)] == [
        "<=2", "(2,5]", "(2,5]", "(5,10]", "(5,10]", "(10,20]", "(10,20]", ">20"
    ]
    assert [m._call_bet_bin(x) for x in (0, .1, 1, 1.1, 2, 2.1, 5, 5.1)] == [
        "0", "(0,1]", "(0,1]", "(1,2]", "(1,2]", "(2,5]", "(2,5]", ">5"
    ]
    assert [m._stack_bin(x) for x in (0, 1, 5, 6, 10, 11, 20, 21, 40, 41)] == [
        "0", "(0,5]", "(0,5]", "(5,10]", "(5,10]", "(10,20]", "(10,20]", "(20,40]", "(20,40]", ">40"
    ]
    assert [m._stack_spread_bin(x) for x in (0, 1, 5, 6, 10, 11, 20, 21)] == [
        "0", "(0,5]", "(0,5]", "(5,10]", "(5,10]", "(10,20]", "(10,20]", ">20"
    ]
    assert [m._spr_bin(x) for x in (None, 1, 1.1, 2, 2.1, 5, 5.1, 10, 10.1)] == [
        "NA", "<=1", "(1,2]", "(1,2]", "(2,5]", "(2,5]", "(5,10]", "(5,10]", ">10"
    ]
    assert [m._history_count_bin(x) for x in (0, 1, 2, 3, 4, 5, 8, 9, 16, 17)] == [
        "0", "1-2", "1-2", "3-4", "3-4", "5-8", "5-8", "9-16", "9-16", "17+"
    ]
    assert [m._ratio_bin(x) for x in (0, .1, .25, .3, .5, .7, 1, 1.5, 2, 3)] == [
        "0", "(0,.25]", "(0,.25]", "(.25,.5]", "(.25,.5]", "(.5,1]", "(.5,1]", "(1,2]", "(1,2]", ">2"
    ]

    assert m._dominant_legal_delta_slot([1, 3], [9.0, .2, 8.0, .4], .3) == 3
    assert m._dominant_legal_delta_slot([1, 3], [0.0, .4, 0.0, .4], .4) == 1
    assert m._dominant_legal_delta_slot([1, 3], [0.0, .4, 0.0, .4], 0.0) is None

    composition = {f"history_action_type_{i}_count": i for i in range(6)}
    assert m._action_composition_key(composition) == "(0,1,2,3,4,5)"

    grouped = m._grouped_tv(
        [{"tv": .1, "x": 0}, {"tv": .3, "x": 0}, {"tv": .2, "x": 1}],
        "x",
    )
    assert grouped["0"]["count"] == 2
    assert abs(grouped["0"]["mean"] - .2) < 1e-12
    assert grouped["1"]["count"] == 1

    rows = [
        _base_row("H2X", 0, .10),
        _base_row("H3X", 0, .08),
        _base_row("H2X", 1, .20),
        _base_row("H3X", 1, .30),
    ]
    paired = m._paired_h3_minus_h2(
        rows, ["D"], [7], h2_name="H2X", h3_name="H3X", policy_count=2
    )
    assert len(paired) == 1
    assert paired[0]["h3_minus_h2_tv"]["count"] == 2
    assert abs(paired[0]["h3_minus_h2_tv"]["mean"] - .04) < 1e-12

    duplicate_failed = False
    try:
        m._paired_h3_minus_h2(
            rows + [dict(rows[0])], ["D"], [7], h2_name="H2X", h3_name="H3X", policy_count=2
        )
    except RuntimeError:
        duplicate_failed = True
    assert duplicate_failed

    missing_failed = False
    try:
        m._paired_h3_minus_h2(
            rows[:-1], ["D"], [7], h2_name="H2X", h3_name="H3X", policy_count=2
        )
    except RuntimeError:
        missing_failed = True
    assert missing_failed

    print("R7.5.3D V1+ forensic enrichment synthetic tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
