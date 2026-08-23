from __future__ import annotations

import math

import r7_5_arch_reset_v1plus_phase2b7_residual_localization as p


def _obs(street: int, events: list[tuple[int, int]]) -> bytes:
    history_count = len(events)
    raw = bytearray(120 + 20 * history_count)
    raw[:8] = b"SPNNIV3\x00"
    raw[9] = int(street)
    raw[116:120] = int(history_count).to_bytes(4, "little", signed=False)
    for index, (event_street, forced) in enumerate(events):
        offset = 120 + 20 * index
        raw[offset + 1] = int(event_street)
        raw[offset + 3] = int(forced)
    return bytes(raw)


def _g(mass: float, tail: float) -> dict:
    return {
        "pilot_tv_mass_share": float(mass),
        "pilot_tail_gt_035_share_of_all_tail": float(tail),
    }


def test_decode_regions() -> None:
    assert p._decode_observation(_obs(0, []))["region"] == "PREFLOP_ROOT"
    assert p._decode_observation(_obs(0, [(0, 0)]))["region"] == "PREFLOP_CONTINUATION_1"
    assert p._decode_observation(_obs(0, [(0, 1), (0, 0), (0, 0)]))["region"] == "PREFLOP_CONTINUATION_2PLUS"
    assert p._decode_observation(_obs(1, [(0, 0)]))["region"] == "FLOP"
    assert p._decode_observation(_obs(2, [(0, 0), (1, 0)]))["region"] == "TURN"
    assert p._decode_observation(_obs(3, [(0, 0), (1, 0), (2, 0)]))["region"] == "RIVER"


def test_bins() -> None:
    assert [p._path_bin(x) for x in (0, 1, 2, 3, 4, 5, 6, 20)] == ["0", "1", "2", "3", "4-5", "4-5", "6+", "6+"]
    assert [p._history_bin(x) for x in (0, 1, 2, 3, 4, 5, 6, 9, 10, 50)] == ["0", "1", "2", "3", "4-5", "4-5", "6-9", "6-9", "10+", "10+"]


def test_group_summary() -> None:
    rows = [
        {"baseline_tv": 0.50, "pilot_tv": 0.40},
        {"baseline_tv": 0.30, "pilot_tv": 0.20},
    ]
    s = p._group_summary(rows, total_tv_mass=1.20, total_tail=2)
    assert s["count"] == 2
    assert math.isclose(s["baseline_tv"]["mean"], 0.40)
    assert math.isclose(s["pilot_tv"]["mean"], 0.30)
    assert math.isclose(s["absolute_mean_improvement"], 0.10)
    assert math.isclose(s["pilot_tv_mass_share"], 0.50)
    assert s["pilot_tail_gt_035_count"] == 1
    assert math.isclose(s["pilot_tail_gt_035_share_of_all_tail"], 0.50)


def test_route_root() -> None:
    regions = {
        "PREFLOP_ROOT": _g(0.40, 0.41),
        "PREFLOP_CONTINUATION_1": _g(0.20, 0.20),
        "PREFLOP_CONTINUATION_2PLUS": _g(0.10, 0.09),
        "FLOP": _g(0.20, 0.20),
        "TURN": _g(0.07, 0.07),
        "RIVER": _g(0.03, 0.03),
    }
    out = p._route_decision(regions, {"0": _g(0.4, 0.4), "1": _g(0.3, 0.3), "2": _g(0.3, 0.3)})
    assert out["classification"] == "ROOT_DOMINANT"


def test_route_continuation() -> None:
    regions = {
        "PREFLOP_ROOT": _g(0.10, 0.10),
        "PREFLOP_CONTINUATION_1": _g(0.25, 0.25),
        "PREFLOP_CONTINUATION_2PLUS": _g(0.20, 0.20),
        "FLOP": _g(0.25, 0.25),
        "TURN": _g(0.15, 0.15),
        "RIVER": _g(0.05, 0.05),
    }
    out = p._route_decision(regions, {"0": _g(0.4, 0.4), "1": _g(0.3, 0.3), "2": _g(0.3, 0.3)})
    assert out["classification"] == "PREFLOP_CONTINUATION_DOMINANT"


def test_route_postflop() -> None:
    regions = {
        "PREFLOP_ROOT": _g(0.10, 0.10),
        "PREFLOP_CONTINUATION_1": _g(0.10, 0.10),
        "PREFLOP_CONTINUATION_2PLUS": _g(0.10, 0.10),
        "FLOP": _g(0.30, 0.30),
        "TURN": _g(0.25, 0.25),
        "RIVER": _g(0.15, 0.15),
    }
    out = p._route_decision(regions, {"0": _g(0.4, 0.4), "1": _g(0.3, 0.3), "2": _g(0.3, 0.3)})
    assert out["classification"] == "POSTFLOP_DOMINANT"


def test_route_scenario_concentrated() -> None:
    regions = {
        "PREFLOP_ROOT": _g(0.20, 0.20),
        "PREFLOP_CONTINUATION_1": _g(0.10, 0.10),
        "PREFLOP_CONTINUATION_2PLUS": _g(0.10, 0.10),
        "FLOP": _g(0.20, 0.20),
        "TURN": _g(0.20, 0.20),
        "RIVER": _g(0.20, 0.20),
    }
    scenarios = {
        "0": _g(0.25, 0.25),
        "1": _g(0.20, 0.20),
        "2": _g(0.15, 0.15),
        "3": _g(0.10, 0.10),
        "4": _g(0.10, 0.10),
        "5": _g(0.10, 0.10),
        "6": _g(0.10, 0.10),
    }
    out = p._route_decision(regions, scenarios)
    assert out["classification"] == "SCENARIO_CONCENTRATED"


def test_route_broad_mixed() -> None:
    regions = {
        "PREFLOP_ROOT": _g(0.20, 0.20),
        "PREFLOP_CONTINUATION_1": _g(0.10, 0.10),
        "PREFLOP_CONTINUATION_2PLUS": _g(0.10, 0.10),
        "FLOP": _g(0.20, 0.20),
        "TURN": _g(0.20, 0.20),
        "RIVER": _g(0.20, 0.20),
    }
    scenarios = {str(i): _g(0.10, 0.10) for i in range(10)}
    out = p._route_decision(regions, scenarios)
    assert out["classification"] == "BROAD_MIXED_RESIDUAL"


def main() -> int:
    test_decode_regions()
    test_bins()
    test_group_summary()
    test_route_root()
    test_route_continuation()
    test_route_postflop()
    test_route_scenario_concentrated()
    test_route_broad_mixed()
    print("R7.5 architecture-reset Phase2B7 residual-localization synthetic tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
