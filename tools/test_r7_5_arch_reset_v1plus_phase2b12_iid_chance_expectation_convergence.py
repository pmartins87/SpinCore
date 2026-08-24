from __future__ import annotations

import math

import r7_5_arch_reset_v1plus_phase2b12_iid_chance_expectation_convergence as p


def _metric(mean_tv: float, sign: float, mad: float, dom: float, tail: float) -> dict:
    return {
        "pair_count": 120,
        "target_mean_abs_diff": {"count": 120, "mean": mad, "p50": mad, "p95": mad, "max": mad},
        "legal_sign_disagreement_fraction": {"count": 120, "mean": sign, "p50": sign, "p95": sign, "max": sign},
        "regret_matching_policy_tv": {"count": 120, "mean": mean_tv, "p50": mean_tv, "p95": mean_tv, "max": mean_tv},
        "dominant_legal_action_mismatch_rate": dom,
        "tail_rate_tv_ge_035": tail,
    }


def _pooled(k64_tv: float, k64_sign: float, k64_dom: float, k64_tail: float, *, k32_tv: float = 0.26) -> dict:
    return {
        "8": _metric(0.45, 0.36, 0.010, 0.45, 0.55),
        "16": _metric(
            p.B11_REPRO["pooled_mean_tv"],
            p.B11_REPRO["pooled_sign"],
            p.B11_REPRO["pooled_target_mad"],
            p.B11_REPRO["pooled_dominant_mismatch"],
            0.40,
        ),
        "32": _metric(k32_tv, 0.21, 0.0050, 0.30, 0.30),
        "64": _metric(k64_tv, k64_sign, 0.0040, k64_dom, k64_tail),
    }


def _by_seed(seed_a64: float, seed_b64: float) -> dict:
    return {
        "1342191342": {
            "8": _metric(0.44, 0.35, 0.01, 0.4, 0.5),
            "16": _metric(p.B11_REPRO["seed_1342191342_mean_tv"], 0.25, 0.006, 0.3, 0.4),
            "32": _metric(0.25, 0.20, 0.005, 0.3, 0.3),
            "64": _metric(seed_a64, 0.18, 0.004, 0.3, 0.25),
        },
        "1801739323": {
            "8": _metric(0.46, 0.37, 0.01, 0.4, 0.5),
            "16": _metric(p.B11_REPRO["seed_1801739323_mean_tv"], 0.25, 0.006, 0.3, 0.4),
            "32": _metric(0.27, 0.20, 0.005, 0.3, 0.3),
            "64": _metric(seed_b64, 0.18, 0.004, 0.3, 0.25),
        },
    }


def test_mean_targets() -> None:
    rows = [tuple(float(i + j) for i in range(10)) for j in (0, 2, 4, 6)]
    got = p._mean_targets(rows)
    expected = tuple(float(i + 3) for i in range(10))
    assert got == expected


def test_pair_metric() -> None:
    left = (1.0, -1.0) + (0.0,) * 8
    right = (-1.0, 1.0) + (0.0,) * 8
    mask = (1, 1) + (0,) * 8
    row = p._pair_metric(left, right, mask)
    assert math.isclose(row["target_mean_abs_diff"], 2.0)
    assert math.isclose(row["legal_sign_disagreement_fraction"], 1.0)
    assert math.isclose(row["regret_matching_policy_tv"], 1.0)
    assert row["dominant_legal_action_mismatch"] == 1
    assert row["tv_ge_035"] == 1


def test_reproduction_gate() -> None:
    pooled = _pooled(0.20, 0.18, 0.30, 0.25)
    by_seed = _by_seed(0.19, 0.21)
    gate = p._reproduction_gate(by_seed, pooled)
    assert gate["pass"] is True


def test_material_pass() -> None:
    pooled = _pooled(0.20, 0.18, 0.30, 0.25, k32_tv=0.25)
    by_seed = _by_seed(0.19, 0.21)
    decision = p._decision(by_seed, pooled)
    assert decision["classification"] == "IID_CHANCE_EXPECTATION_CONVERGES_MATERIALLY"
    assert decision["screen_pass"] is True
    assert decision["small_causal_training_pilot_precommit_allowed"] is True


def test_slow_route() -> None:
    pooled = _pooled(0.27, 0.23, 0.32, 0.34, k32_tv=0.29)
    by_seed = _by_seed(0.26, 0.28)
    decision = p._decision(by_seed, pooled)
    assert decision["classification"] == "IID_CHANCE_EXPECTATION_CONVERGES_SLOWLY"
    assert decision["screen_pass"] is False
    assert decision["small_causal_training_pilot_precommit_allowed"] is False


def test_plateau_route() -> None:
    pooled = _pooled(0.31, 0.24, 0.34, 0.38, k32_tv=0.32)
    by_seed = _by_seed(0.30, 0.32)
    decision = p._decision(by_seed, pooled)
    assert decision["classification"] == "IID_CHANCE_EXPECTATION_PLATEAUS_OR_UNRESOLVED"
    assert decision["small_causal_training_pilot_precommit_allowed"] is False


def main() -> int:
    test_mean_targets()
    test_pair_metric()
    test_reproduction_gate()
    test_material_pass()
    test_slow_route()
    test_plateau_route()
    print("R7.5 architecture-reset Phase2B12 IID chance convergence synthetic tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
