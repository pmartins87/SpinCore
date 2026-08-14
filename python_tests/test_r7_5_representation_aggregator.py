from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from r7_5_aggregate_representation_ablation import apply_frozen_selection


def _summary(
    *,
    gate: bool = True,
    adv: float = 0.50,
    policy: float = 0.08,
    sentinel: float = 0.55,
    cross: float = 0.20,
    speed: float = 1.0,
    params: int = 153350,
    wire: int = 830,
    tokens: int | None = 184,
):
    return {
        "absolute_gate_pass": gate,
        "parameter_count": params,
        "serialized_observation_bytes": wire,
        "active_flop_tokens": tokens,
        "selection_metrics": {
            "worst_domain_advantage_nrmse": adv,
            "worst_domain_policy_tv": policy,
            "worst_domain_sentinel_macro_nrmse": sentinel,
            "worst_domain_cross_fit_p95_tv": cross,
            "worst_domain_policy_inference_seconds_per_sample": speed,
        },
    }


def test_absolute_gate_failure_is_never_revived() -> None:
    summaries = {
        "FAIL_BUT_FAST": _summary(gate=False, adv=0.01, policy=0.01, speed=0.01),
        "PASS": _summary(gate=True, adv=0.60, policy=0.10, speed=2.0),
    }
    winner, trace = apply_frozen_selection(summaries)
    assert winner == "PASS"
    assert trace[0]["survivors"] == ["PASS"]
    assert trace[0]["discarded"] == ["FAIL_BUT_FAST"]


def test_advantage_equivalence_band_keeps_only_candidates_within_0015() -> None:
    summaries = {
        "A": _summary(adv=0.500),
        "B": _summary(adv=0.514, policy=0.07),
        "C": _summary(adv=0.516, policy=0.01),
    }
    winner, trace = apply_frozen_selection(summaries)
    rank2 = next(row for row in trace if row["rank"] == 2)
    assert rank2["threshold"] == 0.515
    assert rank2["survivors"] == ["A", "B"]
    assert winner in {"A", "B"}


def test_policy_band_breaks_advantage_equivalent_candidates() -> None:
    summaries = {
        "A": _summary(adv=0.500, policy=0.090),
        "B": _summary(adv=0.510, policy=0.079),
    }
    winner, trace = apply_frozen_selection(summaries)
    assert winner == "B"
    rank3 = next(row for row in trace if row["rank"] == 3)
    assert rank3["survivors"] == ["B"]


def test_relative_inference_band_is_five_percent() -> None:
    summaries = {
        "A": _summary(speed=1.00, params=153350),
        "B": _summary(speed=1.049, params=152000),
        "C": _summary(speed=1.051, params=1),
    }
    winner, trace = apply_frozen_selection(summaries)
    rank6 = next(row for row in trace if row["rank"] == 6)
    assert abs(rank6["threshold"] - 1.05) < 1e-12
    assert rank6["survivors"] == ["A", "B"]
    # B then wins on lower parameter count among the speed-equivalent survivors.
    assert winner == "B"


def test_conservative_final_tie_keeps_v1_control() -> None:
    summaries = {
        "C0_V1_FROZEN_CONTROL": _summary(params=152438, wire=126, tokens=None),
        "C1_V2_NO_FLOP_TOKEN": _summary(params=152438, wire=126, tokens=0),
    }
    winner, trace = apply_frozen_selection(summaries)
    assert winner == "C0_V1_FROZEN_CONTROL"
    assert trace[-1]["rank"] == 10


def test_v2_only_tie_prefers_fewer_active_flop_tokens() -> None:
    summaries = {
        "C2_V2_H1_CANONICAL_184": _summary(tokens=184),
        "C3_V2_H2_MIN_CHANGE_181": _summary(tokens=181),
        "C4_V2_H3_RECLUSTERED_184": _summary(tokens=184),
    }
    winner, trace = apply_frozen_selection(summaries)
    assert winner == "C3_V2_H2_MIN_CHANGE_181"
    rank9 = next(row for row in trace if row["rank"] == 9)
    assert rank9["survivors"] == ["C3_V2_H2_MIN_CHANGE_181"]
