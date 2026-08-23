from __future__ import annotations

import math

import r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot as b6
import r7_5_arch_reset_v1plus_phase2b8_lagged_preflop_anchor as p


def _obs(street: int, nonforced_preflop: int) -> bytes:
    history_count = 2 + int(nonforced_preflop)
    raw = bytearray(120 + 20 * history_count)
    raw[:8] = b"SPNNIV3\x00"
    raw[9] = int(street)
    raw[116:120] = int(history_count).to_bytes(4, "little", signed=False)
    for index in range(2):
        offset = 120 + 20 * index
        raw[offset + 1] = 0
        raw[offset + 3] = 1
    for index in range(nonforced_preflop):
        offset = 120 + 20 * (2 + index)
        raw[offset + 1] = 0
        raw[offset + 3] = 0
    return bytes(raw)


class DummyBehavior:
    def __init__(self, policy, learned: bool):
        self.policy = tuple(policy)
        self.models = [object()] if learned else []
        self.calls = 0

    def __call__(self, state, observation, legal):
        self.calls += 1
        return self.policy


def _close(a, b, tol=1e-12):
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def test_root_native_unchanged() -> None:
    current = DummyBehavior((0.10, 0.20, 0.30, 0, 0, 0, 0, 0, 0, 0.40), True)
    lagged = DummyBehavior((0.40, 0.30, 0.20, 0, 0, 0, 0, 0, 0, 0.10), True)
    wrapper = p.LaggedBehaviorAnchorPolicy(current, lagged)
    legal = (0, 1, 2, 9)
    out = wrapper(None, _obs(0, 0), legal)
    assert _close(out, current.policy)
    assert wrapper.damped_calls == 0
    assert wrapper.root_preflop_native_calls == 1
    assert lagged.calls == 0


def test_initial_uniform_anchor_matches_phase2b6_exact_algebra() -> None:
    current = DummyBehavior((0.10, 0.20, 0.30, 0, 0, 0, 0, 0, 0, 0.40), True)
    lagged = DummyBehavior((0.25, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0.25), False)
    wrapper = p.LaggedBehaviorAnchorPolicy(current, lagged)
    legal = (0, 1, 2, 9)
    out = wrapper(None, _obs(0, 1), legal)
    expected = b6._mix_uniform(current.policy, legal, p.ANCHOR_WEIGHT)
    assert _close(out, expected)
    assert wrapper.damped_calls == 1
    assert wrapper.lagged_uniform_calls == 1
    assert wrapper.lagged_learned_calls == 0
    assert lagged.calls == 0


def test_learned_lagged_anchor_mix() -> None:
    current = DummyBehavior((0.10, 0.20, 0.30, 0, 0, 0, 0, 0, 0, 0.40), True)
    lagged = DummyBehavior((0.40, 0.30, 0.20, 0, 0, 0, 0, 0, 0, 0.10), True)
    wrapper = p.LaggedBehaviorAnchorPolicy(current, lagged)
    legal = (0, 1, 2, 9)
    out = wrapper(None, _obs(0, 2), legal)
    expected = tuple(0.75 * current.policy[i] + 0.25 * lagged.policy[i] for i in range(10))
    assert _close(out, expected)
    assert math.isclose(sum(out), 1.0)
    assert wrapper.lagged_learned_calls == 1
    assert lagged.calls == 1


def test_postflop_native_unchanged() -> None:
    current = DummyBehavior((0.10, 0.20, 0.30, 0, 0, 0, 0, 0, 0, 0.40), True)
    lagged = DummyBehavior((0.40, 0.30, 0.20, 0, 0, 0, 0, 0, 0, 0.10), True)
    wrapper = p.LaggedBehaviorAnchorPolicy(current, lagged)
    legal = (0, 1, 2, 9)
    out = wrapper(None, _obs(1, 3), legal)
    assert _close(out, current.policy)
    assert wrapper.postflop_native_calls == 1
    assert wrapper.damped_calls == 0
    assert lagged.calls == 0


def test_region_summary() -> None:
    rows = [
        {"region": "PREFLOP_ROOT", "control_tv": 0.30, "candidate_tv": 0.20},
        {"region": "PREFLOP_CONTINUATION_1", "control_tv": 0.25, "candidate_tv": 0.15},
        {"region": "PREFLOP_CONTINUATION_2PLUS", "control_tv": 0.35, "candidate_tv": 0.25},
        {"region": "FLOP", "control_tv": 0.10, "candidate_tv": 0.10},
    ]
    out = p._region_summary(rows)
    assert math.isclose(out["PREFLOP_ROOT"]["candidate_mean_tv"], 0.20)
    assert out["PREFLOP_CONTINUATION_COMBINED"]["count"] == 2
    assert math.isclose(out["PREFLOP_CONTINUATION_COMBINED"]["control_mean_tv"], 0.30)
    assert math.isclose(out["PREFLOP_CONTINUATION_COMBINED"]["candidate_mean_tv"], 0.20)


def test_frozen_contract_constants() -> None:
    assert p.ANCHOR_WEIGHT == 0.25
    assert p.CHUNKS_PER_ITERATION == 4
    assert p.ROOTS_PER_CHUNK == 64
    assert p.TOTAL_ROOTS == 768
    assert p.CAUSAL_ABS_MIN == 0.015
    assert p.CAUSAL_REL_MIN == 0.08
    assert p.B6_COMMON_ROOT_MEAN == 0.25663072380695223
    assert p.B6_COMMON_CONTINUATION_MEAN == 0.1778058850139139


def main() -> int:
    test_root_native_unchanged()
    test_initial_uniform_anchor_matches_phase2b6_exact_algebra()
    test_learned_lagged_anchor_mix()
    test_postflop_native_unchanged()
    test_region_summary()
    test_frozen_contract_constants()
    print("R7.5 architecture-reset Phase2B8 lagged-anchor synthetic tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
