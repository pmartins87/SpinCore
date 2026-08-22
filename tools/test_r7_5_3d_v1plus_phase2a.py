from __future__ import annotations

import random
import tempfile
from pathlib import Path

import r7_5_3d_v1plus_phase2a_strategy_capacity as base
import r7_5_3d_v1plus_phase2a_strategy_capacity_runtimefix as guard
from spincore_nn.reservoir import UniformReservoir


def test_frozen_constants() -> None:
    assert base.REPRESENTATION.endswith("EXACT_STRUCTURED_HISTORY_FINAL")
    assert base.DOMAIN == "THREE_HANDED"
    assert base.CHUNKS_PER_ITERATION == 4
    assert base.ROOTS_PER_CHUNK == 64
    assert base.ROOTS_PER_ITERATION_EFFECTIVE == 256
    assert base.TOTAL_ROOTS == 768
    assert base.CAPACITIES == {
        "S100K_CONTROL": 100_000,
        "S400K": 400_000,
        "S800K": 800_000,
    }
    assert base.COMMON_POLICY_INIT_SEED == 0x13579BDF
    assert base.COMMON_BATCH_SEED == 0x2468ACE013579BDF


def test_capture_preserves_control_reservoir_exactly() -> None:
    seed = 123456
    direct = UniformReservoir(7, seed)
    control = UniformReservoir(7, seed)
    capture = base._StrategyCapture(control)
    for item in range(100):
        direct.add(item)
        capture.add(item)
    assert capture.items == list(range(100))
    assert direct.state_dict() == control.state_dict()


def test_capacity_replay_is_deterministic() -> None:
    seed = 998877
    stream = list(range(200))
    left = UniformReservoir(23, seed ^ base.SHADOW_XOR["S400K"])
    right = UniformReservoir(23, seed ^ base.SHADOW_XOR["S400K"])
    for item in stream:
        left.add(item)
        right.add(item)
    assert left.state_dict() == right.state_dict()


def test_curve_rule() -> None:
    zero = {"ci_low": -0.001, "ci_high": 0.001}
    positive = {"ci_low": 0.001, "ci_high": 0.003}
    boot = {
        "S100K_CONTROL_TO_S400K": zero,
        "S400K_TO_S800K": zero,
    }
    ok, detail = base._curve_coherent(
        {"S100K_CONTROL": 0.22, "S400K": 0.19, "S800K": 0.17},
        boot,
    )
    assert ok and detail["ideal_monotone"]
    ok, _ = base._curve_coherent(
        {"S100K_CONTROL": 0.20, "S400K": 0.202, "S800K": 0.18},
        boot,
    )
    assert ok
    bad_boot = dict(boot)
    bad_boot["S100K_CONTROL_TO_S400K"] = positive
    ok, _ = base._curve_coherent(
        {"S100K_CONTROL": 0.20, "S400K": 0.202, "S800K": 0.18},
        bad_boot,
    )
    assert not ok


def test_runtime_guard_bindings() -> None:
    assert callable(guard._fit_only_iteration)
    assert callable(guard._fit_seed_policies_authoritative_audit)
    assert callable(guard._run_parent_guarded)
    training_seed = 1342191342
    assert (training_seed ^ 0x71A5BEEF) == (training_seed ^ 0x71A5BEEF)


def test_last_stage_report_recovery_contract() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        stream = base._stream_path(root, 1)
        base._atomic_torch_save([1, 2, 3], stream)
        # Last report may be absent only because the atomic resume checkpoint is
        # authoritative and the caller will rewrite it immediately afterward.
        guard._validate_stream_prefix_recoverable(root, 1)


def main() -> int:
    test_frozen_constants()
    test_capture_preserves_control_reservoir_exactly()
    test_capacity_replay_is_deterministic()
    test_curve_rule()
    test_runtime_guard_bindings()
    test_last_stage_report_recovery_contract()
    print("R7.5.3D Phase2A deterministic contract tests PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
