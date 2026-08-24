from __future__ import annotations

import numpy as np

import r7_5_arch_reset_v1plus_phase2c0_structural_reach_factorization as c0
import r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot as base
import r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot_controlfair_v2 as fair


def main() -> int:
    hands = c0._ordered_hands((0, 1))
    wa = np.linspace(0.2, 1.0, len(hands), dtype=np.float64)
    wb = np.linspace(1.0, 0.2, len(hands), dtype=np.float64)

    indices = [fair._control_index(seed) for seed in range(64)]
    if any(index < 0 or index >= base.K for index in indices):
        raise RuntimeError("Phase2C2 fair-control index outside K64")
    if len(set(indices)) < 16:
        raise RuntimeError("Phase2C2 fair-control index namespace lacks deterministic spread")

    seed = 987654321
    raw, _ = fair._original_stratified_joint_indices(hands, wa, wb, seed=seed)
    corrected, stats = fair._fair_stratified_joint_indices(hands, wa, wb, seed=seed)
    index = fair._control_index(seed)
    if len(raw) != base.K or len(corrected) != base.K:
        raise RuntimeError("Phase2C2 fair-control proposal-size drift")
    if corrected[0] != raw[index]:
        raise RuntimeError("Phase2C2 fair-control selected cell was not rotated to index zero")
    if sorted(corrected) != sorted(raw):
        raise RuntimeError("Phase2C2 fair-control rotation changed candidate K64 multiset")
    if int(stats["control_cell_original_index"]) != index:
        raise RuntimeError("Phase2C2 fair-control telemetry index drift")
    if stats["control_cell_selection"] != "UNIFORM_OVER_64_STRATIFIED_CELLS_PRE_OUTPUT_FROZEN":
        raise RuntimeError("Phase2C2 fair-control telemetry mode drift")

    print("R7.5 architecture-reset Phase2C2 fair-control synthetic tests PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
