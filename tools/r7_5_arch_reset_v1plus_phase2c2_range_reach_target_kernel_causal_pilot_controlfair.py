from __future__ import annotations

"""Pre-output control-fairness correction for frozen Phase2C2.

The base Phase2C2 implementation generated an exact randomized 8x8 stratified
posterior set but used cell 0 as the equal-compute K1 control.  A fixed stratum
cell is not itself marginally distributed as the complete posterior.  The
precommit was corrected before any Phase2C2 output: choose one of the 64 cells
uniformly with a separate frozen seed and rotate that cell to position zero.
The candidate K64 arithmetic mean is permutation invariant, so this changes
only the validity of the single-draw control, not candidate semantics or cost.

The wrapper changes base.__file__ before calling main so child arm/seed workers
spawn this same corrected entrypoint and reinstall the patch.
"""

import random

import r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot as base

CONTROL_CELL_NAMESPACE = 0x2C02C011FA1E0001
_original_stratified_joint_indices = base._stratified_joint_indices


def _control_fair_stratified_joint_indices(hands, wa, wb, *, seed: int):
    selected, stats = _original_stratified_joint_indices(hands, wa, wb, seed=int(seed))
    if len(selected) != base.K:
        raise RuntimeError("Phase2C2 control-fair correction requires exact K-cell set")
    rng = random.Random(base._mix64(CONTROL_CELL_NAMESPACE, int(seed)))
    control_index = int(rng.randrange(base.K))
    rotated = list(selected[control_index:]) + list(selected[:control_index])
    if len(rotated) != base.K or rotated[0] != selected[control_index]:
        raise RuntimeError("Phase2C2 control-fair rotation failed")
    patched = dict(stats)
    patched["control_cell_original_index"] = control_index
    patched["control_cell_selection"] = "UNIFORM_OVER_64_STRATIFIED_CELLS_PRE_OUTPUT_FROZEN"
    return rotated, patched


base._stratified_joint_indices = _control_fair_stratified_joint_indices
# _run_parent derives the child entrypoint from base.__file__.  Point it here so
# every child process installs the same frozen correction before executing.
base.__file__ = __file__


def main() -> int:
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
