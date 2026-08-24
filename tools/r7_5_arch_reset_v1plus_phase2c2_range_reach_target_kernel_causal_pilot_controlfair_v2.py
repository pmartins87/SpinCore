from __future__ import annotations

"""Final pre-output fair-control entrypoint for Phase2C2.

A fixed low-stratum cell is not a valid single posterior draw. This wrapper
selects one of the 64 stratified cells uniformly using a separate frozen RNG
namespace and rotates that cell to position zero. The K64 candidate mean is
permutation invariant. No Phase2C2 scientific output existed before this fix.
"""

import random
import r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot as base

CONTROL_CELL_NAMESPACE = 0x2C02C011FA1E0001
_original_stratified_joint_indices = base._stratified_joint_indices
_original_continuation_target_task = base._continuation_target_task
_original_evaluate = base._evaluate


def _control_index(seed: int) -> int:
    return int(random.Random(base._mix64(CONTROL_CELL_NAMESPACE, int(seed))).randrange(base.K))


def _fair_stratified_joint_indices(hands, wa, wb, *, seed: int):
    selected, stats = _original_stratified_joint_indices(hands, wa, wb, seed=int(seed))
    if len(selected) != base.K:
        raise RuntimeError("Phase2C2 fair control requires exact K-cell proposal")
    index = _control_index(int(seed))
    rotated = list(selected[index:]) + list(selected[:index])
    if len(rotated) != base.K or rotated[0] != selected[index]:
        raise RuntimeError("Phase2C2 fair-control rotation failed")
    stats = dict(stats)
    stats["control_cell_original_index"] = index
    stats["control_cell_selection"] = "UNIFORM_OVER_64_STRATIFIED_CELLS_PRE_OUTPUT_FROZEN"
    return rotated, stats


def _continuation_target_task(task: dict) -> dict:
    row = _original_continuation_target_task(task)
    kernel_seed = base._mix64(
        int(task["training_seed"]),
        int(task["global_root"]),
        int(task["iteration"]),
        0xA64,
    )
    row["control_cell_original_index"] = _control_index(kernel_seed)
    row["control_cell_selection"] = "UNIFORM_OVER_64_STRATIFIED_CELLS_PRE_OUTPUT_FROZEN"
    return row


def _evaluate(args) -> dict:
    result = _original_evaluate(args)
    result["training_contract"]["control_continuation_target_used"] = (
        "UNIFORM_RANDOM_CELL_ROTATED_TO_FIRST_OF_SAME_STRATIFIED_K64"
    )
    result["training_contract"]["control_cell_selection_namespace"] = hex(CONTROL_CELL_NAMESPACE)
    return result


base._stratified_joint_indices = _fair_stratified_joint_indices
base._continuation_target_task = _continuation_target_task
base._evaluate = _evaluate
# The parent derives child commands from base.__file__; force children through
# this same corrected entrypoint so the patch is installed in every process.
base.__file__ = __file__


def main() -> int:
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
