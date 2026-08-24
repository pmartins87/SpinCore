from __future__ import annotations

"""Prove that the frozen Phase2C2 fair-control patch survives Windows spawn.

The long pilot computes continuation targets in ProcessPoolExecutor children.
On Windows those children use the spawn start method, so this test verifies the
same bootstrap path instead of relying only on an in-process monkey-patch test.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot as base
import r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot_controlfair_v2 as fair


def _spawn_probe(seed: int) -> dict:
    return {
        "stratified_patch_active": (
            base._stratified_joint_indices is fair._fair_stratified_joint_indices
        ),
        "continuation_patch_active": (
            base._continuation_target_task is fair._continuation_target_task
        ),
        "entrypoint_redirect_active": base.__file__ == fair.__file__,
        "control_index": int(fair._control_index(int(seed))),
    }


def main() -> int:
    seed = 987654321
    expected_index = int(fair._control_index(seed))
    if base.__file__ != fair.__file__:
        raise RuntimeError("Phase2C2 fair-control parent entrypoint redirect is inactive")

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        row = pool.submit(_spawn_probe, seed).result()

    if not bool(row["stratified_patch_active"]):
        raise RuntimeError("Phase2C2 fair stratified patch was lost across spawn")
    if not bool(row["continuation_patch_active"]):
        raise RuntimeError("Phase2C2 fair continuation patch was lost across spawn")
    if not bool(row["entrypoint_redirect_active"]):
        raise RuntimeError("Phase2C2 fair entrypoint redirect was lost across spawn")
    if int(row["control_index"]) != expected_index:
        raise RuntimeError("Phase2C2 fair control index drifted across spawn")

    print(
        "R7.5 architecture-reset Phase2C2 fair-control Windows-spawn test PASS",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
