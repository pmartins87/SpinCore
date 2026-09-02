# R7.5.4A WSL2 i3–i5 driver scope bug — 2026-09-02

Status: **MECHANICAL DRIVER BUG IDENTIFIED; NO NEW SCIENTIFIC WORK STARTED; RECOVERY STATE PRESERVED**

The first execution of `tools/run_r7_5_4a_dense3h_wsl2_i3_i5.sh` successfully passed the frozen Linux/Python/Torch/NumPy runtime check, all 9 recovery tests, and exact SHA/provenance validation of the three fitted iteration-2 checkpoints.

It then printed `Preparing i3c1 (1/32)`, validated seed `1737995611` at fitted i2, but immediately referred to `i2c32` and failed an assertion before a new recovery worker was launched.

Root cause: Bash function variables are global unless declared `local`. The helper `validate()` assigned to the names `seed`, `iter`, `mode`, and `stage`. While the outer driver was at `iter=3`, `stage=1`, the call used to validate the predecessor was `validate <seed> 2 fit 32 ...`; this therefore overwrote the outer loop state to `iter=2`, `stage=32`. The subsequent path was consequently computed as `i2c32` rather than `i3c1`. The existing i2c32 report was then intentionally rejected because the predecessor hash expected for the real i3c1 path did not match the i2c32 report's own direct predecessor.

This is fail-closed mechanical behavior, not a strategic failure and not evidence of checkpoint corruption. The trace contains no worker heartbeat or collection invocation after `Preparing i3c1`; therefore no i3 root was started by the failed driver. Existing fitted i2 and historical partial checkpoints remain unchanged.

Hotfix commit `b385eccceeb9586b6c511f734d226f24ce677ae9` adds `tools/run_r7_5_4a_dense3h_wsl2_i3_i5_scopefix.sh`. The hotfix deterministically regenerates the original driver from commit `02e1261c4ec6c8101186ff81711b3a63c7360d13`, requires exact single-match substitutions, makes helper variables local in `quarantine`, `validate`, `progress`, and `worker`, verifies the patched structure, runs `bash -n`, and only then executes the corrected driver. Scientific source SHA `457996944f76e9f1fa0475691df978f450259641`, recovery SHA `a7eb746b0ac32ef730568150e1e2c2757bb212d2`, runtime, thread count, seeds, root order, root budget, optimizer semantics, and policy semantics are unchanged.
