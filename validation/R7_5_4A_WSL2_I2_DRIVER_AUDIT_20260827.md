# R7.5.4A — WSL2 iteration-2 recovery driver audit

Date: 2026-08-27
Status: **PREPARED / STATICALLY AUDITED / NOT YET RUN**
Scope: local WSL2 continuation of the frozen `PF_DENSE_REFERENCE × THREE_HANDED` recovery from sealed `i2c1` through iteration-2 fit only.

## Evidence basis

The driver `tools/run_r7_5_4a_dense3h_wsl2_i2_remaining.sh` was derived from the immutable recovery implementation at `a7eb746b0ac32ef730568150e1e2c2757bb212d2`.

The frozen orchestration workflow defines iteration 2 as exactly 32 one-root collection stages, `i2c1` through `i2c32`, with `target_iteration=2`, `operation=collect`, `root_budget=1`, and `expected_roots_collected` increasing from 1 to 32. After `i2c32`, `i2finish` consumes `i2c32`, uses `operation=fit`, `root_budget=1`, `expected_roots_collected=32`, and produces stage `i2`.

The reusable recovery step fixes the three training seeds `1737995611`, `645939859`, and `1311335590`, permits at most three parallel seed jobs, uses Python `3.11.15`, Torch `2.13.0+cpu`, NumPy `2.3.5`, and carries the original iteration-1 artifact/checkpoint provenance unchanged through every recovery call.

The recovery worker itself freezes `root_budget` to 1, validates the action-stage contract, preserves `MECHANICAL_MID_ITERATION_CHECKPOINT_ONLY` provenance, supports deterministic partial collection continuation, and sets `production_training_authorized=false` and `ready_for_tables=false` in every report.

## Driver equivalence contract

The local driver preserves the orchestration semantics that matter scientifically:

- exact source execution SHA `457996944f76e9f1fa0475691df978f450259641`;
- exact recovery implementation SHA `a7eb746b0ac32ef730568150e1e2c2757bb212d2`;
- exact three frozen seeds and original-i1 artifact IDs/digests/checkpoint SHA-256 values;
- exact Linux/Python/Torch/NumPy runtime contract and two Torch threads;
- one root per seed per collection stage;
- three seeds in parallel within a stage;
- a hard barrier after all three seeds validate before the next root stage starts;
- `i2c2 → … → i2c32` only after the previously sealed `i2c1` checkpoints validate byte-for-byte;
- exact iteration-2 fit only after all three `i2c32` outputs validate with 32 collected roots;
- no iteration 3 or later work is launched.

The driver adds observational safety checks only: source-file SHA verification, recovery-module byte comparison, report/provenance assertions, checkpoint SHA verification, a single-driver lock, preservation/quarantine of incomplete interrupted outputs, per-stage progress summaries, and safe reuse of already-complete byte-validated stages. These checks do not feed the solver, learner, targets, policy, optimizer, reservoirs, root order, or strategic gates.

## Machine-readable starting seal

The user-supplied `results_i2c1/SUMMARY.json` was independently parsed and matched the expected schema, source SHA, recovery SHA, status, exact three-seed set, exact three checkpoint SHA-256 values, and exact recorded wall times. Its raw SHA-256 was recorded as `16c5e6249efdf9f46afaaf83718026d76d47d467b36c98e24e0d920da0ac813e` in `validation/R7_5_4A_WSL2_I2C1_MACHINE_READABLE_SEAL_20260827.json` on the documentation branch.

The three sealed i2c1 output checkpoints expected by the driver are:

- seed `1737995611`: `0a7e88af09b3cfd2352cdf76e3a882a416ceb8a2a6edc946b576078bbca4e172`;
- seed `645939859`: `1e37a635e2b763c95cc42158cf6f7ed33924eba42ad7d0c27bf2ad024a528987`;
- seed `1311335590`: `0b356aa5eef8dc61de55509afdb9ed8fbf7c6728ea34a08b80f29ff60c873e9b`.

## Restart behavior

A rerun starts from the durable WSL work root `/home/rz9/spincore_r754_dense3h_recovery`. Before reusing any completed stage, the driver validates the checkpoint SHA against its report and validates the report against the exact prior checkpoint and frozen provenance. If an interrupted stage contains only one of `checkpoint.pt` or `report.json`, the incomplete directory is moved to a quarantine path instead of being deleted, then that stage is recomputed from the last validated checkpoint. A complete-looking but contract-invalid stage is not silently accepted or overwritten.

The progress file is exported after every three-seed barrier to `results_i2_recovery/PROGRESS.json`. After the iteration-2 fit, the three `i2` checkpoints/reports and `results_i2/SUMMARY.json` are exported outside the WSL ext4 work root as a second durable copy.

## Governance

This driver is operational recovery tooling only. It does not claim strategic improvement, does not convert the missing cells into PASS before their final frozen checkpoints exist, does not authorize production training/tables, and does not reopen the V1+ architecture reset. The broader sequence remains: finish all three missing dense-3H cells → validate historical 36/36 → run frozen R7.5.4A-160 strategic evaluation → optional non-gating V1+ sidecar diagnosis → R7.5.5 decision/freeze.

The driver intentionally stops after iteration 2. Exact iteration-3+ continuation will be derived from repository evidence before any later local driver is admitted.
