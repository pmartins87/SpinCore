# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED AT 4.5M — BOARD-AVERAGING SMOKE FIXED — CAUSAL ATTRIBUTION REQUIRED BEFORE TRAINING**
Date: 2026-09-18

## Current state

The line has reached:

- LT0: 120k roots;
- LT1: 1.2M roots;
- LT2 Stage A: 1.8M roots;
- LT2 Stage B: 4.5M roots / iteration 7500;
- powered weak-baseline gate confirms Stage B HU Jammer negative;
- HU-preflop target variance is dominated by hidden/chance variation;
- exact1 is not compute-efficient;
- board-only future-board averaging has a measured K4 estimator elbow;
- the first mechanics smoke caught RNG coupling and stopped safely;
- the implementation has been repaired with canonical preflop action-trace replay;
- no K4 training continuation is authorized yet.

Read first:

- `LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_FAILURE_20260918.md`
- `LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_20260918.md`
- `LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_RESULT_20260918.md`

## Preserved milestones

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Never rewrite either preserved checkpoint.

## Strength gate

Stage B HU Jammer is `-5.141` chips/hand with simultaneous family-wise 95% CI `[-9.078,-1.204]`.

Stage B minus Stage A HU Jammer is `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Blind root scaling remains disallowed.

## Estimator mechanism

HU-preflop conditional decomposition:

- future-board variance: **65.88%**;
- opponent-hand posterior variance: **26.10%**;
- residual exact-level-1 opponent-action variance: **1.74%**;
- current-model MSE to conditional mean: **6.27%**.

Board-only K4 reduces label noise and improves branch mismatch/regret enough to justify an implementation experiment.

This does not yet establish causal responsibility for the A->B strength regression.

## First mechanics-smoke failure

The first K1-vs-K4 smoke found different preflop sample counts across future boards.

The reason is structural:

- traverser nodes exact-expand several action branches;
- the traversal is depth-first;
- a branch can enter postflop and consume RNG;
- different future boards can produce different postflop sample paths;
- later preflop branches then see different RNG positions.

Therefore "reset RNG once at root" was not a valid common-random-number construction.

The gate correctly stopped before training.

## Corrected K4 implementation

Default `hu_preflop_board_average_k=1` remains canonical.

For experimental K4:

- board 0 is a completely canonical traversal;
- each sampled preflop opponent action on board 0 is recorded with observation/legal set;
- alternate boards replay that exact preflop opponent-action trace;
- each replayed preflop sample consumes one dummy RNG draw;
- postflop opponent sampling remains ordinary;
- every replayed preflop observation/legal set must match canonical;
- only preflop targets are averaged;
- postflop samples come only from canonical board 0;
- final RNG progression is restored to canonical board-0 state.

The generic collector only gained an overridable sampling hook. Its default implementation is unchanged.

## Active gate

Rerun:

`tools/run_lt2_hu_preflop_board_averaging_smoke.sh`

Required pass:
- identical K1/K4 root and sample counts;
- identical sample identity/order;
- exact postflop target equality;
- nonzero preflop target changes;
- K4 node multiplier measured;
- preserved Stage-B SHA unchanged.

## Scientific rule after smoke

A mechanics PASS is necessary but **not sufficient** to train K4.

Before any candidate continuation, perform Stage-A -> Stage-B causal attribution.

The audit must determine whether the known A->B regression is explained by the same target-sign/policy errors that K4 corrects.

At minimum compare, on matched HU preflop states:

- Stage-A versus Stage-B Advantage raw outputs;
- induced regret-matching policies;
- conditional low-noise target reference;
- sign/branch mismatches;
- action-level changes, especially FOLD / CHECK_CALL / ALL_IN;
- FACING_ALL_IN subset;
- whether K4 corrected targets point toward the better Stage-A behavior or merely toward a benchmark-specific proxy.

Only if this connection is demonstrated may a bounded K4 training pilot be admitted.

## DeepCrusher placement

DeepCrusher remains deferred.

## Immediate direction

1. Preserve Stage A and Stage B.
2. Keep training stopped at iteration 7500.
3. Rerun the corrected mechanics smoke.
4. Stop at PASS or first error.
5. If PASS, build the Stage-A -> Stage-B causal attribution audit.
6. Do not train K4 before that audit.
