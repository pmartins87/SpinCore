# SpinCore Current Work

Date: 2026-09-18
Status: **LT2 STAGE B PASS — 4.5M ROOTS — HU-JAMMER NEGATIVE — BOARD-ONLY K4 ADMITTED — MECHANICS SMOKE NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_RESULT_20260918.md`
- `docs/LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_20260918.md`
- `docs/LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_RESULT_20260917.md`
- `docs/LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500 yet.

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Confirmed practical failure

Powered 30k weak-baseline gate:

- Stage B HU Jammer raw chip EV `-5.141`, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage-B-minus-Stage-A HU-Jammer paired delta `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Long root scaling remains paused.

## Closed optimizer/capacity findings

AveragePolicy: extra fitting does not improve held-out CE.

Advantage: 100 -> 1600 optimizer steps lowers held-out MSE only modestly and does not reproducibly improve production-policy TV.

A larger network is not the next admitted intervention.

## Target-variance mechanism

HU-preflop conditional decomposition:

- future-board variance: **65.88%**;
- opponent-hand posterior variance: **26.10%**;
- residual exact-level-1 opponent-action variance: **1.74%**;
- current-model MSE to conditional mean: **6.27%**.

Total hidden/chance conditional variance: **93.73%**.

## Exact-level decision — closed

The compute-normalized exact0/exact1 sweep established:
- exact1 costs about 2.1x nodes at same K;
- exact0 with more independent hidden deals wins target MSE at matched compute across every tested budget;
- exact1 has no reproducible policy-space benefit sufficient to justify its cost;
- for FACING_ALL_IN, exact0 and exact1 are identical.

Exact1 is not promoted.

## Board-only averaging result — complete

Read-only audit:
- 64 HU-preflop anchors;
- candidate exact0;
- one sampled opponent hand fixed;
- K = 1,2,4,8 future boards;
- 0 training roots;
- 0 optimizer steps.

Overall:

| K | nodes | MSE | TV | argmax | branch mismatch | regret |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75.5 | 0.042638 | 0.5001 | 38.92% | 26.04% | 38.73 |
| 2 | 151.0 | 0.023177 | 0.4966 | 39.84% | 22.05% | 37.01 |
| 4 | 302.1 | 0.013556 | 0.4868 | 41.60% | 17.68% | 35.42 |
| 8 | 604.2 | 0.008715 | 0.4843 | 42.58% | 15.82% | 35.13 |

Paired K4 minus K1:
- MSE `-0.02908`, 95% CI `[-0.03366,-0.02450]`;
- TV `-0.01324`, CI crosses zero;
- regret `-3.31` chips, CI `[-5.80,-0.82]`;
- branch mismatch `-8.36 pp`, CI `[-11.27,-5.45]`.

Paired K8 minus K4:
- MSE improves further;
- TV, regret and argmax show no resolved extra gain;
- branch mismatch improves only another `1.86 pp`;
- node cost doubles.

**K4 is the measured policy-space compute elbow.**

## Facing-all-in

For 25 FACING_ALL_IN anchors:
- K1 TV `0.3615`;
- K4 TV `0.3262`;
- K8 TV `0.3232`.

K8 minus K1 TV = `-0.03836`, 95% CI `[-0.06827,-0.00846]`.

Board averaging is therefore directly relevant to the Jammer-facing subset, unlike exact-opponent branching.

## Admission decision

Admit **HU-preflop future-board averaging K4** as the first candidate training semantic change.

Do not use K8.

Do not start the training pilot until mechanics are verified.

Implementation is opt-in and canonical K1 remains the default.

## Immediate bounded gate — mechanics smoke

Run:

```bash
bash tools/run_lt2_hu_preflop_board_averaging_smoke.sh
```

The smoke compares the same prospective HU roots at K1 and K4 and must prove:
- same sample count/order/identity;
- canonical-board postflop targets unchanged;
- preflop targets actually change under K4;
- K4 node multiplier measured;
- no training-memory writes;
- no optimizer steps;
- Stage-B checkpoint unchanged.

Expected marker:

`LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_PASS`.

After reviewing the smoke, size one bounded causal K4 training pilot from the measured node multiplier. Only a candidate that later beats Stage B on the powered weak-baseline suite can reopen long root scaling.

DeepCrusher remains deferred.

## Immediate user action

Pull current `main` and run `bash tools/run_lt2_hu_preflop_board_averaging_smoke.sh`.

Wait for `LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_PASS`, then send `SpinCore_LT2_hu_preflop_board_averaging_smoke.json`.

Do not resume root training first.
