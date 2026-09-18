# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE B PASS — 4.5M ROOTS — HU-JAMMER NEGATIVE — EXACT0 COMPUTE FRONTIER CONFIRMED — BOARD-ONLY AVERAGING NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_RESULT_20260917.md`
- `docs/LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_20260917.md`
- `docs/LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Confirmed practical failure

Powered 30k weak-baseline gate:

- Stage B HU Jammer raw chip EV `-5.141`, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage-B-minus-Stage-A HU-Jammer paired delta `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Root scaling remains paused.

## Closed optimizer/capacity findings

AveragePolicy: extra fitting does not improve held-out CE.

Advantage: 100 -> 1600 optimizer steps lowers held-out MSE only modestly and does not reproducibly improve production-policy TV.

A larger network is not the next admitted intervention.

## Conditional target mechanism

HU-preflop conditional decomposition:

- future-board variance: **65.88%**;
- opponent-hand posterior variance: **26.10%**;
- residual exact-level-1 opponent-action variance: **1.74%**;
- current-model MSE to conditional mean: **6.27%**.

Total hidden/chance conditional variance: **93.73%**.

## Target-estimator budget sweep — complete

The paired 64-anchor exact0/exact1 budget sweep completed successfully.

Overall same-K K64:
- exact0: 4,822 nodes, MSE `0.001178`, TV `0.3700`;
- exact1: 10,103 nodes, MSE `0.001145`, TV `0.3637`.

Exact1 therefore buys only a tiny same-K gain at roughly twice the node cost.

Matched-compute comparisons use approximately exact0 at 2K versus exact1 at K.

Paired target-MSE differences, exact0 minus exact1, are negative with 95% intervals entirely below zero at every matched budget from K2-vs-K1 through K64-vs-K32.

Conclusion:

**exact0 plus more independent hidden deals is decisively more compute-efficient in target MSE. Exact1 shows no reproducible policy-space advantage large enough to justify its node cost and is not promoted to production training.**

## Policy-space caution

The 64-deal reference itself remains noisy:

- overall 32-vs-32 split-half policy TV: `0.4614`;
- root: `0.5552`;
- continuation-1: `0.5583`;
- continuation-2+: `0.1738`;
- facing-all-in: `0.2470`.

Exact0 averaging curve:

- K1 TV `0.4866`;
- K4 `0.4493`;
- K16 `0.4137`;
- K32 `0.3742`;
- K64 `0.3700`.

MSE falls much faster than policy TV/regret. Do not select K64 merely because raw MSE is smallest.

## HU-Jammer-specific finding

For the 25 FACING_ALL_IN anchors, exact0 and exact1 are identical at every K and at the same node cost.

Once the opponent is already all-in there is no future opponent-action branch for exact-level integration to remove.

Therefore exact1 cannot directly solve the most Jammer-relevant subset.

FACING_ALL_IN:
- current model TV `0.6115`;
- current branch mismatch `84%`;
- exact0 K32 TV `0.2609`;
- exact0 K64 TV `0.2580`.

Chance/hidden-deal averaging is the relevant variance lever.

## Immediate bounded gate — board-only averaging

Full hidden-deal averaging attacks both opponent hand and future board, but online opponent-hand posterior resampling is substantially more complex than future-board resampling.

Future-board variance is the largest single component.

Run:

```bash
bash tools/run_lt2_hu_preflop_board_only_averaging.sh
```

Design:

- same 64 HU-preflop anchors;
- independent reference: 16 posterior hands × 4 future boards, exact1;
- candidate: 16 posterior hands from a separate stream;
- keep each candidate opponent hand fixed;
- average K = 1,2,4,8 independent future boards at exact0;
- no training roots and no optimizer steps.

This directly measures how much of the full-deal gain can be captured by the simpler production-feasible intervention.

## Decision after board-only gate

- If board-only K4/K8 captures most of the full-deal policy improvement, build the first bounded training pilot around future-board averaging.
- If it plateaus far above full-deal averaging, opponent-hand posterior variation must also be addressed.
- If it lowers MSE but barely changes TV/regret, switch next to a policy-aligned Advantage objective rather than brute-force averaging.
- Any training candidate must later beat preserved Stage B on the powered weak-baseline suite before long root scaling resumes.

DeepCrusher remains deferred.

## Immediate user action

Pull current `main` and run `bash tools/run_lt2_hu_preflop_board_only_averaging.sh`. Wait for `LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_PASS`, then send `SpinCore_LT2_hu_preflop_board_only_averaging.json`. Do not resume root training first.
