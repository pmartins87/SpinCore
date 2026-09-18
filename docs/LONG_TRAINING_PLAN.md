# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED AT 4.5M — EXACT0 COMPUTE FRONTIER CONFIRMED — BOARD-ONLY AVERAGING ACTIVE**
Date: 2026-09-17

## Current state

The continuous learning line has reached:

- LT0: 120k roots;
- LT1: 1.2M roots;
- LT2 Stage A: 1.8M roots;
- LT2 Stage B: 4.5M roots / iteration 7500;
- 30k weak-baseline gate confirms Stage B HU Jammer negative;
- AveragePolicy extra fitting is not helpful;
- larger Advantage optimizer budgets do not justify themselves;
- HU-preflop sampled-target error is dominated by hidden/chance variance;
- exact1 is not compute-efficient versus spending the same nodes on more independent hidden deals;
- next step is board-only averaging feasibility, not more roots.

Read first:

- `LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_RESULT_20260917.md`
- `LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_20260917.md`
- `LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_RESULT_20260917.md`

## Core training contract

Current functional line remains unchanged:

- empirical SpinGo 3H/HU/blind/stack sampling;
- WTA chip-EV utility scaled by 1500;
- SPNNIV1 frozen-control representation;
- mature legacy action vocabulary;
- external-sampling Deep CFR;
- separate 3H and HU brains;
- sampled AveragePolicy trajectories;
- 2,000,000-sample reservoir capacity per memory per domain;
- 600 roots per iteration;
- Advantage reset every iteration;
- 100 Advantage optimizer steps per domain per iteration;
- batch size 1024;
- 4000 AveragePolicy optimizer steps per milestone finalization;
- production `exact_opponent_levels=0`;
- production concurrent-fit path.

No production-training semantic change has been admitted yet.

## Preserved milestones

LT1: 1.2M roots, SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.

LT2 Stage A: 1.8M roots / iteration 3000, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

LT2 Stage B: 4.5M roots / iteration 7500, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep all preserved milestones unchanged.

## Strength gate

Stage B HU Jammer is `-5.141` chips/hand with simultaneous family-wise 95% CI `[-9.078,-1.204]`.

Stage B minus Stage A HU Jammer is `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

No long root continuation is authorized.

## Target-variance mechanism

HU-preflop conditional decomposition:

- future-board variance: **65.88%**;
- opponent-hand posterior variance: **26.10%**;
- residual exact-level-1 opponent-action variance: **1.74%**;
- current-model MSE to conditional mean: **6.27%**.

Total hidden/chance conditional variance: **93.73%**.

## Compute-normalized exact-level decision

The completed estimator-budget sweep shows:

- same-K exact1 costs about 2.1x nodes;
- same-K MSE improvement from exact1 is small;
- at approximately matched compute, exact0 with 2K independent hidden deals beats exact1 with K in target MSE at every tested budget;
- paired 95% intervals for those MSE differences are entirely below zero;
- policy-TV/regret differences mostly remain statistically unresolved.

Therefore deeper opponent-action exact branching is not the next training intervention.

In FACING_ALL_IN states exact0 and exact1 are exactly identical, reinforcing that exact branching cannot directly fix the most Jammer-relevant situation.

## Why board-only averaging is next

Full independent hidden-deal averaging is the diagnostic winner, but production implementation of posterior opponent-hand resampling is invasive.

Future-board chance accounts for the largest component and can be resampled without changing the observable information set.

The active audit therefore measures whether future-board averaging alone captures enough of the benefit.

Design:

- 64 deterministic HU-preflop anchors;
- independent 64-deal reference per anchor from 16 posterior hands × 4 boards at exact1;
- candidate uses 16 separate posterior hands;
- each candidate hand is held fixed;
- 8 future boards are generated at exact0;
- board-only averages K = 1,2,4,8;
- no roots and no optimizer steps.

## Decision after board-only audit

If board-only K4/K8 captures most of the full-deal policy improvement, implement a bounded training pilot using future-board averaging only in HU preflop.

If it plateaus materially above the full-deal frontier, opponent-hand conditional resampling or another lower-variance estimator is required.

If target MSE improves but regret-matching policy TV/regret does not, the next experiment should be a policy-aligned Advantage objective rather than larger K.

Every candidate must beat preserved Stage B on the powered weak-baseline suite before root scaling resumes.

## DeepCrusher placement

DeepCrusher remains deferred until the weak-opponent curriculum is strong and stable.

## Immediate direction

1. Preserve Stage A and Stage B.
2. Keep root training stopped at iteration 7500.
3. Run `bash tools/run_lt2_hu_preflop_board_only_averaging.sh`.
4. Review `SpinCore_LT2_hu_preflop_board_only_averaging.json`.
5. Choose the first bounded training semantic change only after that review.
6. Do not move to DeepCrusher yet.
