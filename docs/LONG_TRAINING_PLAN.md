# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED AT 4.5M — HU-JAMMER FAILURE CONFIRMED — TARGETED HU-PREFLOP CONDITIONAL RESAMPLING ACTIVE**
Date: 2026-09-17

## Current state

The continuous learning line has reached:

- LT0: 120k roots;
- LT1: 1.2M roots;
- LT2 Stage A: 1.8M roots;
- LT2 Stage B: 4.5M roots / iteration 7500;
- all four 2M reservoirs in replacement regime;
- Stage A -> Stage B policy movement is material;
- 30k weak-baseline precision gate complete;
- Stage B HU Jammer is statistically negative and worsened relative to Stage A;
- AveragePolicy extra fitting does not improve held-out CE;
- larger Advantage budgets reduce MSE modestly but do not improve production-policy behavior;
- fixed-state repeats show material opponent-action target variance and strong variance reduction from exact level 1;
- complete-reservoir exact-input duplicates are too sparse for a representative conditional-variance estimate;
- immediate next experiment is targeted HU-preflop conditional resampling, not more roots.

Read `LT2_SAME_INPUT_TARGET_VARIANCE_RESULT_20260917.md` and `LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_20260917.md` first.

## Core training contract

Current functional line:

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
- production behavior uses positive-regret matching and masked-softmax fallback when all legal predicted advantages are non-positive;
- production `exact_opponent_levels=0`;
- 31 root workers, one worker numerical thread, 8 parent Torch threads, vectorized batching, production concurrent-fit mode.

## Preserved milestones

LT1: 1.2M roots, SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.

LT2 Stage A: 1.8M roots / iteration 3000, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

LT2 Stage B: 4.5M roots / iteration 7500, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep all preserved milestones unchanged.

## Strength gate freezing root scaling

Stage B HU Jammer is `-5.141` chips/hand with simultaneous family-wise 95% CI `[-9.078,-1.204]`.

Stage B minus Stage A HU Jammer is `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Policy movement without practical strength improvement is not sufficient to admit more roots.

## Diagnostics already closed

More AveragePolicy optimizer steps are not supported.

A 16x Advantage optimizer increase is not supported: 100 -> 1600 steps modestly improves MSE but not the production policy mapping.

The large Advantage target-policy disagreement is not mostly caused by near-tied actions and is not dominated by the all-nonpositive fallback branch.

## Opponent-action target variance

At fixed exact hidden deal:

- production exact level 0 opponent-action noise explains about 35.1% of sampled-target MSE in 3H and 36.7% in HU;
- exact level 1 reduces that measured component by about 78.0% and 80.5%;
- node cost rises about 2.27x and 2.31x.

Exact level 1 therefore remains a serious candidate variance-reduction mechanism.

It is not yet a production-training choice because fixed-deal residual error also contains hidden-card/future-board effects absent from the neural information set.

## Same-input reservoir audit

Scanning all stored Stage-B Advantage memories found very low exact-input repeat coverage.

HU preflop is the best case, but only 527 of 224,491 stored items (0.2348%) are in duplicate exact-input groups and max group size is 3.

The sparse HU-preflop subset yields:
- within-same-input target variance 36.9%;
- model-to-same-input-mean error 63.1%;
- policy TV 0.620;
- target FOLD 38.2% vs model 16.6%;
- target CHECK_CALL 23.7% vs model 44.8%;
- target ALL_IN 23.2% vs model 36.4%.

Those values are suggestive, not representative, because coverage is too low.

## Active targeted conditional audit

The next audit creates its own controlled repeated conditional samples instead of waiting for reservoir collisions.

HU preflop only:

- 16 root anchors;
- 32 continuation-1 anchors;
- 16 continuation-2+ anchors;
- hero cards/public state/action history held fixed;
- all 2450 ordered opponent hands enumerated;
- current Stage-B opponent behavior used to compute exact hand-posterior reach weights for the observed public path;
- 16 stratified posterior hand draws per anchor;
- 4 future boards per hand;
- 4 target repeats per exact hidden deal;
- target collection uses exact opponent level 1.

Balanced nested decomposition:

`sample-target MSE`
`= within-deal action-sampling variance`
`+ future-board variance`
`+ opponent-hand posterior variance`
`+ current-model MSE to conditional mean`.

This is the first current-line diagnostic that directly separates hidden/chance conditional variance from model conditional-mean error on deliberately repeated observable HU preflop states.

## Decision branches

If hidden-hand/future-board variance dominates, architecture growth alone is the wrong intervention. Focus on lower-variance, information-set-correct target estimation.

If model error to the conditional mean dominates, test representation/capacity/per-action calibration in a small HU-preflop candidate before any long training. Localize FOLD/CHECK_CALL/ALL_IN and FACING_ALL_IN states.

If residual within-deal action noise stays large even at exact level 1, test deeper bounded exact branching.

Every candidate must beat preserved Stage B on the powered weak-baseline suite before root scaling resumes.

## DeepCrusher placement

DeepCrusher remains deferred until the weak-opponent curriculum is strong and stable.

## Immediate direction

1. Preserve Stage A and Stage B.
2. Keep root training stopped at iteration 7500.
3. Run `bash tools/run_lt2_hu_preflop_conditional_resampling.sh`.
4. Review `SpinCore_LT2_hu_preflop_conditional_resampling.json`.
5. Select one bounded causal intervention from the measured decomposition.
6. Do not scale roots or move to DeepCrusher yet.
