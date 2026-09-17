# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED AT 4.5M — HU-JAMMER FAILURE CONFIRMED — SAME-INPUT CONDITIONAL-TARGET AUDIT ACTIVE**
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
- larger Advantage budgets reduce MSE modestly but do not improve production-policy TV;
- high-span sampled-target disagreement is real;
- repeated-state audit shows material opponent-action target noise and strong variance reduction from exact level 1;
- immediate next experiment is same-input conditional-target variance, not more roots.

Read `LT2_REPEATED_TARGET_VARIANCE_RESULT_20260917.md` and `LT2_SAME_INPUT_TARGET_VARIANCE_AUDIT_20260917.md` first.

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
- production behavior uses positive-regret matching and a masked-softmax fallback when all legal predicted advantages are non-positive;
- production `exact_opponent_levels=0`;
- 31 root workers, one worker numerical thread, 8 parent Torch threads, vectorized batching, production concurrent-fit mode.

## Preserved milestones

LT1: 1.2M roots, SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.

LT2 Stage A: 1.8M roots / iteration 3000, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

LT2 Stage B: 4.5M roots / iteration 7500, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep all preserved milestones unchanged.

## Strength result that freezes root scaling

The statistically powered 30k gate established:

- Stage B HU Jammer `-5.141` chips/hand, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage B minus Stage A HU Jammer `-1.682`, simultaneous six-claim interval about `[-3.143,-0.222]`.

More roots cannot be admitted merely because policy continues to move.

## Optimizer diagnostics already closed

AveragePolicy: increasing finalization fitting does not improve held-out CE.

Advantage: 100 -> 1600 steps reduces held-out MSE by only about 3.5–3.8% and does not reproducibly improve production-policy TV. Do not multiply the production optimizer budget from this evidence.

## High-span sampled-target result

The 100k-per-domain audit showed that the large policy disagreement is not a near-tie artifact. The `100+ chip` target-span bucket carries about 86% of state weight and essentially all of the signed sampled-target value gap.

HU target-induced action mass was far more fold-heavy than model-induced action mass, directionally compatible with the HU-Jammer weakness. But one stored target is not a gold-standard target because external sampling is noisy and the final deployed checkpoint uses AveragePolicy.

## Repeated-state target-variance result

The next audit fixed each selected hidden deal and future board and repeated the target traversal eight times.

### Production exact level 0

3H:

- sampled-target MSE `0.0141184`;
- within-repeat opponent-action variance `0.00496045`;
- decomposition-by-means noise fraction `35.13%`.

HU:

- sampled-target MSE `0.0211958`;
- within-repeat opponent-action variance `0.00776946`;
- decomposition-by-means noise fraction `36.66%`.

Thus opponent-action external sampling is a material noise source even before counting hidden-card and future-chance variance.

### Exact level 1

3H:

- within-repeat MSE `0.00109172`;
- reduction `77.99%`;
- node multiplier `2.268x`.

HU:

- within-repeat MSE `0.00151854`;
- reduction `80.45%`;
- node multiplier `2.306x`.

For this isolated variance source, `variance ratio × node-cost ratio` is approximately `0.499` in 3H and `0.451` in HU. Therefore level 1 provides about `2.00x` and `2.22x` better statistical efficiency per node for suppressing opponent-action sampling noise.

This makes exact level 1 a serious candidate mechanism, but not yet an admitted long-training setting.

## Why the fixed-deal residual is not network error

After exact level 1, model MSE to the fixed-deal repeat mean is still about `88.6%` of aggregate sampled-target MSE in 3H and `89.1%` in HU.

That does **not** mean the network is responsible for 89% of the error.

The fixed-deal repeat mean is conditioned on opponent private cards and future board cards that SPNNIV1 intentionally does not expose. Consequently the residual mixes:

- hidden-card/future-chance target variance;
- historical target/policy drift for repeated encoded inputs;
- representation/capacity error;
- optimization error.

The next gate separates empirical same-input variance from conditional-mean model error under the actual stored reservoir.

## Immediate same-input audit

Launcher: `tools/run_lt2_same_input_target_variance.sh`.

Design:

- Stage B checkpoint read only and SHA-checked;
- complete stored Advantage reservoirs scanned;
- no roots, no optimizer steps, no memory writes;
- exact grouping key = SPNNIV1 observation bytes + 10-action legal mask;
- duplicate sensitivity tiers >=2, >=4 and >=8 observations;
- explicit duplicate coverage by domain and street;
- training-weight proxy decomposition:

`sample-target MSE = within-same-input target variance + model MSE to weighted same-input mean`.

A deterministic model with the current input cannot fit the within-same-input component. The second component is the learnable conditional-mean error on the covered groups.

Supporting diagnostics retain target/model action mass, TV, argmax, branch mismatch and chip-equivalent value gap.

## Decision branches

If within-same-input variance dominates on well-covered groups, prioritize target-generation variance reduction, weighting, or legitimate observable representation improvements. Larger networks alone cannot fit that component.

If current-model error to the same-input mean dominates, then representation/capacity/per-action calibration becomes a justified branch, especially in HU preflop.

If duplicate coverage is inadequate on a street, do not generalize from another street; build a targeted conditional resampling experiment.

Exact level 1 remains live. A bounded exact-level training candidate is admitted only after the same-input audit shows how much of the remaining objective is learnable under the present input.

Any candidate must then beat preserved Stage B on a statistically powered practical weak-baseline comparison before root scaling resumes.

## DeepCrusher placement

DeepCrusher remains deferred until the weak-opponent curriculum is strong and stable. It is not a current dependency.

## Immediate direction

1. Preserve Stage A and Stage B.
2. Keep root training stopped at iteration 7500.
3. Run `bash tools/run_lt2_same_input_target_variance.sh`.
4. Review `SpinCore_LT2_same_input_target_variance.json`.
5. Choose target-variance reduction versus representation/capacity from the measured conditional decomposition.
6. Do not scale roots or move to DeepCrusher yet.
