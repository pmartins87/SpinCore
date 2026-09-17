# SpinCore Roadmap — active state 2026-09-17

## Active status

- LT0 calibration — **DONE**: 120k roots.
- LT1 production-shaped milestone — **DONE**: 1.2M roots.
- LT1 physical fit optimization — **PASS**.
- LT2 Stage A — **PASS**: 1.8M roots.
- LT2 Stage B — **PASS**: 4.5M roots / iteration 7500.
- Stage B resource gate — **PASS**.
- Policy drift Stage A -> Stage B — **MATERIAL MOVEMENT CONFIRMED**.
- Checkpoint cross-play — **NO REPRODUCIBLE ORDERING**.
- 30k weak-baseline gate — **COMPLETE; HU JAMMER NEGATIVE**.
- AveragePolicy extra-budget hypothesis — **NOT SUPPORTED**.
- Advantage budget sweep V2 — **COMPLETE; MSE IMPROVES BUT PRODUCTION-POLICY TV DOES NOT**.
- 100k Advantage value-sensitivity — **COMPLETE; HIGH-SPAN SAMPLED-TARGET DISAGREEMENT CONFIRMED**.
- Repeated-state target variance / exact-level-1 audit — **COMPLETE; OPPONENT-ACTION NOISE MATERIAL AND LEVEL 1 EFFICIENT FOR THAT COMPONENT**.
- Same-input conditional-target variance audit — **NEXT**.
- Root training beyond iteration 7500 — **PAUSED**.
- DeepCrusher — **DEFERRED**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LT2_REPEATED_TARGET_VARIANCE_RESULT_20260917.md`
- `docs/LT2_SAME_INPUT_TARGET_VARIANCE_AUDIT_20260917.md`
- `docs/LT2_ADVANTAGE_VALUE_SENSITIVITY_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep both unchanged.

## Why roots remain paused

The powered weak-baseline gate established a real practical failure:

- Stage B HU Jammer `-5.141` chips/hand, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage B minus Stage A HU Jammer `-1.682`, simultaneous six-claim CI approximately `[-3.143,-0.222]`.

Extra Stage-A -> Stage-B training measurably worsened this cell.

## What optimizer diagnostics established

AveragePolicy extra fitting does not improve held-out cross-entropy. More policy steps are not the current fix.

Advantage 100 -> 1600 steps lowers MSE modestly but does not reproducibly improve the production policy mapping. More optimizer budget is not a justified global intervention.

## What value sensitivity established

The 100k-per-domain stored-target audit rejected the idea that TV ~0.6 is mostly harmless disagreement on near-tied actions. High-span states dominate the sampled-target value gap.

HU sampled action mass remains directionally concerning: target-induced FOLD `36.5%` vs model-induced `6.6%`, CHECK_CALL `28.5%` vs `40.1%`, ALL_IN `22.3%` vs `41.2%`.

Those target-induced quantities are not a gold-standard policy because stored Advantage targets are noisy external-sampling realizations and final benchmark play uses AveragePolicy.

## What the repeated-state audit established

Using 64 states/domain/street and 8 repeats from the same state/deal:

### Three-handed

- level-0 within-repeat target MSE `0.00496045` out of sampled-target MSE `0.0141184`;
- decomposition-by-means opponent-action noise fraction `35.13%`;
- level 1 reduces within-repeat MSE by `77.99%`;
- node cost multiplier `2.268x`.

### True HU

- level-0 within-repeat target MSE `0.00776946` out of sampled-target MSE `0.0211958`;
- decomposition-by-means opponent-action noise fraction `36.66%`;
- level 1 reduces within-repeat MSE by `80.45%`;
- node cost multiplier `2.306x`.

Compute-normalized for this isolated variance component, level 1 has variance×cost ratios of approximately `0.499` (3H) and `0.451` (HU) relative to production. It is therefore about `2.00x` and `2.22x` more statistically efficient per node for suppressing opponent-action sampling noise.

## Critical interpretation correction

The remaining `model MSE to fixed-deal repeat mean` is **not** pure network approximation error.

The repeat mean is conditioned on hidden opponent cards and future chance that are intentionally absent from the information-set neural input. The residual therefore mixes hidden/chance variance, historical target drift, representation/capacity error and optimizer error.

At exact level 1 the fixed-deal residual contributes about `88.6%` of aggregate sampled-target MSE in 3H and `89.1%` in HU, but those are not approximation-error percentages.

HU preflop reinforces the point: production opponent-action noise is only about `17%` there. Exact level 1 is useful, but it cannot yet be assumed to solve the HU-Jammer weakness.

## Immediate gate — same-input conditional target variance

Launcher: `tools/run_lt2_same_input_target_variance.sh`.

The read-only audit scans the complete stored Stage-B Advantage reservoirs and groups samples by exact SPNNIV1 observation bytes plus exact 10-action legal mask.

For duplicate groups it computes the weighted identity:

`sample-target MSE = within-same-input target variance + model MSE to same-input conditional mean`.

Reported sensitivity tiers are group size >=2, >=4 and >=8. Coverage is reported explicitly by domain/street; no minimum coverage is silently assumed.

Supporting outputs include policy TV/argmax, branch mismatch, same-input target/model action mass and target-value gap.

## Branch after same-input audit

If within-same-input variance dominates on well-covered groups, target-generation/variance reduction or legitimate observable representation changes come before larger networks.

If model error to the same-input conditional mean dominates, representation/capacity/per-action calibration becomes the stronger branch, especially HU FOLD vs CHECK_CALL/ALL_IN.

If exact duplicates are sparse on a street, that street remains unresolved and requires targeted conditional resampling rather than extrapolation.

Exact level 1 remains a live candidate mechanism. A bounded training candidate is admitted only after this separator shows what part of the remaining error is actually learnable from the current input.

Any training candidate must later beat preserved Stage B on a predeclared statistically powered weak-baseline comparison before root scaling resumes.

## Immediate action

```bash
bash tools/run_lt2_same_input_target_variance.sh
```

Wait for `LT2_SAME_INPUT_TARGET_VARIANCE_PASS` and review `SpinCore_LT2_same_input_target_variance.json`. Do not resume root training first.
