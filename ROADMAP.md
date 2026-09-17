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
- 100k Advantage value-sensitivity — **COMPLETE; HIGH-SPAN TARGET DISAGREEMENT CONFIRMED**.
- Repeated-state target variance / exact-level-1 audit — **COMPLETE; OPPONENT-ACTION NOISE MATERIAL, LEVEL 1 EFFICIENT FOR THAT COMPONENT**.
- Same-input conditional-target reservoir audit — **COMPLETE; DUPLICATE COVERAGE TOO SPARSE FOR GLOBAL INFERENCE**.
- HU preflop targeted conditional resampling — **NEXT**.
- Root training beyond iteration 7500 — **PAUSED**.
- DeepCrusher — **DEFERRED**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LT2_SAME_INPUT_TARGET_VARIANCE_RESULT_20260917.md`
- `docs/LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_20260917.md`
- `docs/LT2_REPEATED_TARGET_VARIANCE_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep both unchanged.

## Why root scaling remains frozen

The powered weak-baseline gate established a real practical failure:

- Stage B HU Jammer `-5.141` chips/hand, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage B minus Stage A HU Jammer `-1.682`, simultaneous six-claim CI approximately `[-3.143,-0.222]`.

More roots cannot be justified while that weakness is confirmed and the causal training defect remains unresolved.

## What is already closed

More AveragePolicy fitting does not improve held-out CE.

Advantage 100 -> 1600 optimizer steps lowers MSE modestly but does not reproducibly improve the production-policy behavior.

The large target-policy disagreement is not mostly a near-tie artifact.

A fallback-only regret-sign intervention is not supported.

## Target-variance findings

Fixed-state/deal repetitions show material opponent-action Monte Carlo noise under production exact level 0. Exact level 1 removes roughly 78–80% of that component at about 2.3x node cost and is statistically more efficient per node for suppressing that isolated source.

However fixed-deal repeat means still condition on hidden opponent cards and future boards, so their residual cannot be labeled network approximation error.

The subsequent complete-reservoir exact-input scan could not resolve this globally because duplicate exact inputs are extremely rare.

## Same-input coverage

Threshold >=2 exact-input coverage:

- 3H: preflop 0.0368%, flop 0.0214%, turn 0.0716%, river 0.1150%;
- HU: preflop 0.2348%, flop 0.0195%, turn 0.0506%, river 0.1464%.

No groups >=8. HU preflop max size 3. Postflop groups in this run have zero maximum iteration span.

Sparse HU-preflop duplicates suggest about 36.9% within-input target variance versus 63.1% model-to-mean error, but the covered mass is far too small to generalize.

## Immediate gate — targeted HU-preflop conditional resampling

Launcher:

`tools/run_lt2_hu_preflop_conditional_resampling.sh`

The diagnostic deliberately constructs repeated hidden-state/chance realizations for the same observable HU preflop anchors.

Default anchor strata:
- 16 root;
- 32 continuation-1;
- 16 continuation-2+.

For each anchor:
- all 2450 ordered opponent hands are enumerated;
- posterior reach weights come from the current Stage-B opponent behavior likelihood of the observed public path;
- 16 opponent hands are selected by stratified posterior sampling;
- 4 future boards per hand;
- 4 repeated exact-level-1 target traversals per exact hidden deal.

It decomposes:

`sample-target MSE = opponent-action variance + future-board variance + opponent-hand posterior variance + model conditional-mean error`.

It also reports FACING_ALL_IN anchors separately.

## Branch after targeted resampling

If conditional hidden/chance variance dominates, prioritize target-generation and information-set-correct variance reduction.

If model conditional-mean error dominates, move to a bounded representation/capacity/per-action experiment, localized first to HU preflop and especially FOLD versus CHECK_CALL/ALL_IN.

If exact-level-1 within-deal action noise remains material, evaluate deeper bounded exact branching before architecture changes.

No candidate gets long training until it beats preserved Stage B on a predeclared powered weak-baseline comparison.

## Immediate action

```bash
bash tools/run_lt2_hu_preflop_conditional_resampling.sh
```

Wait for `LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_PASS`, review `SpinCore_LT2_hu_preflop_conditional_resampling.json`, and keep long root training paused.
