# SpinCore Current Work

Date: 2026-09-19
Status: **LT2 STAGE B PASS — JAMMER CURRENT-BEHAVIOR LOSS LOCALIZED 73.86% TO PREFLOP FACING ALL-IN — BROAD ACTION-GAP / REGRET-MATCHING CALIBRATION NEXT — NO TRAINING**

## Active source of truth

Read before new compute:

- `docs/LT2_HU_BEHAVIOR_FIRST_DIVERGENCE_RESULT_20260919.md`
- `docs/LT2_JAMMER_FAI_BROAD_CALIBRATION_20260919.md`
- `docs/LT2_HU_POLICY_CHAIN_RESULT_20260918.md`
- `docs/LT2_TARGET_DRIFT_TRACKING_RESULT_20260918.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Current-behavior first-divergence result

Jammer current behavior:
- total Stage-B-minus-A `-8.4304` chips/hand;
- CI95 `[-12.5155,-4.3453]`;
- divergence rate `59.78%`.

### PREFLOP_FACING_ALL_IN

- frequency `27.24%`;
- 7402 seat-runs;
- contribution `-6.2267`;
- CI95 `[-9.2575,-3.1959]`.

This group explains **73.86%** of the total resolved Jammer current-behavior regression.

### PREFLOP_ROOT

- frequency `32.54%`;
- contribution `-2.2037`;
- CI95 `[-4.9947,+0.5873]`.

Numerically the remaining 26.14%, but unresolved.

### Postflop

For Jammer:
- no FLOP first divergences;
- no TURN;
- no RIVER;
- no PREFLOP_OTHER.

The entire paired current-behavior loss is preflop.

## Outcome-equivalence caution

Inside FAI, clearly FOLD-vs-non-FOLD transitions account for 4761/7402 = **64.32%**.

There are also:
- 2259 `1->9`;
- 367 `9->1`;
- 15 rare transitions involving slot 5.

Raw universal-slot divergence is not the same as strategic-value divergence.

Therefore the next gate does not select on sampled slot mismatch and does not use raw transition count as the primary metric.

## Causal synthesis

We now have:

1. global Jammer current-behavior B-A `-8.43`, resolved;
2. 73.86% of that loss localizes to FAI;
3. Jammer FAI low-noise targets are Stage-A/B stationary;
4. aggregate own-target MSE did not show a matching Stage-B degradation;
5. K4 improves estimator variance but noise is not failure-specific.

The leading hypothesis is now:

**small Advantage action-gap/sign errors are amplified by production regret matching at FAI states.**

Production RM is nonlinear:
- positive outputs are clipped/normalized;
- all-nonpositive outputs enter softmax fallback.

MSE can therefore stay similar while action support and policy EV change sharply.

## Active gate

Run:

```bash
bash tools/run_lt2_jammer_fai_broad_calibration.sh
```

The audit samples 48 broad common FAI states, 8 per forensic seed, **before** sampling the FAI action and without selecting on A/B divergence or terminal outcome.

For each anchor:
- 32 uniform compatible opponent hands;
- 8 future boards/hand;
- common Q-like action-gap reference;
- Stage-A/B raw Advantage;
- exact production RM behavior;
- stage-specific true Advantage target;
- sign/support mistakes;
- all-nonpositive fallback;
- mass on truly negative actions;
- policy regret.

Holdout `20261001..20261006` remains untouched.

DeepCrusher remains deferred.

## Immediate user action

Pull `main` and run:

```bash
bash tools/run_lt2_jammer_fai_broad_calibration.sh
```

Wait for `LT2_JAMMER_FAI_BROAD_CALIBRATION_PASS` or the first error.

Then send `SpinCore_LT2_jammer_fai_broad_calibration.json`.

Do not start any training.
