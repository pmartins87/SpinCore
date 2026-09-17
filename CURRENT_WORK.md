# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE B PASS — 4.5M ROOTS — HU-JAMMER NEGATIVE — HIGH-SPAN ADVANTAGE TARGET DISAGREEMENT CONFIRMED — REPEATED-STATE TARGET-VARIANCE AUDIT NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_ADVANTAGE_VALUE_SENSITIVITY_RESULT_20260917.md`
- `docs/LT2_REPEATED_TARGET_VARIANCE_AUDIT_20260917.md`
- `docs/LT2_ADVANTAGE_BUDGET_SWEEP_V2_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500 until the repeated-target variance audit is reviewed.

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Established external strength result

The statistically powered 30k weak-baseline gate established:

- Stage B HU Jammer raw chip EV `-5.141`, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage-B-minus-Stage-A HU-Jammer paired delta `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

This is the confirmed practical failure that keeps root scaling paused.

## Optimizer findings already closed

AveragePolicy: extra Stage-B fitting does not improve held-out CE. More policy optimizer steps are not supported as the current fix.

Advantage V2: increasing 100 -> 1600 optimizer steps lowers held-out MSE by about 3.8% in 3H and 3.5% in HU, but production-policy TV does not improve reproducibly. A 16x Advantage budget increase is therefore not admitted from loss reduction alone.

## 100k Advantage value-sensitivity result

The read-only audit evaluated 100,000 stored Advantage targets per domain.

Overall sampled-target diagnostics:

- 3H: mean target span `398.3` chips, model-policy regret to sampled-target best action `154.0`, signed target-policy-minus-model-policy sampled-target gap `+101.4`, TV `0.5887`;
- HU: mean target span `494.8` chips, model-policy regret `180.0`, signed gap `+114.2`, TV `0.5949`.

This disagreement is **not** a near-tie artifact. The `100+ chip` span bucket carries `85.98%` of 3H weight and `86.92%` of HU weight, accounting for about `98.6%` and `99.4%` of the respective signed sampled-target gaps. The `<1 chip` bucket has meaningful state weight but essentially zero value gap.

The net loss is also **not dominated by the all-nonpositive fallback branch**. On sampled targets, target-has-positive states contribute approximately `+122.9` chips in 3H and `+149.5` in HU after weighting, whereas target-all-nonpositive states contribute `-21.6` and `-35.4` respectively. A fallback-only intervention is therefore not justified.

The sampled action-mass discrepancy is striking, especially HU:

- HU FOLD target-induced `36.5%` vs model-induced `6.6%`;
- HU CHECK_CALL `28.5%` vs `40.1%`;
- HU ALL_IN `22.3%` vs `41.2%`.

Within HU target-has-positive states, FOLD is `47.7%` target-induced versus only `7.0%` model-induced. This is directionally consistent with weakness against an all-in-heavy opponent, but it is not yet a causal explanation because individual stored targets are noisy external-sampling realizations and the deployed evaluator uses AveragePolicy.

Street gaps are broad rather than isolated to a single street. HU per-decision sampled-target gaps are about preflop `200.8`, flop `149.6`, turn `96.0`, river `87.2` chips-equivalent.

## Critical interpretation limit

Do **not** call the `+101/+114` chip-equivalent gaps realized poker EV. The collector constructs targets from sampled counterfactual recursion. With production `exact_opponent_levels=0`, opponent actions below traverser branches are Monte-Carlo sampled. A large part of apparent target error may therefore be target variance rather than learnable approximation error.

The next causal question is noise versus approximation, not another optimizer budget.

## Immediate bounded gate

Run:

```bash
bash tools/run_lt2_repeated_target_variance_audit.sh
```

Design:

- Stage B source checkpoint SHA-checked and preserved;
- no optimizer steps and no writes to training memories;
- 64 selected states per domain per street;
- first reached decision on preflop/flop/turn/river under current Stage-B Advantage behavior;
- 8 repeated target traversals from the same exact state/deal;
- compare production `exact_opponent_levels=0` with bounded `exact_opponent_levels=1`;
- exact MSE decomposition into repeated-state sampling variance plus model error to the repeat-mean target;
- report node-cost multiplier and lower-noise policy/value diagnostics.

The repeat mean is diagnostic, not a GTO oracle. Fixed-deal repetitions isolate opponent-action sampling noise and do not include across-deal chance/hidden-card variance, so measured noise is a lower bound.

## Decision after repeated-target audit

If within-state sampling variance explains a large share of target MSE and exact level 1 materially reduces it at tolerable cost, test a small isolated exact-level training candidate.

If model error to the repeat mean remains dominant, move to SPNNIV1 representation/capacity and per-action calibration rather than spending more optimizer steps.

If opponent-sampling variance is high but exact level 1 barely helps, investigate chance/hidden-state variance and representation generalization.

Any candidate must beat preserved Stage B on the powered weak-baseline suite before long root training resumes.

DeepCrusher remains deferred.

## Immediate user action

Pull current `main` and run `bash tools/run_lt2_repeated_target_variance_audit.sh`. Wait for `LT2_REPEATED_TARGET_VARIANCE_AUDIT_PASS`, then send `SpinCore_LT2_repeated_target_variance.json`. Do not resume root training first.
