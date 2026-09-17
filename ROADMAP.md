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
- Repeated-state target variance / exact-level-1 audit — **NEXT**.
- Root training beyond iteration 7500 — **PAUSED**.
- DeepCrusher — **DEFERRED**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LT2_ADVANTAGE_VALUE_SENSITIVITY_RESULT_20260917.md`
- `docs/LT2_REPEATED_TARGET_VARIANCE_AUDIT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep both unchanged.

## Why roots remain paused

The powered weak-baseline gate established a real practical failure:

- Stage B HU Jammer `-5.141` chips/hand, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage B minus Stage A HU Jammer `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Extra Stage-A -> Stage-B training measurably worsened this cell.

## What optimizer diagnostics established

AveragePolicy extra fitting does not improve held-out cross-entropy. More policy steps are not the current fix.

Advantage 100 -> 1600 steps lowers MSE modestly but does not reproducibly improve the production policy mapping. More optimizer budget is therefore not a justified global intervention.

## What value sensitivity established

The 100k-per-domain stored-target audit rejected the idea that policy TV ~0.6 is mostly harmless disagreement on nearly tied actions.

Overall sampled-target diagnostics:

- 3H mean target span `398.3` chips-equivalent; model-policy regret `154.0`; signed target-policy-minus-model-policy gap `+101.4`;
- HU mean target span `494.8`; model-policy regret `180.0`; signed gap `+114.2`.

The `100+ chip` target-span bucket accounts for about `98.6%` of the 3H signed gap and `99.4%` of HU. Near-zero-span states contribute essentially nothing.

However the all-nonpositive fallback branch is not the dominant source of net sampled-target loss. Target-has-positive states dominate the positive gap. Therefore do not perform a fallback-only calibration next.

HU action mass is directionally concerning: target-induced FOLD `36.5%` vs model-induced `6.6%`, CHECK_CALL `28.5%` vs `40.1%`, ALL_IN `22.3%` vs `41.2%`. Within HU target-has-positive states FOLD is `47.7%` target-induced vs `7.0%` model-induced.

These quantities are **not realized poker EV**. Individual Advantage targets are noisy external-sampling realizations, while final benchmark play uses AveragePolicy. They identify a serious approximation/target problem but not its cause.

## Immediate gate — repeated-state target variance

Launcher: `tools/run_lt2_repeated_target_variance_audit.sh`.

Design:

- Stage B checkpoint read only;
- no optimizer steps and no training-memory writes;
- 64 independently selected states per domain per street;
- first decision reached on each street under current Stage-B Advantage behavior;
- 8 repeated target traversals from the exact same solver state/deal;
- compare `exact_opponent_levels=0` versus `1` on the same states;
- exact legal-action MSE decomposition: sampled-target MSE = within-repeat target variance + model MSE to repeat mean;
- chip-equivalent RMSE, model/repeat-mean policy TV and target-value gap;
- node-cost multiplier for exact level 1.

Repeated-state variance isolates opponent-action Monte-Carlo noise for a fixed hidden deal and future board. It does **not** measure across-deal chance/hidden-card variance, so it is a lower bound on total target variance.

## Decision branches after target-variance audit

If within-repeat noise is a large fraction of target MSE and exact level 1 reduces it substantially at reasonable node cost, create a small isolated exact-level candidate and evaluate it against preserved Stage B before long training.

If model MSE to repeat-mean targets dominates, inspect SPNNIV1 representation/capacity and per-action sign calibration, especially HU FOLD versus CHECK_CALL/ALL_IN.

If opponent-action noise is large but exact level 1 barely reduces it, investigate chance/hidden-state target variance and representation generalization instead of increasing exact branching blindly.

If street/domain results differ materially, interventions remain localized.

Any candidate must pass a predeclared statistically powered weak-baseline comparison before root scaling resumes.

## Immediate action

```bash
bash tools/run_lt2_repeated_target_variance_audit.sh
```

Wait for `LT2_REPEATED_TARGET_VARIANCE_AUDIT_PASS` and review `SpinCore_LT2_repeated_target_variance.json`. Do not resume root training first.
