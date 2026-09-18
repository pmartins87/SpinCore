# SpinCore Current Work

Date: 2026-09-18
Status: **LT2 STAGE B PASS — CROSS-STREET FULL JSON REVIEW COMPLETE — FUTURE-CHANCE NOT FAILURE-SPECIFIC — TARGET-DRIFT / MODEL-TRACKING AUDIT NEXT — NO TRAINING**

## Active source of truth

Read before new compute:

- `docs/LT2_CROSS_STREET_FULL_JSON_REVIEW_20260918.md`
- `docs/LT2_CROSS_STREET_TARGET_DRIFT_TRACKING_20260918.md`
- `docs/LT2_CROSS_STREET_FUTURE_CHANCE_RESULT_20260918.md`
- `docs/LT2_JAMMER_COMMON_REFERENCE_V2_1_RESULT_20260918.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Full cross-street JSON verdict

The simple rule

`more future-board noise -> Stage-B failure`

is not supported.

### Jammer preflop

Absolute future-board variance:
- FAILURE `0.03834`;
- CONTROL `0.07201`;
- F-C approximate 95% CI `[-0.06195,-0.00538]`.

Controls are significantly noisier than failures.

K4 MSE benefit does not significantly discriminate FAILURE from CONTROL.

### PassiveCaller flop

Relative future-board fraction is higher in FAILURE:
- F `0.6097`;
- C `0.3497`;
- F-C approximate CI `[+0.0456,+0.4743]`.

But absolute future-board variance excess is unresolved:
- F-C approximate CI `[-0.00369,+0.01071]`.

K4 MSE benefit is effectively the same:
- F `-0.012697`;
- C `-0.012868`;
- F-C approximate CI `[-0.01119,+0.01153]`.

### UniformLegal turn

Failure states are more model-dominated in **fraction**, but absolute model error is not higher:
- model absolute F `0.00812`;
- C `0.00881`;
- F-C approximate CI `[-0.01220,+0.01081]`.

K4 regret gain is essentially identical in FAILURE and CONTROL.

## Strategic interpretation

K4 is a real variance-reduction improvement.

It is **not** demonstrated as the cause/fix for Stage-B regression.

Do not:
- train Jammer-specific K4;
- generalize K4 to all streets;
- resume LT2 long training.

The previous cross-street audit only measured the Stage-B target process, so it cannot distinguish:

1. target nonstationarity from Stage A to Stage B;
2. Stage-B approximation / tracking / forgetting failure;
3. both.

That is now the highest-value diagnostic.

## Active gate

Run:

```bash
bash tools/run_lt2_cross_street_target_drift_tracking.sh
```

It reuses the same FAILURE/CONTROL contexts and forensic seeds.

For every anchor it computes paired low-noise Stage-A and Stage-B conditional targets on the same hidden deal, boards and RNG seeds, canonicalizes to an action-gap gauge, then compares:

- target drift A->B;
- Stage-A own-target model error;
- Stage-B own-target model error;
- B-A own-target error;
- model-drift tracking error;
- reference best-action changes.

Holdout seeds `20261001..20261006` remain untouched.

DeepCrusher remains deferred.

## Immediate user action

Pull `main` and run:

```bash
bash tools/run_lt2_cross_street_target_drift_tracking.sh
```

Wait for `LT2_CROSS_STREET_TARGET_DRIFT_TRACKING_PASS` or the first error.

Then send `SpinCore_LT2_cross_street_target_drift_tracking.json`.

Do not start any training.
