# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE B PASS — 4.5M ROOTS — HU-JAMMER NEGATIVE — ADVANTAGE MSE IMPROVES WITH MORE FIT BUT POLICY-TV DOES NOT — VALUE-SENSITIVITY AUDIT NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_ADVANTAGE_BUDGET_SWEEP_V2_RESULT_20260917.md`
- `docs/LT2_ADVANTAGE_VALUE_SENSITIVITY_20260917.md`
- `docs/LT2_FIT_BUDGET_SWEEP_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500 until the value-sensitivity audit is reviewed.

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Established strength result

The statistically powered 30k weak-baseline gate established a real current failure:

- Stage B HU Jammer raw chip EV `-5.141`, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage-B-minus-Stage-A HU-Jammer paired delta `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Thus extra Stage-A -> Stage-B roots measurably worsened this cell. More roots remain paused.

## AveragePolicy budget result

Additional Stage-B AveragePolicy fitting did not improve held-out cross-entropy.

3H: stored `1.089719`; +4000 `1.090321`; +8000 `1.092875`.

HU: stored `1.116161`; +4000 `1.117772`; +8000 `1.119875`.

Therefore simply increasing AveragePolicy optimizer steps is not supported as the current fix.

## Corrected Advantage budget sweep V2

The V2 sweep used the exact production Lean policy mapping and three independent deterministic reset/training replicates over a fixed 25k holdout per domain.

### Three-handed

100 steps mean:

- MSE `0.03194285`;
- policy TV `0.604686`;
- argmax agreement `31.78%`;
- predicted all-nonpositive `12.38%`.

1600 steps mean:

- MSE `0.03073128`;
- policy TV `0.602978`;
- argmax agreement `37.18%`;
- predicted all-nonpositive `10.10%`.

100 -> 1600 reduces MSE about `3.79%`, but TV is effectively unchanged. Target all-nonpositive frequency is `32.13%`.

### True HU

100 steps mean:

- MSE `0.04608811`;
- policy TV `0.587335`;
- argmax agreement `28.64%`;
- predicted all-nonpositive `21.98%`.

1600 steps mean:

- MSE `0.04447139`;
- policy TV `0.604104`;
- argmax agreement `34.52%`;
- predicted all-nonpositive `4.05%`.

100 -> 1600 reduces MSE about `3.51%`, but TV worsens by about `0.01677`. Target all-nonpositive frequency is `37.11%`.

## Decision from V2

The predeclared optimizer branch is resolved: more Advantage fit reduces MSE but does **not** reproducibly improve the actual production-policy distribution. Therefore a 16x production Advantage-budget increase is not admitted from this evidence.

The very large target-versus-predicted all-nonpositive mismatch, especially HU, makes regret-sign/fallback sensitivity a live mechanism. However TV alone can exaggerate strategically harmless disagreement when target action values are nearly tied.

The next diagnostic therefore measures **target-value regret in chip-equivalent units**, not another arbitrary pass score.

## Immediate value-sensitivity audit

Run:

```bash
bash tools/run_lt2_advantage_value_sensitivity.sh
```

Design:

- Stage B read only;
- no roots and no optimizer steps;
- deterministic 100k Advantage-memory samples per domain;
- exact production Lean policy mapping;
- target-value span, target-policy/model-policy value, regret to best target action, branch mismatch;
- target-all-nonpositive versus target-has-positive breakdown;
- preflop/flop/turn/river breakdown;
- descriptive target-span bins in chip-equivalent units.

This answers whether the ~0.6 TV is actually costly in target value and where the cost lives.

## Immediate user action

Pull current `main` and run `bash tools/run_lt2_advantage_value_sensitivity.sh`. Wait for `LT2_ADVANTAGE_VALUE_SENSITIVITY_PASS`, then send `SpinCore_LT2_advantage_value_sensitivity.json`.

Do not resume root training first. DeepCrusher remains deferred.