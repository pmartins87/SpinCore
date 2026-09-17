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
- 30k weak-baseline gate — **COMPLETE; PRECISION TARGET MET; HU JAMMER NEGATIVE**.
- Stored-target fit audit — **COMPLETE**.
- AveragePolicy extra-budget hypothesis — **NOT SUPPORTED**.
- Advantage budget sweep V2 — **COMPLETE; MSE IMPROVES, PRODUCTION-POLICY TV DOES NOT**.
- Advantage value-sensitivity audit — **NEXT**.
- Root training beyond iteration 7500 — **PAUSED**.
- DeepCrusher — **DEFERRED**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LT2_ADVANTAGE_BUDGET_SWEEP_V2_RESULT_20260917.md`
- `docs/LT2_ADVANTAGE_VALUE_SENSITIVITY_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep both unchanged.

## Why roots remain paused

The powered weak-baseline evaluation established:

- Stage B HU Jammer `-5.141`, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage B minus Stage A HU Jammer `-1.682`, simultaneous six-claim CI approximately `[-3.143,-0.222]`.

The policy changed materially but this weak-opponent HU cell became worse. Root count alone is not a justified intervention.

## Fit-budget findings

### AveragePolicy

Additional fitting from the stored Stage-B policy does not improve held-out cross-entropy in either domain. More policy optimizer steps are therefore not the active fix.

### Advantage V2

The corrected three-replicate sweep uses exact functional-production positive-regret matching plus masked-softmax all-nonpositive fallback.

3H, 100 -> 1600 steps:

- MSE `0.03194285 -> 0.03073128` (`~3.79%` lower);
- TV `0.604686 -> 0.602978` (essentially flat);
- argmax `31.78% -> 37.18%`;
- target all-nonpositive `32.13%`, prediction at 1600 `10.10%`.

HU, 100 -> 1600 steps:

- MSE `0.04608811 -> 0.04447139` (`~3.51%` lower);
- TV `0.587335 -> 0.604104` (worse);
- argmax `28.64% -> 34.52%`;
- target all-nonpositive `37.11%`, prediction at 1600 `4.05%`.

Thus more optimizer work improves regression loss but does not reproducibly improve the policy distribution that the trainer actually uses. A 16x budget increase is not admitted from these curves.

## Immediate gate — decision-value sensitivity

Launcher: `tools/run_lt2_advantage_value_sensitivity.sh`.

The audit is read only and uses 100k stored Advantage samples per domain. It converts target and prediction through the exact production policy mapping and measures target-value consequences in chip-equivalent units.

Primary diagnostic quantities:

- target action-value span;
- model-policy and target-policy regret to the best target action;
- signed and positive target-policy/model-policy value gap;
- branch mismatch between positive-regret and all-nonpositive fallback regimes;
- supporting TV and argmax.

Breakdowns:

- target all-nonpositive vs target has positive regret;
- preflop/flop/turn/river;
- target span `<1`, `1-5`, `5-20`, `20-100`, `100+` chips.

These bins are descriptive and are not gates.

## Branch after value-sensitivity audit

If high TV corresponds to low target-value regret concentrated in near-indifferent states, do not treat TV as evidence for a global architecture change. Move to the confirmed HU-Jammer failure directly with a targeted state/action audit.

If model target-value regret is substantial and branch mismatch dominates it, run a bounded regret-sign/fallback calibration experiment before any roots.

If regret is concentrated on specific streets or large target spans, inspect representation/target generation for those states.

Only after a mechanism produces measurable improvement may a small isolated continuation be admitted.

## Immediate action

```bash
bash tools/run_lt2_advantage_value_sensitivity.sh
```

Do not resume root training until the report is reviewed.