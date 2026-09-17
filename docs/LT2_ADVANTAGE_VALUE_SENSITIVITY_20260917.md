# SpinCore — LT2 Advantage value-sensitivity diagnostic

Date: 2026-09-17
Status: **ACTIVE — READ-ONLY DECISION-VALUE DIAGNOSTIC**

## Why this diagnostic exists

The corrected multi-seed Advantage budget sweep established a specific pattern:

- more optimizer steps continue to reduce held-out Advantage MSE;
- production-policy TV does not improve reproducibly;
- argmax agreement improves only modestly and non-monotonically;
- target all-nonpositive frequency is much higher than model-predicted all-nonpositive frequency, especially in HU.

This means MSE and distributional policy distance are no longer sufficient to decide the next intervention. A large TV can be strategically harmless if it occurs where all legal actions have nearly equal target value, or serious if it occurs where target action values differ materially.

The next audit therefore measures **decision value**, not another arbitrary score.

## Source

Stage B checkpoint: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

No roots are collected. No optimizer step is taken. The source checkpoint is hash-checked before and after.

## Sample design

Default: deterministic uniform sample of 100,000 stored Advantage-memory items per domain, evaluated with the stored Stage-B Advantage model.

Both THREE_HANDED and TRUE_HEADS_UP are analyzed independently.

The policy mapping is exactly the functional-production Lean rule:

- normalize positive legal advantages when any positive regret exists;
- otherwise use stable masked softmax over raw legal advantages.

## Core value metrics

For each held-out sample, let the stored target Advantage vector be the local action-value reference.

The audit computes:

- target action-value span in chip-equivalent units;
- target-induced production policy;
- model-induced production policy;
- target-policy regret to the best target action;
- model-policy regret to the best target action;
- signed target-policy-minus-model-policy target-value gap;
- positive model value loss relative to the target-induced policy;
- absolute target-policy/model-policy value gap;
- TV and argmax only as supporting distributional metrics.

The canonical training utility is chip delta divided by 1500, so target-value gaps are multiplied by 1500 for chip-equivalent interpretation.

These are **diagnostic target-value quantities**, not claims of realized hand EV and not exploitability estimates.

## Regret-branch sensitivity

The audit explicitly records whether target and prediction fall on the same side of the production policy discontinuity:

- target has positive legal regret vs target all-nonpositive;
- prediction has positive legal regret vs prediction all-nonpositive;
- branch mismatch rate;
- false-positive positive-regret branch;
- false-negative positive-regret branch.

This is especially important because the V2 budget sweep showed that HU target-all-nonpositive frequency is about 37%, while the 1600-step models predicted all-nonpositive only about 4% on average.

## Breakdowns

Results are separated by:

- target all-nonpositive vs target has positive regret;
- preflop/flop/turn/river using SPNNIV1 street category;
- descriptive target-value-span bins: `<1`, `1-5`, `5-20`, `20-100`, and `100+` chips.

The span bins are descriptive, not PASS thresholds.

## Interpretation

If TV is large but model-policy target-value regret is small and concentrated in low-span states, distributional mismatch is largely strategically indifferent and should not be used alone to justify architecture or optimizer changes.

If target-value regret is substantial, especially in HU and in high-span states, the problem is behaviorally meaningful. Branch-mismatch concentration would then motivate a direct regret-sign/fallback calibration experiment before more roots.

If regret is concentrated by street, that guides a representation/target-generation audit rather than a global model change.

## Launcher

```bash
bash tools/run_lt2_advantage_value_sensitivity.sh
```

Expected completion marker: `LT2_ADVANTAGE_VALUE_SENSITIVITY_PASS`.

Root training remains paused until this result is interpreted.