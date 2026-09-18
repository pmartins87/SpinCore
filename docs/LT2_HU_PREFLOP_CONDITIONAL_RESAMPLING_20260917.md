# SpinCore — LT2 HU preflop conditional resampling audit

Date: 2026-09-17
Status: **COMPLETE — HIDDEN/CHANCE VARIANCE DOMINATES**

This gate has completed successfully. Canonical result:

- `docs/LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_RESULT_20260917.md`

Headline result:

- conditional hidden/chance variance: **93.73%** of sampled-target MSE;
- future-board variance: **65.88%**;
- opponent-hand posterior variance: **26.10%**;
- residual exact-level-1 opponent-action variance: **1.74%**;
- current-model MSE to conditional mean: **6.27%**.

Across 64 anchors, model/conditional-mean regret-matching policy TV is still high (`0.6903`) despite the small raw-MSE share, so the next test must evaluate target estimators in policy space and normalize by actual node cost.

Next gate:

- `docs/LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_20260917.md`
- launcher: `tools/run_lt2_hu_preflop_target_estimator_budget.sh`

Root training remains paused.
