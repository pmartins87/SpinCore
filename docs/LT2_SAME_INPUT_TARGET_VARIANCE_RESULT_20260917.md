# SpinCore — LT2 same-input target-variance result

Date: 2026-09-17
Status: **COMPLETE — EXACT DUPLICATE COVERAGE TOO SPARSE FOR GLOBAL INFERENCE**

## Purpose

The audit scanned the complete Stage-B Advantage reservoirs and grouped samples by exact SPNNIV1 observation bytes plus exact 10-action legal mask. A deterministic model using the present input must emit one prediction for every item in each exact-input group.

The intended decomposition on duplicate groups was:

`sample-target MSE = within-same-input target variance + model MSE to same-input mean target`.

## Coverage result

Exact duplicate coverage is extremely sparse.

Three-handed, group size >=2:
- preflop: 142 / 386,080 items = 0.0368%;
- flop: 89 / 415,570 = 0.0214%;
- turn: 410 / 572,238 = 0.0716%;
- river: 720 / 626,112 = 0.1150%.

True HU, group size >=2:
- preflop: 527 / 224,491 = 0.2348%;
- flop: 72 / 369,907 = 0.0195%;
- turn: 322 / 636,398 = 0.0506%;
- river: 1,126 / 769,204 = 0.1464%.

No group reached size 8. HU preflop max group size is 3. Only one 3H preflop group reached size 4.

Postflop duplicate groups have zero median and zero maximum iteration span in this result, indicating that essentially all exact postflop duplicates came from the same training iteration rather than independent historical revisits. They are therefore especially weak evidence for hidden/chance conditional variance.

## Sparse preflop signal

The only duplicate region with substantial cross-iteration span is preflop.

At threshold >=2:

Three-handed preflop:
- within-same-input variance fraction: 33.80%;
- model-to-same-input-mean fraction: 66.20%;
- model/conditional-mean policy TV: 0.7020;
- signed target-mean-policy minus model-policy gap: +222.9 chip-equivalent;
- target FOLD mass 31.64% vs model 58.67%.

True HU preflop:
- within-same-input variance fraction: 36.86%;
- model-to-same-input-mean fraction: 63.14%;
- model/conditional-mean policy TV: 0.6199;
- signed gap: +170.8 chip-equivalent;
- target FOLD 38.18% vs model 16.61%;
- target CHECK_CALL 23.66% vs model 44.83%;
- target ALL_IN 23.15% vs model 36.38%.

The HU direction remains consistent with the earlier concern that the current Advantage behavior is too continuation-heavy relative to sampled target references, but the duplicate coverage is far too small to treat these fractions or action masses as representative of the complete preflop distribution.

## Decision

The predeclared sparse-coverage branch is active.

Do not:
- infer a global 63/37 learnable/noise split from the duplicate subset;
- enlarge the network from this evidence alone;
- resume long root training;
- promote exact level 1 directly to production training.

The next experiment must deliberately create repeated conditional samples for the same observable HU preflop information states.

## Next gate

`LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_20260917.md`

The targeted audit will preserve the observable HU preflop state while resampling:
- posterior-weighted opponent private hands;
- future boards;
- remaining opponent-action Monte Carlo noise under exact level 1.

This yields a balanced nested decomposition into opponent-action, future-board, opponent-hand, and current-model conditional-mean components.
