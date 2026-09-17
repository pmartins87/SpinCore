# SpinCore — LT2 Advantage budget sweep V2 result

Date: 2026-09-17
Status: **COMPLETE — MORE ADVANTAGE STEPS IMPROVE MSE BUT DO NOT IMPROVE PRODUCTION-POLICY TV**

## Source

Preserved Stage B checkpoint: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

The source checkpoint remained unchanged. The sweep collected no new roots.

## Design

- fixed 25,000-item held-out Advantage sample per domain;
- holdout excluded from optimization;
- budgets `0,25,50,100,200,400,800,1600` cumulative steps;
- three independent deterministic reset/training replicates;
- batch size 1024;
- exact functional-production policy mapping: positive regret matching; if no legal predicted advantage is positive, stable masked softmax over the legal raw advantages;
- aggregate mean/min/max/stdev across replicates.

The purpose was not to enforce a numeric PASS threshold. It was to determine whether increasing the 100-step production fit budget improves behavior-space fidelity as well as MSE.

## Three-handed result

Aggregate mean at 100 steps:

- MSE `0.03194285`;
- production-policy TV `0.604686`;
- argmax agreement `0.317836`;
- predicted all-nonpositive frequency `0.123817`.

Aggregate mean at 1600 steps:

- MSE `0.03073128`;
- production-policy TV `0.602978`;
- argmax agreement `0.371831`;
- predicted all-nonpositive frequency `0.100953`.

Thus 100 -> 1600 improves MSE by about `3.79%` and argmax agreement by about `5.40` percentage points, but TV is essentially unchanged (`-0.00171`). There is no coherent monotone TV improvement across the intermediate budgets.

The held-out target all-nonpositive frequency is `0.321264`, materially above the model-predicted all-nonpositive frequency at both 100 and 1600 steps.

## True-HU result

Aggregate mean at 100 steps:

- MSE `0.04608811`;
- production-policy TV `0.587335`;
- argmax agreement `0.286370`;
- predicted all-nonpositive frequency `0.219828`.

Aggregate mean at 1600 steps:

- MSE `0.04447139`;
- production-policy TV `0.604104`;
- argmax agreement `0.345239`;
- predicted all-nonpositive frequency `0.040489`.

Thus 100 -> 1600 improves MSE by about `3.51%` and argmax agreement by about `5.89` percentage points, while production-policy TV becomes worse by about `0.01677`.

The held-out target all-nonpositive frequency is `0.371145`. At 1600 steps the model predicts all legal advantages non-positive in only about `4.05%` of weighted samples. This large branch-frequency mismatch is a strong reason to inspect regret-sign sensitivity rather than treating lower MSE as sufficient evidence for a larger optimizer budget.

## Interpretation

The predeclared branch is now resolved:

- Advantage MSE does continue to improve beyond 100 steps;
- production-policy TV does **not** improve reproducibly;
- argmax improves modestly, but remains low and non-monotone across budgets;
- the positive-regret versus all-nonpositive fallback branch is badly mismatched, especially in HU at larger budgets.

Therefore a 16x Advantage optimizer-budget increase is **not admitted** from this evidence. The next question is whether the large policy-space TV is strategically important or is concentrated in near-indifferent target states where very different action distributions have little target-value cost.

## Next diagnostic

Run a read-only Advantage value-sensitivity audit on Stage B. It should measure, in chip-equivalent units:

- target advantage span / near-indifference distribution;
- exact target-policy versus model-policy target value;
- regret to the best target action;
- excess target-value regret of the model policy versus the exact target-induced production policy;
- positive-regret/fallback branch confusion;
- conditional results for target-all-nonpositive versus target-has-positive samples;
- street and target-span breakdowns.

This provides a mathematically meaningful bridge between MSE/TV and actual decision value before any optimizer, architecture or root-volume change.

Root training remains paused at iteration 7500. AveragePolicy extra-budget fitting remains unsupported. DeepCrusher remains deferred.