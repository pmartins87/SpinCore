# SpinCore — LT2 training-dynamics fit audit

Date: 2026-09-17
Status: **ACTIVE — READ-ONLY FIT DIAGNOSTIC AFTER 30K WEAK-BASELINE FAILURE**

## Why this audit exists

The 30k multi-seed weak-baseline gate met its predeclared precision target and resolved a real failure mode: Stage B is negative against HU Jammer (`-5.141` chips/hand, simultaneous family-wise 95% CI `[-9.078,-1.204]`). The paired Stage-B-minus-Stage-A HU-Jammer delta is also negative and remains negative after six-claim family-wise correction.

Therefore another blind root block is not justified. The immediate question is whether the learned neural approximations are underfitting the targets already stored in the preserved reservoirs.

## Source checkpoints

- Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.
- Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Both are read only for this audit.

## Current training mechanics being diagnosed

For each domain, the current functional trainer resets the Advantage network every iteration and then fits it from the accumulated Advantage reservoir using the configured `100` optimizer steps. The AveragePolicy network is fit at milestone finalization using `4000` optimizer steps. Stage B has 2M-capacity Advantage and AveragePolicy reservoirs in replacement regime.

These facts motivate the audit but do **not** prove underfitting.

## Audit method

Launcher: `tools/run_lt2_training_dynamics_fit_audit.sh`.

Core analyzer: `tools/audit_lt2_checkpoint_fit.py`.

Default design:

- deterministic uniform sample of 25,000 items from each stored memory;
- both Stage A and Stage B;
- both THREE_HANDED and TRUE_HEADS_UP;
- Advantage and AveragePolicy memories separately;
- vectorized batching;
- read only; no optimizer step; no source-checkpoint mutation.

Advantage diagnostics:

- legal-action MSE of the stored Advantage network;
- MSE of a zero predictor on the same targets;
- `fit_fraction_vs_zero = 1 - model_MSE / zero_MSE`;
- TV distance between regret-matching policy induced by target advantages and by predicted advantages;
- argmax agreement;
- fractions of all-nonpositive target/predicted advantage vectors;
- sampled iteration-age quantiles.

AveragePolicy diagnostics:

- target entropy;
- model cross-entropy;
- excess KL (`CE - target entropy`);
- TV distance between target and predicted action distributions;
- argmax agreement;
- cross-entropy of a uniform-legal predictor;
- fraction of the available uniform-to-target cross-entropy gap closed by the model;
- sampled iteration-age quantiles.

Weighted metrics use the same stored sample weights that the canonical optimizer uses. Unweighted metrics are retained as diagnostics.

## Interpretation policy

There is deliberately no arbitrary PASS score. Interpret Stage A and Stage B comparatively and by mechanism.

- If Stage B HU Advantage fit is materially poor relative to its zero baseline and/or materially worse than Stage A, the next experiment is a bounded optimizer-budget sweep on preserved Stage-B HU Advantage memory.
- If AveragePolicy fit is materially poor, the next experiment is a policy-fit budget sweep from the preserved checkpoint without collecting new roots.
- If both fits are already strong, do not increase optimizer budgets reflexively; inspect target generation, reservoir weighting/age, state/action concentration and HU-specific learning semantics.

A fit audit measures approximation of stored targets, not poker strength. Good fit does not certify the targets; bad fit does establish an approximation problem worth isolating.

## Stop condition

Do not resume long training beyond iteration 7500 until this audit is reviewed and a bounded causal experiment is chosen. DeepCrusher remains deferred.