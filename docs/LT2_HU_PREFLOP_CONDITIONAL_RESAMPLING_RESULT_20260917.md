# SpinCore — LT2 HU preflop conditional resampling result

Date: 2026-09-17
Status: **COMPLETE — HIDDEN/CHANCE TARGET VARIANCE DOMINATES; NETWORK CAPACITY IS NOT THE FIRST INTERVENTION**

## Source and integrity

Stage B checkpoint: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Run completed with:
- `LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_PASS`;
- source checkpoint unchanged;
- 64 HU preflop anchors;
- 16 posterior opponent-hand strata per anchor;
- 4 future boards per selected hand;
- 4 exact-level-1 target repeats per exact hidden deal;
- 16,384 target traversals;
- 0 training roots and 0 optimizer steps.

## Primary decomposition

Equal-anchor aggregate sampled-target MSE: `0.0386818`.

Components:

- future-board variance: `0.0254840` = **65.88%**;
- opponent-hand posterior variance: `0.0100974` = **26.10%**;
- residual within-deal opponent-action variance at exact level 1: `0.0006742` = **1.74%**;
- current-model MSE to conditional mean: `0.0024261` = **6.27%**.

Therefore conditional variance totals **93.73%** of the measured sampled-target MSE, versus **6.27%** current-model conditional-mean error.

Mean-component 95% intervals are well separated:

- conditional variance total: `0.036256 [0.029358, 0.043154]`;
- future-board: `0.025484 [0.020353, 0.030615]`;
- opponent-hand: `0.010097 [0.007653, 0.012542]`;
- exact-level-1 within-deal action noise: `0.000674 [0.000433, 0.000916]`;
- model-to-conditional-mean: `0.002426 [0.001798, 0.003055]`.

The dominant mechanism is not ambiguous: preflop labels are overwhelmingly noisy because the sampled target is conditioned on hidden opponent cards and an already-fixed future board that the information-set input cannot observe.

## Region consistency

The result is consistent across all audited HU preflop strata.

### Root

- conditional variance: **93.30%**;
- model conditional-mean error: **6.70%**;
- future board: **65.81%**;
- opponent hand: **23.28%**;
- exact-level-1 action noise: **4.22%**.

### One-action continuation

- conditional variance: **92.91%**;
- model error: **7.09%**;
- future board: **65.78%**;
- opponent hand: **25.23%**;
- action noise: **1.90%**.

### Two-or-more-action continuation

- conditional variance: **94.83%**;
- model error: **5.17%**;
- future board: **66.02%**;
- opponent hand: **28.51%**;
- action noise: **0.30%**.

This is not a single-region artifact.

## Facing-all-in subset

There are 25 anchors whose immediately previous public action is ALL_IN.

For those states:

- conditional variance: **94.36%**;
- future-board variance: **67.37%**;
- opponent-hand variance: **26.99%**;
- within-deal action noise: **0%**;
- model conditional-mean error: **5.64%**.

The model/conditional-mean policy disagreement is nevertheless large:

- mean policy TV: `0.6590`;
- argmax agreement: `32%`;
- branch mismatch: `72%`;
- model regret against conditional-mean target: `37.39` chip-equivalent on average.

Target conditional-mean action mass:
- FOLD `44.15%`;
- CHECK_CALL `36.95%`;
- ALL_IN `18.90%`.

Current model action mass:
- FOLD `31.04%`;
- CHECK_CALL `34.91%`;
- ALL_IN `34.04%`.

So the model is materially over-allocating the ALL_IN branch in this targeted subset, but the raw MSE decomposition says that simply enlarging the network is not the first causal intervention.

## Policy sensitivity

Across all 64 anchors:

- mean TV(model policy, conditional-mean target policy): `0.6903`;
- argmax agreement: `26.56%`;
- mean model regret under the conditional-mean target: `39.55` chip-equivalent;
- mean signed conditional-mean-policy minus model-policy value gap: `+26.58` chips.

This is important: a relatively small **MSE fraction** can still produce a large **regret-matching policy error**, because the policy transform is highly sensitive to sign and relative advantage errors.

Thus the next test must judge target estimators in policy space, not only raw MSE.

## Quantitative implication for averaging

If independent hidden-deal target variance scaled ideally as `1/K`, the aggregate conditional-variance/model-error ratio is about:

`0.93728 / 0.06272 = 14.94`.

So roughly 15 independent hidden-deal realizations per observable state would be needed merely to reduce estimator variance to the same order as the current model's conditional-mean error.

By stratum the heuristic ratios are approximately:
- root: 13.9;
- continuation-1: 13.1;
- continuation-2+: 18.3;
- facing-all-in: 16.7.

This is a planning approximation, not yet an admitted training budget. Actual compute-normalized policy stabilization must be measured.

## Decision

Do **not**:
- enlarge the network as the next experiment;
- increase optimizer steps again;
- resume long root scaling;
- promote exact level 1 blindly to production training.

The next gate is a paired HU-preflop target-estimator budget sweep. It must compare independent hidden-deal averaging at exact levels 0 and 1 against a higher-budget conditional reference and measure:

- target MSE to reference mean;
- regret-matching policy TV;
- argmax agreement;
- branch mismatch;
- conditional-mean value gap/regret;
- actual node cost.

The question is now: **for a fixed compute budget, is it better to spend nodes on deeper exact opponent branching, or on more independent opponent-hand/future-board realizations of the same information state?**
