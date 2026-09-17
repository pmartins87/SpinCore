# SpinCore — LT2 repeated-target variance result

Date: 2026-09-17
Status: **COMPLETE — OPPONENT-ACTION TARGET NOISE IS MATERIAL; EXACT LEVEL 1 IS COMPUTE-EFFICIENT FOR THAT NOISE, BUT FIXED-DEAL RESIDUAL REMAINS DOMINANT**

## Source

Stage B checkpoint: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

The audit was read only: no roots, no optimizer steps, no memory writes.

Design:

- 64 independently selected states per domain per street;
- 8 repeated target traversals from the exact same state/deal;
- same selected states at `exact_opponent_levels=0` and `1`;
- equal state weight across streets in the aggregate summary;
- exact legal-action decomposition:

`sample-target MSE = within-repeat target variance + model MSE to fixed-deal repeat mean`.

The fixed hidden deal/future board is held constant inside repeats. Therefore the within-repeat term isolates opponent-action external-sampling noise only. It excludes across-deal hidden-card/chance variation.

## Aggregate result

### Three-handed

Exact level 0:

- sampled-target MSE `0.0141184`;
- within-repeat MSE `0.00496045`;
- model MSE to fixed-deal repeat mean `0.00915790`;
- decomposition-by-means noise fraction `35.13%`;
- mean nodes `51.99`;
- signed repeat-mean target-policy minus model-policy value gap `+92.78` chips-equivalent.

Exact level 1:

- sampled-target MSE `0.00955313`;
- within-repeat MSE `0.00109172`;
- model MSE to fixed-deal repeat mean `0.00846141`;
- decomposition-by-means noise fraction `11.43%`;
- mean nodes `117.94`;
- signed value gap `+86.39` chips-equivalent.

Level 1 reduces the directly measured opponent-action variance by `77.99%` while multiplying nodes by `2.268x`.

### True heads-up

Exact level 0:

- sampled-target MSE `0.0211958`;
- within-repeat MSE `0.00776946`;
- model MSE to fixed-deal repeat mean `0.0134263`;
- decomposition-by-means noise fraction `36.66%`;
- mean nodes `79.81`;
- signed repeat-mean target-policy minus model-policy value gap `+107.17` chips-equivalent.

Exact level 1:

- sampled-target MSE `0.0139852`;
- within-repeat MSE `0.00151854`;
- model MSE to fixed-deal repeat mean `0.0124666`;
- decomposition-by-means noise fraction `10.86%`;
- mean nodes `184.06`;
- signed value gap `+106.37` chips-equivalent.

Level 1 reduces directly measured opponent-action variance by `80.45%` while multiplying nodes by `2.306x`.

## Compute-normalized variance efficiency

For a fixed node budget, the variance contribution of an independent sample mean scales approximately as `target_variance * node_cost / compute_budget`.

Using the measured ratios:

- 3H: `0.2201 * 2.268 = 0.499`; exact level 1 is about `2.00x` more statistically efficient for reducing this opponent-action sampling component per node;
- HU: `0.1955 * 2.306 = 0.451`; exact level 1 is about `2.22x` more efficient for this component per node.

This does **not** prove exact level 1 improves final poker strength. It establishes that, for the opponent-action noise isolated by this audit, the extra branching buys more variance reduction than its node cost.

## Street structure

The noise mechanism is not uniform.

HU exact-level-0 decomposition-by-means noise fractions are approximately:

- preflop `17.0%`;
- flop `59.5%`;
- turn `64.5%`;
- river `50.1%`.

Exact level 1 reduces within-repeat MSE by approximately:

- preflop `75.1%`;
- flop `71.1%`;
- turn `82.4%`;
- river `99.2%`.

3H shows the same qualitative pattern: level 1 materially reduces measured opponent-action noise on every street and is especially strong on river.

## Critical correction to interpretation

The remainder after subtracting within-repeat opponent-action variance must **not** be called pure network approximation error.

The repeat mean is conditioned on one fully specified hidden deal and future board. The SPNNIV1 model intentionally does not observe opponents' private cards or future chance. Consequently `model MSE to fixed-deal repeat mean` can contain:

- hidden-opponent-card variance invisible at the information set;
- future-card/chance variance invisible at the information set;
- historical target/policy variation for the same encoded input;
- actual representation/capacity/optimization error.

At exact level 1 this residual is about `88.6%` of aggregate sampled-target MSE in 3H and `89.1%` in HU, but that percentage is **not** an approximation-error estimate.

This point is especially important in HU preflop: opponent-action repeat noise is only about `17%` at production level 0, while the model-versus-fixed-deal residual is large. Before blaming SPNNIV1 capacity, the part of that residual that is unavoidable for identical observable inputs must be measured.

## Decision

Do not resume long root training yet.

Exact level 1 is now a legitimate candidate mechanism because it reduces one measured source of target noise efficiently per node. However starting an expensive exact-level continuation before separating same-input hidden/chance/history variance from learnable conditional-mean error would still be premature.

The next read-only gate groups stored Advantage samples with **exactly identical SPNNIV1 observation bytes and exact legal masks**. A deterministic network must output the same prediction for every sample in such a group. This gives an empirical conditional-target decomposition under the actual Stage-B reservoir distribution.

That audit will determine whether the remaining training loss is mostly:

1. target dispersion that no deterministic function of the current neural input can fit; or
2. error of the current network relative to the empirical same-input conditional mean.

Only the second component is direct evidence for representation/capacity/optimizer insufficiency.
