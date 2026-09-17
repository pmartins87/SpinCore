# SpinCore — LT2 same-input target variance audit

Date: 2026-09-17
Status: **ACTIVE — READ-ONLY CONDITIONAL-TARGET DIAGNOSTIC**

## Why this audit exists

The repeated-state audit showed that one level of exact opponent expansion removes roughly 78–80% of directly measured opponent-action target variance for about 2.3x node cost. That mechanism is real and compute-efficient for that specific noise source.

However the larger residual `model MSE to fixed-deal repeat mean` cannot be interpreted as pure model approximation error. The repeat mean is conditioned on hidden opponent cards and future chance that the neural input correctly does not observe.

The next causal question is therefore:

> Given exactly the same neural input that the network receives, how much target variation remains in the actual Stage-B Advantage reservoir, and how far is the current model from the empirical conditional mean of those identical inputs?

## Exact grouping contract

For each domain and street, stored Stage-B Advantage samples are grouped by:

1. exact SPNNIV1 observation bytes; and
2. exact 10-action legal mask.

Any deterministic network using the current representation must emit the same prediction for all members of one such group.

The audit is read only and uses the complete stored Stage-B Advantage reservoirs. No roots are collected, no optimizer steps are taken, and no memory is changed.

## MSE decomposition

For each duplicate-input group, let `T_i` be stored targets, `w_i` their stored training weights, `T_bar` the weighted group mean, and `P` the current Stage-B model prediction.

On legal actions the weighted identity is:

`E_w ||P - T_i||^2 = E_w ||T_i - T_bar||^2 + ||P - T_bar||^2`.

The first term is **within-same-input target variance** under the reservoir distribution.

The second is **current-model error to the empirical same-input conditional mean**.

The first term is a direct empirical floor for a deterministic function of the exact current input on that sampled distribution. It may mix:

- hidden-card and future-chance sampling;
- opponent-action external-sampling noise;
- historical-policy/target drift for repeated encoded inputs;
- any other randomness not represented in the input.

It is not claimed to be intrinsic game-theoretic irreducible variance.

## Coverage and robustness

Exact duplicate inputs may be common on some streets and sparse on others. Therefore coverage is a primary result, not something to hide.

The audit reports duplicate-group thresholds of at least:

- 2 observations;
- 4 observations;
- 8 observations.

For each threshold it reports:

- fraction of street items covered;
- fraction of stored training weight covered;
- group-count and group-size distribution;
- within-input target MSE;
- model MSE to same-input mean;
- decomposition fractions;
- current model policy TV/argmax versus the same-input mean-target policy;
- signed same-input mean-policy versus model-policy target-value gap;
- action-mass comparison, including FOLD / CHECK_CALL / ALL_IN.

No arbitrary PASS threshold is assigned. Group-size tiers are descriptive sensitivity checks.

## Primary interpretation

If within-same-input variance dominates on well-covered duplicate groups, then large raw target MSE is substantially unlearnable by any deterministic network using the current input. The next work should focus on target generation/variance reduction, weighting, or representation of information that is legitimately observable — not merely larger optimizer budgets.

If model error to the same-input conditional mean dominates, especially in HU preflop and with stable action-mass bias, then representation/capacity/optimization becomes a much stronger causal candidate.

If exact duplicates are too sparse on a street to support inference, that street remains unresolved; do not extrapolate from another street.

## Relation to exact opponent level 1

Exact level 1 is not discarded. The repeated-state audit established that it is approximately 2.0x (3H) and 2.2x (HU) more statistically efficient per node for the opponent-action variance component it removes.

Whether that improvement should be promoted into training depends on the relative size of the remaining same-input conditional variance and learnable mean error. This audit is the final read-only separator before a bounded training intervention.

## Launcher

```bash
bash tools/run_lt2_same_input_target_variance.sh
```

Expected completion marker: `LT2_SAME_INPUT_TARGET_VARIANCE_PASS`.

Long root training remains paused until this result is reviewed.
