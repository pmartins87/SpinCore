# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE B PASS — 4.5M ROOTS — HU-JAMMER NEGATIVE — OPPONENT-SAMPLING NOISE MEASURED — SAME-INPUT CONDITIONAL-TARGET AUDIT NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_REPEATED_TARGET_VARIANCE_RESULT_20260917.md`
- `docs/LT2_SAME_INPUT_TARGET_VARIANCE_AUDIT_20260917.md`
- `docs/LT2_ADVANTAGE_VALUE_SENSITIVITY_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500 until the same-input conditional-target audit is reviewed.

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Confirmed practical failure

The powered 30k weak-baseline gate established:

- Stage B HU Jammer raw chip EV `-5.141`, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage-B-minus-Stage-A HU-Jammer paired delta `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Root scaling remains paused.

## Closed optimizer findings

AveragePolicy: extra fitting does not improve held-out CE.

Advantage: 100 -> 1600 steps reduces held-out MSE by only about 3.5–3.8% and does not reproducibly improve production-policy TV. A global optimizer-budget increase is not admitted.

## Repeated-state target-variance result

The fixed-state/deal audit used 64 states per domain per street, 8 repeats, and compared production `exact_opponent_levels=0` with `1`.

### Three-handed aggregate

Exact level 0:

- sampled-target MSE `0.0141184`;
- within-repeat opponent-action variance `0.00496045`;
- model MSE to fixed-deal repeat mean `0.00915790`;
- decomposition-by-means noise fraction `35.13%`;
- mean nodes `51.99`.

Exact level 1:

- within-repeat variance `0.00109172`;
- noise reduction `77.99%`;
- mean nodes `117.94`, or `2.268x` level 0.

### True HU aggregate

Exact level 0:

- sampled-target MSE `0.0211958`;
- within-repeat opponent-action variance `0.00776946`;
- model MSE to fixed-deal repeat mean `0.0134263`;
- decomposition-by-means noise fraction `36.66%`;
- mean nodes `79.81`.

Exact level 1:

- within-repeat variance `0.00151854`;
- noise reduction `80.45%`;
- mean nodes `184.06`, or `2.306x` level 0.

For the measured opponent-action sampling component, variance × node-cost is about `0.499` of production in 3H and `0.451` in HU. Thus exact level 1 is approximately `2.00x` and `2.22x` more statistically efficient per node for suppressing that specific noise source.

## Important interpretation correction

Do **not** call the remaining model-to-repeat-mean MSE pure approximation error.

The repeated-state mean is conditioned on one hidden opponent-card allocation and future board. Those quantities are intentionally absent from the information-set neural input. Therefore the residual includes some unknown mixture of:

- hidden-card/future-chance variance;
- historical target/policy variation for identical encoded inputs;
- representation/capacity error;
- optimization error.

At exact level 1 the residual is about `88.6%` of aggregate sampled-target MSE in 3H and `89.1%` in HU, but that percentage is not a network-error estimate.

HU preflop is especially important: production opponent-action noise accounts for only about `17%` of MSE there, while the fixed-deal residual remains large. That makes it premature to assume exact level 1 alone fixes the confirmed HU-Jammer problem.

## Immediate bounded gate — same-input conditional target variance

Run:

```bash
bash tools/run_lt2_same_input_target_variance.sh
```

The audit is read only and scans the stored Stage-B Advantage reservoirs. Samples are grouped by exact SPNNIV1 observation bytes plus exact 10-action legal mask. Any deterministic network using the present input must predict the same output inside each group.

For duplicate groups it decomposes training-weighted target MSE into:

1. within-same-input target variance; and
2. current-model error to the empirical same-input mean target.

It reports duplicate coverage and sensitivity at group sizes >=2, >=4 and >=8, by domain and street, plus target/model action mass and policy-value disagreement.

This is the correct next separator between target variance the network cannot fit from the current input and genuinely learnable conditional-mean error.

## Decision after same-input audit

- If within-input variance dominates on well-covered groups, prioritize target-generation/variance reduction or legitimate observable representation improvements; do not merely enlarge the network.
- If model error to the same-input mean dominates, representation/capacity/per-action calibration becomes a stronger causal candidate, especially HU FOLD vs CHECK_CALL/ALL_IN.
- If duplicate coverage is sparse, do not extrapolate; build a targeted conditional resampling audit for that street.
- Exact level 1 remains a live candidate, but no long continuation is admitted until this separator is reviewed.

DeepCrusher remains deferred.

## Immediate user action

Pull current `main` and run `bash tools/run_lt2_same_input_target_variance.sh`. Wait for `LT2_SAME_INPUT_TARGET_VARIANCE_PASS`, then send `SpinCore_LT2_same_input_target_variance.json`. Do not resume root training first.
