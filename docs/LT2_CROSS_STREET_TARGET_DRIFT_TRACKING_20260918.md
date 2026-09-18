# SpinCore — LT2 Stage-A/B target-drift and model-tracking audit

Date: 2026-09-18
Status: **ACTIVE — DISTINGUISH TARGET NONSTATIONARITY FROM MODEL TRACKING / FORGETTING**

## Trigger

The full cross-street future-chance JSON does **not** support:

`more future-board noise -> failure`.

Key findings:

- Jammer controls have significantly more absolute future-board variance than failures.
- PassiveCaller flop failures have a higher *fraction* of board variance, but no resolved excess in absolute board variance.
- UniformLegal turn failures are more model-dominated in fraction, but absolute model error is not elevated versus controls.
- K4 MSE improvements are not failure-specific.

Therefore K4 is a general variance-reduction improvement, not the demonstrated cause of Stage-B regression.

## Unanswered mechanism question

The previous audit measured only the Stage-B target process.

It cannot distinguish:

1. **target nonstationarity**
   - the low-noise conditional target itself changed from Stage A to Stage B;

2. **model tracking / approximation failure**
   - Stage B failed to represent its own current conditional target;

3. **both**.

## Design

Reuse exactly the same FAILURE/CONTROL state-selection design and forensic seeds:

- Jammer preflop FOLD-vs-CONTINUE;
- PassiveCaller FLOP;
- UniformLegal TURN;
- 12 FAILURE + 12 CONTROL per context;
- seeds `20260920..20260925`;
- holdout `20261001..20261006` untouched.

For every anchor:

- exact dealt hidden hand fixed;
- visible board fixed;
- same future-board samples for Stage A and Stage B;
- same target RNG seed for each paired explicit deal/repeat;
- 8 future boards × 4 repeats;
- exact opponent level 1.

## Common gauge

Raw Advantage labels are centered by each stage's current policy value.

Before cross-stage comparison, subtract the mean over legal actions:

`canonical(a) = Advantage(a) - mean_legal(Advantage)`.

This preserves all action-value gaps.

## Metrics

### Target drift

`MSE(target_B - target_A)`

Measures how much the self-play conditional target itself moved.

### Own-target model error

- `MSE(model_A - target_A)`;
- `MSE(model_B - target_B)`;
- paired `B-A`.

Measures approximation/tracking quality at each stage.

### Tracking error

`MSE[(model_B-model_A) - (target_B-target_A)]`.

If target moves and model moves with it, tracking error stays low.

If model drift fails to follow target drift, tracking error is high.

### Cross references

Also report:
- Stage-B model to Stage-A target;
- Stage-A model to Stage-B target;
- reference best-action changes;
- model own-reference best-action agreement;
- sampled Stage-A/B action regret under each stage's own conditional target.

## Interpretation

- high target drift + low own-target degradation -> **nonstationary self-play target drift**;
- low target drift + Stage-B own-target error increase -> **approximation / coverage / forgetting failure**;
- high target drift + high tracking error -> **both**;
- neither -> search outside the Advantage target/model chain.

## Launcher

```bash
bash tools/run_lt2_cross_street_target_drift_tracking.sh
```

Expected marker:

`LT2_CROSS_STREET_TARGET_DRIFT_TRACKING_PASS`

No training is authorized before this matrix is reviewed.
