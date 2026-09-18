# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — K4 IMPROVES ESTIMATION BUT DOES NOT EXPLAIN FAILURES — TARGET-DRIFT / MODEL-TRACKING AUDIT ACTIVE**
Date: 2026-09-18

## Preserved milestones

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Never rewrite either checkpoint.

## Current scientific conclusion

K4 future-board averaging is a genuine target-estimator improvement.

However the full cross-street FAILURE/CONTROL analysis shows that estimator noise is **not failure-specific**.

Important examples:
- Jammer CONTROL absolute future-board variance exceeds FAILURE significantly.
- PassiveCaller FLOP relative board fraction differs, but absolute board variance does not.
- UniformLegal TURN relative model-error fraction differs, but absolute model error does not.
- K4 MSE improvement does not distinguish FAILURE from CONTROL in any context.

Therefore:
- K4 is not yet a justified causal intervention;
- long training remains frozen.

## Missing distinction

The Stage-B-only variance audit cannot tell whether regression came from:

### Target nonstationarity
The conditional self-play target changed between Stage A and Stage B.

### Model tracking failure
The target may be similar, but Stage B no longer represents it accurately.

### Both
The target moved and the model failed to follow.

## Target-drift / model-tracking gate

Canonical contract:

`docs/LT2_CROSS_STREET_TARGET_DRIFT_TRACKING_20260918.md`.

Use the same 72 FAILURE/CONTROL anchors.

For every anchor:
- exact hidden hand fixed;
- same future-board samples A/B;
- same RNG seeds A/B;
- 8 boards × 4 repeats;
- exact1;
- canonical legal-action mean subtraction.

Measure:
- target drift A->B;
- model A error to target A;
- model B error to target B;
- B-A own-target error;
- model drift;
- tracking error;
- reference best-action changes;
- sampled action regret under own-stage target.

## Decision logic

If target drift is large but both models fit own targets:
- investigate self-play nonstationarity and target evolution.

If target drift is small but Stage-B own-target error worsens:
- investigate reservoir coverage, catastrophic interference, representation or optimizer dynamics.

If both are large:
- treat the problem as moving-target + tracking instability.

Only after this distinction is established may an intervention be designed.

## Immediate direction

1. Keep Stage A/B frozen.
2. Run `bash tools/run_lt2_cross_street_target_drift_tracking.sh`.
3. Wait for `LT2_CROSS_STREET_TARGET_DRIFT_TRACKING_PASS`.
4. Send `SpinCore_LT2_cross_street_target_drift_tracking.json`.
5. Keep holdout seeds `20261001..20261006` untouched.
6. Do not train K4 or resume long training.
