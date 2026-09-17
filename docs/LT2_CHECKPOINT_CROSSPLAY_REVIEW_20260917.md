# SpinCore — LT2 Stage A -> Stage B checkpoint cross-play review

Date: 2026-09-17
Status: **TWO INDEPENDENT RUNS COMPLETE — SIGN-UNSTABLE / NO REPRODUCIBLE ORDERING — WEAK-BASELINE VARIANCE GATE NEXT**

## Inputs

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Both evaluations were read-only and paired scenario/deal/hero-seat/per-seat RNG streams.

## Run 1 — 3000 scenarios, seed 20260918

Primary B-minus-A:

- ALL `-1.8129`, CI `[-3.9564,+0.3306]`;
- 3H `-1.9760`, CI `[-4.3207,+0.3687]`;
- HU `-1.6162`, CI `[-5.4070,+2.1747]`.

Additional diagnostics:

- HU direct `-0.6165`, CI `[-10.7519,+9.5188]`;
- 3H invasion `+2.8894`, CI `[-1.2160,+6.9949]`.

## Run 2 — 9000 scenarios, independent seed 20260919

Primary B-minus-A:

- ALL `+0.8625`, CI `[-0.4085,+2.1335]`;
- 3H `+0.9152`, CI `[-0.5940,+2.4244]`;
- HU `+0.7990`, CI `[-1.3337,+2.9317]`.

Additional diagnostics:

- HU direct `+1.5503`, CI `[-4.2649,+7.3656]`;
- 3H invasion `-2.2202`, CI `[-4.4860,+0.0456]`.

All primary signs reversed relative to run 1. Neither run excluded zero.

## Variance interpretation

The 9000-scenario rows contained 22,918 primary seat-runs, but only about 4.19% had a non-zero paired terminal B-minus-A delta. Most common-random-number trajectories therefore cancel exactly; a small minority of divergent sampled actions carry large positive or negative terminal outcomes.

This does not invalidate the cross-play design. It shows that checkpoint ordering is a sparse-difference, high-variance question under this stochastic evaluator.

A formal inverse-variance combination of the two ALL estimates is close to zero rather than favoring either checkpoint. The sign reversal means the first negative result cannot be interpreted as a confirmed Stage-B regression.

## Conclusion

The cross-play evidence establishes neither Stage-B improvement nor Stage-B regression.

What is established independently is that Stage B's action distribution moved materially relative to Stage A. Thus “no learning” is also unsupported.

The correct next question is more basic and more appropriate for the current maturity of SpinCore: **does Stage B actually beat the transparent weak curriculum opponents with adequate statistical precision?**

## Next gate

Use `tools/run_lt2_weak_baseline_variance_review.sh`.

It evaluates Stage A and Stage B on six independent seed blocks totaling 30,000 scenarios per checkpoint, then gives simultaneous family-wise 95% confidence intervals for Stage-B raw chip EV against uniform legal, passive caller and jammer in 3H and HU.

This replaces the premature DeepCrusher-next direction. DeepCrusher remains a later advanced benchmark after weak-opponent strength is established.

## Stop condition

Do not resume training beyond iteration 7500 until the 30k weak-baseline variance gate is reviewed.
