# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — POLICY-CHAIN SPLIT RESOLVED — JAMMER CURRENT-BEHAVIOR LOCALIZATION ACTIVE**
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

K4 improves target estimation but does not identify the Stage-B failure mechanism.

The Stage-A/B target matrix showed:
- Jammer facing-all-in target is stationary A->B;
- Passive flop has some Stage-B own-target fit degradation but not failure-specific;
- Uniform turn remains heterogeneous.

The global HU policy-chain audit now shows that the Stage-B mechanism differs by opponent family.

## Jammer

AveragePolicy B-A:
- `-1.682`, CI95 `[-2.767,-0.597]`.

Current Advantage-induced behavior B-A:
- `-8.430`, CI95 `[-12.515,-4.345]`.

Aggregation-chain delta:
- `+6.748`, CI95 `[+2.521,+10.975]`.

The current Stage-B Advantage/behavior chain contains a strong resolved defect.

AveragePolicy reduces, rather than amplifies, the final current-behavior deterioration.

Because Advantage resets/refits every iteration, this is not by itself proof that the final snapshot caused the cumulative AveragePolicy loss. It is nevertheless the strongest upstream failure currently observed.

## PassiveCaller

AveragePolicy B-A:
- resolved negative.

Current behavior B-A:
- unresolved positive.

This suggests a separate historical aggregation / policy-memory mechanism may exist.

It is secondary until the stronger Jammer upstream defect is localized.

## UniformLegal

No resolved mechanism.

## Current-behavior first-divergence gate

Canonical contract:

`docs/LT2_HU_BEHAVIOR_FIRST_DIVERGENCE_20260918.md`.

Use:
- forensic seeds `20260920..20260925`;
- 5000 scenarios/seed;
- HU only;
- same scenario/deal/hero/baseline/RNG streams.

Run Stage A and Stage B current Advantage-induced behavior in lock-step until the first sampled hero action differs.

Decompose B-A into:
- NO_DIVERGENCE;
- PREFLOP_ROOT;
- PREFLOP_FACING_ALL_IN;
- PREFLOP_OTHER;
- FLOP;
- TURN;
- RIVER.

## Decision logic

If Jammer current-behavior loss is concentrated in PREFLOP_FACING_ALL_IN:
- next inspect broad action-gap / ranking / regret-matching calibration on non-selected states there;
- explicitly measure positive-regret support and all-nonpositive fallback incidence.

If the loss localizes to another state class:
- follow that class instead.

Do not design or train an intervention until this is known.

## Immediate direction

1. Keep Stage A/B frozen.
2. Run `bash tools/run_lt2_hu_behavior_first_divergence.sh`.
3. Wait for `LT2_HU_BEHAVIOR_FIRST_DIVERGENCE_PASS`.
4. Send `SpinCore_LT2_hu_behavior_first_divergence.json`.
5. Keep holdout seeds `20261001..20261006` untouched.
6. Do not train K4 or resume long training.
