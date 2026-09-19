# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — JAMMER LOSS LOCALIZED TO FAI — BROAD ACTION-GAP / RM CALIBRATION ACTIVE**
Date: 2026-09-19

## Preserved milestones

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Never rewrite either checkpoint.

## Current scientific conclusion

The global HU policy-chain audit showed:
- Jammer current behavior regresses much more than AveragePolicy;
- PassiveCaller deployed AveragePolicy regression is not reproduced by final current behavior;
- UniformLegal remains unresolved.

The current-behavior first-divergence audit now localizes Jammer:

- total B-A `-8.4304`, resolved;
- FAI contribution `-6.2267`, resolved;
- ROOT contribution `-2.2037`, unresolved;
- FAI explains **73.86%**;
- no Jammer postflop contribution.

## Why target noise is not the primary next target

Earlier work established:
- Jammer FAI target A->B is effectively stationary;
- Stage-B aggregate FAI own-target MSE does not worsen clearly;
- K4 reduces estimator variance but variance is not failure-specific.

Thus further K4 tuning would not address the best-supported mechanism.

## Leading mechanism

Production current behavior is obtained by lean regret matching.

If any legal Advantage output is positive:
- use positive regrets only.

If all are non-positive:
- use softmax fallback.

This makes policy behavior highly sensitive to:
- sign near zero;
- positive-support membership;
- action ranking;
- relative positive-regret scale.

A model may have similar MSE and still produce much worse policy EV.

## Broad Jammer FAI calibration gate

Canonical contract:

`docs/LT2_JAMMER_FAI_BROAD_CALIBRATION_20260919.md`.

Selection:
- forensic seeds `20260920..20260925`;
- 5000 scenarios/seed;
- all common Stage-A/B current-behavior trajectories that reach FAI before earlier divergence;
- record state before FAI action;
- deterministic 8 anchors/seed;
- no conditioning on FAI action divergence or terminal result.

Reference:
- uniform compatible opponent hands;
- 32 hands × 8 boards;
- common canonical Q-like action-gap gauge.

For each Stage A/B:
- exact production sigma;
- stage-specific true Advantage target;
- raw-target MSE;
- action-gap MSE;
- positive support;
- fallback incidence;
- mass on true-negative actions;
- best-action agreement;
- expected policy regret.

## Decision logic

If Stage B has:
- no material MSE deterioration;
- but more sign/support mistakes, fallback, true-negative mass or policy regret;

then target the **RM-sensitive calibration problem** rather than raw MSE.

If broad FAI calibration does not show a Stage-B defect:
- investigate root-to-FAI trajectory weighting / visitation interaction.

No training resumes before this gate is reviewed.

## Immediate direction

1. Keep Stage A/B frozen.
2. Run `bash tools/run_lt2_jammer_fai_broad_calibration.sh`.
3. Wait for `LT2_JAMMER_FAI_BROAD_CALIBRATION_PASS`.
4. Send `SpinCore_LT2_jammer_fai_broad_calibration.json`.
5. Keep holdout seeds `20261001..20261006` untouched.
6. Do not train K4 or resume long training.
