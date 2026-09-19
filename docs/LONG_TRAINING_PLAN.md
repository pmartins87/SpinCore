# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — JAMMER FAI LOSS RESOLVED IN POPULATION — FIRST INFOSET FOLD-SHIFT AUDIT UNDERPOWERED — POWERED STRUCTURAL CONFIRMATION ACTIVE**
Date: 2026-09-19

## Preserved milestones

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Never rewrite either checkpoint.

## What is already resolved

The natural Jammer population established a genuine FAI current-policy loss:

- expected FAI B-A `-4.18434` chips/hand;
- CI95 `[-6.58924,-1.77944]`.

Stage-B increasing FOLD probability carries the loss:

- B_MORE_FOLD `-5.57116`, resolved;
- B_LESS_FOLD `+1.38682`, resolved beneficial.

The dominant structural blocks were already identified as:

- legal `{FOLD,CALL}`;
- one public action before FAI.

## First low-noise infoset fold-shift test

The valid 96-anchor test used 512 explicit reference deals per anchor.

B_MORE_FOLD:

- policy-value B-A `-5.92295`;
- seed-cluster CI95 `[-19.60209,+7.75619]`.

B_LESS_FOLD:

- `+7.47850`;
- seed-cluster CI95 `[-7.99668,+22.95369]`.

Both signs align with the full-population mechanism, but neither resolves.

This is an underpowered result, not evidence against the mechanism.

## Why we still do not train

The decision-time causal chain is not yet sufficiently isolated.

A training modification now would risk fitting a noisy symptom.

No K4 training, RM-loss change, fallback change, or long-root continuation is authorized.

## Powered structural infoset confirmation

Canonical contract:

`docs/LT2_JAMMER_FAI_STRUCTURAL_INFOSET_CONFIRMATION_20260919.md`.

Frozen inclusion:

- common Jammer FAI;
- legal slots exactly `0,1`;
- one public action before FAI.

These criteria come from the upstream full-population localization, not the first low-noise result.

Within each forensic seed:

- sample 32 B_MORE_FOLD;
- sample 32 B_LESS_FOLD.

Total:

- 384 anchors;
- 196,608 explicit reference deals at 64 hands × 8 boards.

Primary metric:

`policy_value_B_minus_A` for B_MORE_FOLD with seed-cluster CI.

## Decision logic

If B_MORE_FOLD resolves negative:
- confirm a real infoset-level overfold defect in the frozen high-impact structure;
- inspect raw fold/continue Advantage margins, fallback incidence, and estimator noise there;
- design the smallest intervention;
- freeze that intervention before opening any holdout seed.

If B_MORE_FOLD remains unresolved:
- do not increase sample size again by reflex;
- decompose residual variance into state heterogeneity, reference noise, and seed composition.

If B_MORE_FOLD resolves positive:
- reject a direct fold-calibration intervention for this structure;
- investigate evaluation weighting / hidden-chance covariance.

## Immediate direction

1. Keep Stage A/B frozen.
2. Run `bash tools/run_lt2_jammer_fai_structural_infoset_confirmation.sh`.
3. Wait for `LT2_JAMMER_FAI_STRUCTURAL_INFOSET_CONFIRMATION_PASS`.
4. Send `SpinCore_LT2_jammer_fai_structural_infoset_confirmation.json`.
5. Keep holdout `20261001..20261006` untouched.
6. Do not train.
