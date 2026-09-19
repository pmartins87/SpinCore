# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — INSUFFICIENT 100-STEP ADVANTAGE REFIT IDENTIFIED — B400 BROAD GENERALIZATION ACTIVE**
Date: 2026-09-19

## Preserved milestones

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Never rewrite either checkpoint.

## Causal chain now resolved to the fit stage

Previously established:

- Jammer current behavior regresses at Stage B;
- 73.86% localizes to preflop facing-all-in;
- Stage B overfolds;
- low-noise infoset reference confirms that overfold is wrong;
- fold-vs-continue Advantage action-gap drift is the dominant component;
- fallback is an amplifier but a fallback-only patch is insufficient.

Controlled fresh refits now show:

- 100 steps: Stage-B reservoir remains worse;
- 400 steps: Stage-B reservoir is better in all three paired replicates;
- 1600 steps: Stage-B reservoir remains better in all three paired replicates.

Therefore the reservoir contains usable signal and the canonical 100-step fit is the leading failure mechanism.

## Why root training is still paused

The fixed structural cohort is selected around the known Jammer FAI failure.

Before changing the training schedule globally, the 400-step candidate must generalize to the natural HU forensic population.

## Candidate

`advantage_steps=400`.

Reason:

- smallest tested budget that clears the primary cohort in all three replicates;
- 4x canonical optimizer work;
- materially cheaper than 1600.

1600 is reserved for one escalation branch if 400 fails global Jammer generalization.

## Active broad gate

Recreate all three deterministic 400-step Stage-B candidates.

Evaluate on all forensic HU scenarios against:

- UNIFORM_LEGAL;
- PASSIVE_CALLER;
- JAMMER.

Use common random numbers and compare with:

- production Stage A;
- production Stage B.

No roots are generated.

## Decision logic

### B400 improves Jammer in all three replicates and does not create a resolved material regression elsewhere

Freeze 400 as the training-side intervention.

Next:
- create a bounded continuation pilot from Stage B;
- use a small, predefined number of additional iterations;
- evaluate before any long continuation.

### B400 still fails broad Jammer

Run one broad 1600-step escalation.

### B400 introduces a resolved tradeoff

Do not train.
Investigate whether the stronger fit is moving toward a different weakness rather than genuine overall improvement.

## Holdout

Seeds `20261001..20261006` remain sealed until the intervention is frozen and a bounded pilot is ready for final validation.

## Immediate direction

1. Keep Stage A/B frozen.
2. Pull `main`.
3. Run `bash tools/run_lt2_hu_b400_broad_generalization.sh`.
4. Send `SpinCore_LT2_hu_b400_broad_generalization.json`.
5. Do not resume root training.
