# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED — JAMMER FAI INFOSET OVERFOLD CONFIRMED — ACTION-GAP DRIFT DOMINANT — CONTROLLED REFIT ACTIVE**
Date: 2026-09-19

## Preserved milestones

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Never rewrite either checkpoint.

## Confirmed defect

The pre-registered high-impact Jammer FAI structure is:

- legal `{FOLD,CALL}`;
- one public action before FAI.

For B_MORE_FOLD:

- policy-value B-A `-24.54921`;
- CI95 `[-35.58425,-13.51416]`;
- fold-mass B-A `+0.46907`;
- reference FOLD-minus-CONTINUE `-40.86355` chips;
- canonical action-gap MSE worsens significantly;
- class-error mass worsens significantly.

This is a real infoset-level Stage-B overfold against the hand-independent Jammer reference.

## Raw-margin causal decomposition

Using the exact production lean regret-matching map:

- FULL_B `-24.549`;
- GAP_ONLY `-18.719`, resolved;
- OFFSET_ONLY `-4.946`, resolved;
- nonlinear interaction unresolved.

Therefore action-gap drift is the larger causal component.

A diagnostic argmax fallback recovers `+15.415` versus production B but does not prove B=A.

Do not patch fallback alone.

## Why root training remains paused

We still do not know whether the Stage-B action-gap degradation comes from:

1. the Stage-B Advantage reservoir / target signal;
2. the final reset/refit and optimizer budget;
3. reset/fit seed instability.

Blindly adding roots can worsen a data-signal problem and wastes compute if the real issue is the fit stage.

## Active experiment

Controlled fresh-refit audit:

- source checkpoints read-only;
- no roots;
- fixed 384-anchor cohort;
- stored low-noise q reference reused;
- Stage-A and Stage-B HU reservoirs;
- paired init and batch seeds;
- budgets 100, 400, 1600;
- 3 replicates.

### If Stage-B reservoir stays worse at 1600

Next:
- inspect conditional target composition, sampling frequency, iteration weights and hidden-chance noise specifically for the FAI structural class;
- design a target/reservoir intervention.

### If Stage-B catches Stage A as budget rises

Next:
- modify fit schedule only;
- validate on forensic cohort;
- freeze candidate before holdout.

### If replicate variance dominates

Next:
- stabilize the reset/refit procedure;
- consider ensemble/seed selection only if it can be specified without holdout leakage.

## Holdout

Seeds `20261001..20261006` remain sealed until an intervention is frozen.

## Immediate direction

1. Keep Stage A/B frozen.
2. Pull `main`.
3. Run `bash tools/run_lt2_jammer_fai_controlled_refit.sh`.
4. Send `SpinCore_LT2_jammer_fai_controlled_refit.json`.
5. Do not resume root training.
