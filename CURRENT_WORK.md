# SpinCore Current Work

Date: 2026-09-19
Status: **LT2 STAGE B PASS — JAMMER FAI INFOSET OVERFOLD CONFIRMED — ACTION-GAP DRIFT DOMINANT — CONTROLLED REFIT AUDIT NEXT — NO ROOT TRAINING**

## Active source of truth

Read before new compute:

- `docs/LT2_JAMMER_FAI_STRUCTURAL_INFOSET_RESULT_20260919.md`
- `docs/LT2_JAMMER_FAI_RAW_MARGIN_DECOMPOSITION_RESULT_20260919.md`
- `docs/LT2_JAMMER_FAI_CONTROLLED_REFIT_20260919.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Structural overfold is confirmed

In the frozen Jammer FAI structure legal `{FOLD,CALL}`, one public action:

B_MORE_FOLD policy-value B-A:
- `-24.54921` chips;
- CI95 `[-35.58425,-13.51416]`.

Fold-mass B-A:
- `+0.46907`.

Reference FOLD-minus-CONTINUE:
- `-40.86355` chips.

Canonical action-gap MSE and class-error mass both worsen significantly at Stage B.

## Raw-margin decomposition is complete

B_MORE_FOLD:

- FULL_B: `-24.549` `[-35.584,-13.514]`;
- GAP_ONLY: `-18.719` `[-27.359,-10.080]`;
- OFFSET_ONLY: `-4.946` `[-9.166,-0.726]`;
- interaction: `-0.884`, unresolved.

Thus the fold-vs-continue action-gap drift is the larger causal component.

A diagnostic-only argmax replacement for the all-nonpositive fallback recovers:

- `+15.415` chips vs production Stage B;
- CI95 `[+8.623,+22.207]`.

But modified B-vs-A remains `-9.134`, CI crosses zero. Fallback-only patch is not authorized.

## Active gate — controlled fresh refits

Compare the Stage-A and Stage-B HU Advantage reservoirs under:

- identical model init seeds;
- identical batch-sampling seeds;
- budgets 100, 400, 1600;
- 3 replicates;
- same fixed 384-anchor low-noise cohort.

No roots are collected. Source checkpoints remain read-only. Holdout remains sealed.

Goal:

distinguish **reservoir / target-signal drift** from **last-fit instability / insufficient fit**.

## Immediate user action

Pull `main` and run:

```bash
bash tools/run_lt2_jammer_fai_controlled_refit.sh
```

Wait for `LT2_JAMMER_FAI_CONTROLLED_REFIT_PASS` or the first error.

Then send `SpinCore_LT2_jammer_fai_controlled_refit.json`.

Do not start K4 or long training.
