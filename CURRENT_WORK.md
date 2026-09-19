# SpinCore Current Work

Date: 2026-09-19
Status: **LT2 STAGE B PASS — JAMMER FAI LOSS CONFIRMED IN FULL-POPULATION EXPECTED VALUE — STAGE-B OVERFOLDING IS THE HARMFUL COMPONENT — LOW-NOISE INFOSET FOLD-SHIFT GATE NEXT — NO TRAINING**

## Active source of truth

Read before new compute:

- `docs/LT2_JAMMER_FAI_POPULATION_RECONCILIATION_RESULT_20260919.md`
- `docs/LT2_JAMMER_FAI_FOLD_SHIFT_INFOSET_20260919.md`
- `docs/LT2_JAMMER_FAI_BROAD_CALIBRATION_RESULT_20260919.md`
- `docs/LT2_HU_BEHAVIOR_FIRST_DIVERGENCE_RESULT_20260919.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Full-population FAI reconciliation — PASS

Integrity:
- 27,170 HU Jammer seat-runs;
- 13,585 scenario clusters;
- 16,279 common FAI states;
- forensic seeds only;
- holdout untouched;
- read-only.

### Hard reproduction

The paired sampled FAI contribution reproduced the prior result exactly:

`-6.22672064777328` chips/hand.

CI95:
`[-9.25754,-3.19590]`.

Difference from prior:
`0.0`.

### Deterministic expected FAI contribution

Using exact dealt hidden hand/full board and every legal terminal action:

`-4.18434` chips/hand.

CI95:
`[-6.58924,-1.77944]`.

Resolved negative.

Therefore the FAI regression is genuine in expected policy value, not an action-sampling artifact.

Conditional on reaching common FAI:

`-6.9704` chips, seed-cluster CI95 `[-11.2658,-2.6751]`.

## Stage-B overfold localization

Conditional mean fold mass:

- Stage A `0.24964`;
- Stage B `0.37200`;
- B-A `+0.12237`;
- seed-cluster CI95 `[+0.11779,+0.12695]`.

Stage B folds about 12.24 pp more.

### B_MORE_FOLD

- 7,374 rows;
- contribution `-5.57116`;
- CI95 `[-7.61160,-3.53071]`.

Resolved harmful.

### B_LESS_FOLD

- 3,287 rows;
- contribution `+1.38682`;
- CI95 `[+0.21539,+2.55825]`.

Resolved beneficial.

### NO_FOLD_SHIFT

- 5,618 rows;
- contribution numerically zero.

The harmful FAI mechanism is therefore direction-specific: **Stage B increasing FOLD probability**.

## Structural concentration

Legal `{FOLD,CALL}`:
- 7,892 rows;
- contribution `-4.07764`;
- CI95 `[-5.88461,-2.27067]`.

Legal `{FOLD,CALL,ALL_IN}`:
- 8,356 rows;
- contribution `-0.11128`;
- CI crosses zero.

One public action before FAI:
- contribution `-3.52854`;
- CI95 `[-5.83127,-1.22581]`.

Two public actions:
- contribution `-0.65580`;
- unresolved.

Maximum non-FOLD actual-Q spread is exactly zero.

CALL and ALL_IN are outcome-equivalent after the Jammer is already all-in. The meaningful decision class is FOLD versus CONTINUE.

## Why the 48-anchor broad audit looked benign

The 48-anchor broad audit was too small / heterogeneous to expose the full-population directional effect reliably.

It did not invalidate the FAI hypothesis; the full-population test now resolves the loss and identifies the fold direction carrying it.

Do not design a training intervention from actual-deal Q alone, because actual-deal Q uses hidden cards/future board.

## Active gate

Run:

```bash
bash tools/run_lt2_jammer_fai_fold_shift_infoset.sh
```

Pre-reference selection:
- common FAI states only;
- classify only by Stage-B-minus-A production fold-mass sign;
- 8 B_MORE_FOLD + 8 B_LESS_FOLD anchors per forensic seed;
- 96 total;
- no sampled FAI action/outcome/actual-Q/low-noise-Q selection.

Low-noise reference:
- 64 uniform compatible opponent hands;
- 8 future boards/hand;
- 512 deals/anchor.

Primary question:

Does the resolved Stage-B overfold remain harmful under **infoset expectation**?

Holdout `20261001..20261006` remains untouched.

DeepCrusher remains deferred.

## Immediate user action

Pull `main` and run:

```bash
bash tools/run_lt2_jammer_fai_fold_shift_infoset.sh
```

Wait for `LT2_JAMMER_FAI_FOLD_SHIFT_INFOSET_PASS` or the first error.

Then send `SpinCore_LT2_jammer_fai_fold_shift_infoset.json`.

Do not start any training.
