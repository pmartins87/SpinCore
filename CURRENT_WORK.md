# SpinCore Current Work

Date: 2026-09-19
Status: **LT2 STAGE B PASS — FULL-POPULATION JAMMER FAI LOSS RESOLVED — FIRST LOW-NOISE FOLD-SHIFT INFOSET AUDIT DIRECTIONALLY ALIGNED BUT UNDERPOWERED — POWERED STRUCTURAL CONFIRMATION NEXT — NO TRAINING**

## Active source of truth

Read before new compute:

- `docs/LT2_JAMMER_FAI_FOLD_SHIFT_INFOSET_RESULT_20260919.md`
- `docs/LT2_JAMMER_FAI_STRUCTURAL_INFOSET_CONFIRMATION_20260919.md`
- `docs/LT2_JAMMER_FAI_POPULATION_RECONCILIATION_RESULT_20260919.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Full-population FAI result remains canonical

The full natural Jammer population established:

- deterministic expected FAI B-A `-4.18434` chips/hand;
- CI95 `[-6.58924,-1.77944]`;
- Stage-B fold mass +12.24 pp;
- B_MORE_FOLD contribution `-5.57116`, resolved;
- B_LESS_FOLD contribution `+1.38682`, resolved.

The FAI regression is real in evaluation value.

## First low-noise fold-shift infoset audit — valid but inconclusive

Integrity:

- 96 anchors;
- 48 B_MORE_FOLD;
- 48 B_LESS_FOLD;
- 64 hands × 8 boards = 512 deals/anchor;
- read-only;
- holdout untouched;
- maximum Stage-A/B canonical gap discrepancy `2.98e-08`.

Candidate pools exactly match the full-population decomposition:

- B_MORE_FOLD: 7,374;
- B_LESS_FOLD: 3,287;
- NO_FOLD_SHIFT: 5,618.

### B_MORE_FOLD

Infoset policy-value B-A:

- mean `-5.92295` chips;
- seed-cluster CI95 `[-19.60209,+7.75619]`.

Direction matches the full-population harmful effect but is unresolved.

### B_LESS_FOLD

Infoset policy-value B-A:

- mean `+7.47850`;
- seed-cluster CI95 `[-7.99668,+22.95369]`.

Direction matches the full-population beneficial effect but is unresolved.

Therefore the correct verdict is **underpowered / inconclusive**, not confirmation and not falsification.

## Structural focus was already known upstream

Before the 96-anchor infoset audit, the full-population reconciliation had already localized most resolved loss to:

- legal actions exactly `{FOLD,CALL}`;
- one public action before FAI.

A secondary decomposition of the current 96 anchors using those pre-existing criteria gives a negative B_MORE_FOLD signal, but it is not promoted as the final gate because it was not the primary aggregate of the 96-anchor audit.

## Active gate

Run the powered structural confirmation:

```bash
bash tools/run_lt2_jammer_fai_structural_infoset_confirmation.sh
```

Frozen structural inclusion:

- common Jammer FAI;
- legal slots exactly `0,1`;
- exactly one public action before FAI.

Within that structure:

- 32 B_MORE_FOLD anchors per seed;
- 32 B_LESS_FOLD anchors per seed;
- 384 anchors total;
- same 64 hands × 8 boards reference;
- 196,608 explicit reference deals.

Primary confirmatory metric:

`B_MORE_FOLD policy_value_B_minus_A`

with seed-cluster CI.

The selection still does not use sampled FAI action, terminal outcome, actual-deal Q, low-noise Q, or optimal class.

Holdout `20261001..20261006` remains untouched.

DeepCrusher remains deferred.

## Immediate user action

Pull `main` and run:

```bash
bash tools/run_lt2_jammer_fai_structural_infoset_confirmation.sh
```

Wait for `LT2_JAMMER_FAI_STRUCTURAL_INFOSET_CONFIRMATION_PASS` or the first error.

Then send `SpinCore_LT2_jammer_fai_structural_infoset_confirmation.json`.

Do not start any training.
