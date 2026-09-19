# SpinCore Current Work

Date: 2026-09-19
Status: **LT2 STAGE B PASS — JAMMER FAI INFOSET OVERFOLD CONFIRMED — 100-STEP ADVANTAGE REFIT IDENTIFIED AS INSUFFICIENT — B400 BROAD GENERALIZATION NEXT — NO ROOT TRAINING**

## Active source of truth

Read before new compute:

- `docs/LT2_JAMMER_FAI_STRUCTURAL_INFOSET_RESULT_20260919.md`
- `docs/LT2_JAMMER_FAI_RAW_MARGIN_DECOMPOSITION_RESULT_20260919.md`
- `docs/LT2_JAMMER_FAI_CONTROLLED_REFIT_RESULT_20260919.md`
- `docs/LT2_HU_B400_BROAD_GENERALIZATION_20260919.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500.

## Preserved checkpoints

Stage A:
- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Controlled refit result — decisive

The fixed structural cohort reproduces the production B_MORE_FOLD defect.

Fresh paired refits show:

### 100 steps
Stage-B reservoir minus Stage-A reservoir:

- `-11.64417` chips;
- replicate-mean CI95 `[-18.26748,-5.02086]`;
- all 3 replicates negative.

### 400 steps
- `+13.91919`;
- CI95 `[+2.96579,+24.87259]`;
- all 3 replicates positive.

### 1600 steps
- `+7.29093`;
- CI95 `[+3.38533,+11.19653]`;
- all 3 replicates positive.

## Interpretation

The Stage-B 2M-item Advantage reservoir is not poisoned.

The canonical 100-step reset/refit is insufficient at this mature training stage.

This is not just one unlucky final reset: three fresh 100-step trials remain negative.

400 is the minimum tested sufficient budget and therefore the candidate intervention.

1600 remains a fallback escalation because it is 16x canonical optimizer work.

## Active gate — broad B400 generalization

Recreate all three deterministic B400 candidates and evaluate them on the full forensic HU population against:

- UNIFORM_LEGAL;
- PASSIVE_CALLER;
- JAMMER.

Pair against production Stage A and Stage B using common scenario/deal/seat/RNG streams.

Primary question:

does B400 repair the global Jammer current-behavior regression without causing a resolved material regression against the other baselines?

No holdout and no roots.

## Immediate user action

Pull `main` and run:

```bash
bash tools/run_lt2_hu_b400_broad_generalization.sh
```

Wait for `LT2_HU_B400_BROAD_GENERALIZATION_PASS` or the first error.

Then send `SpinCore_LT2_hu_b400_broad_generalization.json`.

Do not resume long training.
