# SpinCore Current Work

Date: 2026-09-20
Status: **ITERATION 8000 FROZEN — DETERMINISTIC OPEN-JAM INFLATION CONFIRMED — PAIRED FRESH-REFIT CAUSAL SPLIT NEXT**

## Root drift confirmation

Across all 13,585 forensic HU roots, iteration 7600 -> 8000:

- TV = `0.73194`;
- argmax disagreement = `83.26%`;
- ALL_IN mass `+56.42 pp`;
- POT_33 mass `-45.68 pp`.

Absolute ALL_IN mass:
- 7600: `19.17%`;
- 8000: `75.58%`.

Absolute POT_33 mass:
- 7600: `52.65%`;
- 8000: `6.97%`.

The shift is broad and strongest in the well-populated >12bb bucket:
- ALL_IN `+61.26 pp`;
- POT_33 `-46.18 pp`.

## What this does NOT yet establish

Each iteration resets the current Advantage network from a different deterministic initialization and freshly fits it.

Therefore a checkpoint-to-checkpoint current-policy difference mixes:
1. reservoir evolution;
2. fresh-fit initialization / batch-sampling realization.

## Active gate

Three paired fresh 400-step refits from the 7600 and 8000 HU Advantage reservoirs.

Within each replicate:
- identical init seed across reservoirs;
- identical batch-sampling seed across reservoirs.

Then evaluate all forensic HU roots deterministically.

No new roots. Holdout sealed.

## Immediate action

```bash
bash tools/run_lt2_hu_paired_fresh400_root_refit.sh
```

Send `SpinCore_LT2_hu_paired_fresh400_root_refit.json`.

Do not train beyond 8000.
