# SpinCore — LT2 HU root-policy drift 7600 -> 8000 result

Date: 2026-09-20  
Status: **OPEN-JAM INFLATION CONFIRMED — NEXT SEPARATE RESERVOIR DRIFT FROM FRESH-REFIT SEED VARIANCE**

## Deterministic result

The root-policy audit evaluated every one of the 13,585 forensic HU scenario roots before any hero action was sampled.

Iteration 7600 -> 8000:

- mean root-policy TV: `0.73194`;
- argmax disagreement: `83.26%`;
- ALL_IN probability mass: `+0.56418`;
- POT_33 probability mass: `-0.45680`;
- CHECK_CALL probability mass: `-0.06521`;
- FOLD probability mass: `-0.04216`.

Absolute root mass:

### Iteration 7600
- ALL_IN: `19.17%`;
- POT_33: `52.65%`;
- CHECK_CALL: `17.93%`;
- FOLD: `10.25%`.

### Iteration 8000
- ALL_IN: `75.58%`;
- POT_33: `6.97%`;
- CHECK_CALL: `11.41%`;
- FOLD: `6.04%`.

This confirms that the sampled first-divergence POT_33 -> ALL_IN pattern was not sampling noise.

## Stack localization

The jam inflation is broad rather than a shallow-stack-only effect.

ALL_IN delta:

- <=3bb: `+18.58 pp`;
- >3-5bb: `+31.42 pp`;
- >5-8bb: `+42.29 pp`;
- >8-12bb: `+52.72 pp`;
- >12bb: `+61.26 pp`.

The largest and best-populated bucket is >12bb (8,433 roots), where POT_33 loses `46.18 pp` while ALL_IN gains `61.26 pp`.

## Important interpretation

This proves the checkpoint-to-checkpoint policy drift, but it does not yet prove that the 400 online iterations changed the underlying reservoir signal.

The current Advantage network is reset from a new deterministic initialization every iteration and then freshly fitted. Therefore iteration 7600 and iteration 8000 are also two different fresh-fit realizations.

The original B400 FAI tests proved that 400 steps are robust on the selected/broad Jammer mechanism, not that the entire root policy is stable across fresh refit seeds.

## Next causal split

Perform three paired fresh 400-step refits from:

- iteration-7600 HU Advantage reservoir;
- iteration-8000 HU Advantage reservoir.

Within each replicate use exactly the same initialization seed and exactly the same batch-sampling seed on both reservoirs.

Then evaluate all 13,585 roots deterministically.

Interpretation:

- if all paired 8000-minus-7600 refits reproduce the large ALL_IN shift, reservoir evolution caused the root drift;
- if paired deltas are small/mixed while absolute policies vary substantially across replicate seeds, fresh-refit instability is the cause;
- if both occur, quantify each component before changing training.

No new roots. Holdout remains sealed.
