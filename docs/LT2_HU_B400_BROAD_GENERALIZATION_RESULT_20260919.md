# SpinCore — LT2 HU B400 broad generalization result

Date: 2026-09-19  
Status: **PASS — B400 REPAIRS GLOBAL HU JAMMER BEHAVIOR AND DOES NOT CREATE A WEAK-BASELINE REGRESSION**

## Integrity

Schema: `SPINCORE_LT2_HU_B400_BROAD_GENERALIZATION_V1`.

The gate used all HU scenarios from forensic seeds `20260920..20260925`:

- 13,585 HU scenario clusters;
- 81,510 paired seat/baseline rows;
- identical scenario, deal, hero seat and RNG streams across policies;
- all three deterministic B400 refits included;
- no replicate selected by outcome;
- no new roots;
- source checkpoints read-only;
- holdout untouched.

## Production reproduction

Against JAMMER:

- PROD_A: `+1.8501` chips/hand;
- PROD_B: `-6.5803`;
- PROD_B minus A: `-8.4304`, CI95 `[-12.5155,-4.3453]`.

This exactly reproduces the previously established Stage-B current-behavior regression.

## JAMMER — primary gate

B400_R0:
- absolute `+10.2463`;
- vs PROD_B `+16.8266`, CI95 `[+12.8183,+20.8348]`;
- vs PROD_A `+8.3962`, CI95 `[+4.2943,+12.4980]`.

B400_R1:
- absolute `+4.3052`;
- vs PROD_B `+10.8855`, CI95 `[+7.1343,+14.6368]`;
- vs PROD_A `+2.4551`, unresolved.

B400_R2:
- absolute `+3.9313`;
- vs PROD_B `+10.5116`, CI95 `[+6.4837,+14.5395]`;
- vs PROD_A `+2.0812`, unresolved.

All three candidates:

- are positive against JAMMER;
- improve significantly versus production Stage B;
- none is worse than production Stage A;
- R0 is significantly better than Stage A.

## PASSIVE_CALLER

All three B400 candidates improve significantly versus production Stage B:

- R0: `+5.8837`, CI95 `[+2.2294,+9.5380]`;
- R1: `+3.2061`, CI95 `[+0.2792,+6.1331]`;
- R2: `+5.5755`, CI95 `[+2.3908,+8.7601]`.

Thus the Jammer repair does not trade off against this weak baseline.

## UNIFORM_LEGAL

Versus production Stage B:

- R0: `+0.5386`, unresolved;
- R1: `+5.1369`, CI95 `[+1.2628,+9.0110]`;
- R2: `+2.7867`, unresolved.

No B400 replicate shows a resolved regression.

## Verdict

The broad gate passes.

The 400-step HU Advantage refit is not merely overfitting the selected FAI anchors. It generalizes across the natural forensic HU population and across all three weak-opponent families.

The intervention is therefore frozen as:

- THREE_HANDED Advantage fit: keep canonical `100` steps;
- TRUE_HEADS_UP Advantage fit: `400` steps.

We intentionally do **not** raise 3H to 400 because the causal evidence and broad validation are HU-specific. This keeps the change minimal and avoids spending compute or changing an unimplicated domain.

## Next gate

Run a bounded online continuation from the preserved Stage-B checkpoint:

- iterations 7501..7600;
- +100 iterations;
- +60,000 roots;
- 3H fit remains 100 steps;
- HU fit uses 400 steps;
- K4 remains off;
- source Stage-B checkpoint remains read-only.

After completion, compare Stage B versus pilot on the full forensic HU policy chain, including both:

- current Advantage behavior;
- deployed AveragePolicy.

If current HU behavior remains repaired, the intervention survives online feedback. AveragePolicy movement is secondary at this short horizon because its reservoir still contains historical policy samples from the old 100-step behavior.
