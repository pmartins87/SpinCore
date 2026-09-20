# SpinCore — LT2 mature HU ensemble root-stability result

Date: 2026-09-20  
Status: **PARTIAL STABILIZATION ONLY — SIZE-4 SHRINKS AGGREGATE ACTION-MASS DISPERSION BUT NOT STATEWISE HIGH-TAIL DISAGREEMENT**

## Integrity

Frozen source:

- iteration 8000 HU Advantage reservoir;
- 2,000,000 stored items / 66,394,758 seen;
- 8 independent fresh 400-step fits;
- same 13,585 forensic HU roots;
- disjoint ensembles of size 1, 2 and 4;
- raw Advantage outputs averaged before unchanged lean regret matching;
- no CFR roots;
- no source-memory writes;
- holdout untouched.

## Single-model instability

Across the 8 single models:

- pairwise mean TV: `0.56430`;
- pairwise p95 TV: `1.00000`;
- argmax disagreement: `61.01%`;
- ALL_IN mass range: `42.97 pp`;
- POT_33 mass range: `40.39 pp`.

Individual ALL_IN mass ranges from `15.87%` to `58.84%`.

## Size-2 ensemble

This is not a useful stabilization:

- mean TV ratio vs single: `0.900`;
- p95 ratio: `1.000`;
- argmax ratio: `0.874`;
- ALL_IN range ratio: `1.105`;
- POT_33 range ratio: `1.120`.

Aggregate action-mass dispersion actually worsens.

## Size-4 ensemble

Size 4 produces a mixed result.

Statewise disagreement:

- mean TV: `0.48142` vs `0.56430` single — about 14.7% lower;
- p95 TV: `1.00000` — no improvement;
- argmax disagreement: `47.18%` vs `61.01%` — about 22.7% lower.

Aggregate action-mass dispersion improves strongly:

- ALL_IN range: `10.05 pp` vs `42.97 pp` — about 76.6% lower;
- POT_33 range: `12.69 pp` vs `40.39 pp` — about 68.6% lower.

The two disjoint size-4 ensembles have:

- ALL_IN: `41.57%` vs `51.62%`;
- POT_33: `30.02%` vs `17.33%`.

## Interpretation

A four-model raw-Advantage ensemble meaningfully stabilizes global action frequencies, but it does not stabilize decisions state-by-state.

This means the ensemble cannot yet be accepted or rejected solely from root-policy TV.

The remaining high-value question is strategic:

**do the two independently composed size-4 ensembles produce similar chip-EV and avoid the Passive/Uniform regressions that motivated this diagnosis?**

If yes, the residual statewise disagreement may be concentrated among strategically near-equivalent decisions.

If no, the ensemble has only cosmetic frequency stabilization and should be rejected.

## Next gate

Recreate the exact same eight deterministic 400-step models from the frozen iteration-8000 reservoir and evaluate the two disjoint size-4 ensembles throughout complete HU hands against:

- UNIFORM_LEGAL;
- PASSIVE_CALLER;
- JAMMER.

Use the same forensic population and common random numbers.

Context policies:

- production Stage B 7500 current behavior;
- production pilot 7600 current behavior;
- production 8000 current behavior.

No training roots and no holdout.
