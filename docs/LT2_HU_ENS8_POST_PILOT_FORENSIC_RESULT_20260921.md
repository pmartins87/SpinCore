# SpinCore — LT2 HU ENS8 post-pilot forensic result

Date: 2026-09-21  
Status: **CURRENT ENS8 ONLINE FEEDBACK PASS — AVERAGEPOLICY SAFE BUT LAGGING — INDEPENDENT LEARNED-ECOSYSTEM CROSSPLAY NEXT**

## Integrity

Read-only design-set audit:

- 13,585 HU scenario clusters;
- 81,510 baseline/seat rows;
- forensic seeds 20260920..20260925;
- no new roots;
- no source-memory writes;
- holdout untouched.

The source ENS8_A at iteration 8000 was reconstructed from the exact frozen reservoir and predeclared member seeds.

## Deterministic root evolution 8000→8100

The ENS8 current policy moved materially:

- mean TV: `0.24275`;
- argmax disagreement: `23.82%`;
- ALL_IN mass: `53.53% → 37.59%`, delta `-15.95 pp`;
- POT_33 mass: `20.50% → 36.48%`, delta `+15.98 pp`;
- FOLD essentially flat: `-0.17 pp`;
- CHECK_CALL essentially flat: `+0.14 pp`.

The de-jamming shift is broad across stack depths and strongest in the high-volume >12bb bucket:

- ALL_IN `44.65% → 26.91%`;
- delta `-17.74 pp`;
- POT_33 delta `+19.15 pp`.

This is substantial policy evolution, not mere numerical jitter.

## Current ENS8 broad EV

### UNIFORM_LEGAL

- ENS8 8000: `+28.097`;
- ENS8 8100: `+32.093`;
- 8100−8000: **`+3.995`, CI95 `[+1.970,+6.021]` — resolved improvement**.

### PASSIVE_CALLER

- ENS8 8000: `+10.961`;
- ENS8 8100: `+12.351`;
- 8100−8000: `+1.390`, CI95 `[-0.090,+2.869]` — unresolved positive/no regression.

### JAMMER

- ENS8 8000: `+15.052`;
- ENS8 8100: `+16.301`;
- 8100−8000: `+1.249`, CI95 `[-0.304,+2.802]` — unresolved positive/no regression.

Versus the prior 7600 current behavior:

- Uniform: `+2.980`, unresolved;
- Passive: **`+8.272`, resolved positive**;
- Jammer: **`+18.698`, resolved positive**.

Therefore the online ENS8 current behavior survives the weak-baseline gate and improves one baseline significantly without a detected tradeoff.

## AveragePolicy 8100

AveragePolicy did not regress, but it did not absorb the current ENS8 improvement in only 100 iterations.

8100−8000:

- Uniform: `+0.086`, CI `[-1.159,+1.331]`;
- Passive: `+0.181`, CI `[-0.819,+1.182]`;
- Jammer: `+0.104`, CI `[-0.887,+1.095]`.

All are essentially flat.

The deployment gap therefore widens because current ENS8 improves faster than the historical AveragePolicy. On Uniform, the chain-gap change is resolved at `-3.909` chips.

This is **deployment lag**, not an AveragePolicy regression.

## Verdict

- online ENS8 current behavior: **PASS**;
- weak-baseline tradeoff check: **PASS**;
- AveragePolicy safety: **PASS / no regression detected**;
- AveragePolicy capture of ENS8 gains: **INCOMPLETE**.

The correct next step is trained-policy crossplay, not more roots yet.

## Independent design-set crossplay

Preregistered seeds:

- 20260926;
- 20260927;
- 20260928;
- 20260929;
- 20260930.

These seed literals had no repository references before preregistration and are outside the sealed holdout 20261001..20261006.

Historical learned-opponent ecosystem:

- AveragePolicy 7600;
- AveragePolicy 8000;
- current behavior 7600;
- ENS8 current 8000.

Primary comparisons under identical opponent assignment/RNG:

- ENS8 8100 minus ENS8 8000;
- AveragePolicy 8100 minus AveragePolicy 8000;
- ENS8 8100 minus AveragePolicy 8100.

Also run seat-balanced direct:

- ENS8 8100 vs ENS8 8000;
- AveragePolicy 8100 vs AveragePolicy 8000;
- ENS8 8100 vs AveragePolicy 8100.

No weak-baseline reuse in the primary ecosystem metric, no new roots, no holdout.

If the current ENS8 remains non-regressing against learned policies, it becomes eligible to be frozen as the HU deployment candidate before holdout.
