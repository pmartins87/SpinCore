# SpinCore — LT2 HU refit-budget root-stability result

Date: 2026-09-20  
Status: **MORE OPTIMIZER STEPS DO NOT SOLVE SAME-MEMORY POLICY INSTABILITY — ENSEMBLE BRANCH NEXT**

## Integrity

Frozen source:

- iteration 8000;
- HU Advantage reservoir: 2,000,000 stored items / 66,394,758 seen;
- 4 independent fit trajectories;
- cumulative budgets 400, 800, 1600, 3200;
- same root corpus: 13,585 forensic HU roots;
- no new training roots;
- no source-memory writes;
- holdout untouched.

## Root-policy stability by cumulative fit budget

### 400 steps

- pairwise mean TV: `0.5950`;
- pairwise p95 TV: `1.0000`;
- argmax disagreement: `63.58%`;
- ALL_IN mass range across replicas: `10.80 pp`;
- POT_33 mass range: `29.14 pp`.

### 800 steps

- pairwise mean TV: `0.5395`;
- pairwise p95 TV: `0.9891`;
- argmax disagreement: `59.57%`;
- ALL_IN range: `44.10 pp`.

This is not a stable improvement.

### 1600 steps

- pairwise mean TV: `0.4278`;
- pairwise p95 TV: `0.9700`;
- argmax disagreement: `46.75%`;
- ALL_IN range: `16.67 pp`;
- POT_33 range: `30.39 pp`.

### 3200 steps

- pairwise mean TV: `0.4090`;
- pairwise p95 TV: `0.9640`;
- argmax disagreement: `47.71%`;
- ALL_IN range: `20.63 pp`;
- POT_33 range: `26.07 pp`.

Relative to 400, 3200 steps reduce mean TV by about 31%, but the p95 remains essentially saturated near 1 and argmax disagreement remains near 48%.

## Non-monotonic trajectory evidence

Individual models continue to move substantially even while training loss remains around the same narrow band.

For example, one replica's root ALL_IN mass evolves:

- 400: `23.11%`;
- 800: `61.08%`;
- 1600: `35.64%`;
- 3200: `12.85%`.

This is not convergence to one stable policy geometry.

## Verdict

The mature-reservoir instability is not fixed by simply buying more optimizer steps.

400 was sufficient to repair the selected Jammer FAI underfit, but globally the fitted Advantage policy remains highly seed/minibatch sensitive.

Escalating beyond 3200 is not justified by these results.

The next branch is variance reduction across independent Advantage fits.

## Next gate

On the frozen iteration-8000 HU reservoir:

- train 8 independent fresh 400-step Advantage replicas;
- evaluate every model on the same 13,585 HU root corpus;
- average raw Advantage outputs before the unchanged lean regret-matching map;
- compare disjoint ensemble sizes 1, 2 and 4.

Primary stability metrics:

- pairwise mean TV;
- pairwise p95 TV;
- argmax disagreement;
- ALL_IN and POT_33 mass dispersion.

This directly tests whether model averaging removes the fit-realization noise at the mature reservoir.

No production change, no roots, no holdout.
