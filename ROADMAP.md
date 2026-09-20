# SpinCore Roadmap — active state 2026-09-20

## Active status

- LT0 — **DONE**.
- LT1 — **DONE**.
- LT2 Stage B — **PASS / PRESERVED**.
- HU400 original FAI repair — **SUPPORTED ACROSS STRUCTURAL, BROAD AND ONLINE GATES**.
- AveragePolicy at 8000 vs Stage B JAMMER — **RESOLVED IMPROVEMENT**.
- 7600 -> 8000 current-behavior PASSIVE loss — **RESOLVED**.
- 7600 -> 8000 current-behavior UNIFORM loss — **RESOLVED**.
- first-divergence localization — **PREFLOP_ROOT DOMINANT**.
- apparent root action shift — **POT_33 -> ALL_IN DOMINANT IN SAMPLED DIVERGENCES**.
- deterministic root probability audit — **NEXT**.
- further training — **PAUSED**.
- holdout — **SEALED**.

## First-divergence evidence

PASSIVE_CALLER root contribution:
- `-3.056`, CI95 `[-5.251,-0.861]`.

UNIFORM_LEGAL root contribution:
- `-7.469`, CI95 `[-10.548,-4.389]`.

Later streets are small/unresolved.

JAMMER retains directionally positive FAI contribution while root is mildly negative/unresolved.

## Interpretation

The 400-step fit is not simply globally bad.

It fixed the originally identified FAI underfit.

During further online training, however, the current Advantage policy drifts at the initial HU preflop root. Sampled transitions strongly indicate excess movement from the 2-BB open (POT_33) toward ALL_IN.

## Next gate

Deterministically compare root policy distributions at 7600 and 8000 over every forensic HU scenario.

Do not modify training until root mass shift is quantified.
