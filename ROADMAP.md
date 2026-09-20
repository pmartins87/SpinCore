# SpinCore Roadmap — active state 2026-09-20

## Active status

- original Jammer FAI underfit — **REPAIRED BY HU400**.
- AveragePolicy at 8000 vs Stage B Jammer — **RESOLVED IMPROVEMENT**.
- root open-jam inflation — **CONFIRMED**.
- reservoir evolution contribution — **REAL BUT MODEST RELATIVE TO PRODUCTION EXTREME**.
- fresh-fit instability — **CONFIRMED**.
- optimizer-budget escalation through 3200 — **INSUFFICIENT / PLATEAUED**.
- mature-reservoir 1/2/4 Advantage ensemble stability gate — **NEXT**.
- further root training — **PAUSED**.
- holdout — **SEALED**.

## Why optimizer escalation stops

At 3200 steps:
- pairwise mean TV still `0.409`;
- p95 TV still `0.964`;
- argmax disagreement still `47.7%`.

The fit does not converge to one stable policy geometry.

## Next branch

Use independent 400-step fits as estimators and average raw Advantage outputs before the unchanged regret-matching map.

Test disjoint ensemble sizes 1, 2 and 4 on the full forensic HU root corpus.

Only if ensemble stability is materially better will it proceed to broad EV/generalization.
