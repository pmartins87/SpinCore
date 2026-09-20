# SpinCore Roadmap — active state 2026-09-20

## Active status

- original Jammer FAI underfit — **REPAIRED BY HU400**.
- AveragePolicy at 8000 vs Stage B Jammer — **RESOLVED IMPROVEMENT**.
- mature single-model fresh-fit instability — **CONFIRMED**.
- more single-model optimizer steps through 3200 — **REJECTED**.
- size-2 ensemble — **NOT USEFUL**.
- size-4 ensemble aggregate action-frequency stabilization — **STRONG**.
- size-4 ensemble statewise TV/p95 stabilization — **WEAK / INCOMPLETE**.
- size-4 broad strategic EV gate — **NEXT**.
- further root training — **PAUSED**.
- holdout — **SEALED**.

## Why broad EV is next

The size-4 ensemble reduces ALL_IN/POT_33 frequency dispersion by roughly 77%/69%, but p95 statewise TV remains 1.0.

The unresolved question is whether the remaining statewise disagreements are strategically important or mostly near-indifferent.

Evaluate the two disjoint size-4 ensembles on the complete forensic HU population against the three established baselines before changing production semantics.
