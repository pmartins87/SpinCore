# SpinCore Roadmap — active state 2026-09-20

## Active status

- original Jammer FAI underfit — **CONFIRMED / REPAIRED BY HU400**.
- AveragePolicy at 8000 vs Stage B Jammer — **RESOLVED IMPROVEMENT**.
- current-behavior passive/uniform regression — **LOCALIZED TO PREFLOP ROOT**.
- deterministic root drift 7600 -> 8000 — **CONFIRMED**.
- ALL_IN root mass: `19.17% -> 75.58%`.
- POT_33 root mass: `52.65% -> 6.97%`.
- next causal split — **RESERVOIR EVOLUTION VS FRESH-REFIT REALIZATION**.
- further roots — **PAUSED**.
- holdout — **SEALED**.

## Next

Run three same-seed paired 400-step fresh refits from the 7600 and 8000 HU reservoirs and compare deterministic root policies.

This distinguishes whether online data changed the learned root signal or whether production current-policy checkpoints are intrinsically unstable across reset/refit seeds.
