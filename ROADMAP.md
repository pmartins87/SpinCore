# SpinCore Roadmap — active state 2026-09-20

## Active status

- LT0 — **DONE**.
- LT1 — **DONE**.
- LT2 Stage A — **PASS**.
- LT2 Stage B — **PASS / PRESERVED**.
- Jammer FAI overfold — **CONFIRMED**.
- 100-step HU fit — **INSUFFICIENT**.
- 400-step HU fit — **STRUCTURAL PASS**.
- B400 broad generalization — **PASS**.
- 100-iteration HU400 online pilot — **PASS**.
- current HU behavior after online feedback — **IMPROVED**.
- deployed AveragePolicy after 100 iterations — **LAGGING / UNRESOLVED**.
- policy-memory refresh to iteration 8000 — **NEXT**.
- long continuation beyond 8000 — **NOT AUTHORIZED**.
- holdout — **SEALED**.

## Pilot evidence

Stage B -> iteration 7600 current behavior:

- JAMMER `+4.183`, resolved;
- PASSIVE_CALLER `+1.363`, unresolved;
- UNIFORM_LEGAL `+8.423`, resolved.

Stage B -> iteration 7600 AveragePolicy:

- JAMMER `+0.679`, unresolved;
- PASSIVE_CALLER `+0.004`, unresolved;
- UNIFORM_LEGAL `+0.764`, unresolved.

Thus the online learning loop does not destroy the HU400 repair, but deployment memory has barely refreshed.

## Next bounded block

Source:
- iteration 7600 SHA `c34f19802d3ad5ad3a5d131b083ffcee0e0867667ae0d57324cf35d5fa246b80`.

Continue:
- 400 iterations;
- target 8000;
- +240k roots;
- 3H 100;
- HU 400;
- K4 off.

## Decision at 8000

Current behavior must still show positive Jammer movement without a resolved regression elsewhere.

AveragePolicy should begin to show measurable Jammer movement.

If AveragePolicy is still essentially flat after cumulative 500 HU400 iterations:
- stop;
- audit policy-memory composition, weighting and AveragePolicy fit;
- do not solve deployment lag merely by adding more roots.

## Immediate action

Run `bash tools/run_lt2_hu_b400_refresh_to_8000.sh`.

Stop at iteration 8000.
