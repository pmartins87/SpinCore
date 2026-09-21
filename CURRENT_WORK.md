# SpinCore Current Work

Date: 2026-09-21
Status: **ENS8 CURRENT ONLINE FEEDBACK PASS AT 8100 — AVERAGEPOLICY SAFE BUT LAGGING — INDEPENDENT LEARNED-POLICY CROSSPLAY NEXT**

## Post-pilot forensic

Current ENS8 8000→8100:

- root mean TV `0.24275`;
- argmax disagreement `23.82%`;
- ALL_IN `53.53% → 37.59%`;
- POT_33 `20.50% → 36.48%`.

Broad EV 8100−8000:
- Uniform **+3.995**, CI [+1.970,+6.021], resolved improvement;
- Passive +1.390, unresolved/no regression;
- Jammer +1.249, unresolved/no regression.

Versus current behavior 7600:
- Passive **+8.272**, resolved;
- Jammer **+18.698**, resolved;
- Uniform +2.980, unresolved.

Current online ENS8 therefore passes the weak-baseline tradeoff gate.

## AveragePolicy

AveragePolicy 8100−8000 is essentially flat on all three baselines:
- Uniform +0.086;
- Passive +0.181;
- Jammer +0.104.

No regression is detected, but the current-policy gain has not transferred to deployment after only 100 iterations.

## Active gate

Use new preregistered design seeds 20260926..20260930, outside holdout, for learned-policy crossplay.

Historical opponent ecosystem:
- AVG7600;
- AVG8000;
- BEH7600;
- ENS8_8000.

Compare current ENS8 8100 and AveragePolicy 8100 against their 8000 counterparts and run direct seat-balanced pairings.

No roots. Holdout sealed.

## Immediate action

```bash
bash tools/run_lt2_hu_ens8_learned_ecosystem_crossplay.sh
```

Wait for `LT2_HU_ENS8_LEARNED_ECOSYSTEM_CROSSPLAY_PASS`, then send
`SpinCore_LT2_hu_ens8_learned_ecosystem_crossplay.json`.

Do not train beyond 8100.
