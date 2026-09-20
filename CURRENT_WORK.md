# SpinCore Current Work

Date: 2026-09-20
Status: **ITERATION 8000 FROZEN — SIZE-4 ENSEMBLE PARTIALLY STABILIZES ROOT FREQUENCIES BUT NOT STATEWISE POLICY — BROAD EV GATE NEXT**

## Mature ensemble stability result

Eight independent fresh400 fits on the frozen iteration-8000 HU reservoir.

Single models:
- mean pair TV `0.5643`;
- p95 `1.0000`;
- argmax disagreement `61.01%`;
- ALL_IN range `42.97 pp`;
- POT_33 range `40.39 pp`.

Size 2 is not useful:
- mean-TV ratio `0.900`;
- p95 ratio `1.000`;
- ALL_IN/POT_33 ranges slightly worse than singles.

Size 4:
- mean TV `0.4814` — only 14.7% lower;
- p95 `1.0000` — unchanged;
- argmax disagreement `47.18%` — 22.7% lower;
- ALL_IN range `10.05 pp` — 76.6% lower;
- POT_33 range `12.69 pp` — 68.6% lower.

## Interpretation

Size 4 strongly stabilizes aggregate action frequencies but does not make the per-state policy geometries agree.

Therefore root-TV alone is insufficient to decide whether the ensemble is strategically useful.

## Active gate

Recreate the exact same 8 models and evaluate the two disjoint size-4 ensembles over complete HU hands against:

- UNIFORM_LEGAL;
- PASSIVE_CALLER;
- JAMMER.

Context:
- production current behavior 7500;
- production current behavior 7600;
- production current behavior 8000.

The key question is whether residual statewise disagreement translates into chip-EV instability.

No roots. Holdout sealed.

## Immediate action

```bash
bash tools/run_lt2_hu_mature_ens4_broad_ev.sh
```

Send `SpinCore_LT2_hu_mature_ens4_broad_ev.json`.

Do not train beyond 8000.
