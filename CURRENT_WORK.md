# SpinCore Current Work

Date: 2026-09-20
Status: **ITERATION 8000 FROZEN — MORE FIT STEPS FAIL TO STABILIZE CURRENT HU POLICY — MATURE-RESERVOIR ENSEMBLE GATE NEXT**

## Refit-budget stability result

Same frozen iteration-8000 HU Advantage reservoir, four independent trajectories:

| steps | mean pair TV | p95 TV | argmax disagreement |
|---:|---:|---:|---:|
| 400 | 0.5950 | 1.0000 | 63.58% |
| 800 | 0.5395 | 0.9891 | 59.57% |
| 1600 | 0.4278 | 0.9700 | 46.75% |
| 3200 | 0.4090 | 0.9640 | 47.71% |

3200 cuts mean TV by only about 31% versus 400 while the high-tail disagreement remains almost saturated.

ALL_IN mass dispersion is not monotonic:
- range 400: 10.80 pp;
- range 800: 44.10 pp;
- range 1600: 16.67 pp;
- range 3200: 20.63 pp.

A representative fit trajectory moves ALL_IN:
`23.11% -> 61.08% -> 35.64% -> 12.85%`
at 400/800/1600/3200.

## Decision

Do not spend more optimizer steps on a single freshly reset model.

The next gate tests direct variance reduction by averaging independent Advantage estimators.

## Active gate

Frozen iteration-8000 HU reservoir:

- 8 independent fresh 400-step fits;
- same 13,585 forensic HU roots;
- disjoint ensemble sizes 1, 2 and 4;
- average raw Advantage outputs before unchanged lean regret matching.

No roots. No checkpoint mutation. Holdout sealed.

## Immediate action

```bash
bash tools/run_lt2_hu_mature_ensemble_root_stability.sh
```

Send `SpinCore_LT2_hu_mature_ensemble_root_stability.json`.

Do not train beyond iteration 8000.
