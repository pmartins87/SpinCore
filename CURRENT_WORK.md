# SpinCore Current Work

Date: 2026-09-20
Status: **ITERATION 8000 FROZEN — RESERVOIR DRIFT + FRESH-FIT INSTABILITY BOTH CONFIRMED — SAME-MEMORY BUDGET STABILITY SWEEP NEXT**

## Paired fresh-400 causal split

Production 7600 -> 8000 root shift:

- ALL_IN `+56.42 pp`;
- POT_33 `-45.68 pp`;
- TV `0.73194`.

Same-seed fresh 400-step refits, 8000 reservoir minus 7600 reservoir:

- ALL_IN mean `+7.97 pp`, replicate CI95 `[+6.31,+9.63]`;
- paired TV mean `0.33561`.

The reservoir therefore moved toward more jamming, but the production checkpoint difference is much larger than the matched-reservoir effect.

## Fresh-fit instability

400-step ALL_IN mass varies strongly across replicate fits even on one fixed reservoir.

7600 reservoir:
- `11.48%, 27.07%, 11.56%`.

8000 reservoir:
- `18.23%, 36.67%, 19.10%`.

One freshly reset 400-step Advantage network is therefore not a stable representation of the mature reservoir.

## Active gate

Use only the frozen iteration-8000 HU reservoir.

Four fit trajectories, each measured at cumulative:

- 400;
- 800;
- 1600;
- 3200 steps.

Measure pairwise root-policy TV, p95, argmax disagreement and ALL_IN/POT_33 mass dispersion.

If more fitting materially contracts the instability, take the smallest stable budget to a broad EV gate.

If not, stop increasing optimizer work and move to the ensemble/stabilization branch already supported by earlier R7.3 evidence.

No new roots. Holdout sealed.

## Immediate action

```bash
bash tools/run_lt2_hu_refit_budget_root_stability.sh
```

Send `SpinCore_LT2_hu_refit_budget_root_stability.json`.

Do not train beyond 8000.
