# SpinCore — LT2 HU paired fresh-400 root refit result

Date: 2026-09-20  
Status: **BOTH MECHANISMS PRESENT — RESERVOIR DRIFT IS REAL, BUT PRODUCTION EXTREME IS DOMINATED BY FRESH-FIT INSTABILITY**

## Integrity

The audit used:

- preserved iteration-7600 HU Advantage reservoir;
- preserved iteration-8000 HU Advantage reservoir;
- three paired fresh 400-step refits;
- identical initialization seed across reservoirs within each replicate;
- identical minibatch RNG seed across reservoirs within each replicate;
- all 13,585 forensic HU roots;
- no new training roots;
- no source-memory writes;
- holdout untouched.

## Production checkpoint difference

The actual iteration-7600 and iteration-8000 current networks differ massively at the HU root:

- ALL_IN: `+56.42 pp`;
- POT_33: `-45.68 pp`;
- TV: `0.73194`.

This is the extreme drift already confirmed in the deterministic root audit.

## Same-seed fresh-refit reservoir effect

Across the three paired 400-step refits, iteration-8000 reservoir minus iteration-7600 reservoir:

- ALL_IN: `+7.97 pp`;
- replicate CI95 `[+6.31,+9.63]`;
- POT_33: `-8.74 pp`, replicate interval unresolved;
- paired root TV: mean `0.33561`.

The ALL_IN direction is positive in every replicate:

- R0: `+6.76 pp`;
- R1: `+9.60 pp`;
- R2: `+7.54 pp`.

Therefore the online reservoir itself has genuinely moved toward more root jamming.

## Fresh-fit realization variance is also large

Absolute ALL_IN mass from the same iteration-7600 reservoir varies across 400-step refits:

- R0: `11.48%`;
- R1: `27.07%`;
- R2: `11.56%`.

For the iteration-8000 reservoir:

- R0: `18.23%`;
- R1: `36.67%`;
- R2: `19.10%`.

POT_33 also varies strongly across fresh fits.

Thus one 400-step freshly reset Advantage network is not a stable representation of the reservoir-wide root policy.

## Causal interpretation

Two mechanisms coexist:

1. **real reservoir evolution** from 7600 to 8000 shifts matched fresh fits by about +8 percentage points of ALL_IN mass;
2. **fit realization instability** is large enough that individual checkpoint networks can land far from the typical fresh-fit behavior.

The production `+56.42 pp` ALL_IN jump is far larger than the matched-reservoir effect. It should not be interpreted as 400 online iterations teaching a +56 pp shove increase by themselves.

This reconnects with the earlier R7.3 diagnosis that independently fitted Advantage networks can produce materially different regret-matching policies even on the same memory.

## Next gate

Before introducing an ensemble or changing the regret map, test the smallest remaining intervention: more complete fitting of the same mature reservoir.

On the frozen iteration-8000 HU reservoir, run four identical-seed fit trajectories and evaluate root policy stability at cumulative budgets:

- 400;
- 800;
- 1600;
- 3200 optimizer steps.

Each replicate uses one initialization and one minibatch stream; the higher budgets continue the exact same fit trajectory.

Measure:

- pairwise root-policy TV;
- pairwise p95 TV;
- argmax disagreement;
- ALL_IN mass dispersion;
- POT_33 mass dispersion.

If increased fit budget strongly contracts same-memory policy dispersion, use the smallest stable budget for a new broad EV gate.

If instability remains large even at 3200, stop escalating optimizer work and move to the already-supported ensemble/stabilization branch.

No long training is authorized.
