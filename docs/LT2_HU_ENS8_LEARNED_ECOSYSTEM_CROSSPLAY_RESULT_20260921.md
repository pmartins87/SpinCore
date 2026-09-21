# SpinCore — LT2 HU ENS8 independent learned-ecosystem crossplay result

Date: 2026-09-21  
Status: **DESIGN-SET CROSSPLAY PASS — ENS8 8100 FROZEN AS FINAL HU CURRENT-POLICY CANDIDATE — HOLDOUT PROTOCOL NOW FIXED**

## Integrity

Independent design seeds:

- 20260926;
- 20260927;
- 20260928;
- 20260929;
- 20260930.

These were preregistered before seeing outcomes and are separate from final holdout seeds 20261001..20261006.

Evaluation:

- 13,511 HU scenario clusters;
- 27,022 ecosystem seat-runs;
- 108,088 total rows;
- no training roots;
- no source-memory writes;
- no optimizer steps on source artifacts;
- holdout untouched.

Historical learned-opponent ecosystem:

- AVG_7600;
- AVG_8000;
- BEH_7600;
- ENS8_8000.

## Learned ecosystem

Absolute mean EV:

- AVG_8000: `-5.772`;
- AVG_8100: `-5.520`;
- ENS8_8000: `+6.086`;
- ENS8_8100: **`+7.538`, CI95 `[+4.575,+10.502]`**.

Paired deltas:

- ENS8_8100 − ENS8_8000: `+1.453`, CI95 `[-0.434,+3.339]`;
- AVG_8100 − AVG_8000: `+0.251`, CI95 `[-0.874,+1.376]`;
- ENS8_8100 − AVG_8100: **`+13.059`, CI95 `[+9.704,+16.414]`**.

ENS8 8100 is positive on four of five preregistered seeds versus ENS8 8000 and has no resolved regression overall.

## Direct seat-balanced crossplay

- ENS8_8100 vs ENS8_8000: `+1.227`, CI95 `[-0.711,+3.165]`;
- AVG_8100 vs AVG_8000: `+0.814`, CI95 `[-2.459,+4.086]`;
- ENS8_8100 vs AVG_8100: **`+12.663`, CI95 `[+9.509,+15.817]`**.

Thus the current ENS8 candidate is strongly superior to the lagged AveragePolicy deployment path on this independent design set, while remaining non-regressing versus the source ENS8.

## Decision

The preregistered design criterion is satisfied:

- no resolved learned-policy regression versus ENS8 8000;
- positive learned-ecosystem absolute EV;
- strong direct and ecosystem advantage over AveragePolicy 8100.

Therefore freeze:

**HU deployment candidate = current ENS8 at iteration 8100**  
**Artifact pair = iteration-8100 checkpoint + iteration-8100 HU ensemble sidecar**

Do not substitute AveragePolicy 8100 as the HU candidate.

## Final holdout

The intervention and decision rules are now frozen before holdout.

Final holdout seeds:

- 20261001..20261006.

Primary acceptance criteria, all preregistered before outcomes:

1. learned-ecosystem ENS8_8100 absolute EV: CI95 lower bound > 0;
2. learned-ecosystem ENS8_8100 − ENS8_8000: CI95 lower bound > -3.0 chips;
3. learned-ecosystem ENS8_8100 − AVG_8100: CI95 lower bound > 0;
4. direct ENS8_8100 vs ENS8_8000: CI95 lower bound > -3.0 chips;
5. direct ENS8_8100 vs AVG_8100: CI95 lower bound > 0;
6. absolute ENS8_8100 EV against each transparent weak baseline (Uniform, Passive, Jammer): CI95 lower bound > 0;
7. ENS8_8100 − ENS8_8000 against each weak baseline: CI95 lower bound > -3.0 chips.

The -3.0 chip margin is a practical non-inferiority bound fixed before holdout, chosen to be slightly wider than the remaining replicated ENS8 composition spread while still excluding a strategically meaningful regression.

One completed holdout run is final. No tuning, member selection, seed changes or reruns based on its outcome.
