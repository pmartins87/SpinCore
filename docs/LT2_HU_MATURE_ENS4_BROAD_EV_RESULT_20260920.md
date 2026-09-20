# SpinCore — LT2 mature HU ENS4 broad-EV result

Date: 2026-09-20  
Status: **ENSEMBLE MECHANISM PASSES BROAD EV; SIZE-4 COMPOSITION VARIANCE REMAINS — REPLICATED SIZE-8 GATE NEXT**

## Integrity

The gate recreated the same eight deterministic fresh-400 HU Advantage fits from the frozen iteration-8000 reservoir.

Two disjoint ensembles were evaluated:

- ENS4_LEFT = replicas 0,1,2,3;
- ENS4_RIGHT = replicas 4,5,6,7.

Each ensemble averaged raw Advantage outputs before the unchanged lean regret-matching map at every hero decision.

Evaluation:

- 13,585 forensic HU scenario clusters;
- 81,510 baseline/seat rows;
- UNIFORM_LEGAL, PASSIVE_CALLER, JAMMER;
- common scenario, deal, seat and hero RNG across policies;
- no new CFR roots;
- no source-memory writes;
- holdout untouched.

## Absolute EV

### ENS4_LEFT

- UNIFORM_LEGAL: `+31.714`, CI95 `[+28.071,+35.358]`;
- PASSIVE_CALLER: `+10.430`, CI95 `[+7.945,+12.914]`;
- JAMMER: `+15.250`, CI95 `[+12.413,+18.087]`.

### ENS4_RIGHT

- UNIFORM_LEGAL: `+27.405`, CI95 `[+23.719,+31.091]`;
- PASSIVE_CALLER: `+8.511`, CI95 `[+5.968,+11.054]`;
- JAMMER: `+10.150`, CI95 `[+7.396,+12.905]`.

Both ensembles are decisively positive against all three baselines.

## Versus production Stage B 7500

ENS4_LEFT:

- UNIFORM_LEGAL: `+11.024`, resolved;
- PASSIVE_CALLER: `+7.714`, resolved;
- JAMMER: `+21.830`, resolved.

ENS4_RIGHT:

- UNIFORM_LEGAL: `+6.715`, resolved;
- PASSIVE_CALLER: `+5.796`, resolved;
- JAMMER: `+16.731`, resolved.

Both disjoint size-4 ensembles outperform Stage B on every baseline.

## Versus production pilot 7600

ENS4_LEFT:

- UNIFORM_LEGAL: `+2.601`, unresolved;
- PASSIVE_CALLER: `+6.352`, resolved positive;
- JAMMER: `+17.647`, resolved positive.

ENS4_RIGHT:

- UNIFORM_LEGAL: `-1.708`, unresolved;
- PASSIVE_CALLER: `+4.433`, resolved positive;
- JAMMER: `+12.548`, resolved positive.

Neither ensemble has a resolved regression versus the strongest prior current-behavior checkpoint on any baseline.

## Versus production 8000

Both ensembles are resolved improvements on all three baselines.

## Composition variance

The two size-4 ensembles are not strategically identical.

RIGHT minus LEFT:

- UNIFORM_LEGAL: `-4.309`, CI95 `[-7.454,-1.164]`;
- PASSIVE_CALLER: `-1.919`, unresolved;
- JAMMER: `-5.100`, CI95 `[-8.193,-2.007]`.

Thus LEFT is significantly stronger than RIGHT on Uniform and Jammer.

This prevents selecting one size-4 composition as production merely because it scored better on this forensic population.

## Verdict

The important result is positive:

**the residual statewise TV of size-4 ensembles does not translate into broad strategic failure.**

Both independent size-4 compositions repair the Jammer problem, avoid the Passive regression and outperform Stage B broadly.

However, ensemble membership still matters materially to EV.

The next step is not target/objective redesign yet. First test whether a larger ensemble suppresses this remaining composition variance.

## Next gate

Create two independent size-8 ensembles from sixteen deterministic fresh-400 fits on the frozen iteration-8000 reservoir:

- ENS8_A = replicas 0..7;
- ENS8_B = replicas 8..15.

Evaluate them separately but on exactly the same forensic scenarios/RNG, then compare:

- absolute EV against all three baselines;
- deltas versus production 7500, 7600 and 8000;
- ENS8_B minus ENS8_A.

No outcome-based member selection, no CFR roots, no holdout.

If both ENS8 groups remain strong and their pairwise EV difference is materially smaller/unresolved, size 8 becomes the leading stabilization candidate.
