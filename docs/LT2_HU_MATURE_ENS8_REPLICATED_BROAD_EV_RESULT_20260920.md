# SpinCore — LT2 mature HU replicated ENS8 broad-EV result

Date: 2026-09-20  
Status: **SIZE-8 IS THE LEADING STABILIZATION CANDIDATE — ONLINE-FEEDBACK PILOT AUTHORIZED IN ISOLATION**

## Integrity

Two independent size-8 ensembles were built from sixteen predeclared deterministic fresh-400 fits on the exact frozen iteration-8000 HU Advantage reservoir.

- ENS8_A = replicas 0..7;
- ENS8_B = replicas 8..15;
- same 13,585 forensic HU scenario clusters;
- same deals, hero seats and hero RNG;
- no CFR roots;
- no source-memory writes;
- holdout untouched.

Raw Advantage outputs were averaged before the unchanged lean regret-matching map.

## Absolute EV

### ENS8_A

- UNIFORM_LEGAL: `+28.097`, CI95 `[+24.442,+31.753]`;
- PASSIVE_CALLER: `+10.961`, CI95 `[+8.496,+13.426]`;
- JAMMER: `+15.052`, CI95 `[+12.305,+17.798]`.

### ENS8_B

- UNIFORM_LEGAL: `+30.034`, CI95 `[+26.488,+33.581]`;
- PASSIVE_CALLER: `+13.462`, CI95 `[+10.512,+16.412]`;
- JAMMER: `+17.212`, CI95 `[+14.036,+20.388]`.

Both independent compositions are decisively positive on all three baselines.

## Versus production Stage B 7500

ENS8_A:

- UNIFORM_LEGAL: `+7.407`, resolved;
- PASSIVE_CALLER: `+8.246`, resolved;
- JAMMER: `+21.632`, resolved.

ENS8_B:

- UNIFORM_LEGAL: `+9.344`, resolved;
- PASSIVE_CALLER: `+10.747`, resolved;
- JAMMER: `+23.792`, resolved.

The original Stage-B weakness is repaired broadly by both independent ENS8 groups.

## Versus production pilot 7600

ENS8_A:

- UNIFORM_LEGAL: `-1.016`, unresolved;
- PASSIVE_CALLER: `+6.883`, resolved positive;
- JAMMER: `+17.449`, resolved positive.

ENS8_B:

- UNIFORM_LEGAL: `+0.921`, unresolved;
- PASSIVE_CALLER: `+9.384`, resolved positive;
- JAMMER: `+19.609`, resolved positive.

Neither independent ENS8 composition has a resolved regression versus 7600 on any baseline.

## Composition variance

ENS8_B minus ENS8_A:

- UNIFORM_LEGAL: `+1.937`, CI95 `[-0.827,+4.701]`, unresolved;
- PASSIVE_CALLER: `+2.501`, CI95 `[+0.157,+4.846]`, barely resolved;
- JAMMER: `+2.160`, CI95 `[-0.868,+5.189]`, unresolved.

Compared with ENS4, the maximum observed composition gap contracts from about `5.10` chips to `2.50` chips and the Uniform/Jammer differences lose significance.

Composition sensitivity is therefore materially reduced, though not mathematically eliminated.

## Decision

Size 8 satisfies the preregistered direction strongly enough to become the leading online-stabilization candidate:

- both independent groups are strong;
- both preserve the Stage-B repair;
- neither regresses versus 7600 on any tested baseline;
- composition sensitivity contracts materially.

Do not select ENS8_B because it happened to score higher.

The online pilot uses **ENS8_A**, the first predeclared group, specifically to avoid outcome-based member selection.

## Online pilot

Source remains the exact frozen iteration-8000 checkpoint.

Pilot target:
- iteration 8100;
- +100 iterations;
- 3H unchanged fresh100;
- HU eight fresh400 estimators;
- exact ENS8_A member fit seeds reused every iteration;
- raw outputs averaged before unchanged lean regret matching;
- K4 off;
- source checkpoint read-only;
- holdout sealed.

Fit RNG is isolated from authoritative policy-sampling RNG so ensemble size does not alter policy RNG merely by consuming more minibatches.

The pilot writes an ordinary checkpoint plus a mandatory HU ensemble sidecar. The ordinary checkpoint alone is not the complete current HU behavior.
