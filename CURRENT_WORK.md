# SpinCore Current Work

Date: 2026-09-20
Status: **ITERATION 8000 FROZEN — SIZE-4 ENSEMBLE PASSES BROAD EV BUT COMPOSITION VARIANCE REMAINS — REPLICATED SIZE-8 GATE NEXT**

## ENS4 broad-EV result

Both disjoint size-4 ensembles are strongly positive against every forensic baseline.

ENS4_LEFT absolute:
- UNIFORM_LEGAL `+31.714`;
- PASSIVE_CALLER `+10.430`;
- JAMMER `+15.250`.

ENS4_RIGHT absolute:
- UNIFORM_LEGAL `+27.405`;
- PASSIVE_CALLER `+8.511`;
- JAMMER `+10.150`.

Both are resolved improvements versus Stage B 7500 on all three baselines.

Versus 7600:
- neither has a resolved Uniform regression;
- both significantly improve PASSIVE_CALLER;
- both significantly improve JAMMER.

Thus the ensemble mechanism passes broad strategic EV.

## Remaining blocker: composition variance

RIGHT minus LEFT:
- UNIFORM_LEGAL `-4.309`, resolved;
- PASSIVE_CALLER `-1.919`, unresolved;
- JAMMER `-5.100`, resolved.

The two size-4 groups are both good, but not strategically equivalent.

Do not select LEFT merely because it won on the forensic population.

## Active gate

Create two independent size-8 ensembles from 16 deterministic fresh400 fits:

- ENS8_A = 0..7;
- ENS8_B = 8..15.

Evaluate both on exactly the same forensic population and RNG schedule.

Goal:
- preserve broad improvements;
- reduce ensemble-composition EV variance.

No roots. Holdout sealed.

## Immediate action

```bash
bash tools/run_lt2_hu_mature_ens8_replicated_broad_ev.sh
```

Send `SpinCore_LT2_hu_mature_ens8_replicated_broad_ev.json`.

Do not train beyond 8000.
