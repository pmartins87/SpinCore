# SpinCore — Long-Training Plan

Status: **ENS8 PILOT 8100 COMPLETE — DESIGN-SET STRATEGIC ADJUDICATION ACTIVE**
Date: 2026-09-20

## Completed pilot

The isolated ENS8 intervention ran from 8000 to 8100:

- +100 iterations;
- +60k roots;
- exact predeclared ENS8_A seed composition reused every iteration;
- 8 fresh400 HU estimators;
- raw-output average;
- unchanged regret matching;
- 3H fresh100 unchanged;
- K4 off;
- holdout untouched.

Mechanical execution passed.

## Why training remains stopped

The training report records fit health and trajectory behavior, not strategic EV.

Before any continuation or holdout, verify that online feedback did not reintroduce:
- root shove inflation;
- PassiveCaller regression;
- Uniform regression;
- deployment/AveragePolicy deterioration.

## Post-pilot forensic gate

On the already-seen forensic seeds:

1. reconstruct ENS8_A from the exact frozen 8000 reservoir;
2. compare deterministic root policies 8000 versus 8100;
3. broad weak-baseline evaluation of ENS8 8100 versus:
   - ENS8 8000;
   - current 7600;
4. compare AveragePolicy 8100 versus AveragePolicy 8000.

No new roots or optimizer steps in source artifacts.

## After this gate

If it passes:
- run trained-policy/cross-play on the design set;
- freeze the final intervention;
- only then unseal holdout.

If it fails:
- localize the new failure before any additional training.

Do not train beyond iteration 8100 meanwhile.
