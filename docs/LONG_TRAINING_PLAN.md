# SpinCore — Long-Training Plan

Status: **8000 FROZEN — OPEN-JAM INFLATION CONFIRMED — CAUSAL SPLIT ACTIVE**
Date: 2026-09-20

## Established

HU400 repaired the original FAI defect and the AveragePolicy at 8000 improved against Jammer.

A separate current-behavior problem arose from 7600 to 8000 at the initial HU preflop root.

Deterministic full-population audit confirms a massive shift:
- ALL_IN +56.42 percentage points;
- POT_33 -45.68 percentage points;
- root TV 0.732;
- argmax disagreement 83.3%.

This is not sampled-action noise.

## Remaining ambiguity

The Advantage model is reset every iteration with an iteration-dependent initialization seed and freshly fitted.

Hence the final current networks at 7600 and 8000 are individual refit realizations.

Before blaming reservoir evolution or the 400-step budget itself, pair fresh refits with identical init and batch seeds across the two reservoirs.

## Gate

Three 400-step paired refits, all 13,585 forensic HU roots.

No training roots and no holdout.

Do not continue long training until this split is resolved.
