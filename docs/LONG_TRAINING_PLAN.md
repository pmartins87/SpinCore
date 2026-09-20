# SpinCore — Long-Training Plan

Status: **8000 FROZEN — TWO-MECHANISM ROOT DRIFT CONFIRMED — FIT-BUDGET STABILITY GATE ACTIVE**
Date: 2026-09-20

## Current causal picture

The original HU Jammer FAI defect was caused by insufficient fitting and is repaired by HU400.

The later root-policy problem is not one simple failure.

Paired fresh-400 refits show:

1. the 8000 reservoir itself has shifted toward more ALL_IN by about 8 percentage points versus 7600;
2. independent 400-step fresh fits on a fixed reservoir still produce widely different root policies.

The actual production checkpoint jump of +56 pp ALL_IN is therefore an extreme individual-fit realization layered on top of a smaller real reservoir drift.

## Next minimal intervention test

Before changing the algorithm, determine whether the mature reservoir simply requires more optimizer convergence globally.

Frozen iteration-8000 HU reservoir:

- 4 independent fit trajectories;
- evaluate at 400, 800, 1600, 3200 cumulative steps;
- common forensic HU root corpus.

Metrics:
- pairwise mean/p95 TV;
- argmax disagreement;
- ALL_IN and POT_33 mass dispersion.

## Branches

If stability improves strongly with budget:
- use smallest stable budget for broad EV validation.

If stability plateaus:
- proceed to ensemble/stability mechanism rather than more long training.

No roots beyond iteration 8000 and no holdout until this is resolved.
