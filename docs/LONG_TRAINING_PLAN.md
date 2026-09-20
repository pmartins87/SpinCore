# SpinCore — Long-Training Plan

Status: **8000 FROZEN — SINGLE-MODEL FIT-BUDGET ESCALATION REJECTED — ENSEMBLE STABILITY GATE ACTIVE**
Date: 2026-09-20

## Current diagnosis

The current HU training-policy problem contains two components:

1. a real but moderate reservoir drift toward more root jamming;
2. a much larger fresh-fit realization instability.

Increasing one fresh model from 400 to 3200 optimizer steps does not remove the instability:

- mean pairwise TV improves only from 0.595 to 0.409;
- p95 stays near 1.0;
- argmax disagreement remains about 48%;
- action-mass trajectories remain non-monotonic.

## Consequence

Do not increase single-model fit budget further.

The already-motivated variance-reduction branch is now the active candidate.

## Gate

On the frozen iteration-8000 HU reservoir:

- 8 fresh 400-step models;
- cache raw Advantage outputs on all forensic HU roots;
- compare disjoint size-1, size-2 and size-4 raw-output ensembles;
- unchanged lean regret matching after averaging.

If size 2 or 4 materially stabilizes root policy, run a broad EV gate before any online integration.

If ensemble also fails, investigate objective / target geometry rather than adding roots.

Holdout remains sealed.
