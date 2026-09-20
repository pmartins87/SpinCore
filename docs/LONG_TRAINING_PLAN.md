# SpinCore — Long-Training Plan

Status: **8000 FROZEN — ROOT PREFLOP CURRENT-POLICY DRIFT UNDER DIAGNOSIS**
Date: 2026-09-20

## Checkpoints

Stage B 7500 SHA:
`3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

HU400 pilot 7600 SHA:
`c34f19802d3ad5ad3a5d131b083ffcee0e0867667ae0d57324cf35d5fa246b80`.

HU400 refresh 8000 SHA:
`773b5d523c7fc5fcbfc3d10cb1f5be6429e50f4283259df9134963db8d274886`.

## What remains valid

The original Jammer facing-all-in defect was a real underfit problem.

HU400 repaired it in fresh refits, broad population tests and online training.

The iteration-8000 AveragePolicy also improves significantly against Jammer without a resolved AveragePolicy weak-baseline regression.

## New blocker

Between 7600 and 8000, current behavior deteriorates against PASSIVE_CALLER and UNIFORM_LEGAL.

First-divergence attribution localizes the dominant resolved loss to PREFLOP_ROOT.

The sampled root transition matrix is dominated by POT_33 -> ALL_IN.

## Current hypothesis

Online HU400 training is producing a new root-policy geometry drift toward open-jamming.

This is separate from the original FAI overfold defect.

Do not conclude that 400 optimizer steps are globally excessive until deterministic root-policy mass and its state distribution are measured.

## Next gate

Read-only deterministic root-policy audit:
- all forensic HU roots;
- no sampled-action selection;
- mass delta per action;
- TV and argmax;
- blind/effective-stack stratification.

No root training beyond 8000 until reviewed.
