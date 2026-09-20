# SpinCore — LT2 HU 7600 -> 8000 current-behavior first-divergence result

Date: 2026-09-20  
Status: **ROOT PREFLOP DRIFT LOCALIZED — PASSIVE AND UNIFORM LOSSES DOMINATED BY PREFLOP_ROOT — NO MORE ROOTS**

## Integrity

Read-only forensic:

- iteration 7600 vs iteration 8000;
- seeds `20260920..20260925`;
- 5000 scenarios/seed;
- 13,585 HU scenario clusters;
- 27,170 seat-runs per baseline;
- no training roots;
- no optimizer steps;
- no memory writes;
- holdout untouched.

## PASSIVE_CALLER

Total current-behavior 8000-minus-7600:

- `-4.28428` chips/hand;
- CI95 `[-6.65670,-1.91187]`.

First-divergence contribution:

- PREFLOP_ROOT: `-3.05609`, CI95 `[-5.25091,-0.86127]`;
- PREFLOP_OTHER: `-1.02882`, unresolved;
- FLOP: `-0.24814`, unresolved;
- TURN/RIVER: approximately zero.

The only resolved component is PREFLOP_ROOT.

## UNIFORM_LEGAL

Total 8000-minus-7600:

- `-9.37435`;
- CI95 `[-13.23204,-5.51666]`.

First-divergence contribution:

- PREFLOP_ROOT: `-7.46864`, CI95 `[-10.54780,-4.38948]`;
- PREFLOP_OTHER: `-1.81174`, nearly but not fully resolved;
- all later streets small/unresolved.

Again, PREFLOP_ROOT dominates.

## JAMMER

Total 8000-minus-7600:

- `+1.34799`;
- CI95 `[-2.23222,+4.92821]`.

Components:

- PREFLOP_FACING_ALL_IN: `+2.21410`, unresolved positive;
- PREFLOP_ROOT: `-0.86610`, unresolved negative.

The original FAI repair therefore remains directionally beneficial while a separate root-policy drift develops.

## Action-transition clue

At PREFLOP_ROOT, the most frequent first-divergence transition is:

- slot 3 -> slot 9: 5,618 seat-runs.

Under the frozen action vocabulary:

- slot 3 = `POT_33`, which preflop is the legacy 2-BB normal open;
- slot 9 = `ALL_IN`.

Other large root transitions toward ALL_IN are:

- CHECK_CALL -> ALL_IN: 1,546;
- FOLD -> ALL_IN: 914.

This is strong evidence that iteration 8000 has shifted root behavior toward substantially more open-jamming.

Counts alone do not prove that the shove-mass shift is the causal value error, so the next gate measures the root probability distributions deterministically before changing training.

## Decision

Do not weaken or remove HU400 yet.

HU400 solved the original FAI underfitting mechanism and the deployed AveragePolicy is now improved.

The new blocker is narrower: online training from 7600 to 8000 causes a preflop-root current-policy drift, apparently toward ALL_IN.

Next: deterministic root-policy drift audit over the full forensic HU population, without sampled hero actions and without terminal outcome selection.
