# SpinCore — LT2 HU B400 refresh-to-8000 result

Date: 2026-09-20  
Status: **MIXED GATE — AVERAGEPOLICY IMPROVES; CURRENT BEHAVIOR DEVELOPS PASSIVE-CALLER REGRESSION; NO MORE ROOTS**

## Integrity

The block continued the preserved iteration-7600 pilot checkpoint to iteration 8000:

- +400 iterations;
- +240,000 roots;
- cumulative HU400 exposure from Stage B: 500 iterations;
- THREE_HANDED Advantage fit = 100;
- TRUE_HEADS_UP Advantage fit = 400;
- K4 off;
- 31 root workers;
- vectorized batches;
- concurrent fit;
- holdout untouched.

Iteration-8000 checkpoint:

- SHA256 `773b5d523c7fc5fcbfc3d10cb1f5be6429e50f4283259df9134963db8d274886`.

## AveragePolicy — deployment result

Stage B 7500 -> iteration 8000:

### JAMMER

- delta: `+1.90206` chips/hand;
- CI95: `[+0.68735,+3.11677]`.

This is the first resolved positive movement of the deployed AveragePolicy against JAMMER.

### PASSIVE_CALLER

- delta: `+0.78649`;
- CI95: `[-0.30808,+1.88107]`.

No resolved regression.

### UNIFORM_LEGAL

- delta: `+1.17571`;
- CI95: `[-0.20461,+2.55603]`.

No resolved regression.

Therefore the deployment policy has now absorbed part of the HU400 repair without a detected weak-baseline tradeoff.

## Current Advantage behavior — training-policy result

Stage B 7500 -> iteration 8000:

### JAMMER

- delta: `+5.53095`;
- CI95: `[+2.10081,+8.96110]`.

Resolved improvement.

### PASSIVE_CALLER

- delta: `-2.92142`;
- CI95: `[-5.54081,-0.30203]`.

This is a newly resolved regression and violates the preregistered current-behavior no-tradeoff condition.

### UNIFORM_LEGAL

- delta: `-0.95105`;
- CI95: `[-4.67461,+2.77251]`.

Unresolved.

## Interpretation

The HU400 intervention successfully repairs the Jammer problem and the AveragePolicy now reflects that repair.

However, continued online training from 7600 to 8000 moved the current Advantage-induced behavior into a new weakness against PASSIVE_CALLER.

That matters even though AveragePolicy is currently healthy, because current behavior is the policy that generates future training trajectories and policy-memory targets. Continuing roots without understanding this tradeoff could eventually contaminate the deployed AveragePolicy.

Therefore iteration 8000 is frozen as a deployment candidate, but long training is paused.

## Next read-only gate

Compare current behavior at iteration 7600 directly against iteration 8000 using the existing full forensic first-divergence audit.

Primary question:

**where does the newly emerged PASSIVE_CALLER behavior loss first appear?**

Groups:

- PREFLOP_ROOT;
- PREFLOP_FACING_ALL_IN;
- PREFLOP_OTHER;
- FLOP;
- TURN;
- RIVER.

No training roots, no optimizer steps, no holdout.

If the Passive loss localizes to one dominant state class, follow that class.

If the loss is diffuse, investigate whether the 400-step refit is increasingly specializing the current behavior toward Jammer-like pressure at the expense of passive exploitation.

Do not continue beyond iteration 8000 until this is resolved.
