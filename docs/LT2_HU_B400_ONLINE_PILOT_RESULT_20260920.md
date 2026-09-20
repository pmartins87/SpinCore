# SpinCore — LT2 HU B400 online pilot result

Date: 2026-09-20  
Status: **PASS — ONLINE FEEDBACK SURVIVES; CURRENT HU BEHAVIOR IMPROVES; AVERAGEPOLICY STILL LAGS**

## Integrity

Pilot:

- source: preserved Stage B iteration 7500;
- target: iteration 7600;
- +100 iterations;
- +60,000 roots;
- THREE_HANDED Advantage fit = 100 steps;
- TRUE_HEADS_UP Advantage fit = 400 steps;
- K4 off;
- 31 workers;
- vectorized batches;
- concurrent fit;
- holdout untouched.

Pilot checkpoint:

- SHA256 `c34f19802d3ad5ad3a5d131b083ffcee0e0867667ae0d57324cf35d5fa246b80`.

## Current Advantage behavior — primary gate

Stage B -> pilot:

### JAMMER

- delta `+4.18296` chips/hand;
- CI95 `[+0.30435,+8.06157]`.

This is a resolved improvement.

Absolute pilot current-behavior EV versus JAMMER:

- `-2.39735`;
- CI95 `[-5.83103,+1.03633]`.

Thus the production-loop intervention repairs a meaningful part of the Stage-B defect, but the final iteration-7600 behavior is not yet proven positive in absolute Jammer EV.

### PASSIVE_CALLER

- delta `+1.36286`;
- CI95 `[-1.50385,+4.22958]`.

No resolved regression.

### UNIFORM_LEGAL

- delta `+8.42330`;
- CI95 `[+4.39799,+12.44860]`.

Resolved improvement.

## AveragePolicy — expected inertia

Stage B -> pilot:

- JAMMER: `+0.67917`, CI95 `[-0.23378,+1.59212]`;
- PASSIVE_CALLER: `+0.00445`, CI95 `[-0.91803,+0.92693]`;
- UNIFORM_LEGAL: `+0.76389`, CI95 `[-0.40032,+1.92811]`.

AveragePolicy has not materially moved after only 100 iterations.

This is consistent with the policy reservoir still being dominated by historical samples from the 100-step HU regime.

## Resource result

The 100-iteration pilot was operationally healthy.

Mean per-iteration fit times:

- THREE_HANDED Advantage: `5.65 s`;
- TRUE_HEADS_UP Advantage: `17.13 s`.

WSL memory remained safe; only about 1 MiB swap was touched at peak.

## Verdict

The HU400 intervention survives online feedback.

This is the result required to leave the 100-iteration pilot.

However, the deployed AveragePolicy has not yet incorporated enough of the corrected behavior. Therefore the next block is not a new algorithmic experiment; it is a bounded policy-memory refresh under the already-frozen HU400 intervention.

## Next gate

Continue from the pilot checkpoint for 400 more iterations:

- 7601..8000;
- +240,000 roots;
- cumulative HU400 exposure from Stage B: 500 iterations;
- 3H remains 100;
- HU remains 400;
- K4 remains off.

After iteration 8000, compare the original Stage B checkpoint directly with the 8000 checkpoint on the full forensic HU policy-chain.

Primary:

- current behavior Jammer delta remains resolved positive;
- no current-behavior resolved regression on other baselines.

Secondary / deployment:

- AveragePolicy Jammer delta should now move measurably positive;
- no resolved AveragePolicy regression on other baselines.

If AveragePolicy remains essentially stationary after 500 cumulative HU400 iterations, stop before another block and inspect policy-memory composition/weighting rather than blindly training.
