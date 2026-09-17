# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED AT 4.5M — HU-JAMMER FAILURE CONFIRMED — ADVANTAGE VALUE-SENSITIVITY AUDIT ACTIVE**
Date: 2026-09-17

## Current state

The continuous learning line has reached:

- LT0: 120k roots;
- LT1: 1.2M roots;
- LT2 Stage A: 1.8M roots;
- LT2 Stage B: 4.5M roots / iteration 7500;
- all four 2M reservoirs in replacement regime;
- Stage A -> Stage B policy movement is material;
- 30k weak-baseline precision gate complete;
- Stage B HU Jammer is statistically negative and worsened relative to Stage A;
- stored-target fit audit complete;
- AveragePolicy extra fitting failed to improve held-out CE;
- corrected Advantage multi-seed budget sweep complete;
- more Advantage steps reduce MSE but do not reduce production-policy TV;
- immediate next experiment is a read-only target-value sensitivity audit, not more roots.

Read `LT2_ADVANTAGE_BUDGET_SWEEP_V2_RESULT_20260917.md` and `LT2_ADVANTAGE_VALUE_SENSITIVITY_20260917.md` first.

## Core training contract

Current functional line:

- empirical SpinGo 3H/HU/blind/stack sampling;
- WTA chip-EV utility scaled by 1500;
- SPNNIV1 frozen-control representation;
- mature legacy action vocabulary;
- external-sampling Deep CFR;
- separate 3H and HU brains;
- sampled AveragePolicy trajectories;
- 2,000,000-sample reservoir capacity per memory per domain;
- 600 roots per iteration;
- Advantage reset every iteration;
- 100 Advantage optimizer steps per domain per iteration;
- batch size 1024;
- 4000 AveragePolicy optimizer steps per milestone finalization;
- production behavior uses positive-regret matching and a masked-softmax fallback when all legal predicted advantages are non-positive;
- 31 root workers, one worker numerical thread, 8 parent Torch threads, vectorized batching, production concurrent-fit mode.

## Preserved milestones

LT1: 1.2M roots, SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.

LT2 Stage A: 1.8M roots / iteration 3000, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

LT2 Stage B: 4.5M roots / iteration 7500, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep all preserved milestones unchanged.

## Strength result that freezes root scaling

The powered 30k gate established:

- Stage B HU Jammer `-5.141` chips/hand, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage B minus Stage A HU Jammer `-1.682`, simultaneous six-claim interval about `[-3.143,-0.222]`.

Therefore extra Stage-A -> Stage-B training measurably degraded this specific weak-opponent HU cell. More roots cannot be admitted merely because policy continues to move.

## What the optimizer diagnostics established

### AveragePolicy

Extra Stage-B policy fitting does not improve held-out CE. The simple hypothesis “4000 finalization steps are insufficient” is not supported.

### Advantage

The corrected V2 sweep used a fixed 25k holdout per domain and three independent reset/training replicates with exact production policy semantics.

3H, 100 -> 1600 steps:

- MSE `0.03194285 -> 0.03073128` (~3.79% lower);
- TV `0.604686 -> 0.602978` (flat);
- argmax `31.78% -> 37.18%`.

HU, 100 -> 1600 steps:

- MSE `0.04608811 -> 0.04447139` (~3.51% lower);
- TV `0.587335 -> 0.604104` (worse);
- argmax `28.64% -> 34.52%`.

The target/predicted policy-branch mismatch is also conspicuous: HU target-all-nonpositive frequency is ~37.11%, while 1600-step predictions are all-nonpositive only ~4.05% on average.

Thus optimizer budget is not a clean behavior-level fix. A 16x Advantage budget is not admitted from MSE alone.

## Why target-value sensitivity is next

Policy TV can be a poor proxy for strategic loss. If target actions are nearly equal in value, two distributions may have TV near 1 while being almost identical in expected target value. Conversely, moderate TV in a high-span state can be costly.

The next audit measures the target-value consequence of the model policy directly, in chip-equivalent units derived from the canonical `/1500` utility scale.

Launcher: `tools/run_lt2_advantage_value_sensitivity.sh`.

Default design:

- Stage B source checkpoint read only and SHA-checked before/after;
- no roots;
- no optimizer steps;
- deterministic 100k Advantage-memory samples per domain;
- target and model predictions both mapped through the exact production Lean policy rule;
- target span, regret to target best action, target-policy/model-policy value gap, branch confusion, TV and argmax;
- breakdown by target branch, street and descriptive chip-span bands.

## Decision branches after value-sensitivity audit

If large TV is mostly near-indifference and target-value loss is small, stop using raw TV as the deciding metric and move directly to a targeted HU-Jammer state/action audit.

If target-value loss is large and concentrated in positive-regret/all-nonpositive branch mismatches, test regret-sign/fallback calibration on held-out targets before any root continuation.

If loss is concentrated by street or large target span, inspect representation and target generation in those state classes.

If a bounded intervention later improves held-out decision value, it must still beat preserved Stage B in a statistically powered weak-baseline comparison before root scaling resumes.

## DeepCrusher placement

DeepCrusher remains deferred until the weak-opponent curriculum is strong and stable. A literal C++ translation may help later but is not a current dependency.

## Immediate direction

1. Preserve Stage A and Stage B.
2. Keep root training stopped at iteration 7500.
3. Run `bash tools/run_lt2_advantage_value_sensitivity.sh`.
4. Review `SpinCore_LT2_advantage_value_sensitivity.json`.
5. Choose between targeted HU-Jammer diagnosis, regret-sign calibration, or representation/target-generation audit from the measured value loss.
6. Do not scale roots or move to DeepCrusher yet.
