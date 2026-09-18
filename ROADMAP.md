# SpinCore Roadmap — active state 2026-09-18

## Active status

- LT0 calibration — **DONE**: 120k roots.
- LT1 production-shaped milestone — **DONE**: 1.2M roots.
- LT1 physical fit optimization — **PASS**.
- LT2 Stage A — **PASS**: 1.8M roots.
- LT2 Stage B — **PASS**: 4.5M roots / iteration 7500.
- Stage B resource gate — **PASS**.
- Policy drift Stage A -> Stage B — **MATERIAL MOVEMENT CONFIRMED**.
- Checkpoint cross-play — **NO REPRODUCIBLE ORDERING**.
- 30k weak-baseline gate — **COMPLETE; HU JAMMER NEGATIVE**.
- AveragePolicy extra-budget hypothesis — **NOT SUPPORTED**.
- Advantage optimizer escalation — **NOT SUPPORTED AS NEXT INTERVENTION**.
- HU-preflop conditional variance decomposition — **COMPLETE; 93.73% HIDDEN/CHANCE VARIANCE**.
- Exact0/exact1 target-estimator sweep — **COMPLETE; EXACT0 + MORE DEALS WINS COMPUTE FRONTIER**.
- Board-only averaging sweep — **COMPLETE; K4 IS ESTIMATOR COMPUTE ELBOW**.
- Board-averaging mechanics smoke first run — **FAILED SAFELY; RNG COUPLING DIAGNOSED**.
- RNG-coupling fix — **IMPLEMENTED; SMOKE RERUN NEXT**.
- Stage-A -> Stage-B causal attribution — **NEXT AFTER SMOKE**.
- K4 causal training pilot — **NOT AUTHORIZED YET**.
- Root training beyond iteration 7500 — **PAUSED**.
- DeepCrusher — **DEFERRED**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_FAILURE_20260918.md`
- `docs/LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_20260918.md`
- `docs/LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_RESULT_20260918.md`
- `docs/LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep both unchanged.

## Why long scaling remains frozen

Stage B HU Jammer remains confirmed negative:
- raw EV `-5.141` chips/hand;
- simultaneous family-wise 95% CI `[-9.078,-1.204]`.

A diagnostic estimator improvement is not enough to reopen training.

## Board-only K4 evidence

Board-only averaging at exact0 reduces target variance and improves branch mismatch/regret at K4, with K8 beyond the policy-space compute elbow.

This justifies testing the estimator mechanically. It does **not** prove that board-noise caused the Stage-A -> Stage-B regression.

## Mechanics-smoke failure

The first K1-vs-K4 smoke failed because preflop sample counts differed across board variants.

Cause:

The collector is depth-first. Postflop branches consume RNG before recursion returns to later preflop branches. Different future boards therefore alter the RNG position seen by those later preflop opponent nodes.

Resetting one global RNG only at traversal start was insufficient.

## Fix

The experimental K4 path now:

1. runs board 0 canonically and records sampled preflop opponent actions;
2. records the corresponding preflop observation and legal set;
3. replays that exact preflop action trace on alternate boards;
4. asserts observation/legal identity at every replayed node;
5. consumes one dummy RNG draw for each replayed preflop sample;
6. leaves postflop external sampling ordinary;
7. restores the canonical board-0 final RNG state afterward;
8. averages only preflop targets and keeps board-0 postflop samples.

Canonical K1 remains default and unchanged.

## Immediate gate

Rerun:

```bash
bash tools/run_lt2_hu_preflop_board_averaging_smoke.sh
```

Expected marker:

`LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_PASS`.

## Branch after smoke

Even on PASS, **do not start a K4 continuation**.

Next build a Stage-A -> Stage-B forensic audit that asks whether:

- the states in which Stage B moved away from Stage A are the same states with high noisy-target sign/policy instability;
- the change is concentrated in HU preflop/FACING_ALL_IN or elsewhere;
- K4 moves the target/reference direction toward the better Stage-A behavior in those same states.

Only if that causal link survives the audit should K4 be trained.

This explicitly prevents benchmark-specific overfitting to Jammer.

## Immediate action

Rerun the fixed smoke and stop at PASS or first error. Keep root training paused.
