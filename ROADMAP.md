# SpinCore Roadmap — active state 2026-09-17

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
- Repeated-state target variance — **COMPLETE; EXACT LEVEL 1 REDUCES OPPONENT-ACTION NOISE**.
- Same-input reservoir audit — **COMPLETE; DUPLICATE COVERAGE TOO SPARSE**.
- HU-preflop conditional resampling — **COMPLETE; 93.73% HIDDEN/CHANCE VARIANCE**.
- HU-preflop target-estimator budget sweep — **NEXT**.
- Root training beyond iteration 7500 — **PAUSED**.
- DeepCrusher — **DEFERRED**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_RESULT_20260917.md`
- `docs/LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_20260917.md`
- `docs/LT2_REPEATED_TARGET_VARIANCE_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep both unchanged.

## Strength failure freezing scaling

Stage B HU Jammer: `-5.141` chips/hand, simultaneous family-wise 95% CI `[-9.078,-1.204]`.

Stage B minus Stage A HU Jammer: `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

No more roots until the learning mechanism changes and a bounded candidate beats the preserved checkpoint.

## Mechanism now established

The targeted HU-preflop conditional audit resolves the major ambiguity left by fixed-deal target repetitions.

Across 64 anchors:

- future-board variance: **65.88%** of sampled-target MSE;
- opponent-hand posterior variance: **26.10%**;
- residual exact-level-1 opponent-action variance: **1.74%**;
- current-model error to conditional mean: **6.27%**.

Total conditional hidden/chance variance: **93.73%**.

The same qualitative split holds at root, continuation-1, continuation-2+, and in the 25 FACING_ALL_IN anchors.

Therefore simply enlarging the network or adding optimizer steps is not the next justified experiment.

## But policy disagreement remains large

Current model against the high-budget conditional-mean target:

- mean regret-matching policy TV: `0.6903`;
- argmax agreement: `26.56%`;
- branch mismatch: `34.38%`;
- model regret under conditional mean: `39.55` chips-equivalent;
- signed conditional-mean-policy minus model-policy gap: `+26.58` chips.

FACING_ALL_IN:
- TV `0.6590`;
- branch mismatch `72%`;
- target FOLD/CHECK_CALL/ALL_IN mass `44.15/36.95/18.90%`;
- model mass `31.04/34.91/34.04%`.

This means raw target MSE and decision quality are not interchangeable.

## Immediate gate — compute-normalized target estimator

Launcher:

`tools/run_lt2_hu_preflop_target_estimator_budget.sh`

Design:

- same 64 HU-preflop anchors;
- 64 independent hidden-deal exact-level-1 samples form the reference mean;
- 32/32 split-half reference diagnostic;
- separate 64-deal candidate pool;
- exact0 and exact1 evaluated on the same candidate hidden deals;
- candidate averages K = 1, 2, 4, 8, 16, 32, 64;
- actual traversal nodes measured.

Metrics:
- target MSE to reference;
- policy TV;
- argmax;
- branch mismatch;
- reference-target value gap/regret;
- node cost.

## Branch after estimator sweep

If exact0 with more independent hidden deals gives better policy accuracy per node than exact1, build a bounded chance-averaged estimator pilot around that frontier.

If exact1 still wins at matched compute, retain deeper opponent branching.

If K reduces target MSE strongly while policy TV/regret remains high, switch the next training intervention toward sign/ranking/regret-policy-aligned supervision rather than plain MSE.

If a moderate K reaches near the reference split-half floor, use that budget in a small HU-preflop causal training pilot.

Only after a bounded candidate beats Stage B on the powered weak-baseline suite may root scaling resume.

## Immediate action

```bash
bash tools/run_lt2_hu_preflop_target_estimator_budget.sh
```

Wait for `LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_PASS`. Keep long training paused.
