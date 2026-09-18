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
- Repeated-state target variance — **COMPLETE**.
- Same-input reservoir audit — **COMPLETE; DUPLICATES TOO SPARSE**.
- HU-preflop conditional resampling — **COMPLETE; HIDDEN/CHANCE VARIANCE DOMINANT**.
- HU-preflop target-estimator budget sweep — **COMPLETE; EXACT0 + MORE DEALS IS COMPUTE FRONTIER**.
- HU-preflop board-only averaging — **NEXT**.
- Root training beyond iteration 7500 — **PAUSED**.
- DeepCrusher — **DEFERRED**.

Canonical current files:

- `CURRENT_WORK.md`
- `docs/LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_RESULT_20260917.md`
- `docs/LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_20260917.md`
- `docs/LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep both unchanged.

## Strength failure freezing scaling

Stage B HU Jammer remains confirmed negative. No more long roots until a bounded causal intervention beats preserved Stage B on the powered weak-baseline suite.

## Mechanism now established

HU preflop sampled-target MSE is dominated by hidden/chance variation:

- future board **65.88%**;
- opponent-hand posterior **26.10%**;
- exact-level-1 action noise **1.74%**;
- current-model conditional-mean error **6.27%**.

## Exact branching decision

The compute-normalized estimator sweep compared exact0/exact1 on paired candidate hidden deals.

Exact1 costs about 2.1x nodes at the same K.

At matched node budgets, exact0 with approximately twice as many independent hidden deals has lower target MSE at every tested budget, with paired 95% intervals excluding zero.

Policy-TV/regret differences at those matched budgets mostly remain unresolved, so exact1 has no demonstrated policy-space benefit sufficient to pay its cost.

**Exact1 is not promoted.**

## Jammer-specific structure

FACING_ALL_IN exact0 and exact1 are identical at every K because no future opponent action remains to exactify.

This makes chance averaging, not opponent-action branching, the relevant variance mechanism for the confirmed HU-Jammer weakness.

## Next implementation-feasibility gate

The next read-only audit tests future-board-only averaging at exact0 while keeping one sampled opponent hand fixed.

Why:

- future-board variance is the largest single component;
- future-board resampling is far easier to integrate into production training than posterior opponent-hand resampling;
- full hidden-deal averaging is useful diagnostically but operationally more invasive.

Launcher:

`tools/run_lt2_hu_preflop_board_only_averaging.sh`

Design:
- same 64 anchors;
- reference = 16 posterior hands × 4 boards, exact1;
- candidate = 16 separate posterior hands, each held fixed while 8 future boards are generated;
- K = 1,2,4,8 board averages at exact0;
- no roots and no optimizer steps.

## Branch after board-only audit

If K4/K8 captures most of the full-deal policy-space improvement, implement a bounded HU-preflop board-averaging training pilot.

If board-only leaves a large gap, opponent-hand posterior variation must be incorporated.

If MSE falls but TV/regret remains largely unchanged, prioritize a policy-aligned Advantage objective.

Only a candidate that later beats Stage B on the powered weak-baseline suite can reopen long root scaling.

## Immediate action

```bash
bash tools/run_lt2_hu_preflop_board_only_averaging.sh
```

Wait for `LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_PASS`. Keep root training paused.
