# SpinCore Current Work

Date: 2026-09-17
Status: **LT2 STAGE B PASS — 4.5M ROOTS — HU-JAMMER NEGATIVE — HIDDEN/CHANCE TARGET VARIANCE DOMINANT — TARGET-ESTIMATOR BUDGET SWEEP NEXT**

## Active source of truth

Read before new compute:

- `docs/LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_RESULT_20260917.md`
- `docs/LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_20260917.md`
- `docs/LT2_REPEATED_TARGET_VARIANCE_RESULT_20260917.md`
- `docs/LT2_WEAK_BASELINE_VARIANCE_RESULT_20260917.md`
- `docs/LONG_TRAINING_PLAN.md`

Preserve Stage A and Stage B. Do not continue root training beyond iteration 7500 until the target-estimator compute frontier is reviewed.

## Preserved checkpoints

Stage A: iteration 3000 / 1.8M roots, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B: iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Confirmed practical failure

Powered 30k weak-baseline gate:

- Stage B HU Jammer raw chip EV `-5.141`, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage-B-minus-Stage-A HU-Jammer paired delta `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Root scaling remains paused.

## Closed optimizer findings

AveragePolicy: extra fitting does not improve held-out CE.

Advantage: 100 -> 1600 steps reduces held-out MSE only modestly and does not reproducibly improve production-policy TV.

A larger network is not the next admitted intervention.

## Conditional target decomposition — resolved

Targeted HU-preflop resampling completed on 64 anchors with 16,384 target traversals.

Equal-anchor decomposition:

- future-board variance: **65.88%**;
- opponent-hand posterior variance: **26.10%**;
- residual exact-level-1 opponent-action variance: **1.74%**;
- current-model MSE to conditional mean: **6.27%**.

Total hidden/chance conditional variance: **93.73%**.

Mean conditional-variance component `0.036256 [0.029358,0.043154]` versus model-to-conditional-mean component `0.002426 [0.001798,0.003055]`.

The mechanism is consistent across root, continuation-1, continuation-2+, and the 25 FACING_ALL_IN anchors.

## Important policy-space finding

Raw MSE understates the operational disagreement.

Across all 64 anchors:

- mean model vs conditional-mean regret-matching policy TV: `0.6903`;
- argmax agreement: `26.56%`;
- model regret under the conditional-mean target: `39.55` chip-equivalent;
- signed conditional-mean-policy minus model-policy value gap: `+26.58` chips.

FACING_ALL_IN:

- TV `0.6590`;
- branch mismatch `72%`;
- target action mass FOLD/CHECK_CALL/ALL_IN `44.15/36.95/18.90%`;
- model action mass `31.04/34.91/34.04%`.

Thus target noise is the dominant raw-loss mechanism, while the regret-matching transform remains highly sensitive to the residual model error.

## Immediate bounded gate — target-estimator budget sweep

Run:

```bash
bash tools/run_lt2_hu_preflop_target_estimator_budget.sh
```

Design:

- same 64 deterministic HU-preflop anchors;
- independent 64-deal exact-level-1 conditional reference per anchor;
- split-half reference diagnostic;
- separate paired 64-deal candidate pool;
- same candidate hidden deals evaluated at exact levels 0 and 1;
- target averages at K = 1, 2, 4, 8, 16, 32, 64 independent hidden deals;
- actual node cost measured;
- no training roots and no optimizer steps.

Primary comparison is policy-space quality per node:
- target MSE to reference;
- regret-matching policy TV;
- argmax;
- branch mismatch;
- reference-value gap/regret;
- node cost.

## Decision after target-estimator sweep

- If exact0 plus more independent hidden deals dominates exact1 at matched compute, prioritize chance/hidden-deal averaging.
- If exact1 remains superior at matched node budget, keep deeper opponent branching in the estimator.
- If averaging lowers MSE but policy TV/regret stays high, move next to sign/ranking/regret-policy-aligned learning rather than raw MSE.
- If a moderate K approaches the reference split-half floor in policy space, build a bounded HU-preflop chance-averaged training pilot.
- Any training candidate must later beat preserved Stage B on the powered weak-baseline suite before root scaling resumes.

DeepCrusher remains deferred.

## Immediate user action

Pull current `main` and run `bash tools/run_lt2_hu_preflop_target_estimator_budget.sh`. Wait for `LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_PASS`, then send `SpinCore_LT2_hu_preflop_target_estimator_budget.json`. Do not resume root training first.
