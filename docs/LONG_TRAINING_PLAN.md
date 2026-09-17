# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED AT 4.5M — HU-JAMMER FAILURE CONFIRMED — REPEATED-STATE TARGET-VARIANCE AUDIT ACTIVE**
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
- AveragePolicy extra fitting does not improve held-out CE;
- larger Advantage budgets reduce MSE modestly but do not improve production-policy TV;
- 100k-per-domain target-value sensitivity shows large high-span sampled-target disagreement;
- immediate next experiment is a repeated-state target-variance decomposition, not more roots.

Read `LT2_ADVANTAGE_VALUE_SENSITIVITY_RESULT_20260917.md` and `LT2_REPEATED_TARGET_VARIANCE_AUDIT_20260917.md` first.

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
- production `exact_opponent_levels=0`;
- 31 root workers, one worker numerical thread, 8 parent Torch threads, vectorized batching, production concurrent-fit mode.

## Preserved milestones

LT1: 1.2M roots, SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.

LT2 Stage A: 1.8M roots / iteration 3000, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

LT2 Stage B: 4.5M roots / iteration 7500, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep all preserved milestones unchanged.

## Strength result that freezes root scaling

The statistically powered 30k weak-baseline gate established:

- Stage B HU Jammer `-5.141` chips/hand, simultaneous family-wise 95% CI `[-9.078,-1.204]`;
- Stage B minus Stage A HU Jammer `-1.682`, simultaneous six-claim interval about `[-3.143,-0.222]`.

More roots cannot be admitted merely because policy continues to move.

## Optimizer diagnostics

### AveragePolicy

Additional fitting from the stored Stage-B policy does not improve held-out cross-entropy. Increasing the 4000-step finalization budget is not supported as the current fix.

### Advantage

The corrected three-replicate V2 sweep showed:

3H, 100 -> 1600 steps:

- MSE `0.03194285 -> 0.03073128` (~3.79% lower);
- production-policy TV `0.604686 -> 0.602978` (flat);
- argmax `31.78% -> 37.18%`.

HU:

- MSE `0.04608811 -> 0.04447139` (~3.51% lower);
- TV `0.587335 -> 0.604104` (worse);
- argmax `28.64% -> 34.52%`.

Thus more fitting improves regression loss but not the behavior distribution reliably. Do not multiply the production Advantage budget by 16 from this evidence.

## 100k sampled-target value result

The next audit asked whether the ~0.6 TV is merely disagreement among near-equivalent actions. It is not.

Overall:

- 3H mean target span `398.3` chip-equivalent; model-policy regret to sampled-target best action `154.0`; signed target-policy-minus-model-policy sampled-target gap `+101.4`;
- HU mean span `494.8`; model-policy regret `180.0`; signed gap `+114.2`.

The `100+ chip` target-span bucket carries about `86%` of state weight in both domains and contributes approximately `98.6%` of the 3H signed gap and `99.4%` of HU. The `<1 chip` bucket contributes essentially zero value loss.

The all-nonpositive fallback is not the main net-loss source. Target-has-positive states dominate the positive sampled-target value gap.

The model-induced Advantage behavior is strongly less fold-heavy than the target-induced sampled policy, especially HU:

- overall HU FOLD `6.6%` model vs `36.5%` target-induced;
- CHECK_CALL `40.1%` vs `28.5%`;
- ALL_IN `41.2%` vs `22.3%`;
- in HU target-has-positive states, FOLD `7.0%` model vs `47.7%` target-induced.

This is directionally compatible with the HU-Jammer weakness, but the target-induced policy is not a gold-standard policy. Stored Advantage targets come from Monte-Carlo external sampling, and final benchmark inference uses AveragePolicy.

## Why target variance is now the key causal fork

At a traverser node the collector evaluates every legal hero action and sets each target to `action_value - current_policy_node_value`. Under production `exact_opponent_levels=0`, opponent decisions below those branches are sampled. Consequently one exact information state/deal can produce different target vectors across repeated traversals.

The 100k audit measures disagreement against one sampled target at a time. It cannot tell whether poor fit is dominated by:

- external-sampling target variance;
- model approximation/capacity;
- SPNNIV1 representation/generalization;
- or some mixture.

Changing architecture or root count before this decomposition would be premature.

## Immediate repeated-state audit

Launcher: `tools/run_lt2_repeated_target_variance_audit.sh`.

Default design:

- Stage B source checkpoint SHA-checked and preserved;
- no optimizer steps and no training-memory writes;
- diagnostic trajectories use stored Stage-B Advantage behavior;
- select first decision reached on each street;
- 64 selected states per domain per street;
- 8 repeated Advantage-target traversals from the exact same state/deal;
- evaluate the same states at `exact_opponent_levels=0` and `1`.

For repeated sampled targets `T_r`, repeat mean `T_bar`, and model prediction `P`, the legal-action decomposition is exact:

`mean_r ||P-T_r||^2 = mean_r ||T_r-T_bar||^2 + ||P-T_bar||^2`.

The first term is directly observed opponent-action external-sampling noise for the fixed state/deal. The second is model error to a lower-noise local target estimate. The repeat mean is diagnostic only, not a GTO oracle.

Because the hidden deal/future board is fixed inside repeats, this measured noise excludes across-deal chance/hidden-card variance and is therefore a lower bound on total conditional target variance.

The audit also reports policy/value disagreement against the repeat mean and the node-cost multiplier from exact level 1.

## Decision branches

If within-repeat noise explains a large portion of sampled-target MSE and exact level 1 substantially reduces it at acceptable node cost, build only a small isolated exact-level candidate and compare it against preserved Stage B before any long run.

If model error to repeat-mean targets dominates, move to SPNNIV1 representation/capacity and per-action sign calibration, especially the HU FOLD versus CHECK_CALL/ALL_IN imbalance.

If opponent-action noise is high but exact level 1 barely helps, investigate across-deal chance/hidden-state variance and representation generalization rather than increasing exact branching blindly.

If effects differ by street/domain, keep intervention localized.

Any candidate must show a statistically powered practical improvement against the existing weak-baseline suite before root scaling resumes.

## DeepCrusher placement

DeepCrusher remains deferred until the weak-opponent curriculum is strong and stable. It is not a current dependency.

## Immediate direction

1. Preserve Stage A and Stage B.
2. Keep root training stopped at iteration 7500.
3. Run `bash tools/run_lt2_repeated_target_variance_audit.sh`.
4. Review `SpinCore_LT2_repeated_target_variance.json`.
5. Choose the next isolated intervention from the measured noise-versus-approximation decomposition.
6. Do not scale roots or move to DeepCrusher yet.
