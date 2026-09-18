# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED AT 4.5M — HU-JAMMER FAILURE CONFIRMED — TARGET-ESTIMATOR BUDGET SWEEP ACTIVE**
Date: 2026-09-17

## Current state

The continuous learning line has reached:

- LT0: 120k roots;
- LT1: 1.2M roots;
- LT2 Stage A: 1.8M roots;
- LT2 Stage B: 4.5M roots / iteration 7500;
- all four 2M reservoirs in replacement regime;
- Stage A -> Stage B policy movement is material;
- 30k weak-baseline gate confirms Stage B HU Jammer negative;
- AveragePolicy extra fitting is not helpful;
- larger Advantage optimizer budgets improve MSE only modestly and do not improve production-policy behavior;
- fixed-deal repeats show exact level 1 reduces opponent-action noise;
- exact same-input reservoir collisions are too sparse for inference;
- targeted HU-preflop conditional resampling shows **93.73% of sampled-target MSE is hidden/chance conditional variance**;
- next step is compute-normalized target-estimator selection, not more roots or a larger network.

Read first:

- `LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_RESULT_20260917.md`
- `LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_20260917.md`

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
- production behavior uses positive-regret matching and masked-softmax fallback when all legal predicted advantages are non-positive;
- production `exact_opponent_levels=0`;
- 31 root workers, one worker numerical thread, 8 parent Torch threads, vectorized batching, production concurrent-fit mode.

## Preserved milestones

LT1: 1.2M roots, SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.

LT2 Stage A: 1.8M roots / iteration 3000, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

LT2 Stage B: 4.5M roots / iteration 7500, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Keep all preserved milestones unchanged.

## Strength gate freezing root scaling

Stage B HU Jammer is `-5.141` chips/hand with simultaneous family-wise 95% CI `[-9.078,-1.204]`.

Stage B minus Stage A HU Jammer is `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

Policy movement without practical strength improvement is not enough to continue roots.

## Mechanism result

Targeted conditional resampling on 64 HU-preflop anchors produced:

- future-board variance: **65.88%**;
- opponent-hand posterior variance: **26.10%**;
- residual exact-level-1 opponent-action variance: **1.74%**;
- current-model MSE to conditional mean: **6.27%**.

Total hidden/chance conditional variance: **93.73%**.

This result is consistent across root, continuation-1, continuation-2+, and FACING_ALL_IN.

Thus the training target is dominated by information that is correctly absent from the observable information-set input. A deterministic model can only learn the conditional mean, not individual sampled realizations.

## Policy-space caveat

The current model still differs strongly from the high-budget conditional-mean policy:

- mean policy TV: `0.6903`;
- argmax agreement: `26.56%`;
- model regret under the conditional mean: `39.55` chips-equivalent.

So "model is only 6.27% of raw MSE" does not mean "model is good enough." Regret matching is sensitive to sign and relative errors.

The correct next intervention must reduce label variance and then measure policy-space improvement.

## Active target-estimator gate

The next read-only audit compares the compute frontier of two ways to spend target-generation nodes:

1. deeper opponent branching (`exact_opponent_levels=1`);
2. more independent opponent-hand/future-board realizations of the same observable HU-preflop state.

Design:

- 64 deterministic HU-preflop anchors;
- independent 64-deal exact-level-1 reference per anchor;
- reference split 32/32 to estimate its own Monte-Carlo floor;
- separate 64-deal candidate pool;
- exact0 and exact1 evaluated on the same candidate hidden deals;
- averages K = 1, 2, 4, 8, 16, 32, 64;
- actual nodes measured.

The key outputs are target MSE, regret-matching policy TV, argmax, branch mismatch, reference-target value gap/regret, and nodes.

## Decision branches

If exact0 plus more independent hidden deals dominates exact1 at matched node cost, implement chance/hidden-deal averaging first.

If exact1 remains superior at matched compute, retain exact branching in the estimator.

If averaging reduces target MSE but not policy TV/regret, the next bounded experiment should modify the Advantage learning objective toward sign/ranking/regret-policy alignment.

If a moderate K approaches the reference split-half floor, use that K in a bounded HU-preflop chance-averaged training pilot.

Every training intervention must beat preserved Stage B on the powered weak-baseline suite before root scaling resumes.

## DeepCrusher placement

DeepCrusher remains deferred until the weak-opponent curriculum is strong and stable.

## Immediate direction

1. Preserve Stage A and Stage B.
2. Keep root training stopped at iteration 7500.
3. Run `bash tools/run_lt2_hu_preflop_target_estimator_budget.sh`.
4. Review `SpinCore_LT2_hu_preflop_target_estimator_budget.json`.
5. Select one bounded causal training intervention from the measured compute frontier.
6. Do not scale roots or move to DeepCrusher yet.
