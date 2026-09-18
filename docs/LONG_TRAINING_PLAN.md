# SpinCore — Long-Training Plan

Status: **LT2 STAGE B PASS — ROOT TRAINING PAUSED AT 4.5M — HU-PREFLOP BOARD-AVERAGING K4 ADMITTED — MECHANICS SMOKE ACTIVE**
Date: 2026-09-18

## Current state

The continuous learning line has reached:

- LT0: 120k roots;
- LT1: 1.2M roots;
- LT2 Stage A: 1.8M roots;
- LT2 Stage B: 4.5M roots / iteration 7500;
- powered weak-baseline gate confirms Stage B HU Jammer negative;
- extra AveragePolicy fitting is not helpful;
- larger Advantage optimizer budgets do not justify themselves;
- HU-preflop sampled-target error is dominated by hidden/chance variance;
- exact1 is not compute-efficient;
- board-only future-board averaging has a measured K4 policy-space elbow;
- the next step is a read-only implementation smoke, not long training.

Read first:

- `LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_RESULT_20260918.md`
- `LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_20260918.md`
- `LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_RESULT_20260917.md`

## Canonical training contract

The preserved Stage-B line remains:

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
- production `exact_opponent_levels=0`;
- canonical `hu_preflop_board_average_k=1`;
- 31 root workers, vectorized batching, concurrent-fit iteration mode.

The preserved checkpoints are never rewritten.

## Preserved milestones

LT1: 1.2M roots, SHA256 `beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337`.

LT2 Stage A: 1.8M roots / iteration 3000, SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

LT2 Stage B: 4.5M roots / iteration 7500, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

## Strength gate

Stage B HU Jammer is `-5.141` chips/hand with simultaneous family-wise 95% CI `[-9.078,-1.204]`.

Stage B minus Stage A HU Jammer is `-1.682`, simultaneous six-claim interval approximately `[-3.143,-0.222]`.

No blind root scaling is authorized.

## Target-variance mechanism

HU-preflop conditional decomposition:

- future-board variance: **65.88%**;
- opponent-hand posterior variance: **26.10%**;
- residual exact-level-1 opponent-action variance: **1.74%**;
- current-model MSE to conditional mean: **6.27%**.

Total hidden/chance conditional variance: **93.73%**.

## Closed exact-level branch

Exact1 costs about 2.1x as many nodes at same K.

Matched-compute exact0 with more independent hidden deals wins target MSE across every tested budget, while policy-space differences do not justify exact1.

In FACING_ALL_IN, exact0 and exact1 are identical.

Exact1 stays off.

## Board-only feasibility result

Board-only exact0 future-board averaging while fixing the sampled opponent hand:

| K | nodes | MSE | TV | argmax | branch mismatch | regret |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75.5 | 0.042638 | 0.5001 | 38.92% | 26.04% | 38.73 |
| 2 | 151.0 | 0.023177 | 0.4966 | 39.84% | 22.05% | 37.01 |
| 4 | 302.1 | 0.013556 | 0.4868 | 41.60% | 17.68% | 35.42 |
| 8 | 604.2 | 0.008715 | 0.4843 | 42.58% | 15.82% | 35.13 |

K4 vs K1 gives resolved improvements in:
- target MSE;
- regret;
- positive-regret branch mismatch.

K8 doubles K4 compute but adds no resolved TV/regret/argmax benefit.

**K4 is the first admitted causal training candidate.**

This is not yet an authorization to train.

## Candidate semantic

The opt-in parameter is `hu_preflop_board_average_k`.

Default `1` is canonical.

Candidate K4 must:
- affect TRUE_HEADS_UP only;
- preserve the canonical root's hole cards;
- retain canonical future board as board 0;
- add three independent future boards conditional on those holes;
- replay the same external-sampling RNG state across all four boards;
- average only preflop Advantage targets;
- retain postflop samples from canonical board 0;
- preserve sample count/order/observation/legal/weight/iteration;
- restore canonical-board RNG progression after averaging.

Thus the intervention targets the measured future-board noise without multiplying reservoir sample density or changing postflop labels.

## Active mechanics gate

Before any training continuation, compare identical prospective Stage-B HU roots at K1 and K4.

Launcher:

`tools/run_lt2_hu_preflop_board_averaging_smoke.sh`

The smoke performs:
- no training-memory writes;
- no optimizer steps;
- no checkpoint write.

Required pass conditions:
- same root/sample identity;
- postflop target equality;
- nonzero preflop target changes;
- K4 node cost > K1;
- preserved source SHA unchanged.

## Pilot sizing after smoke

Do not preselect the root count.

Use the measured K4/K1 node multiplier from the smoke to choose a bounded compute budget that is large enough to move the saturated HU policy memory but small enough to stop cheaply if the semantic fails.

The eventual candidate must be isolated from Stage B and must pass the same powered weak-baseline suite before long training can resume.

## DeepCrusher placement

DeepCrusher remains deferred until the weak-opponent curriculum is strong and stable.

## Immediate direction

1. Preserve Stage A and Stage B.
2. Keep long root training stopped at iteration 7500.
3. Run `bash tools/run_lt2_hu_preflop_board_averaging_smoke.sh`.
4. Review `SpinCore_LT2_hu_preflop_board_averaging_smoke.json`.
5. Size the bounded K4 causal training pilot from measured compute.
6. Do not move to DeepCrusher yet.
