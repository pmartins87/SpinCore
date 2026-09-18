# SpinCore — LT2 HU preflop board-only averaging result

Date: 2026-09-18
Status: **COMPLETE — BOARD-ONLY AVERAGING CAPTURES MOST OF THE PRACTICAL POLICY GAP; K4 IS THE COMPUTE ELBOW; MECHANICS SMOKE NEXT**

## Source and integrity

Stage B checkpoint:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Report:
- schema `SPINCORE_LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_V1`;
- 64 anchors = 16 root + 32 continuation-1 + 16 continuation-2+;
- read-only;
- 0 training roots;
- 0 optimizer steps;
- candidate exact-opponent level 0;
- one fixed sampled opponent hand per estimator;
- K = 1,2,4,8 future boards.

## Overall curve

Equal-anchor aggregate:

| K | nodes | target MSE | policy TV | argmax | branch mismatch | regret chips |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75.5 | 0.042638 | 0.5001 | 38.92% | 26.04% | 38.73 |
| 2 | 151.0 | 0.023177 | 0.4966 | 39.84% | 22.05% | 37.01 |
| 4 | 302.1 | 0.013556 | 0.4868 | 41.60% | 17.68% | 35.42 |
| 8 | 604.2 | 0.008715 | 0.4843 | 42.58% | 15.82% | 35.13 |

The Stage-B model against the same reference has:
- target MSE `0.001730`;
- policy TV `0.6852`;
- argmax agreement `34.38%`;
- branch mismatch `35.94%`;
- regret `27.56` chips.

The estimator itself is not a candidate playing policy. Its role is to supply lower-variance Advantage labels.

## Paired budget effects

Across the same 64 anchors, K4 minus K1:

- target MSE: `-0.02908`, 95% CI `[-0.03366,-0.02450]`;
- policy TV: `-0.01324`, CI `[-0.03359,+0.00711]`;
- regret: `-3.31` chips, CI `[-5.80,-0.82]`;
- argmax agreement: `+2.69 pp`, CI `[-0.30,+5.67]`;
- branch mismatch: `-8.36 pp`, CI `[-11.27,-5.45]`.

Thus K4 gives a large and resolved reduction in target MSE, a resolved reduction in regret, and a large resolved improvement in positive-regret branch classification. Mean TV improves only modestly and its paired interval crosses zero.

K8 minus K4:

- target MSE: `-0.00484`, 95% CI `[-0.00569,-0.00399]`;
- policy TV: `-0.00254`, CI `[-0.01643,+0.01134]`;
- regret: `-0.29` chips, CI `[-1.43,+0.85]`;
- argmax agreement: `+0.98 pp`, CI `[-1.19,+3.15]`;
- branch mismatch: `-1.86 pp`, CI `[-3.39,-0.32]`.

K8 therefore doubles target-generation nodes relative to K4 but adds no resolved improvement in TV, regret, or argmax. Only a small additional branch-mismatch improvement remains.

**K4 is the measured compute elbow.**

## Facing-all-in subset

There are 25 FACING_ALL_IN anchors.

| K | nodes | MSE | policy TV | argmax | branch mismatch | regret chips |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.6 | 0.060052 | 0.3615 | 53.53% | 30.72% | 49.49 |
| 2 | 7.2 | 0.033251 | 0.3399 | 55.25% | 27.94% | 47.94 |
| 4 | 14.4 | 0.019502 | 0.3262 | 56.75% | 26.75% | 46.73 |
| 8 | 28.8 | 0.012995 | 0.3232 | 57.00% | 27.00% | 46.41 |

K8 minus K1:
- TV `-0.03836`, 95% CI `[-0.06827,-0.00846]`;
- branch mismatch `-3.72 pp`, CI `[-7.23,-0.21]`;
- regret improvement is not resolved.

K4 -> K8 has no resolved policy-space gain in this subset.

This is particularly relevant because exact-opponent branching had zero benefit once the opponent was already all-in. Board chance averaging is the first measured estimator change that actually improves policy-space agreement on those states.

## Comparison with full hidden-deal averaging

The earlier full hidden-deal exact0 sweep gave:
- K4: MSE `0.011604`, TV `0.4493`, nodes `301.4`;
- K8: MSE `0.006810`, TV `0.4455`, nodes `602.8`.

Board-only:
- K4: MSE `0.013556`, TV `0.4868`, nodes `302.1`;
- K8: MSE `0.008715`, TV `0.4843`, nodes `604.2`.

These are separate independent diagnostic runs, so the cross-run differences are descriptive rather than paired.

At essentially identical node cost, board-only remains somewhat worse than full hidden-deal averaging, as expected because opponent-hand uncertainty remains. However, relative to the current-model TV around 0.68, board-only K4/K8 closes roughly 84-85% of the model-to-full-deal policy-TV gap while avoiding online posterior opponent-hand resampling.

That is enough to justify a bounded causal training experiment.

## Decision

Admit **HU preflop future-board averaging K4** as the first candidate training semantic change.

Do not use K8:
- it doubles K4 node cost;
- TV/regret/argmax do not show resolved additional improvement;
- K4 is the measured elbow.

Do not yet resume long training.

Before spending roots, the implementation must pass a mechanics smoke proving:
1. same deterministic HU roots at K1 and K4;
2. identical sample count/order/identity;
3. canonical-board postflop targets unchanged;
4. only preflop targets are averaged;
5. K4 changes at least one preflop target;
6. K4 node cost is measured;
7. Stage-B checkpoint remains unchanged.

Launcher:
`tools/run_lt2_hu_preflop_board_averaging_smoke.sh`.

Expected marker:
`LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_PASS`.

Only after that smoke is reviewed will the bounded training-pilot root budget be selected.
