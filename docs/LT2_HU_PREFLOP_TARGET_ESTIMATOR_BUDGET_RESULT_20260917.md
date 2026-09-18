# SpinCore — LT2 HU preflop target-estimator budget result

Date: 2026-09-17
Status: **COMPLETE — EXACT0 + MORE INDEPENDENT HIDDEN DEALS IS THE COMPUTE-EFFICIENT FRONTIER; EXACT1 NOT ADMITTED**

## Source and integrity

Stage B checkpoint:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Run completed with:
- `LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_PASS`;
- source checkpoint unchanged;
- 64 HU-preflop anchors;
- independent 64-deal exact-level-1 reference per anchor;
- separate paired candidate pool;
- candidate exact levels 0 and 1 on the same hidden deals;
- K = 1,2,4,8,16,32,64;
- 0 training roots and 0 optimizer steps.

## Overall result

Reference split-half diagnostic:
- target MSE: `0.004453`;
- regret-matching policy TV: `0.4614`;
- argmax agreement: `43.75%`;
- branch mismatch: `18.75%`.

The reference itself is therefore still noisy in policy space. Treat it as a conditional Monte-Carlo reference, not an oracle or a strict policy floor.

Current Stage-B model against the 64-deal reference:
- target MSE: `0.001676`;
- policy TV: `0.6825`;
- argmax agreement: `31.25%`;
- branch mismatch: `39.06%`;
- candidate-policy regret to reference best action: `29.50` chips;
- signed reference-policy minus model-policy value gap: `+18.31` chips.

## Same-K exact0 versus exact1

Exact1 costs about 2.1x as many traversal nodes overall.

At K=64:
- exact0: 4,822 nodes, MSE `0.001178`, TV `0.3700`, regret `28.10`;
- exact1: 10,103 nodes, MSE `0.001145`, TV `0.3637`, regret `26.03`.

The tiny same-K gain from exact1 is bought at roughly double the node cost.

## Matched-compute comparison

Because exact1 costs about twice as much, the relevant comparison is approximately exact0 at 2K versus exact1 at K.

Paired 64-anchor target-MSE differences, exact0 minus exact1:

- exact0 K2 vs exact1 K1: `-0.01529`, 95% CI `[-0.01884,-0.01175]`;
- exact0 K4 vs exact1 K2: `-0.00883`, CI `[-0.01119,-0.00647]`;
- exact0 K8 vs exact1 K4: `-0.00363`, CI `[-0.00462,-0.00264]`;
- exact0 K16 vs exact1 K8: `-0.00188`, CI `[-0.00246,-0.00130]`;
- exact0 K32 vs exact1 K16: `-0.00138`, CI `[-0.00188,-0.00089]`;
- exact0 K64 vs exact1 K32: `-0.00144`, CI `[-0.00214,-0.00074]`.

Every matched-compute MSE comparison favors spending the nodes on **more independent hidden deals at exact0**, not on exact1 branching.

Policy-TV/regret paired differences at those matched budgets mostly have intervals crossing zero. Therefore the correct statement is:

**exact0 is decisively more compute-efficient in target MSE, while exact1 shows no reproducible policy-space advantage large enough to justify its node cost.**

Exact1 is not promoted to production training.

## Averaging curve at exact0

Overall exact0:

| K | nodes | MSE | policy TV | argmax | branch mismatch | regret chips |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 0.042729 | 0.4866 | 42.58% | 25.20% | 40.58 |
| 2 | 151 | 0.022730 | 0.4669 | 44.92% | 18.99% | 38.21 |
| 4 | 301 | 0.011604 | 0.4493 | 46.09% | 15.82% | 36.39 |
| 8 | 603 | 0.006810 | 0.4455 | 45.70% | 15.23% | 36.22 |
| 16 | 1,206 | 0.004374 | 0.4137 | 47.27% | 16.41% | 33.50 |
| 32 | 2,411 | 0.002720 | 0.3742 | 53.12% | 10.94% | 29.54 |
| 64 | 4,822 | 0.001178 | 0.3700 | 50.00% | 14.06% | 28.10 |

MSE falls strongly and approximately with averaging. Policy-space improvement is much slower and non-monotone.

This confirms both prior findings at once:
1. hidden/chance target variance is real and reducible;
2. raw MSE reduction does not translate proportionally into regret-matching policy stabilization.

## Region structure

Reference split-half policy TV remains high at:
- root: `0.5552`;
- one-action continuation: `0.5583`.

It is lower at:
- continuation-2+: `0.1738`;
- facing-all-in: `0.2470`.

Thus 64 hidden deals are not a precise policy oracle for the most important root/one-action strata.

## Facing-all-in result

There are 25 FACING_ALL_IN anchors.

For these states exact0 and exact1 are **identical at every K and at the same node cost**.

That is expected: once the opponent is already all-in there is no future opponent action for exact-level branching to integrate.

Consequently exact1 cannot directly solve the HU-Jammer failure in the very subset most directly tied to facing a jam.

FACING_ALL_IN:
- current model TV: `0.6115`;
- current branch mismatch: `84%`;
- exact0 K32 TV: `0.2609`;
- exact0 K64 TV: `0.2580`.

Chance/hidden-deal averaging is therefore the relevant variance lever there.

## Decision

Do not:
- promote exact1 to production training;
- resume long root scaling;
- enlarge the network;
- increase optimizer steps again;
- select K64 merely because it has the smallest MSE.

The next question is implementation-oriented.

Full independent hidden-deal averaging attacks both the 65.9% future-board component and the 26.1% opponent-hand component, but online opponent-hand posterior resampling is materially more complex than resampling future boards.

Future-board variance is the single largest component and future-board resampling is much easier to integrate into the current preflop traversal.

The next bounded gate therefore measures **board-only averaging at exact0**, keeping one sampled opponent hand fixed, against an independent full conditional reference.

It will determine how much of the full-deal benefit can be captured by the simpler production-feasible intervention before modifying training semantics.
