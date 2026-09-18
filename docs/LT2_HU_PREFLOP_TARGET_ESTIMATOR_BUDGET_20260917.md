# SpinCore — LT2 HU preflop target-estimator budget sweep

Date: 2026-09-17
Status: **ACTIVE — READ-ONLY COMPUTE-NORMALIZED TARGET-ESTIMATOR GATE**

## Trigger

The completed HU-preflop conditional decomposition found:

- 93.73% conditional hidden/chance variance;
- 65.88% future-board variance;
- 26.10% opponent-hand posterior variance;
- 1.74% residual exact-level-1 opponent-action variance;
- 6.27% current-model error to the conditional mean.

The current model can therefore be much closer in raw MSE to the conditional mean than to individual sampled targets while still disagreeing strongly in regret-matching policy space.

The next question is not "bigger network or more optimizer steps?" It is:

**For a fixed node budget, should target-generation compute be spent on deeper exact opponent branching or on averaging more independent hidden-hand/future-board realizations of the same information state?**

## Source

Stage B checkpoint:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Read only. No roots and no optimizer steps.

## Anchors

Same deterministic HU-preflop design as the prior conditional audit:

- 16 root;
- 32 one-action continuation;
- 16 two-or-more-action continuation.

The FACING_ALL_IN subset is reported separately.

## Conditional posterior

For each anchor all 2,450 ordered opponent hands consistent with hero cards are enumerated and weighted by the current Stage-B opponent behavior reach probability of the observed public path.

## Independent reference

For every anchor:

- 64 independent posterior hidden-hand/future-board deals;
- exact opponent level 1;
- one target traversal per hidden deal;
- mean target becomes the high-budget conditional reference.

The 64 reference samples are split into 32/32 halves. Their target MSE and regret-matching policy TV provide a direct estimate of the reference Monte-Carlo floor.

The reference remains a current-target-process estimate, not a GTO oracle.

## Candidate pools

A separate independent pool of 64 hidden-hand/future-board deals is generated per anchor.

The same candidate hidden deals are evaluated at:

- exact opponent level 0;
- exact opponent level 1.

Candidate target averages use non-overlapping blocks of:

`K = 1, 2, 4, 8, 16, 32, 64`.

Metrics are first averaged within each anchor and then across anchors with equal anchor weight.

## Decision metrics

For every exact level and K:

- target MSE to the independent reference mean;
- regret-matching policy TV to reference;
- argmax agreement;
- positive-regret branch mismatch;
- signed reference-policy minus candidate-policy value gap;
- candidate policy regret to the best action under the reference target;
- actual traversal nodes.

The current Stage-B Advantage model is scored against the same reference for context.

## Interpretation

Raw MSE alone is not sufficient because the prior audit found mean policy TV around 0.69 despite model conditional-mean MSE representing only 6.27% of sampled-target MSE.

The preferred estimator is the compute-efficient policy-stability frontier, not automatically the lowest-variance or deepest-exact option.

Expected qualitative branches:

- If exact0 with more independent hidden deals dominates exact1 at matched node cost, prioritize chance/hidden-deal averaging rather than deeper opponent branching.
- If exact1 remains better at matched compute, retain exact branching as part of the estimator.
- If K materially reduces target MSE but policy TV/regret remains stubbornly high, the next intervention should move from raw advantage MSE toward sign/ranking/regret-policy-aligned learning.
- If K around 16 approaches the reference split-half floor in policy space, a bounded chance-averaged HU-preflop training pilot becomes justified.

No long root training is authorized by this diagnostic alone.

## Launcher

```bash
bash tools/run_lt2_hu_preflop_target_estimator_budget.sh
```

Expected marker:

`LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_PASS`
