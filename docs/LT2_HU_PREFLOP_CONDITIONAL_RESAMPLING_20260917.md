# SpinCore — LT2 HU preflop conditional resampling audit

Date: 2026-09-17
Status: **ACTIVE — READ-ONLY TARGETED CONDITIONAL DECOMPOSITION**

## Why this audit exists

The complete Stage-B reservoir scan found exact same-input duplicates too sparse for representative inference. HU preflop has the best duplicate coverage, but even there only 0.2348% of stored items belong to a duplicate group and the largest group has size 3.

The confirmed practical failure is also HU-specific: Stage B is statistically negative against the Jammer. The next separator is therefore localized to HU preflop rather than expanding a global architecture experiment.

## Source

Stage B checkpoint:
- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

The checkpoint is read only and hash-checked before and after. No optimizer steps and no training-memory writes occur.

## Anchor design

Default HU preflop anchors:
- 16 preflop root states;
- 32 one-action continuation states;
- 16 two-or-more-action continuation states.

Anchors are generated from deterministic Stage-B Advantage-behavior trajectories. Continuation-1 is deliberately oversampled because one-action continuations contain the canonical situation of responding after one opponent action, including states whose immediately preceding action is ALL_IN.

Results also report the subset with `last_action_slot == ALL_IN`.

## Conditional hidden-state posterior

For each anchor:

1. Hero hole cards, public stacks, blind state, action history, actor, legal actions and exact SPNNIV1 observation are fixed.
2. All 2,450 ordered opponent two-card assignments consistent with the hero cards are enumerated.
3. For every candidate opponent hand, the public action path is replayed under an explicit deal.
4. The reach weight is the product of the current Stage-B opponent behavior probabilities for the opponent's observed actions in that path.
5. Hero-action probabilities are constant across candidate opponent hands because hero cards and public observation are fixed, so those factors cancel from the conditional posterior.
6. Opponent hands are selected with deterministic randomized stratification from the normalized exact enumerated posterior.

The audit records positive posterior support, effective sample size and maximum hand probability per state.

## Future chance and target repetitions

For each selected opponent hand:
- sample 4 independent ordered future boards uniformly from remaining cards;
- for each exact hidden deal, recompute the Advantage target 4 times;
- target traversal uses `exact_opponent_levels=1` to suppress most opponent-action sampling noise already identified by the previous audit.

Default design:
- 16 posterior opponent-hand strata/state;
- 4 boards/hand;
- 4 target repeats/deal;
- 64 anchors total.

This is 256 target evaluations per anchor and 16,384 target traversals total. It remains bounded and collects no training roots.

## Exact balanced decomposition

For the finite nested sample, on legal actions:

`sample-target MSE`
`= within-deal opponent-action variance`
`+ future-board variance within opponent hand`
`+ opponent-hand posterior variance`
`+ model MSE to the conditional mean target`.

The first three terms are conditional target variance that one deterministic prediction cannot fit for that exact observable input under this target process.

The final term is the current model's error to the estimated conditional mean target and is therefore the component most directly relevant to representation/capacity/optimization.

## Interpretation limits

The estimated conditional mean is not GTO and not realized poker EV. It is the mean of the current Stage-B target-generation process under the current Stage-B behavior posterior.

The equal-anchor aggregate is diagnostic and is not a natural-visitation EV estimate.

There is no arbitrary PASS score. The decision is based on measured component shares, their consistency across root/continuation strata, and the policy/value disagreement to the conditional mean.

## Decision branches

If opponent-hand plus future-board variance dominates, larger networks cannot solve the main target disagreement from the current observable state. Prioritize target estimators/variance reduction and information-set-correct training design.

If model error to the conditional mean dominates, representation/capacity/per-action calibration becomes a strong causal candidate. The HU FOLD versus CHECK_CALL/ALL_IN discrepancy should then be localized by continuation type, especially the FACING_ALL_IN subset.

If within-deal opponent-action variance remains unexpectedly large even at exact level 1, revisit exact branching depth before architecture changes.

Any intervention remains bounded and must later beat preserved Stage B on the statistically powered weak-baseline suite before long root scaling resumes.

## Launcher

```bash
bash tools/run_lt2_hu_preflop_conditional_resampling.sh
```

Expected marker:

`LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_PASS`
