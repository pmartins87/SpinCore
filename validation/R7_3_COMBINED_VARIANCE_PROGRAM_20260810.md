# R7.3 dual-variance convergence program — 2026-08-10

`READY FOR TABLES = NO`. Frozen gates remain unchanged:

- Advantage weighted normalized RMSE `<= 0.75`
- AveragePolicy weighted mean TV `<= 0.12`
- cross-seed mean TV `<= 0.15`
- cross-seed p95 TV `<= 0.35`

## 1. Current causal model

The physical generation-2 evidence now separates the R7.3 failure into two upstream mechanisms rather than one generic “seed variance” problem.

### A. CFR target / memory variance

Opponent external sampling changes the Advantage targets and target support observed by each seed. Own-reach sampling separately fragments AveragePolicy support. Exact bootstrap controls prove both effects before any neural feedback is present.

The best bounded intervention so far is partial opponent expectation:

- level 1, 256 roots / two iterations: mean/p95 `0.223853 / 0.807704`;
- level 2: `0.191695 / 0.669413`;
- authoritative level 0: `0.313641 / 0.882657`.

Level 2 therefore reduces mean by about 38.9% and p95 by about 24.2% at the short-screen scale, at about `9.35x` traversal-node cost.

At 640 roots / five iterations, deck-exact level 2 remains the strongest completed target-variance intervention:

- mean TV `0.381601`;
- p50 `0.327918`;
- p95 `0.852469`;
- both per-seed Advantage/Policy fit gates PASS.

It is materially better than corrected 640 (`0.477649 / 0.902403`) but still far from the frozen `0.15 / 0.35` gates.

Exhaustive opponent enumeration is closed as a practical path: at only 64 roots it required about `781x` the baseline tree work, saturated a 400k Advantage reservoir after roughly 1.68M samples seen per seed, and made final policy stability worse rather than better.

### B. Advantage function-approximation / regret-sign variance

A completely separate experiment froze one Advantage reservoir and trained four independent AdvantageNets on exactly the same samples.

All four reached weighted NRMSE `0.472–0.487`, but their production hard-regret policies still had:

- pairwise mean TV `0.224349`;
- pairwise p95 `0.757529`;
- identical positive-regret support on only `55.49%` of observations.

A 2x2 factorial showed both sources of fit randomness are material:

- same init / different minibatch order: mean TV `0.201320`;
- different init / same minibatch order: `0.221737`;
- both different: `0.224532`.

Thus R7.3 cannot be solved only by improving the traversal estimator. Even identical targets can be mapped by independent nonconvex regression fits into materially different positive-regret supports.

## 2. Five-iteration evidence joins the two mechanisms

Fresh regret-policy mean TV after each Advantage refit:

| iteration | baseline | Advantage x4 |
|---:|---:|---:|
| 1 | `0.476820` | `0.300327` |
| 2 | `0.578266` | `0.520984` |
| 3 | `0.562001` | `0.496018` |
| 4 | `0.553982` | `0.475241` |
| 5 | `0.538547` | `0.522414` |

p95 is `1.0` for both modes at every iteration.

Four external-sampling paths therefore remove a large amount of iteration-1 target noise, but the benefit falls from about 37% at iteration 1 to about 3% by iteration 5. The regression/sign-support mechanism is already severe at the first fitted AdvantageNet and later CFR feedback repeatedly reintroduces divergence.

Deck-exact coupled Advantage-x4 at 640 confirms that plain replication is not enough:

- mean `0.467601`;
- p95 `0.935642`;
- all individual fit gates PASS.

This definitively closes plain x4 as the primary acceptance solution.

## 3. Fit-stability interventions already screened

### Common fit randomness

Using common Advantage initialization and common per-iteration minibatch RNG across algorithm seeds produced only modest improvement. It does not solve different-target-memory approximation by itself.

### Raw Advantage ensembling on one frozen memory

Eight independent same-memory models were grouped into disjoint ensembles. Raw Advantage predictions were averaged before unchanged hard regret matching.

- single-model ensemble comparison mean TV: `0.222009`;
- 2-model: `0.171537`;
- 4-model: `0.110164`;
- 4-model / 1-model mean ratio: `0.4962`.

p95 fell from about `0.770524` to `0.487173`. The 4-model ensemble also improved effective weighted NRMSE into roughly the `0.40–0.42` range.

This is the strongest demonstrated treatment of **same-memory fit variance** so far.

End-to-end ensembling without target-variance reduction is much weaker: at 256 roots / two iterations, size 1/2/4 final cross-seed means are approximately `0.3248 / 0.3213 / 0.3136`, with p95 about `0.8733 / 0.8609 / 0.8592`. Thus ensembling alone cannot repair divergent CFR memories; it needs to be paired with a target-variance treatment.

### Behavior-aware auxiliary Advantage objective

On one frozen memory, adding a smooth regret-policy auxiliary term reduces cross-fit hard-regret disagreement but trades against raw NRMSE. A weight sweep found `0.10` is the strongest tested gate-safe value:

- pairwise mean TV `0.207362`;
- p95 `0.701394`;
- support equality `0.583008`;
- worst replica NRMSE `0.731640` — still inside the frozen `0.75` gate.

Weight `0.15` improves policy stability further but violates the Advantage fit gate, so it is not eligible.

## 4. Correlation strategies closed or deprioritized

- common opponent-path random numbers: only modest improvement;
- antithetic/quarter-turn x4: mean essentially flat and p95 worse;
- raw independent x8/x16: exact oracle shows diminishing returns after x4 and p95 remains saturated;
- full exact enumeration: excessive tree/memory cost and worse end-to-end behavior;
- more unique roots: 640 -> 1280 was essentially flat while policy fit degraded.

None remains on the primary path.

## 5. Active combined experiments

The project is no longer testing one tweak per interaction. The following independent questions are running in parallel.

### Partial exact level 2 + Advantage ensemble

`tools/run_r7_3_partial_exact_ensemble_e2e.py` combines the strongest bounded target-variance treatment with 1-, 2- and 4-model Advantage ensembles. It directly tests whether the two independently proven mechanisms are complementary end-to-end.

Promotion rule: use the smallest ensemble whose incremental stability benefit justifies its repeated neural-fit cost. A strong 256-root result would then be promoted to a deck-exact 640 acceptance candidate.

### Behavior-aware Advantage objective end-to-end

`tools/run_r7_3_advantage_objective_e2e.py` compares recovered weighted MSE against the gate-safe `MSE + 0.10 * policy_aux` objective while keeping traversal, hard regret matching and AveragePolicy semantics unchanged. This determines whether the same-memory improvement survives CFR feedback.

### Partial exact level 2 + common fit randomness

This combined screen completed with essentially no incremental value: common-versus-independent fit-randomness ratios were roughly `0.980` mean and `0.997` p95. It is closed as a primary combination.

### Ensemble mapping order

The same frozen models were compared under:

1. average raw Advantage, then hard regret matching;
2. hard regret match each member, then average policies.

Neither mapping decisively dominates: raw averaging is slightly better on mean disagreement, while policy-mixture averaging is slightly better in the p95 tail. This is a secondary design choice, not the primary remaining bottleneck.

### Exact weighted target aggregation

`tools/run_r7_3_advantage_target_aggregation.py` tests a particularly low-risk variance-control idea. Duplicate samples with identical `(observation, legal-mask)` are replaced by their LCFR-weighted mean target and summed weight.

For weighted squared error this transformation preserves the exact population minimizer:

`sum_i w_i ||f-y_i||^2 = W ||f-y_bar||^2 + constant`.

The experiment measures whether removing contradictory duplicate minibatch draws stabilizes independent fits while evaluating all models back on the original unaggregated reservoir.

### Behavior-aware multistart selection

`tools/run_r7_3_advantage_multistart_selection.py` trains eight independent models on one frozen memory. Two disjoint groups of four select the gate-passing model whose hard-regret policy is closest to the reservoir's own target-regret policy. This tests whether behavior-aware restart selection can capture much of the ensemble benefit without ensemble inference at runtime.

### Level-2 strong-fit 640

A separate deck-exact 640 level-2 run doubles the Advantage optimizer ceiling from 4096 to 8192 steps/iteration. This tests whether the level-2 acceptance result is being limited by neural fit capacity rather than the estimator itself.

## 6. Current engineering objective

The next useful SpinCore candidate must reduce **both** upstream components:

`total instability ≈ target/memory variance + Advantage fit/sign-support variance + downstream policy fit/generalization`.

The primary candidate family is therefore:

`bounded partial opponent expectation + a proven Advantage fit-stability mechanism`.

The least expensive fit-stability mechanism that survives end-to-end testing should be preferred. Current contenders are, in order of direct evidence:

1. raw-Advantage ensembling;
2. behavior-aware multistart selection;
3. exact duplicate-target aggregation;
4. gate-safe behavior-aware training objective.

Only after a combined 256-root experiment shows a strong effect should the corresponding combination be promoted to 640 acceptance scale.

## 7. Non-negotiable promotion requirements

Any algorithmic change that eventually clears R7.3 must still:

- preserve the frozen gates;
- receive an explicit versioned estimator/training/checkpoint contract;
- pass continuous versus stop/restore/continue deterministic recertification;
- pass the larger HU + 3H R7.4 program before production training;
- keep `READY FOR TABLES = NO` until all remaining roadmap gates are physically passed.
